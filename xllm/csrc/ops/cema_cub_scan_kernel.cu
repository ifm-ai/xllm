#include <ATen/AccumulateType.h>
#include <ATen/DeviceGuard.h>
#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/MaybeOwned.h>
#include <c10/util/complex.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <type_traits>

#include "blas.h"
#include "complex_utils.cuh"
#include "cuda_utils.cuh"
#include "ops/cema_cub_scan.h"
#include "reduce.cuh"
#include "utils.h"

namespace xllm {
namespace ops {
namespace {
constexpr int64_t kMaxNumThreads = 256;
constexpr int N_MAX = 16;
constexpr int N_TILE = 4;

constexpr int64_t kNumThreadsHi = 64;
constexpr int64_t kNumThreadsMi = 64;
constexpr int64_t kNumThreadsLo = 64;

constexpr int64_t getChunkNum(int64_t B, int64_t N) {
  if (B <= 2) return kNumThreadsHi;
  if (B <= 4) return kNumThreadsMi;
  return kNumThreadsLo;
}

struct ComplexRightAffineProd {
    __device__ __forceinline__ float4 identity() const {
        return make_float4(1.f, 0.f, 0.f, 0.f);
    }
    __device__ __forceinline__ float4 operator()(const float4 &a, const float4 &b) const {
        float first_real = fmaf(-a.y, b.y, a.x * b.x);
        float first_imag = fmaf(a.y, b.x, a.x * b.y);
        float second_real = fmaf(b.x, a.z, fmaf(-b.y, a.w, b.z));
        float second_imag = fmaf(b.x, a.w, fmaf(b.y, a.z, b.w));

        return make_float4(first_real, first_imag, second_real, second_imag);
    }
};

struct ComplexLeftAffineProd {
    __device__ __forceinline__ float4 identity() const {
        return make_float4(1.f, 0.f, 0.f, 0.f);
    }
    __device__ __forceinline__ float4 operator()(const float4 &a, const float4 &b) const {
        float first_real = fmaf(-a.y, b.y, a.x * b.x);
        float first_imag = fmaf(a.y, b.x, a.x * b.y);
        float second_real = fmaf(a.x, b.z, fmaf(-a.y, b.w, a.z));
        float second_imag = fmaf(a.x, b.w, fmaf(a.y, b.z, a.w));

        return make_float4(first_real, first_imag, second_real, second_imag);
    }
};

struct ReversedComplexLeftAffineProd {
    __device__ __forceinline__ float4 operator()(const float4& a, const float4& b) const {
        return ComplexLeftAffineProd{}(b, a);
    }
};

template <typename T, typename T_ACC, int64_t CHUNK=256>
__global__ void CEMACubFwdKernel(
        int64_t B, int64_t D, int64_t N, int64_t L,
        const T* __restrict__ x,
        const T_ACC* __restrict__ p,
        const c10::complex<T_ACC>* __restrict__ q,
        const c10::complex<T_ACC>* __restrict__ gamma,
        const bool* __restrict__ bos_mask,
        const c10::complex<T_ACC>* __restrict__ h0,
        c10::complex<T_ACC>* __restrict__ h,
        T* __restrict__ y,
        c10::complex<T_ACC>* __restrict__ chunk_decay,
        c10::complex<T_ACC>* __restrict__ chunk_gain) {
    const int64_t d = blockIdx.x;
    const int64_t b = blockIdx.y;
    const int64_t tid = threadIdx.x;

    if (d >= D || b >= B || tid >= CHUNK) return;
    const int64_t num_chunks = (L + CHUNK - 1) / CHUNK;

    ComplexRightAffineProd op;
    const float4 identity = op.identity();

    extern __shared__ float4 fwd_shm[];
    float4* prev_shm = fwd_shm;

    if (tid == 0) {
        for (int n = 0; n < N; ++n) {
            c10::complex<T_ACC> h0_v = h0 ? h0[(b * D + d) * N + n] : c10::complex<T_ACC>(0);
            prev_shm[n] = make_float4(1.f, 0.f, h0_v.real(), h0_v.imag());
        }
    }
    __syncthreads();

    using BlockScan  = cub::BlockScan<float4, CHUNK, cub::BLOCK_SCAN_WARP_SCANS>;
    __shared__ typename BlockScan::TempStorage scan_smem;

    struct PrefixCB {
        float4 running;
        ComplexRightAffineProd op;
        __device__ PrefixCB(float4 init) : running(init) {}
        __device__ float4 operator()(float4 block_total) {
            float4 old = running;
            running = op(running, block_total);
            return old;
        }
    };

    for (int64_t c = 0; c < num_chunks; ++c) {
        int64_t t = c * CHUNK + tid;
        bool valid = (t < L);

        T_ACC y_contrib = T_ACC(0);
        float x_v = 0.f, mask = 1.f;
        if (valid) {
            x_v = static_cast<float>(x[(b * D + d) * L + t]);
            bool bos = bos_mask ? bos_mask[b * L + t] : false;
            mask = bos ? 0.f : 1.f;
        }

        for (int64_t n = 0; n < N; ++n) {
            const float p_v = static_cast<float>(p[d * N + n]);
            const c10::complex<T_ACC>  q_v = static_cast<c10::complex<float>>(q[d * N + n]);
            const c10::complex<T_ACC>  g_v = static_cast<c10::complex<float>>(gamma[d * N + n]);

            float4 v = identity;
            if (valid) {
                v = make_float4(mask * q_v.real(),
                                mask * q_v.imag(),
                                p_v * x_v,
                                0.f);
            }

            float4 prev = prev_shm[n];

            if (tid == 0) {
                int64_t idx = (((b * D + d) * N + n) * num_chunks + c);
                chunk_decay[idx] = {prev.x, prev.y};
                chunk_gain[idx] = {prev.z, prev.w};
            }

            PrefixCB pcb(prev);
            float4 prefix;
            BlockScan(scan_smem).ExclusiveScan(v, prefix, op, pcb);

            if (valid) {
                float4 local_h  = op(prefix, v);

                y_contrib = fmaf(local_h.z, static_cast<float>(g_v.real()), y_contrib);
                y_contrib = fmaf(-local_h.w, static_cast<float>(g_v.imag()), y_contrib);

                if (t == L - 1) h[(b * D + d) * N + n] = {local_h.z, local_h.w};
            }

            __syncthreads();
            if (tid == 0) prev_shm[n] = pcb.running;
            __syncthreads();
        }
        if (valid) y[(b * D + d) * L + t] = static_cast<T>(y_contrib);
    }
}

template <typename T, typename T_ACC, int64_t B, int64_t N, int64_t CHUNK=256, int64_t U = 4>
__global__ void CEMACubFwdKernel(
        int64_t D, int64_t L,
        const T* __restrict__ x,
        const T_ACC* __restrict__ p,
        const c10::complex<T_ACC>* __restrict__ q,
        const c10::complex<T_ACC>* __restrict__ gamma,
        const bool* __restrict__ bos_mask,
        const c10::complex<T_ACC>* __restrict__ h0,
        c10::complex<T_ACC>* __restrict__ h,
        T* __restrict__ y,
        c10::complex<T_ACC>* __restrict__ chunk_decay,
        c10::complex<T_ACC>* __restrict__ chunk_gain) {
    const int64_t d = blockIdx.x;
    const int64_t b = blockIdx.y;
    const int64_t tid = threadIdx.x;

    if (d >= D || b >= B || tid >= CHUNK) return;
    const int64_t num_chunks = (L + CHUNK - 1) / CHUNK;

    ComplexRightAffineProd op;
    const float4 identity = op.identity();

    extern __shared__ float4 fwd_shm[];
    float4* prev_shm = fwd_shm;

    if (tid == 0) {
        #pragma unroll
        for (int n = 0; n < N; ++n) {
            c10::complex<T_ACC> h0_v = h0 ? h0[(b * D + d) * N + n] : c10::complex<T_ACC>(0);
            prev_shm[n] = make_float4(1.f, 0.f, h0_v.real(), h0_v.imag());
        }
    }
    __syncthreads();

    using BlockScan  = cub::BlockScan<float4, CHUNK, cub::BLOCK_SCAN_WARP_SCANS>;
    __shared__ typename BlockScan::TempStorage scan_smem;

    struct PrefixCB {
        float4 running;
        ComplexRightAffineProd op;
        __device__ PrefixCB(float4 init) : running(init) {}
        __device__ float4 operator()(float4 block_total) {
            float4 old = running;
            running = op(running, block_total);
            return old;
        }
    };

    __shared__ float p_shared[N_MAX];
    __shared__ float2 q_shared[N_MAX];
    __shared__ float2 gamma_shared[N_MAX];

    if (threadIdx.x < N) {
        int64_t n = threadIdx.x;
        p_shared[n] = static_cast<float>(p[d * N + n]);
        c10::complex<float> qv = static_cast<c10::complex<float>>(q[d * N + n]);
        c10::complex<float> gamma_v = static_cast<c10::complex<float>>(gamma[d * N + n]);
        q_shared[n] = make_float2(qv.real(), qv.imag());
        gamma_shared[n] = make_float2(gamma_v.real(), gamma_v.imag());
    }
    __syncthreads();

    for (int64_t c = 0; c < num_chunks; ++c) {
        int64_t t = c * CHUNK + tid;
        bool valid = (t < L);
        T_ACC y_contrib = T_ACC(0);
        float x_v = 0.f, mask = 1.f;
        if (valid) {
            x_v = static_cast<float>(x[(b * D + d) * L + t]);
            bool bos = bos_mask ? bos_mask[b * L + t] : false;
            mask = bos ? 0.f : 1.f;
        }

        #pragma unroll U
        for (int64_t n = 0; n < N; ++n) {
            float4 v = identity;
            if (valid) {
                v = make_float4(mask * q_shared[n].x,
                                mask * q_shared[n].y,
                                p_shared[n] * x_v,
                                0.f);
            }

            float4 prev = prev_shm[n];

            if (tid == 0) {
                int64_t idx = (((b * D + d) * N + n) * num_chunks + c);
                chunk_decay[idx] = {prev.x, prev.y};
                chunk_gain[idx] = {prev.z, prev.w};
            }

            PrefixCB pcb(prev);
            float4 prefix;
            BlockScan(scan_smem).ExclusiveScan(v, prefix, op, pcb);

            if (valid) {
                float4 local_h  = op(prefix, v);

                y_contrib = fmaf(local_h.z, static_cast<float>(gamma_shared[n].x), y_contrib);
                y_contrib = fmaf(-local_h.w, static_cast<float>(gamma_shared[n].y), y_contrib);

                if (t == L - 1) h[(b * D + d) * N + n] = {local_h.z, local_h.w};
            }

            __syncthreads();
            if (tid == 0) prev_shm[n] = pcb.running;
            __syncthreads();
        }

        if (valid) y[(b * D + d) * L + t] = static_cast<T>(y_contrib);
    }
}

template <typename T, typename T_ACC, int64_t CHUNK=256>
__global__ void CEMACubFwdRecalcKernel(
        int64_t B, int64_t D, int64_t N, int64_t L,
        const T* __restrict__ x,
        const T_ACC* __restrict__ p,
        const c10::complex<T_ACC>* __restrict__ q,
        const c10::complex<T_ACC>* __restrict__ gamma,
        const bool* __restrict__ bos_mask,
        const c10::complex<T_ACC>* __restrict__ chunk_decay,
        const c10::complex<T_ACC>* __restrict__ chunk_gain,
        c10::complex<T_ACC>* __restrict__ h,
        T* __restrict__ y) {
    const int64_t d = blockIdx.x;
    const int64_t c = blockIdx.y;
    const int64_t tid = threadIdx.x;

    if (d >= D || tid >= CHUNK) return;
    const int64_t num_chunks = (L + CHUNK - 1) / CHUNK;
    if (c >= num_chunks) return;

    ComplexRightAffineProd op;
    const float4 identity = op.identity();

    using BlockScan  = cub::BlockScan<float4, CHUNK, cub::BLOCK_SCAN_WARP_SCANS>;
    __shared__ typename BlockScan::TempStorage scan_smem;

    const int64_t t = c * CHUNK + tid;
    const bool valid = (t < L);

    for (int64_t b = 0; b < B; ++b) {
        T_ACC y_contrib = T_ACC(0);
        float x_v = 0.f, mask = 1.f;
        if (valid) {
            x_v = static_cast<float>(x[(b * D + d) * L + t]);
            bool bos = bos_mask ? bos_mask[b * L + t] : false;
            mask = bos ? 0.f : 1.f;
        }

        for (int64_t n = 0; n < N; ++n) {
            const int64_t idx = (((b * D + d) * N + n) * num_chunks + c);

            const c10::complex<T_ACC> decay = chunk_decay[idx];
            const c10::complex<T_ACC> gain = chunk_gain[idx];
            float4 prev = make_float4(
                static_cast<float>(decay.real()),  static_cast<float>(decay.imag()),
                static_cast<float>(gain.real()),  static_cast<float>(gain.imag()));

            float4 v = identity;
            if (valid) {
                const float p_v = static_cast<float>(p[d * N + n]);
                const c10::complex<T_ACC> q_v =
                    static_cast<c10::complex<float>>(q[d * N + n]);

                v = make_float4(mask * static_cast<float>(q_v.real()),
                                mask * static_cast<float>(q_v.imag()),
                                p_v * x_v,
                                0.f);
            }

            struct PrefixCB {
                float4 running;
                ComplexRightAffineProd op;
                __device__ PrefixCB(float4 init) : running(init) {}
                __device__ float4 operator()(float4 block_total) {
                    float4 old = running;
                    running = op(running, block_total);
                    return old;
                }
            } pcb(prev);

            float4 prefix;
            BlockScan(scan_smem).ExclusiveScan(v, prefix, op, pcb);

            if (valid) {
                const float4 local_h = op(prefix, v);
                const c10::complex<T_ACC> g_v =
                    static_cast<c10::complex<float>>(gamma[d * N + n]);

                y_contrib = fmaf(local_h.z, static_cast<float>(g_v.real()), y_contrib);
                y_contrib = fmaf(-local_h.w, static_cast<float>(g_v.imag()), y_contrib);

                if (t == L - 1) {
                    h[(b * D + d) * N + n] = {static_cast<T_ACC>(local_h.z),
                                              static_cast<T_ACC>(local_h.w)};
                }
            }
            __syncthreads();
        }

        if (valid) y[(b * D + d) * L + t] = static_cast<T>(y_contrib);
    }
}

template <typename T, typename T_ACC, int64_t B, int64_t N, int64_t CHUNK=256>
__global__ void CEMACubFwdRecalcKernel(
        int64_t D, int64_t L,
        const T* __restrict__ x,
        const T_ACC* __restrict__ p,
        const c10::complex<T_ACC>* __restrict__ q,
        const c10::complex<T_ACC>* __restrict__ gamma,
        const bool* __restrict__ bos_mask,
        const c10::complex<T_ACC>* __restrict__ chunk_decay,
        const c10::complex<T_ACC>* __restrict__ chunk_gain,
        c10::complex<T_ACC>* __restrict__ h,
        T* __restrict__ y) {
    const int64_t d   = blockIdx.x;
    const int64_t c   = blockIdx.y;
    const int64_t tid = threadIdx.x;

    if (d >= D || tid >= CHUNK) return;
    const int64_t num_chunks = (L + CHUNK - 1) / CHUNK;
    if (c >= num_chunks) return;

    ComplexRightAffineProd op;
    const float4 identity = op.identity();

    using BlockScan  = cub::BlockScan<float4, CHUNK, cub::BLOCK_SCAN_WARP_SCANS>;
    __shared__ typename BlockScan::TempStorage scan_smem;

    const int64_t t = c * CHUNK + tid;
    const bool valid = (t < L);

    __shared__ float p_shared[N_MAX];
    __shared__ float2 q_shared[N_MAX];
    __shared__ float2 gamma_shared[N_MAX];
    __shared__ float2 decay_shared[N_MAX];
    __shared__ float2 gain_shared[N_MAX];

    if (threadIdx.x < N) {
        int64_t n = threadIdx.x;
        p_shared[n] = static_cast<float>(p[d * N + n]);
        c10::complex<float> qv = static_cast<c10::complex<float>>(q[d * N + n]);
        c10::complex<float> gamma_v = static_cast<c10::complex<float>>(gamma[d * N + n]);
        q_shared[n] = make_float2(qv.real(), qv.imag());
        gamma_shared[n] = make_float2(gamma_v.real(), gamma_v.imag());
    }
    __syncthreads();

    #pragma unroll
    for (int64_t b = 0; b < B; ++b) {
        T_ACC y_contrib = T_ACC(0);
        float x_v = 0.f, mask = 1.f;
        if (valid) {
            x_v = static_cast<float>(x[(b * D + d) * L + t]);
            bool bos = bos_mask ? bos_mask[b * L + t] : false;
            mask = bos ? 0.f : 1.f;
        }
        if (threadIdx.x < N) {
            int n = threadIdx.x;
            int64_t idx = (((b * D + d) * N + n) * num_chunks + c);
            c10::complex<T_ACC> decay = chunk_decay[idx];
            c10::complex<T_ACC> gain = chunk_gain[idx];
            decay_shared[n] = make_float2(static_cast<float>(decay.real()), static_cast<float>(decay.imag()));
            gain_shared[n] = make_float2(static_cast<float>(gain.real()), static_cast<float>(gain.imag()));
        }
        __syncthreads();

        #pragma unroll 4
        for (int64_t n = 0; n < N; ++n) {
            float4 prev = make_float4(decay_shared[n].x, decay_shared[n].y, gain_shared[n].x, gain_shared[n].y);

            float4 v = identity;
            if (valid) {
                v = make_float4(mask * q_shared[n].x,
                                mask * q_shared[n].y,
                                p_shared[n] * x_v,
                                0.f);
            }

            struct PrefixCB {
                float4 running;
                ComplexRightAffineProd op;
                __device__ PrefixCB(float4 init) : running(init) {}
                __device__ float4 operator()(float4 block_total) {
                    float4 old = running;
                    running = op(running, block_total);
                    return old;
                }
            } pcb(prev);

            float4 prefix;
            BlockScan(scan_smem).ExclusiveScan(v, prefix, op, pcb);

            if (valid) {
                const float4 local_h = op(prefix, v);
                const c10::complex<T_ACC> g_v =
                    static_cast<c10::complex<float>>(gamma[d * N + n]);

                y_contrib = fmaf(local_h.z, gamma_shared[n].x, y_contrib);
                y_contrib = fmaf(-local_h.w, gamma_shared[n].y, y_contrib);

                if (t == L - 1) {
                    h[(b * D + d) * N + n] = {static_cast<T_ACC>(local_h.z),
                                              static_cast<T_ACC>(local_h.w)};
                }
            }

            __syncthreads();
        }

        if (valid) {
            y[(b * D + d) * L + t] = static_cast<T>(y_contrib);
        }
    }
}

template <typename T, typename T_ACC, int64_t CHUNK = 256>
__global__ void CEMACubBwdKernel(
        int64_t B, int64_t D, int64_t N, int64_t L,
        const T* __restrict__ y_grad,
        const c10::complex<T_ACC>* __restrict__ h_last_grad,
        const c10::complex<T_ACC>* __restrict__ chunk_decay,
        const c10::complex<T_ACC>* __restrict__ chunk_gain,
        const T* __restrict__ x,
        const T_ACC* __restrict__ p,
        const c10::complex<T_ACC>* __restrict__ q,
        const c10::complex<T_ACC>* __restrict__ gamma,
        const bool* __restrict__ bos_mask,
        T* __restrict__ x_grad,
        T_ACC* __restrict__ p_grad,
        c10::complex<T_ACC>* __restrict__ q_grad,
        c10::complex<T_ACC>* __restrict__ gamma_grad,
        c10::complex<T_ACC>* __restrict__ h0_grad) {

    const int64_t d = blockIdx.x;
    const int64_t tid = threadIdx.x;
    if (d >= D || tid >= CHUNK) return;

    const int64_t num_chunks = (L + CHUNK - 1) / CHUNK;

    // Init & load
    ComplexRightAffineProd prefixOp;
    ComplexLeftAffineProd suffixOp;
    ReversedComplexLeftAffineProd rev_op;

    float4 identity = suffixOp.identity();

    extern __shared__ float bwd_shm[];
    float4* next_shm = reinterpret_cast<float4*>(bwd_shm);
    float4* revBuf = next_shm + B * N;
    T_ACC* reduce_shm = reinterpret_cast<T_ACC*>(revBuf + CHUNK);

    __shared__ float4 block_total;

    if (tid == 0){
        for (int b=0; b < B; ++b) {
            for (int n=0; n < N; ++n){
                next_shm[b * N + n] = identity;
            }
        }
    }
    __syncthreads();

    using BlockScan = cub::BlockScan<float4, CHUNK, cub::BLOCK_SCAN_WARP_SCANS>;
    __shared__ typename BlockScan::TempStorage scan_smem;

    struct PrefixCB {
        float4 running;
        ComplexRightAffineProd op;
        __device__ PrefixCB(float4 init) : running(init) {}
        __device__ float4 operator()(float4 block_total) {
            float4 old = running;
            running = op(running, block_total);
            return old;
        }
    };

    T_ACC local_p_grad[N_MAX] = {0.f};
    T_ACC local_q_grad_real[N_MAX] = {0.f};
    T_ACC local_q_grad_imag[N_MAX] = {0.f};
    T_ACC local_gamma_grad_real[N_MAX] = {0.f};
    T_ACC local_gamma_grad_imag[N_MAX] = {0.f};

    for (int64_t chunk = num_chunks - 1; chunk >= 0; --chunk) {
        int64_t t = chunk * CHUNK + tid;
        bool valid = (t < L);

        for (int64_t b = 0; b < B; ++b) {
            T_ACC x_grad_contrib = T_ACC(0);

            for (int64_t n = 0; n < N; ++n) {
                // Scan h
                // Load
                float p_v = static_cast<float>(p[d * N + n]);
                c10::complex<float> q_v = static_cast<c10::complex<float>>(q[d * N + n]);
                c10::complex<float> gamma_v = static_cast<c10::complex<float>>(gamma[d * N + n]);
                float4 v = identity;
                if (valid) {
                    bool bos = bos_mask ? bos_mask[b * L + t] : 0;
                    float x_v = static_cast<float>(x[(b * D + d) * L + t]);
                    v = make_float4(bos? 0.f : q_v.real(),
                                    bos? 0.f : q_v.imag(),
                                    p_v * x_v,
                                    0.f);
                }

                int64_t idx = (((b * D + d) * N + n) * num_chunks + chunk);
                c10::complex<T_ACC> prev_decay = chunk_decay[idx];
                c10::complex<T_ACC> prev_gain = chunk_gain[idx];
                float4 prev = make_float4(prev_decay.real(), prev_decay.imag(), prev_gain.real(), prev_gain.imag());
                PrefixCB pcb(prev);
                float4 ht_prev;
                BlockScan(scan_smem).ExclusiveScan(v, ht_prev, prefixOp, pcb);
                float4 ht = prefixOp(ht_prev, v);

                // Scan h_grad
                // Load
                float4 w = identity;
                if (valid) {
                    bool next_bos = false;
                    if (bos_mask){
                        if (t != L - 1) next_bos = bos_mask[b * L + t + 1];
                        else next_bos = true;
                    }
                    float y_grad_v = static_cast<float>(y_grad[(b * D + d) * L + t]);
                    float add_last_r = 0.f, add_last_i = 0.f;
                    if (h_last_grad && t == L - 1) {
                        c10::complex<T_ACC> hl = h_last_grad[(b * D + d) * N + n];
                        add_last_r = hl.real();  add_last_i = hl.imag();
                    }
                    w = make_float4(next_bos ? 0.f : q_v.real(),
                                    next_bos ? 0.f : -q_v.imag(),
                                    y_grad_v * gamma_v.real() + add_last_r,
                                    -y_grad_v * gamma_v.imag() + add_last_i);
                }

                revBuf[CHUNK - 1 - tid] = w;
                __syncthreads();
                float4 w_rev = revBuf[tid];

                float4 inclusive_suffix_rev;
                BlockScan(scan_smem).InclusiveScan(w_rev, inclusive_suffix_rev, rev_op);

                revBuf[CHUNK - 1 - tid] = inclusive_suffix_rev;
                __syncthreads();
                float4 ht_grad_within_chunk = revBuf[tid];

                if (tid == CHUNK - 1) {
                    block_total = inclusive_suffix_rev;
                }
                __syncthreads();

                float4 next = next_shm[b * N + n];
                float4 ht_grad = suffixOp(ht_grad_within_chunk, next);

                // Add next result
                if (valid) {
                    bool bos = false;
                    if (bos_mask != nullptr) bos = bos_mask[b * L + t];

                    float grad_h_real = ht_grad.z;
                    float grad_h_imag = ht_grad.w;

                    // x grad
                    // x_grad_contrib += static_cast<T>(p_v) * static_cast<T>(grad_h_real);
                    x_grad_contrib += p_v * grad_h_real;

                    // p grad
                    float x_v = static_cast<float>(x[(b * D + d) * L + t]);
                    local_p_grad[n] += x_v * grad_h_real;

                    // q grad
                    if (!bos){
                        float q_grad_real = ht_prev.z * grad_h_real + ht_prev.w * grad_h_imag;
                        float q_grad_imag = ht_prev.z * grad_h_imag - ht_prev.w * grad_h_real;

                        local_q_grad_real[n] += q_grad_real;
                        local_q_grad_imag[n] += q_grad_imag;
                    }

                    // gamma grad
                    float y_grad_v = static_cast<float>(y_grad[(b * D + d) * L + t]);
                    float gamma_grad_real = ht.z * y_grad_v;
                    float gamma_grad_imag = -ht.w * y_grad_v;

                    local_gamma_grad_real[n] += gamma_grad_real;
                    local_gamma_grad_imag[n] += gamma_grad_imag;

                    // h0 grad
                    if (chunk == 0 && tid == 0 && !bos) {
                        float h0_grad_real = grad_h_real * q_v.real() + grad_h_imag * q_v.imag();
                        float h0_grad_imag = grad_h_imag * q_v.real() - grad_h_real * q_v.imag();

                        h0_grad[(b * D + d) * N + n] = c10::complex<T_ACC>(h0_grad_real, h0_grad_imag);
                    }
                }

                __syncthreads();
                if (tid == 0) {
                    next = suffixOp(block_total, next);
                    next_shm[b * N + n] = next;
                }
                __syncthreads();
            }
            if (valid) x_grad[(b * D + d) * L + t] = static_cast<T>(x_grad_contrib);
        }
    }

    for (int64_t n = 0; n < N; ++n) {
        T_ACC p_sum = static_cast<T_ACC>(reduce::BlockReduce(local_p_grad[n], reduce_shm));
        T_ACC q_real_sum = static_cast<T_ACC>(reduce::BlockReduce(local_q_grad_real[n], reduce_shm));
        T_ACC q_imag_sum = static_cast<T_ACC>(reduce::BlockReduce(local_q_grad_imag[n], reduce_shm));
        T_ACC gamma_real_sum = static_cast<T_ACC>(reduce::BlockReduce(local_gamma_grad_real[n], reduce_shm));
        T_ACC gamma_imag_sum = static_cast<T_ACC>(reduce::BlockReduce(local_gamma_grad_imag[n], reduce_shm));

        if (tid == 0) {
            p_grad[d * N + n] = p_sum;
            q_grad[d * N + n] = c10::complex<T_ACC>(q_real_sum, q_imag_sum);
            gamma_grad[d * N + n] = c10::complex<T_ACC>(gamma_real_sum, gamma_imag_sum);
        }
        __syncthreads();
    }
}

template <typename T, typename T_ACC, int64_t B, int64_t N, int64_t CHUNK = 256, int64_t UB = 4, int64_t UN = 4>
__global__ void CEMACubBwdKernel(
        int64_t D, int64_t L,
        const T* __restrict__ y_grad,
        const c10::complex<T_ACC>* __restrict__ h_last_grad,
        const c10::complex<T_ACC>* __restrict__ chunk_decay,
        const c10::complex<T_ACC>* __restrict__ chunk_gain,
        const T* __restrict__ x,
        const T_ACC* __restrict__ p,
        const c10::complex<T_ACC>* __restrict__ q,
        const c10::complex<T_ACC>* __restrict__ gamma,
        const bool* __restrict__ bos_mask,
        T* __restrict__ x_grad,
        T_ACC* __restrict__ p_grad,
        c10::complex<T_ACC>* __restrict__ q_grad,
        c10::complex<T_ACC>* __restrict__ gamma_grad,
        c10::complex<T_ACC>* __restrict__ h0_grad) {

    const int64_t d = blockIdx.x;
    const int64_t tid = threadIdx.x;
    if (d >= D || tid >= CHUNK) return;

    const int64_t num_chunks = (L + CHUNK - 1) / CHUNK;

    // Init & load
    ComplexRightAffineProd prefixOp;
    ComplexLeftAffineProd suffixOp;
    ReversedComplexLeftAffineProd rev_op;

    float4 identity = suffixOp.identity();

    extern __shared__ float bwd_shm[];
    float4* next_shm = reinterpret_cast<float4*>(bwd_shm);
    float4* revBuf = next_shm + B * N;
    T_ACC* reduce_shm = reinterpret_cast<T_ACC*>(revBuf + CHUNK);

    __shared__ float4 block_total;

    if (tid == 0){
        #pragma unroll
        for (int b=0; b < B; ++b) {
            #pragma unroll
            for (int n=0; n < N; ++n){
                next_shm[b * N + n] = identity;
            }
        }
    }
    __syncthreads();

    using BlockScan = cub::BlockScan<float4, CHUNK, cub::BLOCK_SCAN_WARP_SCANS>;
    __shared__ typename BlockScan::TempStorage scan_smem;

    struct PrefixCB {
        float4 running;
        ComplexRightAffineProd op;
        __device__ PrefixCB(float4 init) : running(init) {}
        __device__ float4 operator()(float4 block_total) {
            float4 old = running;
            running = op(running, block_total);
            return old;
        }
    };

    T_ACC local_p_grad[N] = {0.f};
    T_ACC local_q_grad_real[N] = {0.f};
    T_ACC local_q_grad_imag[N] = {0.f};
    T_ACC local_gamma_grad_real[N] = {0.f};
    T_ACC local_gamma_grad_imag[N] = {0.f};

    __shared__ float  p_shared[N];
    __shared__ float2 q_shared[N];
    __shared__ float2 gamma_shared[N];
    __shared__ float2 decay_shared[N];
    __shared__ float2 gain_shared[N];

    if (threadIdx.x < N) {
        int64_t n = threadIdx.x;
        p_shared[n] = static_cast<float>(p[d * N + n]);
        c10::complex<float> qv = static_cast<c10::complex<float>>(q[d * N + n]);
        c10::complex<float> gamma_v = static_cast<c10::complex<float>>(gamma[d * N + n]);
        q_shared[n] = make_float2(qv.real(), qv.imag());
        gamma_shared[n] = make_float2(gamma_v.real(), gamma_v.imag());
    }
    __syncthreads();

    for (int64_t chunk = num_chunks - 1; chunk >= 0; --chunk) {
        int64_t t = chunk * CHUNK + tid, t_rev = chunk * CHUNK + (CHUNK - 1 - tid);
        bool valid = (t < L), valid_rev = (t_rev < L);

        #pragma unroll UB
        for (int64_t b = 0; b < B; ++b) {
            T_ACC x_grad_contrib = T_ACC(0);

            float x_v = 0.f, y_grad_v = 0.f;
            bool  bos = false;
            if (valid) {
                x_v = static_cast<float>(x[(b * D + d) * L + t]);
                y_grad_v = static_cast<float>(y_grad[(b * D + d) * L + t]);
                bos = bos_mask ? bos_mask[b * L + t] : false;
            }
            const float f_valid = valid ? 1.f : 0.f;
            const float f_mask = (f_valid && !bos) ? 1.f : 0.f;

            float y_grad_v_rev = 0.f;
            bool  next_bos_rev = false;
            if (valid_rev) {
                y_grad_v_rev = static_cast<float>(y_grad[(b * D + d) * L + t_rev]);
                next_bos_rev = bos_mask ? ((t_rev != L - 1) ? bos_mask[b * L + t_rev + 1] : true) : false;
            }
            const float f_valid_rev = valid_rev ? 1.f : 0.f;
            const float f_next_rev = (f_valid_rev && !next_bos_rev) ? 1.f : 0.f;


            if (threadIdx.x < N) {
                int n = threadIdx.x;
                int64_t idx = (((b * D + d) * N + n) * num_chunks + chunk);
                c10::complex<T_ACC> decay = chunk_decay[idx];
                c10::complex<T_ACC> gain = chunk_gain[idx];
                decay_shared[n] = make_float2(static_cast<float>(decay.real()), static_cast<float>(decay.imag()));
                gain_shared[n] = make_float2(static_cast<float>(gain.real()), static_cast<float>(gain.imag()));
            }
            __syncthreads();

            #pragma unroll UN
            for (int64_t n = 0; n < N; ++n) {
                // Scan h
                // Load
                float2 q_shared_n = q_shared[n];
                float  p_shared_n = p_shared[n];
                float4 v;
                v.x = f_mask * q_shared_n.x;
                v.y = f_mask * q_shared_n.y;
                v.z = p_shared_n * x_v;
                v.w = 0.f;

                float4 prev = make_float4(decay_shared[n].x, decay_shared[n].y, gain_shared[n].x, gain_shared[n].y);
                PrefixCB pcb(prev);
                float4 ht_prev;
                BlockScan(scan_smem).ExclusiveScan(v, ht_prev, prefixOp, pcb);
                float4 ht = prefixOp(ht_prev, v);

                // Scan h_grad
                // Load
                float2 g_shared_n = gamma_shared[n];

                float add_last_r = 0.f, add_last_i = 0.f;
                if (h_last_grad && valid_rev && t_rev == L - 1) {
                    c10::complex<T_ACC> hl = h_last_grad[(b * D + d) * N + n];
                    add_last_r = hl.real(); add_last_i = hl.imag();
                }

                float y_grad_gate = y_grad_v_rev * f_valid_rev;
                float4 w_rev;
                w_rev.x = f_next_rev *  q_shared_n.x;
                w_rev.y = f_next_rev * -q_shared_n.y;
                w_rev.z = fmaf(y_grad_gate, g_shared_n.x, add_last_r);
                w_rev.w = fmaf(-y_grad_gate, g_shared_n.y, add_last_i);

                float4 inclusive_suffix_rev;
                BlockScan(scan_smem).InclusiveScan(w_rev, inclusive_suffix_rev, rev_op);

                revBuf[CHUNK - 1 - tid] = inclusive_suffix_rev;
                __syncthreads();
                float4 ht_grad_within_chunk = revBuf[tid];

                //

                if (tid == CHUNK - 1) {
                    block_total = inclusive_suffix_rev;
                }
                __syncthreads();

                float4 next = next_shm[b * N + n];
                float4 ht_grad = suffixOp(ht_grad_within_chunk, next);

                // Add next result
                if (valid) {
                    float grad_h_real = ht_grad.z;
                    float grad_h_imag = ht_grad.w;

                    // x grad
                    x_grad_contrib = fmaf(p_shared_n, grad_h_real, x_grad_contrib);

                    // p grad
                    local_p_grad[n] = fmaf(x_v, grad_h_real, local_p_grad[n]);

                    // q grad
                    float q_grad_real = f_mask * fmaf(ht_prev.w,  grad_h_imag, ht_prev.z * grad_h_real);
                    float q_grad_imag = f_mask * fmaf(-ht_prev.w, grad_h_real, ht_prev.z * grad_h_imag);

                    local_q_grad_real[n] += q_grad_real;
                    local_q_grad_imag[n] += q_grad_imag;

                    // gamma grad
                    local_gamma_grad_real[n] = fmaf(ht.z, y_grad_v, local_gamma_grad_real[n]);
                    local_gamma_grad_imag[n] = fmaf(-ht.w, y_grad_v, local_gamma_grad_imag[n]);

                    // h0 grad
                    if (chunk == 0 && tid == 0 && !bos) {
                        float h0_grad_real = fmaf(grad_h_imag, q_shared_n.y, grad_h_real * q_shared_n.x);
                        float h0_grad_imag = fmaf(-grad_h_real, q_shared_n.y, grad_h_imag * q_shared_n.x);

                        h0_grad[(b * D + d) * N + n] = c10::complex<T_ACC>(h0_grad_real, h0_grad_imag);
                    }
                }

                __syncthreads();
                if (tid == 0) {
                    next = suffixOp(block_total, next);
                    next_shm[b * N + n] = next;
                }
                __syncthreads();
            }
            if (valid) x_grad[(b * D + d) * L + t] = static_cast<T>(x_grad_contrib);
        }
    }

    #pragma unroll
    for (int64_t n = 0; n < N; ++n) {
        T_ACC p_sum = static_cast<T_ACC>(reduce::BlockReduce(local_p_grad[n], reduce_shm));
        T_ACC q_real_sum = static_cast<T_ACC>(reduce::BlockReduce(local_q_grad_real[n], reduce_shm));
        T_ACC q_imag_sum = static_cast<T_ACC>(reduce::BlockReduce(local_q_grad_imag[n], reduce_shm));
        T_ACC gamma_real_sum = static_cast<T_ACC>(reduce::BlockReduce(local_gamma_grad_real[n], reduce_shm));
        T_ACC gamma_imag_sum = static_cast<T_ACC>(reduce::BlockReduce(local_gamma_grad_imag[n], reduce_shm));

        if (tid == 0) {
            p_grad[d * N + n] = p_sum;
            q_grad[d * N + n] = c10::complex<T_ACC>(q_real_sum, q_imag_sum);
            gamma_grad[d * N + n] = c10::complex<T_ACC>(gamma_real_sum, gamma_imag_sum);
        }
        __syncthreads();
    }
}

#define DISPATCH_FWD_KERNEL(                                          \
    KernelFunc, T, T_ACC, shm_size, cuda_stream, B, D, N, L, ...)                \
  do {                                                                 \
    if (B == 1){ \
        if (N == 4) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 1, 4, kNumThreadsHi, 4>, \
                dim3(D, B), kNumThreadsHi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 8) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 1, 8, kNumThreadsHi, 4>, \
                dim3(D, B), kNumThreadsHi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 16) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 1, 16, kNumThreadsHi, 2>, \
                dim3(D, B), kNumThreadsHi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else { \
            KernelFunc<T, T_ACC, kNumThreadsHi>                               \
            <<<dim3(D, B), kNumThreadsHi, shm_size, cuda_stream>>>(1, D, N, L, __VA_ARGS__); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); \
        } \
    } \
    else if (B == 2){ \
        if (N == 4) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2, 4, kNumThreadsHi, 4>, \
                dim3(D, B), kNumThreadsHi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 8) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2, 8, kNumThreadsHi, 4>, \
                dim3(D, B), kNumThreadsHi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 16) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2, 16, kNumThreadsHi, 2>, \
                dim3(D, B), kNumThreadsHi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else { \
            KernelFunc<T, T_ACC, kNumThreadsHi>                               \
            <<<dim3(D, B), kNumThreadsHi, shm_size, cuda_stream>>>(2, D, N, L, __VA_ARGS__); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); \
        } \
    } \
    else if (B == 4){ \
        if (N == 4) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4, 4, kNumThreadsMi, 4>, \
                dim3(D, B), kNumThreadsMi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 8) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4, 8, kNumThreadsMi, 4>, \
                dim3(D, B), kNumThreadsMi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 16) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4, 16, kNumThreadsMi, 2>, \
                dim3(D, B), kNumThreadsMi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else { \
            KernelFunc<T, T_ACC, kNumThreadsMi>                               \
            <<<dim3(D, B), kNumThreadsMi, shm_size, cuda_stream>>>(B, D, N, L, __VA_ARGS__); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); \
        } \
    } \
    else if (B == 8){ \
        if (N == 4) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8, 4, kNumThreadsLo, 4>, \
                dim3(D, B), kNumThreadsLo, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 8) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8, 8, kNumThreadsLo, 4>, \
                dim3(D, B), kNumThreadsLo, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 16) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8, 16, kNumThreadsLo, 2>, \
                dim3(D, B), kNumThreadsLo, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else { \
            KernelFunc<T, T_ACC, kNumThreadsLo>                               \
            <<<dim3(D, B), kNumThreadsLo, shm_size, cuda_stream>>>(B, D, N, L, __VA_ARGS__); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); \
        } \
    } \
    else { \
        KernelFunc<T, T_ACC, kNumThreadsLo>                               \
            <<<dim3(D, B), kNumThreadsLo, shm_size, cuda_stream>>>(B, D, N, L, __VA_ARGS__); \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                    \
    } \
  } while (false)

#define DISPATCH_BWD_KERNEL(                                          \
    KernelFunc, T, T_ACC, shm_size, cuda_stream, B, D, N, L, ...)                \
  do {                                                                 \
    if (B == 1){ \
        if (N == 4) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 1, 4, kNumThreadsHi, 1, 4>, \
                D, kNumThreadsHi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 8) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 1, 8, kNumThreadsHi, 1, 4>, \
                D, kNumThreadsHi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 16) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 1, 16, kNumThreadsHi, 1, 2>, \
                D, kNumThreadsHi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else { \
            KernelFunc<T, T_ACC, kNumThreadsHi>                               \
            <<<D, kNumThreadsHi, shm_size, cuda_stream>>>(1, D, N, L, __VA_ARGS__); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); \
        } \
    } \
    else if (B == 2){ \
        if (N == 4) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2, 4, kNumThreadsHi, 2, 4>, \
                D, kNumThreadsHi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 8) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2, 8, kNumThreadsHi, 2, 4>, \
                D, kNumThreadsHi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 16) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2, 16, kNumThreadsHi, 2, 2>, \
                D, kNumThreadsHi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else { \
            KernelFunc<T, T_ACC, kNumThreadsHi>                               \
            <<<D, kNumThreadsHi, shm_size, cuda_stream>>>(2, D, N, L, __VA_ARGS__); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); \
        } \
    } \
    else if (B == 4){ \
        if (N == 4) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4, 4, kNumThreadsMi, 2, 4>, \
                D, kNumThreadsMi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 8) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4, 8, kNumThreadsMi, 1, 4>, \
                D, kNumThreadsMi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 16) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4, 16, kNumThreadsMi, 1, 2>, \
                D, kNumThreadsMi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else { \
            KernelFunc<T, T_ACC, kNumThreadsMi>                               \
            <<<D, kNumThreadsMi, shm_size, cuda_stream>>>(B, D, N, L, __VA_ARGS__); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); \
        } \
    } \
    else if (B == 8){ \
        if (N == 4) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8, 4, kNumThreadsLo, 2, 4>, \
                D, kNumThreadsLo, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 8) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8, 8, kNumThreadsLo, 1, 4>, \
                D, kNumThreadsLo, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 16) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8, 16, kNumThreadsLo, 1, 2>, \
                D, kNumThreadsLo, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else { \
            KernelFunc<T, T_ACC, kNumThreadsLo>                               \
            <<<D, kNumThreadsLo, shm_size, cuda_stream>>>(B, D, N, L, __VA_ARGS__); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); \
        } \
    } \
    else { \
        KernelFunc<T, T_ACC, kNumThreadsLo>                               \
            <<<D, kNumThreadsLo, shm_size, cuda_stream>>>(B, D, N, L, __VA_ARGS__); \
        C10_CUDA_KERNEL_LAUNCH_CHECK();                                    \
    } \
  } while (false)

#define DISPATCH_RECALC_KERNEL(KernelFunc, T, T_ACC, shm_size, cuda_stream, B, D, N, L, K, ...) \
  do { \
    if (B == 1) { \
        if (N == 4) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 1, 4, kNumThreadsHi>, \
                dim3(D, K), kNumThreadsHi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } else if (N == 8) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 1, 8, kNumThreadsHi>, \
                dim3(D, K), kNumThreadsHi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } else if (N == 16) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 1, 16, kNumThreadsHi>, \
                dim3(D, K), kNumThreadsHi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } else { \
            KernelFunc<T, T_ACC, kNumThreadsHi> \
                <<<dim3(D, K), kNumThreadsHi, shm_size, cuda_stream>>>(B, D, N, L, __VA_ARGS__); \
            C10_CUDA_KERNEL_LAUNCH_CHECK(); \
        } \
    } else if (B == 2) { \
        if (N == 4) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2, 4, kNumThreadsHi>, \
                dim3(D, K), kNumThreadsHi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } else if (N == 8) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2, 8, kNumThreadsHi>, \
                dim3(D, K), kNumThreadsHi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } else if (N == 16) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2, 16, kNumThreadsHi>, \
                dim3(D, K), kNumThreadsHi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } else { \
            KernelFunc<T, T_ACC, kNumThreadsHi> \
                <<<dim3(D, K), kNumThreadsHi, shm_size, cuda_stream>>>(B, D, N, L, __VA_ARGS__); \
            C10_CUDA_KERNEL_LAUNCH_CHECK(); \
        } \
    } else if (B == 4) { \
        if (N == 4) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4, 4, kNumThreadsMi>, \
                dim3(D, K), kNumThreadsMi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } else if (N == 8) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4, 8, kNumThreadsMi>, \
                dim3(D, K), kNumThreadsMi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } else if (N == 16) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4, 16, kNumThreadsMi>, \
                dim3(D, K), kNumThreadsMi, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } else { \
            KernelFunc<T, T_ACC, kNumThreadsMi> \
                <<<dim3(D, K), kNumThreadsMi, shm_size, cuda_stream>>>(B, D, N, L, __VA_ARGS__); \
            C10_CUDA_KERNEL_LAUNCH_CHECK(); \
        } \
    } else if (B == 8) { \
        if (N == 4) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8, 4, kNumThreadsLo>, \
                dim3(D, K), kNumThreadsLo, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } else if (N == 8) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8, 8, kNumThreadsLo>, \
                dim3(D, K), kNumThreadsLo, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } else if (N == 16) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8, 16, kNumThreadsLo>, \
                dim3(D, K), kNumThreadsLo, shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } else { \
            KernelFunc<T, T_ACC, kNumThreadsLo> \
                <<<dim3(D, K), kNumThreadsLo, shm_size, cuda_stream>>>(B, D, N, L, __VA_ARGS__); \
            C10_CUDA_KERNEL_LAUNCH_CHECK(); \
        } \
    } else { \
        KernelFunc<T, T_ACC, kNumThreadsLo> \
            <<<dim3(D, K), kNumThreadsLo, shm_size, cuda_stream>>>(B, D, N, L, __VA_ARGS__); \
        C10_CUDA_KERNEL_LAUNCH_CHECK(); \
    } \
  } while (false)

template <typename T>
void CEMAScanCUDAFwdImpl(const torch::Tensor& x,
                        const torch::Tensor& p,
                        const torch::Tensor& q,
                        const torch::Tensor& gamma,
                        const torch::Tensor& bos_mask,
                        const torch::Tensor& h0,
                        torch::Tensor& h,
                        torch::Tensor& y,
                        torch::Tensor& chunk_decay,
                        torch::Tensor& chunk_gain) {
    using T_ACC = at::acc_type<T, true>;

    const int64_t B = x.size(0);
    const int64_t D = x.size(1);
    const int64_t L = x.size(2);
    const int64_t N = p.size(1);

    const T* x_data = x.data_ptr<T>();
    const T_ACC* p_data = p.data_ptr<T_ACC>();
    const c10::complex<T_ACC>* q_data = q.data_ptr<c10::complex<T_ACC>>();
    const c10::complex<T_ACC>* gamma_data = gamma.data_ptr<c10::complex<T_ACC>>();

    const bool* bos_mask_data = bos_mask.defined() ? bos_mask.data_ptr<bool>() : nullptr;
    const c10::complex<T_ACC>* h0_data = h0.defined() ? h0.data_ptr<c10::complex<T_ACC>>() : nullptr;

    c10::complex<T_ACC>* h_data = h.data_ptr<c10::complex<T_ACC>>();
    T* y_data = y.data_ptr<T>();
    c10::complex<T_ACC>* chunk_decay_data = chunk_decay.data_ptr<c10::complex<T_ACC>>();
    c10::complex<T_ACC>* chunk_gain_data = chunk_gain.data_ptr<c10::complex<T_ACC>>();

    at::cuda::OptionalCUDAGuard guard(at::device_of(x));
    cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

    const int64_t num_threads = getChunkNum(B, N);
    const size_t shm_size = N * sizeof(float4);

    DISPATCH_FWD_KERNEL(
        CEMACubFwdKernel, T, T_ACC,
        shm_size, cuda_stream,
        B, D, N, L, /*num_threads,*/
        x_data, p_data, q_data, gamma_data,
        bos_mask_data, h0_data,
        h_data, y_data, chunk_decay_data, chunk_gain_data);
}

template <typename T>
void CEMAScanCUDAFwdRecalcImpl(const torch::Tensor& x,
                        const torch::Tensor& p,
                        const torch::Tensor& q,
                        const torch::Tensor& gamma,
                        const torch::Tensor& bos_mask,
                        const torch::Tensor& chunk_decay,
                        const torch::Tensor& chunk_gain,
                        torch::Tensor& h,
                        torch::Tensor& y) {
    using T_ACC = at::acc_type<T, true>;

    const int64_t B = x.size(0);
    const int64_t D = x.size(1);
    const int64_t L = x.size(2);
    const int64_t N = p.size(1);

    const int64_t num_threads = getChunkNum(B, N);
    const int64_t K = (L + num_threads - 1) / num_threads;

    const T* x_data = x.data_ptr<T>();
    const T_ACC* p_data = p.data_ptr<T_ACC>();
    const c10::complex<T_ACC>* q_data = q.data_ptr<c10::complex<T_ACC>>();
    const c10::complex<T_ACC>* gamma_data = gamma.data_ptr<c10::complex<T_ACC>>();

    const bool* bos_mask_data = bos_mask.defined() ? bos_mask.data_ptr<bool>() : nullptr;

    c10::complex<T_ACC>* h_data = h.data_ptr<c10::complex<T_ACC>>();
    T* y_data = y.data_ptr<T>();
    c10::complex<T_ACC>* chunk_decay_data = chunk_decay.data_ptr<c10::complex<T_ACC>>();
    c10::complex<T_ACC>* chunk_gain_data = chunk_gain.data_ptr<c10::complex<T_ACC>>();

    at::cuda::OptionalCUDAGuard guard(at::device_of(x));
    cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

    const size_t shm_size = 0;

    DISPATCH_RECALC_KERNEL(
        CEMACubFwdRecalcKernel, T, T_ACC,
        shm_size, cuda_stream,
        B, D, N, L, K, /*num_threads,*/
        x_data, p_data, q_data, gamma_data,
        bos_mask_data,
        chunk_decay_data, chunk_gain_data,
        h_data, y_data);
}

template <typename T>
void CEMAScanCUDABwdImpl(const torch::Tensor& y_grad,
                        const torch::Tensor& h_last_grad,
                        const torch::Tensor& chunk_decay,
                        const torch::Tensor& chunk_gain,
                        const torch::Tensor& x,
                        const torch::Tensor& p,
                        const torch::Tensor& q,
                        const torch::Tensor& gamma,
                        const torch::Tensor& bos_mask,
                        torch::Tensor& x_grad,
                        torch::Tensor& p_grad,
                        torch::Tensor& q_grad,
                        torch::Tensor& gamma_grad,
                        torch::Tensor& h0_grad) {
    using T_ACC = at::acc_type<T, true>;

    const int64_t B = x.size(0);
    const int64_t D = x.size(1);
    const int64_t L = x.size(2);
    const int64_t N = p.size(1);

    const T* y_grad_data = y_grad.data_ptr<T>();
    const c10::complex<T_ACC>* chunk_decay_data = chunk_decay.data_ptr<c10::complex<T_ACC>>();
    const c10::complex<T_ACC>* chunk_gain_data = chunk_gain.data_ptr<c10::complex<T_ACC>>();
    const T* x_data = x.data_ptr<T>();
    const T_ACC* p_data = p.data_ptr<T_ACC>();
    const c10::complex<T_ACC>* q_data = q.data_ptr<c10::complex<T_ACC>>();
    const c10::complex<T_ACC>* gamma_data = gamma.data_ptr<c10::complex<T_ACC>>();

    const bool* bos_mask_data = bos_mask.defined() ? bos_mask.data_ptr<bool>() : nullptr;
    const c10::complex<T_ACC>* h_last_grad_data =
        h_last_grad.defined() ? h_last_grad.data_ptr<c10::complex<T_ACC>>() : nullptr;

    T* x_grad_data = x_grad.data_ptr<T>();
    T_ACC* p_grad_data = p_grad.data_ptr<T_ACC>();
    c10::complex<T_ACC>* q_grad_data = q_grad.data_ptr<c10::complex<T_ACC>>();
    c10::complex<T_ACC>* gamma_grad_data = gamma_grad.data_ptr<c10::complex<T_ACC>>();
    c10::complex<T_ACC>* h0_grad_data = h0_grad.data_ptr<c10::complex<T_ACC>>();

    at::cuda::OptionalCUDAGuard guard(at::device_of(x));
    cudaStream_t cuda_stream = at::cuda::getCurrentCUDAStream();

    const int64_t num_threads = getChunkNum(B, N);
    const size_t num_warps   = (num_threads + cuda_utils::kWarpSize - 1) / cuda_utils::kWarpSize;

    const size_t shm_size = (B * N + num_threads) * sizeof(float4) + num_warps * sizeof(T_ACC);

    DISPATCH_BWD_KERNEL(
        CEMACubBwdKernel, T, T_ACC,
        shm_size, cuda_stream,
        B, D, N, L, /*num_threads,*/
        y_grad_data, h_last_grad_data,
        chunk_decay_data, chunk_gain_data,
        x_data, p_data, q_data, gamma_data,
        bos_mask_data,
        x_grad_data, p_grad_data, q_grad_data, gamma_grad_data, h0_grad_data);
}

#undef DISPATCH_FWD_KERNEL
#undef DISPATCH_BWD_KERNEL
#undef DISPATCH_RECALC_KERNEL
// #undef DISPATCH_SCAN_BS1_KERNEL
} // namespace

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor> CEMACUBScanCUDAFwd(const torch::Tensor& x,
                                                            const torch::Tensor& p,
                                                            const torch::Tensor& q,
                                                            const torch::Tensor& gamma,
                                                            const c10::optional<torch::Tensor>& bos_mask,
                                                            const c10::optional<torch::Tensor>& h0) {
    const int64_t B = x.size(0);
    const int64_t D = x.size(1);
    const int64_t L = x.size(2);
    const int64_t N = p.size(1);

    const int64_t chunk_size = getChunkNum(B, N);
    const int64_t num_chunks = (L + chunk_size - 1) / chunk_size;

    c10::MaybeOwned<torch::Tensor> bos_mask_maybe_owned =
        at::borrow_from_optional_tensor(bos_mask);
    c10::MaybeOwned<torch::Tensor> h0_maybe_owned =
        at::borrow_from_optional_tensor(h0);
    torch::Tensor h = torch::empty(
        {B, D, N}, q.options().memory_format(at::MemoryFormat::Contiguous));
    torch::Tensor y = torch::zeros(
        {B, D, L}, x.options().memory_format(at::MemoryFormat::Contiguous));
    torch::Tensor chunk_decay = torch::empty(
        {B, D, N, num_chunks}, q.options().memory_format(at::MemoryFormat::Contiguous));
    torch::Tensor chunk_gain = torch::empty_like(chunk_decay);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(), "CEMAScanCUDAFwd", [&]() {
        CEMAScanCUDAFwdImpl<scalar_t>(*(x.expect_contiguous()),
            *(p.expect_contiguous()), *(q.expect_contiguous()), *(gamma.expect_contiguous()),
            *(bos_mask_maybe_owned->expect_contiguous()), *(h0_maybe_owned->expect_contiguous()),
            h, y, chunk_decay, chunk_gain);
    });
    return std::make_tuple<torch::Tensor, torch::Tensor,
                           torch::Tensor, torch::Tensor>(std::move(y),
                                                         std::move(h),
                                                         std::move(chunk_decay),
                                                         std::move(chunk_gain));
}

std::tuple<torch::Tensor, torch::Tensor> CEMACUBScanCUDAFwdRecalc(const torch::Tensor& x,
                                                            const torch::Tensor& p,
                                                            const torch::Tensor& q,
                                                            const torch::Tensor& gamma,
                                                            const c10::optional<torch::Tensor>& bos_mask,
                                                            const torch::Tensor& chunk_decay,
                                                            const torch::Tensor& chunk_gain) {
    const int64_t B = x.size(0);
    const int64_t D = x.size(1);
    const int64_t L = x.size(2);
    const int64_t N = p.size(1);

    const int64_t chunk_size = getChunkNum(B, N);
    const int64_t num_chunks = (L + chunk_size - 1) / chunk_size;

    c10::MaybeOwned<torch::Tensor> bos_mask_maybe_owned =
        at::borrow_from_optional_tensor(bos_mask);
    torch::Tensor h = torch::empty(
        {B, D, N}, q.options().memory_format(at::MemoryFormat::Contiguous));
    torch::Tensor y = torch::zeros(
        {B, D, L}, x.options().memory_format(at::MemoryFormat::Contiguous));
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(), "CEMAScanCUDAFwdRecalc", [&]() {
        CEMAScanCUDAFwdRecalcImpl<scalar_t>(*(x.expect_contiguous()),
            *(p.expect_contiguous()), *(q.expect_contiguous()), *(gamma.expect_contiguous()),
            *(bos_mask_maybe_owned->expect_contiguous()),
            *(chunk_decay.expect_contiguous()), *(chunk_gain.expect_contiguous()), h, y);
    });
    return std::make_tuple<torch::Tensor, torch::Tensor>(std::move(y), std::move(h));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor> CEMACUBScanCUDABwd(const torch::Tensor& y_grad,
                                                            const c10::optional<torch::Tensor>& h_last_grad,
                                                            const torch::Tensor& chunk_decay,
                                                            const torch::Tensor& chunk_gain,
                                                            const torch::Tensor& x,
                                                            const torch::Tensor& p,
                                                            const torch::Tensor& q,
                                                            const torch::Tensor& gamma,
                                                            const c10::optional<torch::Tensor>& bos_mask){
    const int64_t B = x.size(0);
    const int64_t D = x.size(1);
    const int64_t L = x.size(2);
    const int64_t N = p.size(1);

    c10::MaybeOwned<torch::Tensor> bos_mask_maybe_owned =
        at::borrow_from_optional_tensor(bos_mask);
    c10::MaybeOwned<torch::Tensor> h_last_grad_maybe_owned =
        at::borrow_from_optional_tensor(h_last_grad);

    torch::Tensor x_grad = torch::zeros(
        {B, D, L}, x.options().memory_format(at::MemoryFormat::Contiguous));
    torch::Tensor p_grad = torch::zeros(
        {D, N}, p.options().memory_format(at::MemoryFormat::Contiguous));
    torch::Tensor q_grad = torch::zeros(
        {D, N}, q.options().memory_format(at::MemoryFormat::Contiguous));
    torch::Tensor gamma_grad = torch::zeros(
        {D, N}, gamma.options().memory_format(at::MemoryFormat::Contiguous));
    torch::Tensor h0_grad = torch::zeros(
        {B, D, N}, q.options().memory_format(at::MemoryFormat::Contiguous));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, x.scalar_type(), "CEMAScanCUDABwd", [&]() {
        CEMAScanCUDABwdImpl<scalar_t>(*(y_grad.expect_contiguous()),
            *(h_last_grad_maybe_owned->expect_contiguous()),
            *(chunk_decay.expect_contiguous()), *(chunk_gain.expect_contiguous()),
            *(x.expect_contiguous()), *(p.expect_contiguous()), *(q.expect_contiguous()), *(gamma.expect_contiguous()),
            *(bos_mask_maybe_owned->expect_contiguous()),
            x_grad, p_grad, q_grad, gamma_grad, h0_grad);
    });

    return std::make_tuple<torch::Tensor, torch::Tensor, torch::Tensor,
                           torch::Tensor, torch::Tensor>(std::move(x_grad),
                                                         std::move(p_grad),
                                                         std::move(q_grad),
                                                         std::move(gamma_grad),
                                                         std::move(h0_grad));
}

} // namespace ops
} // namespace xllm
