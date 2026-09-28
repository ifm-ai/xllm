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
#include <cooperative_groups.h>
#include <cuda/pipeline>

#include <type_traits>

#include "blas.h"
#include "complex_utils.cuh"
#include "cuda_utils.cuh"
#include "ops/cema_blelloch_scan.h"
#include "reduce.cuh"
#include "utils.h"

namespace cg = cooperative_groups;

namespace xllm {
namespace ops {
namespace {

constexpr int chunk_sz = 32;
constexpr int d_tile = 4;
constexpr unsigned mask=0xffffffffu;
constexpr int kWarpSize=32;

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

template<typename T, typename T_ACC>
__global__ void CEMAScanFwdKernel(
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
    c10::complex<T_ACC>* __restrict__ chunk_gain){
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int d = blockIdx.x * d_tile + warp;

    if (d >= D) return;

    const int64_t num_chunks = (L + chunk_sz - 1) / chunk_sz;

    ComplexRightAffineProd op;

    extern __shared__ float fwd_shm[];
    float* p_shared = fwd_shm; // d_tile * N
    c10::complex<float>* q_shared = reinterpret_cast<c10::complex<float>*>(p_shared + d_tile * N); // d_tile * N
    c10::complex<float>* gamma_shared = q_shared + d_tile * N; // d_tile * N
    float4* running = reinterpret_cast<float4*>(gamma_shared + d_tile * N); // B * d_tile * N

    if (lane < N) {
        int n = lane;
        p_shared[warp * N + n] = static_cast<float>(p[d * N + n]);
        c10::complex<float> q_v = static_cast<c10::complex<float>>(q[d * N + n]);
        c10::complex<float> gamma_v = static_cast<c10::complex<float>>(gamma[d * N + n]);
        q_shared[warp * N + n] = q_v;
        gamma_shared[warp * N + n] = gamma_v;

        for (int b = 0; b < B; ++b){
            c10::complex<T_ACC> h0_v = h0 ? h0[(b * D + d) * N + n] : c10::complex<T_ACC>(0);
            running[(warp * B + b) * N + n] = make_float4(1.f, 0.f, h0_v.real(), h0_v.imag());
        }
    }
    __syncthreads();

    for (int c = 0; c < num_chunks; ++c) {
        int64_t t = c * chunk_sz + lane;
        bool valid = (t < L);

        for (int b = 0; b < B; ++b){
            T_ACC y_contrib = T_ACC(0);
            float x_v = 0.f, f_mask = 1.f;
            if (valid) {
                x_v = static_cast<float>(x[(b * D + d) * L + t]);
                bool bos = bos_mask ? bos_mask[b * L + t] : false;
                f_mask = bos ? 0.f : 1.f;
            }

            for (int n = 0; n < N; ++n) {
                // Load for prefix scan
                float4 ht = op.identity();
                if (valid) {
                    float p_n = p_shared[warp * N + n];
                    c10::complex<float> q_n = q_shared[warp * N + n];
                    ht = make_float4(f_mask * q_n.real(),
                                    f_mask * q_n.imag(),
                                    p_n * x_v,
                                    0.f);
                }

                float4 carry = running[(warp * B + b) * N + n];

                // Warp scan float4
                {
                    for (int delta = 1; delta < kWarpSize; delta <<= 1) {
                        float4 up;
                        up.x = __shfl_up_sync(mask, ht.x, delta, kWarpSize);
                        up.y = __shfl_up_sync(mask, ht.y, delta, kWarpSize);
                        up.z = __shfl_up_sync(mask, ht.z, delta, kWarpSize);
                        up.w = __shfl_up_sync(mask, ht.w, delta, kWarpSize);
                        if ((lane % kWarpSize) >= delta) ht = op(up, ht);
                    }
                }

                ht = op(carry, ht);

                // y contribution
                if (valid) {
                    c10::complex<float> gamma_n = gamma_shared[warp * N + n];
                    y_contrib = fmaf(ht.z, static_cast<float>(gamma_n.real()), y_contrib);
                    y_contrib = fmaf(-ht.w, static_cast<float>(gamma_n.imag()), y_contrib);

                    if (t == L - 1) h[(b * D + d) * N + n] = {ht.z, ht.w};
                }

                // Store prev running
                if (lane == 0) {
                    int idx = ((b * D + d) * num_chunks + c) * N + n; // [B, D, C, N]
                    chunk_decay[idx] = {carry.x, carry.y};
                    chunk_gain[idx] = {carry.z, carry.w};
                }

                // Update running
                if (lane == 31) {
                    running[(warp * B + b) * N + n] = ht;
                }
            }

            // Update y
            if (valid) y[(b * D + d) * L + t] = static_cast<T>(y_contrib);
        }
    }
}

template<typename T, typename T_ACC, int64_t B, int64_t N>
__global__ void CEMAScanFwdKernel(
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
    c10::complex<T_ACC>* __restrict__ chunk_gain){
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int d = blockIdx.x * d_tile + warp;

    if (d >= D) return;

    const int64_t num_chunks = (L + chunk_sz - 1) / chunk_sz;

    ComplexRightAffineProd op;

    extern __shared__ float fwd_shm[];
    float* p_shared = fwd_shm; // d_tile * N
    c10::complex<float>* q_shared = reinterpret_cast<c10::complex<float>*>(p_shared + d_tile * N); // d_tile * N
    c10::complex<float>* gamma_shared = q_shared + d_tile * N; // d_tile * N
    float4* running = reinterpret_cast<float4*>(gamma_shared + d_tile * N); // B * d_tile * N

    if (lane < N) {
        int n = lane;
        p_shared[warp * N + n] = static_cast<float>(p[d * N + n]);
        c10::complex<float> q_v = static_cast<c10::complex<float>>(q[d * N + n]);
        c10::complex<float> gamma_v = static_cast<c10::complex<float>>(gamma[d * N + n]);
        q_shared[warp * N + n] = q_v;
        gamma_shared[warp * N + n] = gamma_v;

        #pragma unroll
        for (int b = 0; b < B; ++b){
            c10::complex<T_ACC> h0_v = h0 ? h0[(b * D + d) * N + n] : c10::complex<T_ACC>(0);
            running[(warp * B + b) * N + n] = make_float4(1.f, 0.f, h0_v.real(), h0_v.imag());
        }
    }
    __syncthreads();

    for (int c = 0; c < num_chunks; ++c) {
        int64_t t = c * chunk_sz + lane;
        bool valid = (t < L);

        #pragma unroll 1
        for (int b = 0; b < B; ++b){
            T_ACC y_contrib = T_ACC(0);
            float x_v = 0.f, f_mask = 1.f;
            if (valid) {
                x_v = static_cast<float>(x[(b * D + d) * L + t]);
                bool bos = bos_mask ? bos_mask[b * L + t] : false;
                f_mask = bos ? 0.f : 1.f;
            }

            #pragma unroll 4
            for (int n = 0; n < N; ++n) {
                // Load for prefix scan
                float4 ht = op.identity();
                if (valid) {
                    float p_n = p_shared[warp * N + n];
                    c10::complex<float> q_n = q_shared[warp * N + n];
                    ht = make_float4(f_mask * q_n.real(),
                                    f_mask * q_n.imag(),
                                    p_n * x_v,
                                    0.f);
                }

                float4 carry = running[(warp * B + b) * N + n];

                // Warp scan float4
                {
                    for (int delta = 1; delta < kWarpSize; delta <<= 1) {
                        float4 up;
                        up.x = __shfl_up_sync(mask, ht.x, delta, kWarpSize);
                        up.y = __shfl_up_sync(mask, ht.y, delta, kWarpSize);
                        up.z = __shfl_up_sync(mask, ht.z, delta, kWarpSize);
                        up.w = __shfl_up_sync(mask, ht.w, delta, kWarpSize);
                        if ((lane % kWarpSize) >= delta) ht = op(up, ht);
                    }
                }

                ht = op(carry, ht);

                // y contribution
                if (valid) {
                    c10::complex<float> gamma_n = gamma_shared[warp * N + n];
                    y_contrib = fmaf(ht.z, static_cast<float>(gamma_n.real()), y_contrib);
                    y_contrib = fmaf(-ht.w, static_cast<float>(gamma_n.imag()), y_contrib);

                    if (t == L - 1) h[(b * D + d) * N + n] = {ht.z, ht.w};
                }

                // Store prev running
                if (lane == 0) {
                    int idx = ((b * D + d) * num_chunks + c) * N + n; // [B, D, C, N]
                    chunk_decay[idx] = {carry.x, carry.y};
                    chunk_gain[idx] = {carry.z, carry.w};
                }

                // Update running
                if (lane == 31) {
                    running[(warp * B + b) * N + n] = ht;
                }
            }

            // Update y
            if (valid) y[(b * D + d) * L + t] = static_cast<T>(y_contrib);
        }
    }
}

template <typename T, typename T_ACC>
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
    c10::complex<T_ACC>* __restrict__ h0_grad){
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int d = blockIdx.x * d_tile + warp;

    if (d >= D) return;

    const int64_t num_chunks = (L + chunk_sz - 1) / chunk_sz;

    ComplexRightAffineProd prefixOp;
    ComplexLeftAffineProd suffixOp;

    extern __shared__ float bwd_shm[];
    float* p_shared = bwd_shm; // d_tile * N
    float* p_grad_shared = p_shared + d_tile * N; // d_tile * N
    c10::complex<float>* q_shared = reinterpret_cast<c10::complex<float>*>(p_grad_shared + d_tile * N); // d_tile * N
    c10::complex<float>* q_grad_shared = q_shared + d_tile * N; // d_tile * N
    c10::complex<float>* gamma_shared = q_grad_shared + d_tile * N; // d_tile * N
    c10::complex<float>* gamma_grad_shared = gamma_shared + d_tile * N; // d_tile * N
    c10::complex<float>* decay_shared = gamma_grad_shared + d_tile * N; // d_tile * N
    c10::complex<float>* gain_shared = decay_shared + d_tile * N; // d_tile * N
    float4* running = reinterpret_cast<float4*>(gain_shared + d_tile * N); // B * d_tile * N

    if (lane < N) {
        int n = lane;
        p_shared[warp * N + n] = static_cast<float>(p[d * N + n]);
        c10::complex<float> q_v = static_cast<c10::complex<float>>(q[d * N + n]);
        c10::complex<float> gamma_v = static_cast<c10::complex<float>>(gamma[d * N + n]);
        q_shared[warp * N + n] = q_v;
        gamma_shared[warp * N + n] = gamma_v;

        p_grad_shared[warp * N + n] = 0.f;
        q_grad_shared[warp * N + n] = {0.f, 0.f};
        gamma_grad_shared[warp * N + n] = {0.f, 0.f};

        for (int b = 0; b < B; ++b) {
            c10::complex<T_ACC> h_last_grad_v = h_last_grad ? h_last_grad[(b * D + d) * N + n] : c10::complex<T_ACC>(0);
            running[(warp * B + b) * N + n] = make_float4(1.f, 0.f, h_last_grad_v.real(), h_last_grad_v.imag());
        }
    }
    __syncthreads();

    for (int c = num_chunks - 1; c >= 0; --c) {
        int64_t t = c * chunk_sz + lane;
        bool valid = (t < L);

        for (int b = 0; b < B; ++b){
            T_ACC x_grad_contrib = T_ACC(0);
            float x_v = 0.f, y_grad_v = 0.f, f_mask = 1.f, next_mask = 1.f;

            if (valid) {
                x_v = static_cast<float>(x[(b * D + d) * L + t]);
                y_grad_v = static_cast<float>(y_grad[(b * D + d) * L + t]);
                bool bos = bos_mask ? bos_mask[b * L + t] : false;
                bool next_bos = bos_mask ? (t < (L - 1)? bos_mask[b * L + t + 1] : true): false;
                f_mask = bos ? 0.f : 1.f;
                next_mask = next_bos ? 0.f : 1.f;
            }

            if (lane < N) {
                int n = lane;
                int idx = ((b * D + d) * num_chunks + c) * N + n; // [B, D, C, N]
                c10::complex<float> decay = static_cast<c10::complex<float>>(chunk_decay[idx]);
                c10::complex<float> gain = static_cast<c10::complex<float>>(chunk_gain[idx]);
                decay_shared[warp * N + n] = decay;
                gain_shared[warp * N + n] = gain;
            }

            for (int n = 0; n < N; ++n){
                // Load for prefix scan
                float4 ht = prefixOp.identity();
                float4 ht_prev;
                float4 ht_grad = suffixOp.identity();

                float p_n = p_shared[warp * N + n];
                c10::complex<float> q_n = q_shared[warp * N + n];
                c10::complex<float> gamma_n = gamma_shared[warp * N + n];

                {
                    if (valid) {
                        ht = make_float4(f_mask * q_n.real(),
                                        f_mask * q_n.imag(),
                                        p_n * x_v,
                                        0.f);
                    }

                    c10::complex<float> decay = decay_shared[warp * N + n];
                    c10::complex<float> gain = gain_shared[warp * N + n];
                    float4 carry = make_float4(decay.real(), decay.imag(), gain.real(), gain.imag());

                    // Warp prefix scan h
                    // Inclusive
                    for (int delta = 1; delta < kWarpSize; delta <<= 1) {
                        float4 up;
                        up.x = __shfl_up_sync(mask, ht.x, delta, kWarpSize);
                        up.y = __shfl_up_sync(mask, ht.y, delta, kWarpSize);
                        up.z = __shfl_up_sync(mask, ht.z, delta, kWarpSize);
                        up.w = __shfl_up_sync(mask, ht.w, delta, kWarpSize);
                        if ((lane % kWarpSize) >= delta) ht = prefixOp(up, ht);
                    }

                    ht = prefixOp(carry, ht); // h_t

                    // Exclusive
                    ht_prev.x = __shfl_up_sync(mask, ht.x, 1, kWarpSize);
                    ht_prev.y = __shfl_up_sync(mask, ht.y, 1, kWarpSize);
                    ht_prev.z = __shfl_up_sync(mask, ht.z, 1, kWarpSize);
                    ht_prev.w = __shfl_up_sync(mask, ht.w, 1, kWarpSize);

                    if (lane == 0) {
                        ht_prev = carry;
                    }

                    // Load for suffix scan
                    carry = running[(warp * B + b) * N + n];

                    if (valid) {
                        ht_grad = make_float4((t == L - 1)? 1.f : next_mask * q_n.real(),
                                            (t == L - 1)? 0.f : next_mask * -q_n.imag(),
                                            y_grad_v * gamma_n.real(),
                                            -y_grad_v * gamma_n.imag());
                    }

                    // Warp suffix scan h grad
                    for (int delta = 1; delta < kWarpSize; delta <<= 1) {
                        float4 down;
                        down.x = __shfl_down_sync(mask, ht_grad.x, delta, kWarpSize);
                        down.y = __shfl_down_sync(mask, ht_grad.y, delta, kWarpSize);
                        down.z = __shfl_down_sync(mask, ht_grad.z, delta, kWarpSize);
                        down.w = __shfl_down_sync(mask, ht_grad.w, delta, kWarpSize);

                        if (lane + delta < kWarpSize) ht_grad = suffixOp(ht_grad, down);
                    }

                    ht_grad = suffixOp(ht_grad, carry);
                }

                // grad
                {
                    float local_p_grad = 0;
                    float local_q_grad_real = 0, local_q_grad_imag = 0;
                    float local_gamma_grad_real = 0, local_gamma_grad_imag = 0;

                    if (valid) {
                        // x grad
                        x_grad_contrib = fmaf(p_n, ht_grad.z, x_grad_contrib);

                        // p grad
                        local_p_grad = x_v * ht_grad.z;

                        // q grad
                        local_q_grad_real = f_mask * fmaf(ht_prev.w,  ht_grad.w, ht_prev.z * ht_grad.z);
                        local_q_grad_imag = f_mask * fmaf(-ht_prev.w, ht_grad.z, ht_prev.z * ht_grad.w);

                        // gamma grad
                        local_gamma_grad_real = ht.z * y_grad_v;
                        local_gamma_grad_imag = -ht.w * y_grad_v;
                    }

                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        local_p_grad += __shfl_down_sync(mask, local_p_grad, offset);
                        local_q_grad_real += __shfl_down_sync(mask, local_q_grad_real, offset);
                        local_q_grad_imag += __shfl_down_sync(mask, local_q_grad_imag, offset);
                        local_gamma_grad_real += __shfl_down_sync(mask, local_gamma_grad_real, offset);
                        local_gamma_grad_imag += __shfl_down_sync(mask, local_gamma_grad_imag, offset);
                    }

                    if (lane == 0) {
                        // Accumulate grad
                        p_grad_shared[warp * N + n] += local_p_grad;
                        q_grad_shared[warp * N + n] += c10::complex<float>(local_q_grad_real, local_q_grad_imag);
                        gamma_grad_shared[warp * N + n] += c10::complex<float>(local_gamma_grad_real, local_gamma_grad_imag);

                        // Update running
                        running[(warp * B + b) * N + n] = ht_grad;
                    }
                }
            }

            // Update x grad
            if (valid) x_grad[(b * D + d) * L + t] = static_cast<T>(x_grad_contrib);
        }
    }
    if (lane < N) {
        int n = lane;
        p_grad[d * N + n] = static_cast<T_ACC>(p_grad_shared[warp * N + n]);
        q_grad[d * N + n] = static_cast<c10::complex<T_ACC>>(q_grad_shared[warp * N + n]);
        gamma_grad[d * N + n] = static_cast<c10::complex<T_ACC>>(gamma_grad_shared[warp * N + n]);

        for (int b = 0; b < B; ++b) {
            float4 h_grad_running = running[(warp * B + b) * N + n];
            bool bos_at_0 = bos_mask ? bos_mask[b * L] : false;
            float mask_val = bos_at_0 ? 0.f : 1.f;
            c10::complex<float> q_v = q_shared[warp * N + n];

            float h0_grad_real = mask_val * fmaf(h_grad_running.w, q_v.imag(), h_grad_running.z * q_v.real());
            float h0_grad_imag = mask_val * fmaf(-h_grad_running.z, q_v.imag(), h_grad_running.w * q_v.real());

            h0_grad[(b * D + d) * N + n] = {h0_grad_real, h0_grad_imag};
        }
    }
}

template <typename T, typename T_ACC, int64_t B, int64_t N>
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
    c10::complex<T_ACC>* __restrict__ h0_grad){
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int d = blockIdx.x * d_tile + warp;

    if (d >= D) return;

    const int64_t num_chunks = (L + chunk_sz - 1) / chunk_sz;

    ComplexRightAffineProd prefixOp;
    ComplexLeftAffineProd suffixOp;

    extern __shared__ float bwd_shm[];
    float* p_shared = bwd_shm; // d_tile * N
    float* p_grad_shared = p_shared + d_tile * N; // d_tile * N
    c10::complex<float>* q_shared = reinterpret_cast<c10::complex<float>*>(p_grad_shared + d_tile * N); // d_tile * N
    c10::complex<float>* q_grad_shared = q_shared + d_tile * N; // d_tile * N
    c10::complex<float>* gamma_shared = q_grad_shared + d_tile * N; // d_tile * N
    c10::complex<float>* gamma_grad_shared = gamma_shared + d_tile * N; // d_tile * N
    c10::complex<float>* decay_shared = gamma_grad_shared + d_tile * N; // d_tile * N
    c10::complex<float>* gain_shared = decay_shared + d_tile * N; // d_tile * N
    float4* running = reinterpret_cast<float4*>(gain_shared + d_tile * N); // B * d_tile * N

    if (lane < N) {
        int n = lane;
        p_shared[warp * N + n] = static_cast<float>(p[d * N + n]);
        c10::complex<float> q_v = static_cast<c10::complex<float>>(q[d * N + n]);
        c10::complex<float> gamma_v = static_cast<c10::complex<float>>(gamma[d * N + n]);
        q_shared[warp * N + n] = q_v;
        gamma_shared[warp * N + n] = gamma_v;

        p_grad_shared[warp * N + n] = 0.f;
        q_grad_shared[warp * N + n] = {0.f, 0.f};
        gamma_grad_shared[warp * N + n] = {0.f, 0.f};

        for (int b = 0; b < B; ++b) {
            c10::complex<T_ACC> h_last_grad_v = h_last_grad ? h_last_grad[(b * D + d) * N + n] : c10::complex<T_ACC>(0);
            running[(warp * B + b) * N + n] = make_float4(1.f, 0.f, h_last_grad_v.real(), h_last_grad_v.imag());
        }
    }
    __syncthreads();

    for (int c = num_chunks - 1; c >= 0; --c) {
        int64_t t = c * chunk_sz + lane;
        bool valid = (t < L);

        #pragma unroll 1
        for (int b = 0; b < B; ++b){
            T_ACC x_grad_contrib = T_ACC(0);
            float x_v = 0.f, y_grad_v = 0.f, f_mask = 1.f, next_mask = 1.f;

            if (valid) {
                x_v = static_cast<float>(x[(b * D + d) * L + t]);
                y_grad_v = static_cast<float>(y_grad[(b * D + d) * L + t]);
                bool bos = bos_mask ? bos_mask[b * L + t] : false;
                bool next_bos = bos_mask ? (t < (L - 1)? bos_mask[b * L + t + 1] : true): false;
                f_mask = bos ? 0.f : 1.f;
                next_mask = next_bos ? 0.f : 1.f;
            }

            if (lane < N) {
                int n = lane;
                int idx = ((b * D + d) * num_chunks + c) * N + n; // [B, D, C, N]
                c10::complex<float> decay = static_cast<c10::complex<float>>(chunk_decay[idx]);
                c10::complex<float> gain = static_cast<c10::complex<float>>(chunk_gain[idx]);
                decay_shared[warp * N + n] = decay;
                gain_shared[warp * N + n] = gain;
            }

            #pragma unroll 4
            for (int n = 0; n < N; ++n){
                // Load for prefix scan
                float4 ht = prefixOp.identity();
                float4 ht_prev;
                float4 ht_grad = suffixOp.identity();

                float p_n = p_shared[warp * N + n];
                c10::complex<float> q_n = q_shared[warp * N + n];
                c10::complex<float> gamma_n = gamma_shared[warp * N + n];

                {
                    if (valid) {
                        ht = make_float4(f_mask * q_n.real(),
                                        f_mask * q_n.imag(),
                                        p_n * x_v,
                                        0.f);
                    }

                    c10::complex<float> decay = decay_shared[warp * N + n];
                    c10::complex<float> gain = gain_shared[warp * N + n];
                    float4 carry = make_float4(decay.real(), decay.imag(), gain.real(), gain.imag());

                    // Warp prefix scan h
                    // Inclusive
                    for (int delta = 1; delta < kWarpSize; delta <<= 1) {
                        float4 up;
                        up.x = __shfl_up_sync(mask, ht.x, delta, kWarpSize);
                        up.y = __shfl_up_sync(mask, ht.y, delta, kWarpSize);
                        up.z = __shfl_up_sync(mask, ht.z, delta, kWarpSize);
                        up.w = __shfl_up_sync(mask, ht.w, delta, kWarpSize);
                        if ((lane % kWarpSize) >= delta) ht = prefixOp(up, ht);
                    }

                    ht = prefixOp(carry, ht); // h_t

                    // Exclusive
                    ht_prev.x = __shfl_up_sync(mask, ht.x, 1, kWarpSize);
                    ht_prev.y = __shfl_up_sync(mask, ht.y, 1, kWarpSize);
                    ht_prev.z = __shfl_up_sync(mask, ht.z, 1, kWarpSize);
                    ht_prev.w = __shfl_up_sync(mask, ht.w, 1, kWarpSize);

                    if (lane == 0) {
                        ht_prev = carry;
                    }

                    // Load for suffix scan
                    carry = running[(warp * B + b) * N + n];

                    if (valid) {
                        ht_grad = make_float4((t == L - 1)? 1.f : next_mask * q_n.real(),
                                            (t == L - 1)? 0.f : next_mask * -q_n.imag(),
                                            y_grad_v * gamma_n.real(),
                                            -y_grad_v * gamma_n.imag());
                    }

                    // Warp suffix scan h grad
                    for (int delta = 1; delta < kWarpSize; delta <<= 1) {
                        float4 down;
                        down.x = __shfl_down_sync(mask, ht_grad.x, delta, kWarpSize);
                        down.y = __shfl_down_sync(mask, ht_grad.y, delta, kWarpSize);
                        down.z = __shfl_down_sync(mask, ht_grad.z, delta, kWarpSize);
                        down.w = __shfl_down_sync(mask, ht_grad.w, delta, kWarpSize);

                        if (lane + delta < kWarpSize) ht_grad = suffixOp(ht_grad, down);
                    }

                    ht_grad = suffixOp(ht_grad, carry);
                }

                // grad
                {
                    float local_p_grad = 0;
                    float local_q_grad_real = 0, local_q_grad_imag = 0;
                    float local_gamma_grad_real = 0, local_gamma_grad_imag = 0;

                    if (valid) {
                        // x grad
                        x_grad_contrib = fmaf(p_n, ht_grad.z, x_grad_contrib);

                        // p grad
                        local_p_grad = x_v * ht_grad.z;

                        // q grad
                        local_q_grad_real = f_mask * fmaf(ht_prev.w,  ht_grad.w, ht_prev.z * ht_grad.z);
                        local_q_grad_imag = f_mask * fmaf(-ht_prev.w, ht_grad.z, ht_prev.z * ht_grad.w);

                        // gamma grad
                        local_gamma_grad_real = ht.z * y_grad_v;
                        local_gamma_grad_imag = -ht.w * y_grad_v;
                    }

                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        local_p_grad += __shfl_down_sync(mask, local_p_grad, offset);
                        local_q_grad_real += __shfl_down_sync(mask, local_q_grad_real, offset);
                        local_q_grad_imag += __shfl_down_sync(mask, local_q_grad_imag, offset);
                        local_gamma_grad_real += __shfl_down_sync(mask, local_gamma_grad_real, offset);
                        local_gamma_grad_imag += __shfl_down_sync(mask, local_gamma_grad_imag, offset);
                    }

                    if (lane == 0) {
                        // Accumulate grad
                        p_grad_shared[warp * N + n] += local_p_grad;
                        q_grad_shared[warp * N + n] += c10::complex<float>(local_q_grad_real, local_q_grad_imag);
                        gamma_grad_shared[warp * N + n] += c10::complex<float>(local_gamma_grad_real, local_gamma_grad_imag);

                        // Update running
                        running[(warp * B + b) * N + n] = ht_grad;
                    }
                }
            }

            // Update x grad
            if (valid) x_grad[(b * D + d) * L + t] = static_cast<T>(x_grad_contrib);
        }
    }
    if (lane < N) {
        int n = lane;
        p_grad[d * N + n] = static_cast<T_ACC>(p_grad_shared[warp * N + n]);
        q_grad[d * N + n] = static_cast<c10::complex<T_ACC>>(q_grad_shared[warp * N + n]);
        gamma_grad[d * N + n] = static_cast<c10::complex<T_ACC>>(gamma_grad_shared[warp * N + n]);

        #pragma unroll
        for (int b = 0; b < B; ++b) {
            float4 h_grad_running = running[(warp * B + b) * N + n];
            bool bos_at_0 = bos_mask ? bos_mask[b * L] : false;
            float mask_val = bos_at_0 ? 0.f : 1.f;
            c10::complex<float> q_v = q_shared[warp * N + n];

            float h0_grad_real = mask_val * fmaf(h_grad_running.w, q_v.imag(), h_grad_running.z * q_v.real());
            float h0_grad_imag = mask_val * fmaf(-h_grad_running.z, q_v.imag(), h_grad_running.w * q_v.real());

            h0_grad[(b * D + d) * N + n] = {h0_grad_real, h0_grad_imag};
        }
    }
}

#define DISPATCH_FWD_KERNEL(                                          \
    KernelFunc, T, T_ACC, shm_size, cuda_stream, B, D, N, L, ...)                \
  do {       \
    if (B == 1){ \
        if (N == 4) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 1, 4>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 8) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 1, 8>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 16) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 1, 16>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else { \
            KernelFunc<T, T_ACC>                               \
                <<<dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream>>>(1, D, N, L, __VA_ARGS__); \
                    C10_CUDA_KERNEL_LAUNCH_CHECK(); \
        } \
    } \
    else if (B == 2){ \
        if (N == 4) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2, 4>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 8) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2, 8>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 16) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2, 16>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else { \
            KernelFunc<T, T_ACC>                               \
                <<<dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream>>>(2, D, N, L, __VA_ARGS__); \
                    C10_CUDA_KERNEL_LAUNCH_CHECK(); \
        } \
    } \
    else if (B == 4){ \
        if (N == 4) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4, 4>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 8) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4, 8>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 16) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4, 16>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else { \
            KernelFunc<T, T_ACC>                               \
                <<<dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream>>>(4, D, N, L, __VA_ARGS__); \
                    C10_CUDA_KERNEL_LAUNCH_CHECK(); \
        } \
    } \
    else if (B == 8){ \
        if (N == 4) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8, 4>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 8) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8, 8>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 16) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8, 16>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else { \
            KernelFunc<T, T_ACC>                               \
                <<<dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream>>>(8, D, N, L, __VA_ARGS__); \
                    C10_CUDA_KERNEL_LAUNCH_CHECK(); \
        } \
    } \
    else { \
        KernelFunc<T, T_ACC>                               \
                <<<dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream>>>(B, D, N, L, __VA_ARGS__); \
            C10_CUDA_KERNEL_LAUNCH_CHECK(); \
    } \
  } while (false)


#define DISPATCH_BWD_KERNEL(                                          \
    KernelFunc, T, T_ACC, shm_size, cuda_stream, B, D, N, L, ...)                \
  do {       \
    if (B == 1){ \
        if (N == 4) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 1, 4>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 8) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 1, 8>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 16) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 1, 16>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else { \
            KernelFunc<T, T_ACC>                               \
                <<<dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream>>>(1, D, N, L, __VA_ARGS__); \
                    C10_CUDA_KERNEL_LAUNCH_CHECK(); \
        } \
    } \
    else if (B == 2){ \
        if (N == 4) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2, 4>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 8) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2, 8>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 16) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 2, 16>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else { \
            KernelFunc<T, T_ACC>                               \
                <<<dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream>>>(2, D, N, L, __VA_ARGS__); \
                    C10_CUDA_KERNEL_LAUNCH_CHECK(); \
        } \
    } \
    else if (B == 4){ \
        if (N == 4) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4, 4>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 8) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4, 8>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 16) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 4, 16>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else { \
            KernelFunc<T, T_ACC>                               \
                <<<dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream>>>(4, D, N, L, __VA_ARGS__); \
                    C10_CUDA_KERNEL_LAUNCH_CHECK(); \
        } \
    } \
    else if (B == 8){ \
        if (N == 4) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8, 4>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 8) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8, 8>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else if (N == 16) { \
            cuda_utils::LaunchKernel(KernelFunc<T, T_ACC, 8, 16>, \
                dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream, D, L, __VA_ARGS__); \
        } \
        else { \
            KernelFunc<T, T_ACC>                               \
                <<<dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream>>>(8, D, N, L, __VA_ARGS__); \
                    C10_CUDA_KERNEL_LAUNCH_CHECK(); \
        } \
    } \
    else { \
        KernelFunc<T, T_ACC>                               \
                <<<dim3((D + d_tile - 1) / d_tile, 1, 1), \
                dim3(d_tile * 32, 1, 1), shm_size, cuda_stream>>>(B, D, N, L, __VA_ARGS__); \
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

    const size_t shm_size = (d_tile * N) * sizeof(float)
                        + 2 * (d_tile * N) * sizeof(c10::complex<float>)
                        + (B * d_tile * N) * sizeof(float4);

    DISPATCH_FWD_KERNEL(
        CEMAScanFwdKernel, T, T_ACC,
        shm_size, cuda_stream,
        B, D, N, L,
        x_data, p_data, q_data, gamma_data,
        bos_mask_data, h0_data,
        h_data, y_data, chunk_decay_data, chunk_gain_data);
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

    const size_t shm_size = 2 * d_tile * N * sizeof(float)
                            + 6 * d_tile * N * sizeof(c10::complex<float>)
                            + (B * d_tile * N) * sizeof(float4);;

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
}

std::tuple<torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor> CEMAScanCUDAFwd(const torch::Tensor& x,
                                                        const torch::Tensor& p,
                                                        const torch::Tensor& q,
                                                        const torch::Tensor& gamma,
                                                        const c10::optional<torch::Tensor>& bos_mask,
                                                        const c10::optional<torch::Tensor>& h0) {
    const int64_t B = x.size(0);
    const int64_t D = x.size(1);
    const int64_t L = x.size(2);
    const int64_t N = p.size(1);

    TORCH_CHECK(p.size(1) <= 32, "CEMAScanCUDAFwd only supports N <= 32 (got N=", p.size(1), ").");

    const int64_t num_chunks = (L + chunk_sz - 1) / chunk_sz;

    c10::MaybeOwned<torch::Tensor> bos_mask_maybe_owned =
        at::borrow_from_optional_tensor(bos_mask);
    c10::MaybeOwned<torch::Tensor> h0_maybe_owned =
        at::borrow_from_optional_tensor(h0);
    torch::Tensor h = torch::empty(
        {B, D, N}, q.options().memory_format(at::MemoryFormat::Contiguous));
    torch::Tensor y = torch::zeros(
        {B, D, L}, x.options().memory_format(at::MemoryFormat::Contiguous));
    torch::Tensor chunk_decay = torch::empty(
        {B, D, num_chunks, N}, q.options().memory_format(at::MemoryFormat::Contiguous));
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor,
torch::Tensor, torch::Tensor> CEMAScanCUDABwd(const torch::Tensor& y_grad,
                                            const c10::optional<torch::Tensor>& h_last_grad,
                                            const torch::Tensor& chunk_decay,
                                            const torch::Tensor& chunk_gain,
                                            const torch::Tensor& x,
                                            const torch::Tensor& p,
                                            const torch::Tensor& q,
                                            const torch::Tensor& gamma,
                                            const c10::optional<torch::Tensor>& bos_mask) {
    const int64_t B = x.size(0);
    const int64_t D = x.size(1);
    const int64_t L = x.size(2);
    const int64_t N = p.size(1);

    TORCH_CHECK(p.size(1) <= 32, "CEMAScanCUDABwd only supports N <= 32 (got N=", p.size(1), ").");

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
}
}
