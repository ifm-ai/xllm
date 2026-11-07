#pragma once

// Shared implementation of GroupLayerNorm and GroupRMSNorm.
//
// x of shape [outer_size, num_channels] is viewed as num_segments = outer_size * num_groups
// contiguous segments of channels_per_group (cpg) elements: segment s is group s % num_groups
// of row s / num_groups, and mean[s] / rstd[s] are its statistics.
//
// Each segment is normalized by a team of threads. Teams of up to a warp are packed into blocks
// of kSubWarpBlockSize threads and reduce with warp shuffles; larger teams are a whole block and
// reduce through shared memory. Segments longer than a block can hold (kMaxBlockChannels) use a
// thread block cluster as the team (sm_90+), whose blocks combine their block totals through
// distributed shared memory. Every thread keeps at most kElemsPerThread elements of its segment
// in registers and moves kVec elements per access. Teams loop over segments.
//
// The affine backward pass also accumulates dgamma/dbeta. Its grid is sized so that the number
// of teams is a multiple of num_groups, which pins every team to one group: the team sums
// dgamma/dbeta over all of its rows in registers and writes a single partial row.
// SumPartialsKernel reduces the [num_phases, num_channels] partials.

#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <cooperative_groups.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <type_traits>
#include <utility>

#include "../cuda_utils.cuh"
#include "../reduce.cuh"
#include "../utils.h"

namespace xllm {
namespace ops {
namespace group_norm {

constexpr int64_t kVecBytes = 16;
constexpr int64_t kElemsPerThread = 8;
constexpr int64_t kMaxBlockSize = 1024;
constexpr int64_t kMaxBlockChannels = kElemsPerThread * kMaxBlockSize;
constexpr int64_t kMaxClusterSize = 8;  // the portable limit
constexpr int64_t kMaxChannelsPerGroup = kMaxClusterSize * kMaxBlockChannels;
constexpr int64_t kSubWarpBlockSize = 256;

template <typename T, int kVec>
struct alignas(sizeof(T) * kVec) Vec {
  T val[kVec];
};

template <int kVec, typename SRC, typename DST>
__device__ __forceinline__ void LoadVec(const SRC* ptr, DST* out) {
  const Vec<SRC, kVec> v = *reinterpret_cast<const Vec<SRC, kVec>*>(ptr);
#pragma unroll
  for (int k = 0; k < kVec; ++k) {
    out[k] = static_cast<DST>(v.val[k]);
  }
}

template <int kVec, typename SRC, typename DST>
__device__ __forceinline__ void StoreVec(const SRC* in, DST* ptr) {
  Vec<DST, kVec> v;
#pragma unroll
  for (int k = 0; k < kVec; ++k) {
    v.val[k] = static_cast<DST>(in[k]);
  }
  *reinterpret_cast<Vec<DST, kVec>*>(ptr) = v;
}

template <int kVec, typename T>
__device__ __forceinline__ void FillVec(T value, T* out) {
#pragma unroll
  for (int k = 0; k < kVec; ++k) {
    out[k] = value;
  }
}

// The threads that normalize one segment.
enum class TeamKind {
  kSubWarp,  // up to a warp; kSubWarpBlockSize / team_size teams share a block
  kBlock,    // a whole block
  kCluster,  // a thread block cluster (sm_90+)
};

// The cluster helpers are only reachable from kCluster kernels, which are launched on sm_90+.
__device__ __forceinline__ unsigned ClusterBlocks() {
#if defined(_CG_HAS_CLUSTER_GROUP)
  return cooperative_groups::this_cluster().num_blocks();
#else
  __trap();
  return 1;
#endif
}

__device__ __forceinline__ unsigned ClusterRank() {
#if defined(_CG_HAS_CLUSTER_GROUP)
  return cooperative_groups::this_cluster().block_rank();
#else
  __trap();
  return 0;
#endif
}

__device__ __forceinline__ void ClusterSync() {
#if defined(_CG_HAS_CLUSTER_GROUP)
  cooperative_groups::this_cluster().sync();
#else
  __trap();
#endif
}

// ClusterSum state in shared memory, double buffered by round parity: an mbarrier that completes
// once every block of the cluster has stored its values, and one slot per sending block.
template <typename T>
struct ClusterSlots {
  uint64_t bar[2];
  alignas(16) T recv[2][kMaxClusterSize][2];
};

template <typename T>
__device__ __forceinline__ ClusterSlots<T>& GetClusterSlots() {
  __shared__ ClusterSlots<T> slots;
  return slots;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ uint32_t SmemAddr(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// The address of the same shared memory location in block `rank` of the cluster.
__device__ __forceinline__ uint32_t MapRank(uint32_t addr, uint32_t rank) {
  uint32_t out;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(out) : "r"(addr), "r"(rank));
  return out;
}

__device__ __forceinline__ void ArriveExpectTx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void WaitParity(uint32_t bar, uint32_t parity) {
  uint32_t done;
  do {
    asm volatile(
        "{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}"
        : "=r"(done) : "r"(bar), "r"(parity) : "memory");
  } while (!done);
}

// Stores a and b to remote shared memory and completes their bytes on the remote mbarrier.
template <typename T>
__device__ __forceinline__ void StoreAsync(uint32_t dst, T a, T b, uint32_t bar) {
  if constexpr (sizeof(T) == 4) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];"
                 :: "r"(dst), "r"(__float_as_uint(a)), "r"(__float_as_uint(b)), "r"(bar) : "memory");
  } else {
    static_assert(sizeof(T) == 8);
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b64 [%0], {%1, %2}, [%3];"
                 :: "r"(dst), "l"(__double_as_longlong(a)), "l"(__double_as_longlong(b)), "r"(bar) : "memory");
  }
}
#endif

// Sums each of the kN (<= 2) values, already block totals held by every thread, over the blocks of
// the cluster. Thread r of every block sends the block's values to block r with st.async, so
// no cluster-wide barrier or memory fence is needed per round. Blocks add the slots in rank
// order. After waiting for round k, a block re-arms the same
// mbarrier for round k + 2. The other blocks can only send round k + 2 after this block has
// sent round k + 1, which it does after a __syncthreads that follows all its reads of round k,
// so both the mbarrier and the slots are free again by then.
template <int kN, typename T>
__device__ __forceinline__ void ClusterSum(T (&v)[kN], int& round) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  static_assert(kN == 1 || kN == 2);
  ClusterSlots<T>& slots = GetClusterSlots<T>();
  const uint32_t num_blocks = ClusterBlocks();
  const uint32_t tx_bytes = num_blocks * 2 * sizeof(T);
  if (round == 0) {
    if (threadIdx.x == 0) {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(SmemAddr(&slots.bar[i])) : "memory");
        ArriveExpectTx(SmemAddr(&slots.bar[i]), tx_bytes);
      }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    ClusterSync();  // no block may send before every mbarrier of the cluster is initialized
  }
  const int buf = round & 1;
  const uint32_t parity = (round >> 1) & 1;
  ++round;
  const uint32_t bar = SmemAddr(&slots.bar[buf]);
  if (threadIdx.x < num_blocks) {
    const uint32_t dst = MapRank(SmemAddr(&slots.recv[buf][ClusterRank()][0]), threadIdx.x);
    StoreAsync<T>(dst, v[0], kN > 1 ? v[kN - 1] : T(0), MapRank(bar, threadIdx.x));
  }
  WaitParity(bar, parity);
  if (threadIdx.x == 0) {
    ArriveExpectTx(bar, tx_bytes);
  }
#pragma unroll
  for (int k = 0; k < kN; ++k) {
    v[k] = T(0);
  }
  for (uint32_t r = 0; r < num_blocks; ++r) {
#pragma unroll
    for (int k = 0; k < kN; ++k) {
      v[k] += slots.recv[buf][r][k];
    }
  }
#else
  __trap();
#endif
}

// Where the calling thread sits: team `index` of the block, thread `lane` of the team.
template <TeamKind kKind>
struct Team {
  int size;
  int per_block;
  int index;
  int lane;
  mutable int round = 0;  // kCluster: ClusterSum rounds so far (picks its buffer and mbarrier phase)

  __device__ explicit Team(int team_size) {
    if constexpr (kKind == TeamKind::kSubWarp) {
      size = team_size;
      per_block = blockDim.x / team_size;
      index = threadIdx.x / team_size;
      lane = threadIdx.x % team_size;
    } else {
      size = blockDim.x * blocks();
      per_block = 1;
      index = 0;
      lane = rank() * blockDim.x + threadIdx.x;
    }
  }

  __device__ static unsigned blocks() {
    if constexpr (kKind == TeamKind::kCluster) {
      return ClusterBlocks();
    } else {
      return 1;
    }
  }

  __device__ static unsigned rank() {
    if constexpr (kKind == TeamKind::kCluster) {
      return ClusterRank();
    } else {
      return 0;
    }
  }

  // id() of team 0 of the block; uniform over the block, and over the cluster for kCluster.
  __device__ int64_t first() const { return blockIdx.x / blocks() * per_block; }
  __device__ int64_t id() const { return static_cast<int64_t>(blockIdx.x / blocks()) * per_block + index; }
  __device__ int64_t count() const { return static_cast<int64_t>(gridDim.x / blocks()) * per_block; }

  // Called once at the end of the kernel by every thread.
  __device__ void Finish() const {
    if constexpr (kKind == TeamKind::kCluster) {
      ClusterSync();  // no block exits while st.async stores to or from it may be in flight
    }
  }
};

// Sum over the threads of a team; every thread of the block (and cluster) must call it.
// round is Team::round, only used by kCluster.
template <TeamKind kKind, typename T>
__device__ __forceinline__ T TeamSum(T x, int64_t team_size, T* shm, int& round) {
  if constexpr (kKind == TeamKind::kSubWarp) {
    // Teams are aligned powers of two, so xor offsets below team_size stay inside the team.
    for (int offset = team_size >> 1; offset > 0; offset >>= 1) {
      x += cuda_utils::WarpShflXor(x, offset);
    }
    return x;
  } else {
    x = reduce::BlockAllReduce(x, shm, reduce::SumOp<T>());
    if constexpr (kKind == TeamKind::kCluster) {
      T v[1] = {x};
      ClusterSum(v, round);
      x = v[0];
    }
    return x;
  }
}

// TeamSum of two values with a single round of shared memory; shm holds 2 * kWarpSize values.
template <TeamKind kKind, typename T>
__device__ __forceinline__ void TeamSum2(T& a, T& b, int64_t team_size, T* shm, int& round) {
  if constexpr (kKind != TeamKind::kSubWarp) {
    const int64_t lid = threadIdx.x % cuda_utils::kWarpSize;
    const int64_t wid = threadIdx.x / cuda_utils::kWarpSize;
    const int64_t num_warps = blockDim.x / cuda_utils::kWarpSize;
    a = reduce::WarpReduce(a, reduce::SumOp<T>());
    b = reduce::WarpReduce(b, reduce::SumOp<T>());
    __syncthreads();  // shm may still be read by the previous reduction
    if (lid == 0) {
      shm[wid] = a;
      shm[cuda_utils::kWarpSize + wid] = b;
    }
    __syncthreads();
    a = lid < num_warps ? shm[lid] : T(0);
    b = lid < num_warps ? shm[cuda_utils::kWarpSize + lid] : T(0);
    a = reduce::WarpAllReduce(a, reduce::SumOp<T>());
    b = reduce::WarpAllReduce(b, reduce::SumOp<T>());
    if constexpr (kKind == TeamKind::kCluster) {
      T v[2] = {a, b};
      ClusterSum(v, round);
      a = v[0];
      b = v[1];
    }
  } else {
    for (int offset = team_size >> 1; offset > 0; offset >>= 1) {
      a += cuda_utils::WarpShflXor(a, offset);
      b += cuda_utils::WarpShflXor(b, offset);
    }
  }
}

// minBlocksPerMultiprocessor for __launch_bounds__; 0 omits the occupancy bound.
template <typename T, int kVec, TeamKind kTeam>
constexpr int ClusterMinBlocks(bool light_kernel) {
  return kTeam == TeamKind::kCluster && kVec > 1 && sizeof(T) <= 4 && light_kernel ? 2 : 0;
}

template <typename T, typename T_ACC>
struct FwdParams {
  int64_t num_segments;
  int64_t num_groups;
  int64_t channels_per_group;
  int64_t team_size;
  T_ACC eps;
  const T* x;
  const T* gamma;
  const T* beta;
  T* y;
  T_ACC* mean;
  T_ACC* rstd;
};

template <typename T, typename T_ACC, int kVec, TeamKind kTeam, bool kRMS, bool kAffine>
__global__ void __launch_bounds__(kTeam == TeamKind::kSubWarp ? kSubWarpBlockSize : kMaxBlockSize,
                                  ClusterMinBlocks<T, kVec, kTeam>(true))
GroupNormFwdKernel(FwdParams<T, T_ACC> p) {
  constexpr int kVecsPerThread = kElemsPerThread / kVec;
  __shared__ T_ACC shm[cuda_utils::kWarpSize];
  const Team<kTeam> team(p.team_size);
  const int64_t cpg = p.channels_per_group;
  const int num_vecs = static_cast<int>(cpg / kVec);
  const T_ACC inv_n = T_ACC(1) / static_cast<T_ACC>(cpg);

  // The loop bound is uniform over the block (and cluster) so that team reductions stay convergent.
  for (int64_t first = team.first(); first < p.num_segments; first += team.count()) {
    const int64_t s = first + team.index;
    const bool valid = s < p.num_segments;
    T_ACC v[kVecsPerThread][kVec];
    bool active[kVecsPerThread];
    T_ACC sum = T_ACC(0);
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      const int j = team.lane + i * team.size;
      active[i] = valid && j < num_vecs;
      if (active[i]) {
        LoadVec<kVec>(p.x + s * cpg + j * kVec, v[i]);
      } else {
        FillVec<kVec>(T_ACC(0), v[i]);
      }
#pragma unroll
      for (int k = 0; k < kVec; ++k) {
        sum += v[i][k];
      }
    }

    // Two-pass variance: the mean is subtracted before squaring.
    T_ACC mean = T_ACC(0);
    if constexpr (!kRMS) {
      mean = TeamSum<kTeam>(sum, team.size, shm, team.round) * inv_n;
    }
    T_ACC m2 = T_ACC(0);
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      if (active[i]) {
#pragma unroll
        for (int k = 0; k < kVec; ++k) {
          m2 += cuda_utils::Square(v[i][k] - mean);
        }
      }
    }
    const T_ACC rstd = c10::cuda::compat::rsqrt(TeamSum<kTeam>(m2, team.size, shm, team.round) * inv_n + p.eps);
    if (valid && team.lane == 0) {
      if constexpr (!kRMS) {
        p.mean[s] = mean;
      }
      p.rstd[s] = rstd;
    }

    const int64_t c0 = (s % p.num_groups) * cpg;
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      if (active[i]) {
        const int j = team.lane + i * team.size;
        T_ACC w[kVec];
        T_ACC b[kVec];
        if constexpr (kAffine) {
          LoadVec<kVec>(p.gamma + c0 + j * kVec, w);
          if constexpr (!kRMS) {
            LoadVec<kVec>(p.beta + c0 + j * kVec, b);
          }
        }
#pragma unroll
        for (int k = 0; k < kVec; ++k) {
          T_ACC out = (v[i][k] - mean) * rstd;
          if constexpr (kAffine) {
            out *= w[k];
            if constexpr (!kRMS) {
              out += b[k];
            }
          }
          v[i][k] = out;
        }
        StoreVec<kVec>(v[i], p.y + s * cpg + j * kVec);
      }
    }
  }
  team.Finish();
}

template <typename T, typename T_ACC>
struct BwdParams {
  int64_t num_segments;
  int64_t num_groups;
  int64_t channels_per_group;
  int64_t team_size;
  const T* dy;
  const T* x_or_y;
  const T_ACC* mean;
  const T_ACC* rstd;
  const T* gamma;
  const T* beta;
  T* dx;
  // Affine only: [num_teams / num_groups, num_channels] partial sums of dgamma and dbeta.
  T_ACC* dgamma_partial;
  T_ACC* dbeta_partial;
};

// With kMemoryEfficient, x_or_y is the forward output y and x_hat is recovered as
// (y - beta) / gamma; otherwise it is x and x_hat = (x - mean) * rstd.
//   dx = rstd * (g - mean(g) - x_hat * mean(g * x_hat)),  g = dy * gamma
// RMS norm drops the mean(g) term. Recovering x_hat with affine parameters requires nonzero gamma.
template <typename T, typename T_ACC, int kVec, TeamKind kTeam, bool kRMS, bool kAffine, bool kMemoryEfficient>
__global__ void __launch_bounds__(kTeam == TeamKind::kSubWarp ? kSubWarpBlockSize : kMaxBlockSize,
                                  ClusterMinBlocks<T, kVec, kTeam>(!kAffine))
GroupNormBwdKernel(BwdParams<T, T_ACC> p) {
  constexpr int kVecsPerThread = kElemsPerThread / kVec;
  constexpr bool kUseBeta = kAffine && kMemoryEfficient && !kRMS;
  __shared__ T_ACC shm[2 * cuda_utils::kWarpSize];
  const Team<kTeam> team(p.team_size);
  const int64_t cpg = p.channels_per_group;
  const int num_vecs = static_cast<int>(cpg / kVec);
  const int64_t num_channels = p.num_groups * cpg;
  const T_ACC inv_n = T_ACC(1) / static_cast<T_ACC>(cpg);

  // For kAffine the launch makes team.count() a multiple of num_groups, so this team only
  // ever sees group (team.id() % num_groups). gamma/beta are loaded per row.
  const int64_t c0 = (team.id() % p.num_groups) * cpg;
  T_ACC dw[kVecsPerThread][kVec];
  T_ACC db[kVecsPerThread][kVec];
  if constexpr (kAffine) {
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      FillVec<kVec>(T_ACC(0), dw[i]);
      FillVec<kVec>(T_ACC(0), db[i]);
    }
  }

  for (int64_t first = team.first(); first < p.num_segments; first += team.count()) {
    const int64_t s = first + team.index;
    const bool valid = s < p.num_segments;
    T_ACC g[kVecsPerThread][kVec];
    T_ACC x_hat[kVecsPerThread][kVec];
    bool active[kVecsPerThread];
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      const int j = team.lane + i * team.size;
      active[i] = valid && j < num_vecs;
      if (active[i]) {
        LoadVec<kVec>(p.dy + s * cpg + j * kVec, g[i]);
        LoadVec<kVec>(p.x_or_y + s * cpg + j * kVec, x_hat[i]);
      }
    }
    const T_ACC rstd = valid ? p.rstd[s] : T_ACC(0);
    T_ACC mean = T_ACC(0);
    if constexpr (!kRMS && !kMemoryEfficient) {
      mean = valid ? p.mean[s] : T_ACC(0);
    }

    T_ACC sum_g = T_ACC(0);
    T_ACC sum_g_x_hat = T_ACC(0);
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      if (active[i]) {
        const int j = team.lane + i * team.size;
        T_ACC w[kVec];
        T_ACC b[kVec];
        if constexpr (kAffine) {
          LoadVec<kVec>(p.gamma + c0 + j * kVec, w);
          if constexpr (kUseBeta) {
            LoadVec<kVec>(p.beta + c0 + j * kVec, b);
          }
        }
#pragma unroll
        for (int k = 0; k < kVec; ++k) {
          T_ACC xh;
          if constexpr (kMemoryEfficient) {
            xh = x_hat[i][k];
            if constexpr (kUseBeta) {
              xh -= b[k];
            }
            if constexpr (kAffine) {
              xh /= w[k];
            }
          } else {
            xh = (x_hat[i][k] - mean) * rstd;
          }
          T_ACC gk = g[i][k];
          if constexpr (kAffine) {
            dw[i][k] += gk * xh;
            if constexpr (!kRMS) {
              db[i][k] += gk;
            }
            gk *= w[k];
          }
          sum_g += gk;
          sum_g_x_hat += gk * xh;
          g[i][k] = gk;
          x_hat[i][k] = xh;
        }
      }
    }
    T_ACC mean_g = T_ACC(0);
    if constexpr (kRMS) {
      sum_g_x_hat = TeamSum<kTeam>(sum_g_x_hat, team.size, shm, team.round);
    } else {
      TeamSum2<kTeam>(sum_g, sum_g_x_hat, team.size, shm, team.round);
      mean_g = sum_g * inv_n;
    }
    const T_ACC mean_g_x_hat = sum_g_x_hat * inv_n;

#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      if (active[i]) {
        const int j = team.lane + i * team.size;
#pragma unroll
        for (int k = 0; k < kVec; ++k) {
          g[i][k] = rstd * (g[i][k] - mean_g - x_hat[i][k] * mean_g_x_hat);
        }
        StoreVec<kVec>(g[i], p.dx + s * cpg + j * kVec);
      }
    }
  }

  if constexpr (kAffine) {
    const int64_t phase = team.id() / p.num_groups;
    T_ACC* dw_out = p.dgamma_partial + phase * num_channels + c0;
    T_ACC* db_out = kRMS ? nullptr : p.dbeta_partial + phase * num_channels + c0;
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      const int j = team.lane + i * team.size;
      if (j < num_vecs) {
        StoreVec<kVec>(dw[i], dw_out + j * kVec);
        if constexpr (!kRMS) {
          StoreVec<kVec>(db[i], db_out + j * kVec);
        }
      }
    }
  }
  team.Finish();
}

// out[c] = sum_p partial[p, c], launched with (kWarpSize, kWarpSize) blocks.
template <typename T, typename T_ACC>
__global__ void SumPartialsKernel(int64_t num_phases, int64_t num_channels,
                                  const T_ACC* __restrict__ partial, T* __restrict__ out) {
  __shared__ T_ACC shm[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];
  const int64_t c = blockIdx.x * cuda_utils::kWarpSize + threadIdx.x;
  T_ACC sum = T_ACC(0);
  if (c < num_channels) {
    for (int64_t i = threadIdx.y; i < num_phases; i += blockDim.y) {
      sum += partial[i * num_channels + c];
    }
  }
  shm[threadIdx.y][threadIdx.x] = sum;
  __syncthreads();
  // Transpose so that each warp holds all phase lanes of one channel.
  sum = reduce::WarpReduce(shm[threadIdx.x][threadIdx.y], reduce::SumOp<T_ACC>());
  const int64_t c_out = blockIdx.x * cuda_utils::kWarpSize + threadIdx.y;
  if (threadIdx.x == 0 && c_out < num_channels) {
    out[c_out] = static_cast<T>(sum);
  }
}

struct Layout {
  int64_t vec;              // elements per memory access
  TeamKind kind;
  int64_t team_size;        // threads per segment
  int64_t block_size;
  int64_t teams_per_block;  // > 1 only for kSubWarp
  int64_t cluster_size;     // blocks per team for kCluster, else 1
  int64_t threads;          // threads that hold elements of a segment
};

inline int64_t NextPow2(int64_t x) {
  int64_t p = 1;
  while (p < x) {
    p <<= 1;
  }
  return p;
}

template <typename T>
Layout MakeLayout(int64_t channels_per_group, std::initializer_list<const void*> ptrs) {
  constexpr int64_t kMaxVec = kVecBytes / sizeof(T);
  bool vectorize = channels_per_group % kMaxVec == 0;
  for (const void* ptr : ptrs) {
    vectorize &= reinterpret_cast<std::uintptr_t>(ptr) % kVecBytes == 0;
  }
  Layout layout;
  layout.vec = vectorize ? kMaxVec : 1;
  const int64_t vecs_per_thread = kElemsPerThread / layout.vec;
  const int64_t threads = utils::DivUp(channels_per_group / layout.vec, vecs_per_thread);
  layout.threads = threads;
  if (threads <= kMaxBlockSize) {
    layout.team_size = NextPow2(threads);
    layout.kind = layout.team_size > cuda_utils::kWarpSize ? TeamKind::kBlock : TeamKind::kSubWarp;
    layout.block_size = layout.kind == TeamKind::kBlock ? layout.team_size : kSubWarpBlockSize;
    layout.teams_per_block = layout.block_size / layout.team_size;
    layout.cluster_size = 1;
  } else {
    TORCH_CHECK(at::cuda::getCurrentDeviceProperties()->major >= 9, "channels_per_group > ", kMaxBlockChannels,
                " needs thread block clusters (sm_90 or newer)");
    // The block and cluster sizes depend on the kernel; see FitLayout.
    layout.kind = TeamKind::kCluster;
    layout.team_size = 0;
    layout.block_size = 0;
    layout.teams_per_block = 1;
    layout.cluster_size = 0;
  }
  return layout;
}

// Calls f(std::integral_constant<int, kVec>, std::integral_constant<TeamKind, kTeam>) for the layout.
template <typename T, typename F>
void DispatchLayout(const Layout& layout, F&& f) {
  constexpr int kMaxVec = kVecBytes / sizeof(T);
  auto with_vec = [&](auto vec) {
    switch (layout.kind) {
      case TeamKind::kSubWarp:
        f(vec, std::integral_constant<TeamKind, TeamKind::kSubWarp>{});
        break;
      case TeamKind::kBlock:
        f(vec, std::integral_constant<TeamKind, TeamKind::kBlock>{});
        break;
      case TeamKind::kCluster:
        f(vec, std::integral_constant<TeamKind, TeamKind::kCluster>{});
        break;
    }
  };
  if (layout.vec == 1) {
    with_vec(std::integral_constant<int, 1>{});
  } else {
    with_vec(std::integral_constant<int, kMaxVec>{});
  }
}

inline cudaLaunchConfig_t ClusterLaunchConfig(const Layout& layout, int64_t num_clusters, cudaStream_t stream,
                                              cudaLaunchAttribute* attr) {
  attr->id = cudaLaunchAttributeClusterDimension;
  attr->val.clusterDim.x = layout.cluster_size;
  attr->val.clusterDim.y = 1;
  attr->val.clusterDim.z = 1;
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(num_clusters * layout.cluster_size);
  config.blockDim = dim3(layout.block_size);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  config.attrs = attr;
  config.numAttrs = 1;
  return config;
}

// Completes the layout for `kernel` and returns it with the number of launch units, blocks or for
// kCluster whole clusters, that fit on the GPU at once. A cluster team spreads its threads over
// blocks of at most 1024 or 512 threads; each cluster normalizes one segment at a time, so the
// shape with the most resident clusters is selected, preferring smaller blocks on a tie.
template <typename Kernel>
std::pair<Layout, int64_t> FitLayout(Kernel kernel, const Layout& layout, cudaStream_t stream) {
  if (layout.kind != TeamKind::kCluster) {
    int blocks_per_sm = 0;
    AT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, layout.block_size, 0));
    return {layout, std::max(blocks_per_sm, 1) * static_cast<int64_t>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount)};
  }
  Layout best = layout;
  int best_clusters = -1;
  for (const int64_t max_block : {kMaxBlockSize, kMaxBlockSize / 2}) {
    Layout fit = layout;
    fit.cluster_size = utils::DivUp(layout.threads, max_block);
    if (fit.cluster_size > kMaxClusterSize) {
      continue;
    }
    fit.block_size = utils::DivUp(utils::DivUp(layout.threads, fit.cluster_size), cuda_utils::kWarpSize) * cuda_utils::kWarpSize;
    fit.team_size = fit.block_size * fit.cluster_size;
    cudaLaunchAttribute attr;
    const cudaLaunchConfig_t config = ClusterLaunchConfig(fit, 1, stream, &attr);
    int num_clusters = 0;
    AT_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&num_clusters, kernel, &config));
    if (num_clusters >= best_clusters) {
      best = fit;
      best_clusters = num_clusters;
    }
  }
  TORCH_CHECK(best_clusters > 0, "no cluster of ", best.cluster_size, " blocks of ", best.block_size, " threads fits");
  return {best, best_clusters};
}

template <typename Kernel, typename Params>
void LaunchUnits(Kernel kernel, const Layout& layout, int64_t num_units, cudaStream_t stream, const Params& params) {
  if (layout.kind == TeamKind::kCluster) {
    cudaLaunchAttribute attr;
    const cudaLaunchConfig_t config = ClusterLaunchConfig(layout, num_units, stream, &attr);
    AT_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, params));
  } else {
    cuda_utils::LaunchKernel(kernel, dim3(num_units), dim3(layout.block_size), 0, stream, params);
  }
}

// gamma and beta may be nullptr (no affine / RMS norm); mean is unused for RMS norm.
template <typename T, bool kRMS>
void GroupNormFwd(int64_t outer_size, int64_t num_groups, int64_t channels_per_group, double eps,
                  const T* x, const T* gamma, const T* beta, T* y,
                  at::acc_type<T, true>* mean, at::acc_type<T, true>* rstd) {
  using T_ACC = at::acc_type<T, true>;
  const int64_t num_segments = outer_size * num_groups;
  if (num_segments == 0) {
    return;
  }
  const Layout layout = MakeLayout<T>(channels_per_group, {x, gamma, beta, y});
  const FwdParams<T, T_ACC> params{num_segments, num_groups, channels_per_group, layout.team_size,
                                   static_cast<T_ACC>(eps), x, gamma, beta, y, mean, rstd};
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  DispatchLayout<T>(layout, [&](auto vec, auto team) {
    constexpr int kVec = decltype(vec)::value;
    constexpr TeamKind kTeam = decltype(team)::value;
    auto launch = [&](auto kernel) {
      const auto [fit, resident_units] = FitLayout(kernel, layout, stream);
      const int64_t num_units = std::min(resident_units, utils::DivUp(num_segments, fit.teams_per_block));
      LaunchUnits(kernel, fit, num_units, stream, params);
    };
    if (gamma != nullptr) {
      launch(GroupNormFwdKernel<T, T_ACC, kVec, kTeam, kRMS, true>);
    } else {
      launch(GroupNormFwdKernel<T, T_ACC, kVec, kTeam, kRMS, false>);
    }
  });
}

// gamma == nullptr means no affine parameters, and dgamma/dbeta are then not written.
// beta is only read for LayerNorm with memory_efficient; mean is unused for RMS norm.
template <typename T, bool kRMS>
void GroupNormBwd(int64_t outer_size, int64_t num_groups, int64_t channels_per_group, bool memory_efficient,
                  const T* dy, const T* x_or_y, const at::acc_type<T, true>* mean,
                  const at::acc_type<T, true>* rstd, const T* gamma, const T* beta,
                  T* dx, T* dgamma, T* dbeta, const torch::TensorOptions& options) {
  using T_ACC = at::acc_type<T, true>;
  const int64_t num_segments = outer_size * num_groups;
  const int64_t num_channels = num_groups * channels_per_group;
  const bool affine = gamma != nullptr;
  if (num_segments == 0) {
    return;
  }
  const Layout layout = MakeLayout<T>(channels_per_group, {dy, x_or_y, gamma, beta, dx});
  BwdParams<T, T_ACC> params{num_segments, num_groups, channels_per_group, layout.team_size,
                             dy, x_or_y, mean, rstd, gamma, beta, dx, nullptr, nullptr};
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  torch::Tensor partial;
  int64_t num_phases = 0;

  DispatchLayout<T>(layout, [&](auto vec, auto team) {
    constexpr int kVec = decltype(vec)::value;
    constexpr TeamKind kTeam = decltype(team)::value;
    auto launch = [&](auto kernel) {
      const auto [fit, resident_units] = FitLayout(kernel, layout, stream);
      int64_t num_units = std::min(resident_units, utils::DivUp(num_segments, fit.teams_per_block));
      if (affine) {
        // The number of teams must be a multiple of num_groups (see GroupNormBwdKernel).
        const int64_t step = num_groups / std::gcd(num_groups, fit.teams_per_block);
        num_units = std::max<int64_t>(num_units / step, 1) * step;
        num_phases = num_units * fit.teams_per_block / num_groups;
        partial = torch::empty({kRMS ? 1 : 2, num_phases, num_channels}, options.dtype(c10::CppTypeToScalarType<T_ACC>::value));
        params.dgamma_partial = partial.data_ptr<T_ACC>();
        params.dbeta_partial = kRMS ? nullptr : params.dgamma_partial + num_phases * num_channels;
      }
      LaunchUnits(kernel, fit, num_units, stream, params);
    };
    auto with_affine = [&](auto affine_c) {
      constexpr bool kAffine = decltype(affine_c)::value;
      if (memory_efficient) {
        launch(GroupNormBwdKernel<T, T_ACC, kVec, kTeam, kRMS, kAffine, true>);
      } else {
        launch(GroupNormBwdKernel<T, T_ACC, kVec, kTeam, kRMS, kAffine, false>);
      }
    };
    if (affine) {
      with_affine(std::true_type{});
    } else {
      with_affine(std::false_type{});
    }
  });

  if (affine) {
    const dim3 grid(utils::DivUp<int64_t>(num_channels, cuda_utils::kWarpSize));
    const dim3 block(cuda_utils::kWarpSize, cuda_utils::kWarpSize);
    cuda_utils::LaunchKernel(SumPartialsKernel<T, T_ACC>, grid, block, 0, stream,
                             num_phases, num_channels, params.dgamma_partial, dgamma);
    if constexpr (!kRMS) {
      cuda_utils::LaunchKernel(SumPartialsKernel<T, T_ACC>, grid, block, 0, stream,
                               num_phases, num_channels, params.dbeta_partial, dbeta);
    }
  }
}

// Checks shared by the GroupLayerNorm / GroupRMSNorm entry points; returns channels_per_group.
inline int64_t CheckShapes(const torch::Tensor& x, int64_t num_channels, int64_t num_groups) {
  TORCH_CHECK(x.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(x.layout() == c10::kStrided, "input must have strided layout");
  TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16 ||
              x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble,
              "input dtype must be float16, bfloat16, float32, or float64");
  TORCH_CHECK(num_channels > 0, "num_channels must be positive, got ", num_channels);
  TORCH_CHECK(x.dim() == 3, "expected a 3D input, got ", x.dim(), "D");
  TORCH_CHECK(x.size(-1) == num_channels, "num_channels (", num_channels, ") must match x.size(-1) (", x.size(-1), ")");
  TORCH_CHECK(num_groups >= 1 && num_channels % num_groups == 0,
              "num_channels (", num_channels, ") must be divisible by num_groups (", num_groups, ")");
  const int64_t channels_per_group = num_channels / num_groups;
  TORCH_CHECK(channels_per_group <= kMaxChannelsPerGroup, "channels_per_group must be at most ",
              kMaxChannelsPerGroup, ", got ", channels_per_group);
  return channels_per_group;
}

inline void CheckParam(const torch::Tensor& param, const torch::Tensor& x, int64_t num_channels, const char* name) {
  TORCH_CHECK(param.device() == x.device(), name, " must be on the same device as the input");
  TORCH_CHECK(param.layout() == c10::kStrided, name, " must have strided layout");
  TORCH_CHECK(param.numel() == num_channels, name, " must have ", num_channels, " elements, got ", param.numel());
  TORCH_CHECK(param.scalar_type() == x.scalar_type(), name, " must have the same dtype as the input");
}

inline void CheckGradient(const torch::Tensor& dy, const torch::Tensor& x) {
  TORCH_CHECK(dy.device() == x.device(), "y_grad must be on the same device as the input");
  TORCH_CHECK(dy.layout() == c10::kStrided, "y_grad must have strided layout");
  TORCH_CHECK(dy.sizes() == x.sizes(), "y_grad must have the same shape as the input");
  TORCH_CHECK(dy.scalar_type() == x.scalar_type(), "y_grad must have the same dtype as the input");
}

inline void CheckStats(const torch::Tensor& stats, const torch::Tensor& x, int64_t count, const char* name) {
  TORCH_CHECK(stats.device() == x.device(), name, " must be on the same device as the input");
  TORCH_CHECK(stats.layout() == c10::kStrided, name, " must have strided layout");
  TORCH_CHECK(stats.numel() == count, name, " must have ", count, " elements, got ", stats.numel());
  TORCH_CHECK(stats.scalar_type() == at::toAccumulateType(x.scalar_type(), true),
              name, " dtype must be float64 for float64 input, otherwise float32");
}

}  // namespace group_norm
}  // namespace ops
}  // namespace xllm
