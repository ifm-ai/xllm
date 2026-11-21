// The generic group norm kernels, forward and backward, for every shape the public ops accept: any channels per group
// up to 65536, any operand alignment, fp16, bf16 and fp32. fallback.cu launches them for every call the dispatcher
// (dispatch.cu) gives no specialized kernel. Derived from ../group_norm_kernel.cuh.
#pragma once

#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <mutex>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "../../cuda_utils.cuh"
#include "../../reduce.cuh"
#include "../../utils.h"

// x of shape [rows, C] is viewed as rows * groups segments of cpg (channels per group) contiguous elements: segment s
// is group s % groups of row s / groups, and mean[s] / rstd[s] are its statistics. A team of threads normalizes one
// segment at a time and loops over segments. A team is up to a warp (several teams share a block and reduce with warp
// shuffles), a whole block (shared-memory reductions), or a thread block cluster, whose blocks add their block totals
// through distributed shared memory. A thread keeps kElemsPerThread elements of its segment in registers (kFwdWideElems
// in the forward pass of segments wider than a block) and moves kVec elements an access.
//
// The affine backward pass also sums dgamma and dbeta. Its grid holds a multiple of num_groups teams, so a team only
// ever sees one group: it sums its rows in registers and writes one row of partial sums, which a second kernel folds.
// Every sum runs in a fixed order (no atomics), so every output is bitwise reproducible.

namespace xllm::ops::blackwell::generic {

constexpr int64_t kVecBytes = 16;
// The fp32 affine backward's 32-byte access: one 256-bit load or store where the 16-byte form needs two. Eight floats
// are still one vector a thread, so the team, block and grid of the 16-byte form are unchanged.
constexpr int64_t kVecBytesWide = 32;
constexpr int64_t kElemsPerThread = 8;
// Forward only: a segment too wide for one block at kElemsPerThread takes kFwdWideElems elements a thread, a cluster of
// a quarter of the blocks, so that its reductions fan in over fewer blocks; it is still read and written once.
constexpr int64_t kFwdWideElems = 32;
constexpr int64_t kMaxBlockSize = 1024;
constexpr int64_t kMaxClusterSize = 8;  // the portable limit
constexpr int64_t kSubWarpBlockSize = 256;
// The backward hoists the affine parameters out of the row loop (from y; also at 65536 channels) and stages rows
// through shared memory only from kLongRows rows: on shorter calls both measured slower. Calls of fewer than
// kPdlMaxRows rows are launched dependent (programmatic dependent launch); longer calls measured slower with it.
constexpr int64_t kLongRows = 512;
constexpr int64_t kPdlMaxRows = 2048;
static_assert(kLongRows <= kPdlMaxRows, "a backward call of fewer than kLongRows rows is always launched dependent");
// The fused folds (one launch for both partial planes) pay only while the plane is small: at 2.4M cells the fused form
// measured 11.9% slower than two launches. The channel bound is a precaution for the widest groups.
constexpr int64_t kFuseMaxPartialCells = 1363968;
constexpr int64_t kFuseMaxChannels = 32768;

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

// ClusterSum state in shared memory, double buffered by round parity: an mbarrier that completes once every block of
// the cluster has stored its values, and one slot per sending block.
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

// Bulk copy of one contiguous run of global memory into this block's shared memory, completed on an mbarrier by its
// byte count. `bytes` must be a multiple of 16 and both addresses 16-byte aligned.
__device__ __forceinline__ void BulkCopyShared(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
#endif

// Stages one row of dy and x_or_y for the block, issued by one thread: `runs` runs of `run_elems` elements, run i at
// element off0 + i * stride_elems of both tensors, packed run after run into sh_dy and sh_xy; the mbarrier `bar`
// expects `bytes` in all. A no-op outside an sm_90+ device pass.
template <typename T>
__device__ __forceinline__ void StageRowRuns(uint32_t bar, uint32_t bytes, const T* dy, const T* xy, T* sh_dy,
                                             T* sh_xy, int64_t off0, int64_t stride_elems, int runs, int run_elems) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  ArriveExpectTx(bar, bytes);
  const uint32_t run_bytes = static_cast<uint32_t>(run_elems * sizeof(T));
  for (int i = 0; i < runs; ++i) {
    const int64_t off = off0 + static_cast<int64_t>(i) * stride_elems;
    const int dst = i * run_elems;
    BulkCopyShared(SmemAddr(sh_dy + dst), dy + off, run_bytes, bar);
    BulkCopyShared(SmemAddr(sh_xy + dst), xy + off, run_bytes, bar);
  }
#else
  (void)bar;
  (void)bytes;
  (void)dy;
  (void)xy;
  (void)sh_dy;
  (void)sh_xy;
  (void)off0;
  (void)stride_elems;
  (void)runs;
  (void)run_elems;
#endif
}

// One 16-byte asynchronous copy from global memory into a shared-memory slot, with no register in between; the bytes
// are there once CpAsyncCommit and then CpAsyncWaitAll have run. A cp.async is ordered after every earlier instruction
// of its thread (PTX ISA, memory consistency model, "Asynchronous Operations"), so a thread may refill a slot right
// after reading it.
__device__ __forceinline__ void CpAsync16(void* dst_shared, const void* src_global) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst_shared));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(src_global) : "memory");
#endif
}

__device__ __forceinline__ void CpAsyncCommit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

// Waits for every group this thread has committed (an empty group is legal) and makes their bytes visible to it.
__device__ __forceinline__ void CpAsyncWaitAll() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

// Sums each of the kN (<= 2) values, already block totals held by every thread, over the blocks of the cluster. Thread
// r of every block sends the block's values to block r with st.async, so no cluster-wide barrier or memory fence is
// needed per round, and every block adds the slots in rank order (kUnroll: unrolled for a full cluster). After waiting
// for round k, a block re-arms the same mbarrier for round k + 2. The other blocks can only send round k + 2 after this
// block has sent round k + 1, which it does after a __syncthreads that follows all its reads of round k, so both the
// mbarrier and the slots are free again by then.
template <int kN, bool kUnroll, typename T>
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
  if constexpr (kUnroll) {
    if (num_blocks == kMaxClusterSize) {
#pragma unroll
      for (uint32_t r = 0; r < kMaxClusterSize; ++r) {
#pragma unroll
        for (int k = 0; k < kN; ++k) {
          v[k] += slots.recv[buf][r][k];
        }
      }
    } else {
      for (uint32_t r = 0; r < num_blocks; ++r) {
#pragma unroll
        for (int k = 0; k < kN; ++k) {
          v[k] += slots.recv[buf][r][k];
        }
      }
    }
  } else {
    for (uint32_t r = 0; r < num_blocks; ++r) {
#pragma unroll
      for (int k = 0; k < kN; ++k) {
        v[k] += slots.recv[buf][r][k];
      }
    }
  }
#else
  __trap();
#endif
}

// Where the calling thread sits: team `index` of the block, thread `lane` of the team. kTeamsPerBlock > 1 is the
// backward's multi-team block (kMultiTeams below): that many teams of team_size threads share one block.
template <TeamKind kKind, int kTeamsPerBlock = 1>
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
    } else if constexpr (kKind == TeamKind::kBlock && kTeamsPerBlock > 1) {
      size = team_size;
      per_block = kTeamsPerBlock;
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

// Sum over the threads of a team; every thread of the block (and cluster) must call it. `round` is Team::round, read
// by kCluster teams. kParity: the cross-warp slots alternate between two buffers by round parity, as TeamSum2 explains;
// shm then holds 2 * kWarpSize values.
template <TeamKind kKind, bool kUnroll = false, bool kParity = false, typename T>
__device__ __forceinline__ T TeamSum(T x, int64_t team_size, T* shm, int& round) {
  if constexpr (kKind == TeamKind::kSubWarp) {
    // Teams are aligned powers of two, so xor offsets below team_size stay inside the team.
    for (int offset = team_size >> 1; offset > 0; offset >>= 1) {
      x += cuda_utils::WarpShflXor(x, offset);
    }
    return x;
  } else {
    if constexpr (kParity) {
      const int64_t lid1 = threadIdx.x % cuda_utils::kWarpSize;
      const int64_t wid1 = threadIdx.x / cuda_utils::kWarpSize;
      const int64_t nw1 = blockDim.x / cuda_utils::kWarpSize;
      T* slot1 = shm + (round & 1) * cuda_utils::kWarpSize;
      x = reduce::WarpReduce(x, reduce::SumOp<T>());
      if (lid1 == 0) {
        slot1[wid1] = x;
      }
      __syncthreads();
      x = lid1 < nw1 ? slot1[lid1] : T(0);
      x = reduce::WarpAllReduce(x, reduce::SumOp<T>());
    } else {
      x = reduce::BlockAllReduce(x, shm, reduce::SumOp<T>());
    }
    if constexpr (kKind == TeamKind::kCluster) {
      T v[1] = {x};
      ClusterSum<1, kUnroll>(v, round);
      x = v[0];
    }
    return x;
  }
}

// TeamSum of two values with a single round of shared memory; shm holds 2 * kWarpSize values, or kTeamsPerBlock times
// that for a multi-team block, whose teams reduce in their own slot pairs and all pass the same two barriers. With
// kParity the slot pair alternates between two buffers by round parity, so the barrier that waits for the previous
// round's reads is not needed: a warp reaches the write of round r + 2 only after the second barrier of round r + 1,
// which every warp passes after its own reads of round r.
template <TeamKind kKind, bool kUnroll, int kTeamsPerBlock = 1, bool kParity = false, typename T>
__device__ __forceinline__ void TeamSum2(T& a, T& b, int64_t team_size, T* shm, int& round) {
  if constexpr (kKind != TeamKind::kSubWarp) {
    const int64_t lid = threadIdx.x % cuda_utils::kWarpSize;
    int64_t wid = 0;
    int64_t num_warps = 0;
    T* slot = shm;
    if constexpr (kTeamsPerBlock > 1) {
      const int64_t ts = team_size;
      wid = (threadIdx.x % ts) / cuda_utils::kWarpSize;
      num_warps = ts / cuda_utils::kWarpSize;
      slot = shm + (threadIdx.x / ts) * 2 * cuda_utils::kWarpSize;
    } else {
      wid = threadIdx.x / cuda_utils::kWarpSize;
      num_warps = blockDim.x / cuda_utils::kWarpSize;
    }
    a = reduce::WarpReduce(a, reduce::SumOp<T>());
    b = reduce::WarpReduce(b, reduce::SumOp<T>());
    if constexpr (kParity) {
      slot = shm + (round & 1) * 2 * cuda_utils::kWarpSize;
    } else {
      __syncthreads();  // shm may still be read by the previous reduction
    }
    if (lid == 0) {
      slot[wid] = a;
      slot[cuda_utils::kWarpSize + wid] = b;
    }
    __syncthreads();
    a = lid < num_warps ? slot[lid] : T(0);
    b = lid < num_warps ? slot[cuda_utils::kWarpSize + lid] : T(0);
    a = reduce::WarpAllReduce(a, reduce::SumOp<T>());
    b = reduce::WarpAllReduce(b, reduce::SumOp<T>());
    if constexpr (kKind == TeamKind::kCluster) {
      T v[2] = {a, b};
      ClusterSum<2, kUnroll>(v, round);
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

// The forward's 2-byte cluster form at 16-byte vectors and kFwdWideElems elements: blocks of 512 threads (three an SM)
// keep the segment in a static shared tile between the passes instead of registers.
template <typename T, int kVec, TeamKind kTeam, int kElems, int kMaxBlock>
constexpr bool UseSharedWideInput() {
  return sizeof(T) == 2 && kTeam == TeamKind::kCluster && kVec == 8 && kElems == kFwdWideElems &&
         kMaxBlock == kMaxBlockSize / 2;
}

// The forward's fp32 affine cluster form at 16-byte vectors and kFwdWideElems elements: the next segment's bytes are
// copied into shared memory (cp.async) while this one is reduced and written, which hides the load latency otherwise
// exposed at the first reduction. The form already spends its registers on the segment, so the slots cost no
// occupancy.
template <typename T, bool kAffine, int kVec, TeamKind kTeam, int kElems, int kMaxBlock>
constexpr bool UseWidePrefetch() {
  return sizeof(T) == 4 && kAffine && sizeof(T) * kVec == 16 && kTeam == TeamKind::kCluster &&
         kElems == kFwdWideElems && kMaxBlock == kMaxBlockSize;
}

// UseWidePrefetch's slots, one vector per (vector of a thread, thread of the largest block), and the bytes that align
// them to kWidePrefetchAlign at run time: 128 KB of dynamic shared memory (a static array may take only 48 KB on every
// target).
constexpr size_t kWidePrefetchAlign = 128;
template <typename T, int kVec, int kElems, int kMaxBlock>
constexpr size_t WidePrefetchSmemBytes() {
  return sizeof(Vec<T, kVec>) * (kElems / kVec) * kMaxBlock + kWidePrefetchAlign;
}

// minBlocksPerMultiprocessor for __launch_bounds__ (0 sets no bound): three blocks for the shared-tile form, two for a
// light kernel (the forward, the backward without affine) in clusters of vectors at kElemsPerThread elements.
template <typename T, int kVec, TeamKind kTeam, int kElems = kElemsPerThread,
          int kMaxBlock = kMaxBlockSize>
constexpr int ClusterMinBlocks(bool light_kernel) {
  if constexpr (UseSharedWideInput<T, kVec, kTeam, kElems, kMaxBlock>()) {
    return 3;
  }
  return kTeam == TeamKind::kCluster && kVec > 1 && sizeof(T) <= 4 && light_kernel && kElems <= kElemsPerThread
             ? 2
             : 0;
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

// Forward. A team normalizes one segment at a time: it sums the segment for the mean, sums the squared deviations from
// the mean for rstd (two passes over the values its threads hold), then writes y = (x - mean) * rstd (* gamma + beta).
// Thread `lane` of a team holds the vectors j = lane + i * team size, i < kElems / kVec, of the segment. The three arms
// differ only in where a thread keeps its values between the passes: registers, a static shared tile
// (UseSharedWideInput) or prefetch slots in dynamic shared memory (UseWidePrefetch).
template <typename T, typename T_ACC, int kVec, int kElems, TeamKind kTeam, bool kRMS, bool kAffine,
          int kMaxBlock = kMaxBlockSize>
__global__ void __launch_bounds__(kTeam == TeamKind::kSubWarp ? kSubWarpBlockSize : kMaxBlock,
                                  ClusterMinBlocks<T, kVec, kTeam, kElems, kMaxBlock>(true))
GroupNormFwdKernel(FwdParams<T, T_ACC> p) {
  static_assert(kElems % kVec == 0, "a forward layout's elements per thread must be whole vectors");
  constexpr int kVecsPerThread = kElems / kVec;
  __shared__ T_ACC shm[cuda_utils::kWarpSize];
  const Team<kTeam> team(p.team_size);
  const int64_t cpg = p.channels_per_group;
  const int num_vecs = static_cast<int>(cpg / kVec);
  const T_ACC inv_n = T_ACC(1) / static_cast<T_ACC>(cpg);

  // The loop bound is uniform over the block (and cluster) so that team reductions stay convergent.
  for (int64_t first = team.first(); first < p.num_segments; first += team.count()) {
    const int64_t s = first + team.index;
    const bool valid = s < p.num_segments;
    if constexpr (UseSharedWideInput<T, kVec, kTeam, kElems, kMaxBlock>()) {
      // A static tile, so that the occupancy queries count it; vector-first, so 16-byte accesses are conflict free.
      __shared__ Vec<T, kVec> input_tile[kVecsPerThread][kMaxBlock];
      bool active[kVecsPerThread];
      T_ACC sum = T_ACC(0);
#pragma unroll
      for (int i = 0; i < kVecsPerThread; ++i) {
        const int j = team.lane + i * team.size;
        active[i] = valid && j < num_vecs;
        Vec<T, kVec> x_native;
        if (active[i]) {
          LoadVec<kVec>(p.x + s * cpg + j * kVec, x_native.val);
          StoreVec<kVec>(x_native.val, input_tile[i][threadIdx.x].val);
        } else {
          FillVec<kVec>(T(0), x_native.val);
        }
#pragma unroll
        for (int k = 0; k < kVec; ++k) {
          sum += static_cast<T_ACC>(x_native.val[k]);
        }
      }

      // A thread reads back only its own tile entries: a compiler barrier orders the reads after the stores, and the
      // team barriers of the reductions below order the next segment's stores after them.
      asm volatile("" ::: "memory");

      T_ACC mean = T_ACC(0);
      if constexpr (!kRMS) {
        mean = TeamSum<kTeam>(sum, team.size, shm, team.round) * inv_n;
      }
      T_ACC m2 = T_ACC(0);
#pragma unroll
      for (int i = 0; i < kVecsPerThread; ++i) {
        if (active[i]) {
          Vec<T, kVec> x_native;
          LoadVec<kVec>(input_tile[i][threadIdx.x].val, x_native.val);
#pragma unroll
          for (int k = 0; k < kVec; ++k) {
            const T_ACC x = static_cast<T_ACC>(x_native.val[k]);
            m2 += cuda_utils::Square(x - mean);
          }
        }
      }
      const T_ACC rstd =
          c10::cuda::compat::rsqrt(TeamSum<kTeam>(m2, team.size, shm, team.round) * inv_n + p.eps);
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
          Vec<T, kVec> x_native;
          LoadVec<kVec>(input_tile[i][threadIdx.x].val, x_native.val);
          T_ACC w[kVec];
          T_ACC b[kVec];
          if constexpr (kAffine) {
            LoadVec<kVec>(p.gamma + c0 + j * kVec, w);
            if constexpr (!kRMS) {
              LoadVec<kVec>(p.beta + c0 + j * kVec, b);
            }
          }
          T_ACC out_vec[kVec];
#pragma unroll
          for (int k = 0; k < kVec; ++k) {
            T_ACC out = (static_cast<T_ACC>(x_native.val[k]) - mean) * rstd;
            if constexpr (kAffine) {
              out *= w[k];
              if constexpr (!kRMS) {
                out += b[k];
              }
            }
            out_vec[k] = out;
          }
          StoreVec<kVec>(out_vec, p.y + s * cpg + j * kVec);
        }
      }
    } else if constexpr (UseWidePrefetch<T, kAffine, kVec, kTeam, kElems, kMaxBlock>()) {
      // One cluster a segment. A thread's slots hold exactly its own vectors of a segment, so the mapping, the 16-byte
      // accesses, the summation order and the arithmetic are the register arm's; only the fetch moves: the next
      // segment's copy starts once this one's values are out of the slots and is waited on one iteration later. A
      // thread fills and reads only its own slots, so no barrier is needed (see CpAsync16 for the refill). The slots
      // are aligned to kWidePrefetchAlign here rather than in the declaration, which would pad the shared layout of
      // every kernel in the module; at 16-byte alignment the arm measured 13% slower.
      extern __shared__ __align__(16) unsigned char wide_prefetch_smem[];
      const uint32_t pad = static_cast<uint32_t>(kWidePrefetchAlign -
                                                 __cvta_generic_to_shared(wide_prefetch_smem) % kWidePrefetchAlign) %
                           kWidePrefetchAlign;
      auto* const staged = reinterpret_cast<Vec<T, kVec>(*)[kMaxBlock]>(wide_prefetch_smem + pad);
      // The loop below takes every segment this cluster owns, so it runs on the first pass only.
      if (first == team.first()) {
        const int64_t step = team.count();
        auto stage = [&](int64_t seg) {
          const int64_t base = seg * cpg;
#pragma unroll
          for (int i = 0; i < kVecsPerThread; ++i) {
            const int j = team.lane + i * team.size;
            if (j < num_vecs) {
              CpAsync16(&staged[i][threadIdx.x].val[0], p.x + base + j * kVec);
            } else {
              FillVec<kVec>(T(0), staged[i][threadIdx.x].val);  // no bytes to fetch for this slot
            }
          }
        };
        stage(team.first());
        CpAsyncCommit();
        for (int64_t f = team.first(); f < p.num_segments; f += step) {
          CpAsyncWaitAll();  // the copy started one iteration ago is in the slots
          T_ACC v[kVecsPerThread][kVec];
          bool active[kVecsPerThread];
#pragma unroll
          for (int i = 0; i < kVecsPerThread; ++i) {
            const int j = team.lane + i * team.size;
            active[i] = j < num_vecs;
            LoadVec<kVec>(staged[i][threadIdx.x].val, v[i]);
          }
          if (f + step < p.num_segments) {
            stage(f + step);
          }
          CpAsyncCommit();

          T_ACC sum = T_ACC(0);
#pragma unroll
          for (int i = 0; i < kVecsPerThread; ++i) {
#pragma unroll
            for (int k = 0; k < kVec; ++k) {
              sum += v[i][k];
            }
          }

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
          const T_ACC rstd =
              c10::cuda::compat::rsqrt(TeamSum<kTeam>(m2, team.size, shm, team.round) * inv_n + p.eps);
          if (team.lane == 0) {
            if constexpr (!kRMS) {
              p.mean[f] = mean;
            }
            p.rstd[f] = rstd;
          }

          const int64_t c0 = (f % p.num_groups) * cpg;
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
              StoreVec<kVec>(v[i], p.y + f * cpg + j * kVec);
            }
          }
        }
      }
    } else {
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
  // Affine only: [num_phases, num_channels] partial sums of dgamma and dbeta; row p holds those of teams
  // p * num_groups to p * num_groups + num_groups - 1, one a group (of blocks, for the multi-team form).
  T_ACC* dgamma_partial;
  T_ACC* dbeta_partial;
};

// The multi-team block of the fp32 affine LayerNorm backward from x at kMultiTeamCpg channels a group (a team of
// kMultiTeamSize threads at 16-byte vectors): kMultiTeams teams take kMultiTeams rows of one group and add their
// dgamma/dbeta in shared memory into one partial row, which leaves the fold a quarter of the rows.
constexpr int kMultiTeams = 4;
constexpr int kMultiTeamSize = 64;
constexpr int kMultiTeamCpg = 512;
constexpr int kMultiTeamVec = static_cast<int>(kVecBytes / sizeof(float));
// The multi-team body's static shared words: a TeamSum2 slot pair a team and the combine buffer.
constexpr int64_t MultiTeamShmWords(int64_t teams, bool rms) {
  return teams * 2 * cuda_utils::kWarpSize + teams * kMultiTeamCpg * (rms ? 1 : 2);
}
static_assert(MultiTeamShmWords(kMultiTeams, false) * static_cast<int64_t>(sizeof(float)) <=
                  cuda_utils::kMaxStaticSharedMemorySize,
              "the multi-team body must fit the static shared memory of a block");

// The backward forms that recover x_hat from y with gamma (and beta) loaded once and 1/gamma kept in registers instead
// of a load and a divide a row: bf16, and the fp32 RMS cluster form at 16-byte vectors. Elsewhere the hoist measured
// slower (more registers and fewer blocks an SM, or a stack frame).
template <typename T, bool kRMS, TeamKind kTeam, int kVec>
constexpr bool HoistableForm() {
  return std::is_same_v<T, at::BFloat16> ||
         (std::is_same_v<T, float> && kRMS && kTeam == TeamKind::kCluster && kVec == kVecBytes / sizeof(T));
}

// Backward. From x, x_hat = (x - mean) * rstd; from y (kMemoryEfficient), x_hat = (y - beta) / gamma, which needs
// nonzero gamma. With g = dy * gamma (dy without affine parameters),
//   dx = rstd * (g - mean(g) - x_hat * mean(g * x_hat)),
// and RMS norm drops the mean(g) term. The mapping is the forward's at kElemsPerThread elements a thread. With kAffine
// a thread also sums dgamma += dy * x_hat and dbeta += dy over the rows its team visits, in row order, and the team
// writes them as one partial row (GroupNormBwd keeps a team on one group). Variants, all chosen on the host:
//   kHoistOn        from y: gamma (and beta) loaded once and 1/gamma kept (HoistableForm);
//   kStageOn        rows staged through shared memory one row ahead: bf16 bulk copies (block and cluster teams), bf16
//                   LayerNorm per-lane cp.async (sub-warp teams, from y), fp32 bulk copies into a dynamic frame;
//   kTeamsPerBlock  kMultiTeams for the multi-team block;
//   kPDL            launched dependent: waits for the previous kernel on the stream before anything else.
template <typename T, typename T_ACC, int kVec, TeamKind kTeam, bool kRMS, bool kAffine, bool kMemoryEfficient,
          bool kHoistOn = false, bool kStageOn = false, int kTeamsPerBlock = 1, bool kPDL = false>
__global__ void __launch_bounds__(kTeam == TeamKind::kSubWarp ? kSubWarpBlockSize : kMaxBlockSize,
                                  ClusterMinBlocks<T, kVec, kTeam>(!kAffine))
GroupNormBwdKernel(BwdParams<T, T_ACC> p) {
  if constexpr (kPDL) {
    cudaGridDependencySynchronize();
  }
  constexpr int kVecsPerThread = kElemsPerThread / kVec;
  constexpr bool kUseBeta = kAffine && kMemoryEfficient && !kRMS;
  constexpr bool kHoistType = HoistableForm<T, kRMS, kTeam, kVec>();
  static_assert(!kHoistOn || kHoistType, "kHoistOn may only be set on a hoist-eligible instantiation");
  constexpr bool kHoist = kHoistOn && kAffine && kMemoryEfficient && kHoistType;
  // The fp32 affine cluster forms alternate two reduction buffers instead of waiting at a barrier (TeamSum2).
  constexpr bool kOneBarrierSlot = std::is_same<T, float>::value && kTeam == TeamKind::kCluster &&
                                   kAffine && kTeamsPerBlock == 1;
  constexpr int64_t kShmWords =
      kTeamsPerBlock > 1 ? MultiTeamShmWords(kTeamsPerBlock, kRMS)
                         : ((kOneBarrierSlot && !kRMS) ? 4 : 2) * cuda_utils::kWarpSize;
  // Aligned for the StoreVec<4> stores of the multi-team combine buffer.
  __shared__ __align__(16) T_ACC shm[kShmWords];
  // LayerNorm from x adds a full cluster's slots in an unrolled loop (same order).
  constexpr bool kSlotUnroll = !kRMS && !kMemoryEfficient;
  // The multi-team buffer is kMultiTeamCpg channels wide, the only width the host admits. A block-uniform return ahead
  // of every barrier guards it; a trap would give the kernel a stack frame.
  if constexpr (kTeamsPerBlock > 1) {
    if (p.channels_per_group != kMultiTeamCpg) {
      return;
    }
  }
  const Team<kTeam, kTeamsPerBlock> team(p.team_size);
  const int64_t cpg = p.channels_per_group;
  const int num_vecs = static_cast<int>(cpg / kVec);
  const int64_t num_channels = p.num_groups * cpg;
  const T_ACC inv_n = T_ACC(1) / static_cast<T_ACC>(cpg);

  // Row staging. A block or cluster team's slice of a row is contiguous (lane = rank * blockDim + threadIdx, run i at
  // rank * blockDim + i * team size), so one thread stages it with a bulk copy a tensor and run, and the block waits on
  // an mbarrier; the next row is issued once every thread has read this one out (one __syncthreads). A sub-warp team's
  // slice is not contiguous: every lane copies its own vectors into its own slots with cp.async. Every staged form runs
  // only at 16-byte vectors (kStageVecOk), which the host chose after testing all five pointers: the alignment
  // cp.async.bulk needs. The bf16 frames are static shared memory, the fp32 frame (64 KB) dynamic.
  constexpr bool kStageArch =
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      true;
#else
      false;  // the host pass; no device pass below sm_90 is built
#endif
  constexpr bool kStage =
      kStageOn && (kTeam == TeamKind::kCluster || kTeam == TeamKind::kBlock) && kAffine && sizeof(T) == 2 &&
      kStageArch && kTeamsPerBlock == 1;
  constexpr bool kStageSubShape = kStageOn && kTeam == TeamKind::kSubWarp && kAffine && kHoist &&
                                  kTeamsPerBlock == 1 && !kRMS && sizeof(T) == 2;
  constexpr int kStageElems =
      kStage ? kVecsPerThread * static_cast<int>(kMaxBlockSize) * kVec
             : (kStageSubShape ? kVecsPerThread * static_cast<int>(kSubWarpBlockSize) * kVec : 1);
  __shared__ __align__(16) T stage_dy[kStageElems];
  __shared__ __align__(16) T stage_xy[kStageElems];
  __shared__ uint64_t stage_bar;
  constexpr bool kStageDynShape =
      kStageOn && kTeam == TeamKind::kCluster && kAffine && std::is_same_v<T, float> && kTeamsPerBlock == 1;
  constexpr int kStageDynElems = kVecsPerThread * static_cast<int>(kMaxBlockSize) * kVec;
  constexpr bool kStageVecOk = kVec == kVecBytes / sizeof(T);
  constexpr bool kStageSub = kStageSubShape && !kStage && kStageArch && kStageVecOk;
  constexpr bool kStageDyn = kStageDynShape && kStageArch && kStageVecOk;
  // At function scope: `extern __shared__` in a nested block has a null base. Two slices, dy then x_or_y.
  extern __shared__ __align__(16) unsigned char stage_dyn_frame[];
  T* stage_dy_dyn = reinterpret_cast<T*>(stage_dyn_frame);
  T* stage_xy_dyn = stage_dy_dyn + kStageDynElems;
  // The host's tests, term for term: a row slice that does not fill its team exactly is not staged at all.
  const bool stage_ok = kStage && kStageVecOk &&
                        num_vecs == static_cast<int>(team.size) * kVecsPerThread &&
                        (kTeam == TeamKind::kCluster ? cpg >= 32768 : (cpg >= 2048 && cpg <= 8192)) &&
                        p.num_segments / p.num_groups >= kLongRows &&
                        ((cpg * static_cast<int64_t>(sizeof(T))) % 16) == 0;
  uint32_t stage_bar_addr = 0;
  const uint32_t stage_bytes = static_cast<uint32_t>(2 * kVecsPerThread * blockDim.x * kVec * sizeof(T));
  uint32_t stage_parity = 0;
  bool stage_armed = false;
  const bool stage_dyn_ok =
      kStageDyn && num_vecs == static_cast<int>(team.size) * kVecsPerThread && cpg >= 32768 &&
      p.num_segments / p.num_groups >= kLongRows && ((cpg * static_cast<int64_t>(sizeof(T))) % 16) == 0;
  // The bulk stage's frame and test: the static frame for bf16, the dynamic one for fp32 (one of the two at most).
  T* const frame_dy = kStageDyn ? stage_dy_dyn : stage_dy;
  T* const frame_xy = kStageDyn ? stage_xy_dyn : stage_xy;
  const bool bulk_ok = kStageDyn ? stage_dyn_ok : stage_ok;
  if constexpr (kStage || kStageDyn) {
    if (stage_ok || stage_dyn_ok) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      stage_bar_addr = SmemAddr(&stage_bar);
      if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(stage_bar_addr) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
      }
#endif
      __syncthreads();  // no thread waits on a barrier another thread has not initialized
    }
  }

  // The group a team owns. With one team a block the grid is a multiple of num_groups teams, so a team only ever sees
  // group team.id() % num_groups. A multi-team block hosts kTeamsPerBlock teams of one group (kTeamsPerBlock rows of
  // it): its group is id / (teams / num_groups), and each team walks its own residue class of the group's rows.
  int64_t c0 = 0;
  int64_t first_row = 0;
  int64_t teams_per_group = 0;
  if constexpr (kTeamsPerBlock > 1) {
    teams_per_group = (static_cast<int64_t>(gridDim.x) * kTeamsPerBlock) / p.num_groups;
    const int64_t group = team.id() / teams_per_group;
    c0 = group * cpg;
    // block-uniform: team 0's row within the group, which every team of the block offsets by its index
    first_row = (static_cast<int64_t>(blockIdx.x) * kTeamsPerBlock) % teams_per_group;
  } else {
    c0 = (team.id() % p.num_groups) * cpg;
  }
  T_ACC dw[kVecsPerThread][kVec];
  T_ACC db[kVecsPerThread][kVec];
  if constexpr (kAffine) {
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      FillVec<kVec>(T_ACC(0), dw[i]);
      FillVec<kVec>(T_ACC(0), db[i]);
    }
  }

  // The hoist: gamma, beta and 1/gamma of the thread's channels, which no row changes, loaded once.
  T_ACC wr[kVecsPerThread][kVec];
  T_ACC rv[kVecsPerThread][kVec];
  T_ACC bv[kVecsPerThread][kVec];
  if constexpr (kHoist) {
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      const int j = team.lane + i * team.size;
      if (j < num_vecs) {
        LoadVec<kVec>(p.gamma + c0 + j * kVec, wr[i]);
        if constexpr (kUseBeta) {
          LoadVec<kVec>(p.beta + c0 + j * kVec, bv[i]);
        }
#pragma unroll
        for (int k = 0; k < kVec; ++k) rv[i][k] = T_ACC(1) / wr[i][k];
      }
    }
  }

  // Segments of one group are num_groups apart, the stride between the teams of a multi-team block.
  const int64_t stride = kTeamsPerBlock > 1 ? p.num_groups : 1;
  const int64_t first = kTeamsPerBlock > 1 ? first_row * p.num_groups + c0 / cpg
                                           : team.first();
  const bool stage_sub_ok =
      kStageSub && num_vecs <= static_cast<int>(team.size) * kVecsPerThread && num_vecs > 0;
  for (int64_t base = first; base < p.num_segments; base += team.count()) {
    const int64_t s = base + team.index * stride;
    const bool valid = s < p.num_segments;
    T_ACC g[kVecsPerThread][kVec];
    T_ACC x_hat[kVecsPerThread][kVec];
    bool active[kVecsPerThread];
    if constexpr (kStage || kStageDyn) {
      if (bulk_ok && valid) {
        if (!stage_armed) {
          if (threadIdx.x == 0) {
            StageRowRuns<T>(stage_bar_addr, stage_bytes, p.dy, p.x_or_y, frame_dy, frame_xy,
                            s * cpg + static_cast<int64_t>(team.rank()) * blockDim.x * kVec,
                            static_cast<int64_t>(team.size) * kVec, kVecsPerThread,
                            static_cast<int>(blockDim.x) * kVec);
          }
          stage_armed = true;
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
        WaitParity(stage_bar_addr, stage_parity);  // this row's bytes are in the frame
        stage_parity ^= 1;
#endif
      }
    }
    if constexpr (kStageSub) {
      if (stage_sub_ok) {
        // The first row's copies are issued here and waited for at once; every later row was issued one row ahead.
        if (!stage_armed) {
          if (valid) {
#pragma unroll
            for (int i = 0; i < kVecsPerThread; ++i) {
              const int j = team.lane + i * team.size;
              if (j < num_vecs) {
                CpAsync16(stage_dy + i * static_cast<int>(blockDim.x) * kVec + threadIdx.x * kVec,
                          p.dy + s * cpg + j * kVec);
                CpAsync16(stage_xy + i * static_cast<int>(blockDim.x) * kVec + threadIdx.x * kVec,
                          p.x_or_y + s * cpg + j * kVec);
              }
            }
            CpAsyncCommit();
          }
          stage_armed = true;
        }
        CpAsyncWaitAll();
      }
    }
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      const int j = team.lane + i * team.size;
      active[i] = valid && j < num_vecs;
      if (active[i]) {
        if ((kStage && stage_ok) || (kStageSub && stage_sub_ok) || (kStageDyn && stage_dyn_ok)) {
          LoadVec<kVec>(frame_dy + i * static_cast<int>(blockDim.x) * kVec + threadIdx.x * kVec, g[i]);
          LoadVec<kVec>(frame_xy + i * static_cast<int>(blockDim.x) * kVec + threadIdx.x * kVec, x_hat[i]);
        } else {
          LoadVec<kVec>(p.dy + s * cpg + j * kVec, g[i]);
          LoadVec<kVec>(p.x_or_y + s * cpg + j * kVec, x_hat[i]);
        }
      }
    }
    if constexpr (kStage || kStageDyn) {
      if (bulk_ok) {
        __syncthreads();  // every thread has read this row out of the frame; it may be refilled now
        const int64_t next = s + team.count();
        if (threadIdx.x == 0 && next < p.num_segments) {
          StageRowRuns<T>(stage_bar_addr, stage_bytes, p.dy, p.x_or_y, frame_dy, frame_xy,
                          next * cpg + static_cast<int64_t>(team.rank()) * blockDim.x * kVec,
                          static_cast<int64_t>(team.size) * kVec, kVecsPerThread,
                          static_cast<int>(blockDim.x) * kVec);
        }
      }
    }
    if constexpr (kStageSub) {
      if (stage_sub_ok) {
        __syncwarp();  // every lane has read its own slots out; the warp may refill them
        const int64_t next = s + team.count();
        if (next < p.num_segments) {
#pragma unroll
          for (int i = 0; i < kVecsPerThread; ++i) {
            const int j = team.lane + i * team.size;
            if (j < num_vecs) {
              CpAsync16(stage_dy + i * static_cast<int>(blockDim.x) * kVec + threadIdx.x * kVec,
                        p.dy + next * cpg + j * kVec);
              CpAsync16(stage_xy + i * static_cast<int>(blockDim.x) * kVec + threadIdx.x * kVec,
                        p.x_or_y + next * cpg + j * kVec);
            }
          }
        }
        CpAsyncCommit();
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
        if constexpr (kAffine && !kHoist) {
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
              if constexpr (kHoist) {
                xh -= bv[i][k];
              } else {
                xh -= b[k];
              }
            }
            if constexpr (kAffine) {
              if constexpr (kHoist) {
                xh *= rv[i][k];
              } else {
                xh /= w[k];
              }
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
            if constexpr (kHoist) {
              gk *= wr[i][k];
            } else {
              gk *= w[k];
            }
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
      sum_g_x_hat = TeamSum<kTeam, kSlotUnroll, kOneBarrierSlot>(sum_g_x_hat, team.size, shm, team.round);
    } else {
      TeamSum2<kTeam, kSlotUnroll, kTeamsPerBlock, kOneBarrierSlot>(sum_g, sum_g_x_hat, team.size, shm, team.round);
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

  if constexpr (kAffine && kTeamsPerBlock > 1) {
    // One partial row a block: the teams store their sums in shared memory (every reduction slot is free by now) and
    // team 0 adds them in team order, (team 0's rows) + (team 1's rows) + ..., and stores the row. The width is capped
    // at kMultiTeamCpg, the buffer's, which the entry guard has made the call's.
    const int stage_cpg_raw = static_cast<int>(p.channels_per_group);
    const int stage_cpg = stage_cpg_raw < kMultiTeamCpg ? stage_cpg_raw : kMultiTeamCpg;
    const int stage_vecs = stage_cpg / kVec;
    T_ACC* stage_w = shm + kTeamsPerBlock * 2 * cuda_utils::kWarpSize;
    T_ACC* my_w = stage_w + team.index * stage_cpg;
    T_ACC* stage_b = kRMS ? nullptr : stage_w + kTeamsPerBlock * stage_cpg;
    T_ACC* my_b = kRMS ? nullptr : stage_b + team.index * stage_cpg;
#pragma unroll
    for (int i = 0; i < kVecsPerThread; ++i) {
      const int j = team.lane + i * team.size;
      if (j < stage_vecs) {
        StoreVec<kVec>(dw[i], my_w + j * kVec);
        if constexpr (!kRMS) {
          StoreVec<kVec>(db[i], my_b + j * kVec);
        }
      }
    }
    __syncthreads();
    if (team.index == 0) {
      // The block's row is its residue class in the group, blockIdx.x % (teams_per_group / kTeamsPerBlock): every
      // row of [0, num_phases) once a group.
      const int64_t phase = first_row / kTeamsPerBlock;
      T_ACC* dw_out = p.dgamma_partial + phase * num_channels + c0;
      T_ACC* db_out = kRMS ? nullptr : p.dbeta_partial + phase * num_channels + c0;
#pragma unroll
      for (int i = 0; i < kVecsPerThread; ++i) {
        const int j = team.lane + i * team.size;
        if (j < stage_vecs) {
          T_ACC aw[kVec];
          T_ACC ab[kVec];
#pragma unroll
          for (int k = 0; k < kVec; ++k) {
            aw[k] = stage_w[j * kVec + k];
            if constexpr (!kRMS) {
              ab[k] = stage_b[j * kVec + k];
            } else {
              ab[k] = T_ACC(0);
            }
          }
#pragma unroll
          for (int t = 1; t < kTeamsPerBlock; ++t) {
#pragma unroll
            for (int k = 0; k < kVec; ++k) {
              aw[k] += stage_w[t * stage_cpg + j * kVec + k];
              if constexpr (!kRMS) {
                ab[k] += stage_b[t * stage_cpg + j * kVec + k];
              }
            }
          }
          StoreVec<kVec>(aw, dw_out + j * kVec);
          if constexpr (!kRMS) {
            StoreVec<kVec>(ab, db_out + j * kVec);
          }
        }
      }
    }
  } else if constexpr (kAffine) {
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
  // Lets a fold launched dependent on this kernel be scheduled early: every LayerNorm fold so launched, and the RMS
  // fold above kFuseMaxChannels channels (the vec4 tree). Its cudaGridDependencySynchronize() still waits for this grid
  // to complete and its stores to be visible.
  if constexpr (kAffine) {
    if (!kRMS || num_channels > kFuseMaxChannels) {
      cudaTriggerProgrammaticLaunchCompletion();
    }
  }
  team.Finish();
}

// The folds of the partial rows into dgamma (and dbeta). In each, one thread or lane owns a channel and adds its phases
// r, r + 32, r + 64, ... of each residue r < 32 in ascending order; the 32 residue sums are then added by the
// 16/8/4/2/1 shuffle (or register) tree, so the order of every sum is fixed. They differ in how many channels a block
// owns and how many planes a launch folds; FoldPartials picks one by the plane's size and the device.

// Four channel components of one transposed cell form one 16-byte record.
template <typename T_ACC>
struct alignas(16) AccRecord4 {
  T_ACC v[4];
};

// One plane, (kWarpSize, kWarpSize) blocks of 128 channels: thread (x, y) adds residue y of the four channels
// 4 * (32 * block + x) + 0..3, the block transposes the four sums through shared memory as one record a thread, and
// warp y folds the residues of its four channels.
template <typename T, typename T_ACC>
__global__ void SumPartialsKernel(int64_t num_phases, int64_t num_channels, const T_ACC* __restrict__ partial,
                                  T* __restrict__ out) {
  constexpr int kChannelsPerLane = 4;
  constexpr int kCells = cuda_utils::kWarpSize * (cuda_utils::kWarpSize + 1);
  __shared__ __align__(16) T_ACC tile[kCells * kChannelsPerLane];
  AccRecord4<T_ACC>* const rec = reinterpret_cast<AccRecord4<T_ACC>*>(tile);
  static_assert(kCells * sizeof(AccRecord4<T_ACC>) <= sizeof(tile), "the packed transpose must fit its shared array");
  const int64_t c0 = 4 * (blockIdx.x * cuda_utils::kWarpSize + threadIdx.x);
  T_ACC sums[kChannelsPerLane] = {};
  if (c0 + kChannelsPerLane - 1 < num_channels) {
    for (int64_t i = threadIdx.y; i < num_phases; i += blockDim.y) {
      const T_ACC* q = partial + i * num_channels + c0;
#pragma unroll
      for (int k = 0; k < kChannelsPerLane; ++k) {
        sums[k] += q[k];
      }
    }
  } else if (c0 < num_channels) {
    // Only the last lane of the plane can be partial.
    for (int64_t i = threadIdx.y; i < num_phases; i += blockDim.y) {
#pragma unroll
      for (int k = 0; k < kChannelsPerLane; ++k) {
        if (c0 + k < num_channels) {
          sums[k] += partial[i * num_channels + c0 + k];
        }
      }
    }
  }
  AccRecord4<T_ACC> packed;
#pragma unroll
  for (int k = 0; k < kChannelsPerLane; ++k) {
    packed.v[k] = sums[k];
  }
  rec[threadIdx.y * (cuda_utils::kWarpSize + 1) + threadIdx.x] = packed;
  __syncthreads();
  const AccRecord4<T_ACC> folded =
      rec[threadIdx.x * (cuda_utils::kWarpSize + 1) + threadIdx.y];
#pragma unroll
  for (int k = 0; k < kChannelsPerLane; ++k) {
    sums[k] = reduce::WarpReduce(folded.v[k], reduce::SumOp<T_ACC>());
  }
  const int64_t c_out = 4 * (blockIdx.x * cuda_utils::kWarpSize + threadIdx.y);
  if (threadIdx.x == 0) {
#pragma unroll
    for (int k = 0; k < kChannelsPerLane; ++k) {
      if (c_out + k < num_channels) {
        out[c_out + k] = static_cast<T>(sums[k]);
      }
    }
  }
}

// One plane, blocks of 512 threads over 16 channels: threadIdx.x & 15 is the channel and threadIdx.x >> 4 the residue,
// so a load covers two coalesced 16-channel runs; after one barrier, warp w folds channel w.
template <typename T, typename T_ACC>
__global__ __launch_bounds__(512) void SumPartialsWarpKernel(int64_t num_phases, int64_t num_channels,
                                                            const T_ACC* __restrict__ partial, T* __restrict__ out) {
  __shared__ T_ACC shm[cuda_utils::kWarpSize][16 + 1];  // [residue][channel], padded for conflict-free reads below
  const int txc = threadIdx.x & 15;
  const int tyr = threadIdx.x >> 4;
  const int64_t c = (int64_t)blockIdx.x * 16 + txc;
  T_ACC sum = T_ACC(0);
  if (c < num_channels) {
    for (int64_t i = tyr; i < num_phases; i += cuda_utils::kWarpSize) {
      sum += partial[i * num_channels + c];
    }
  }
  shm[tyr][txc] = sum;
  __syncthreads();
  const int lane = threadIdx.x & (cuda_utils::kWarpSize - 1);
  const int w = threadIdx.x / cuda_utils::kWarpSize;
  const int64_t c_out = (int64_t)blockIdx.x * 16 + w;
  T_ACC v = shm[lane][w];
  v = reduce::WarpReduce(v, reduce::SumOp<T_ACC>());
  if (lane == 0 && c_out < num_channels) {
    out[c_out] = static_cast<T>(v);
  }
}

// Both planes (blockIdx.y), at most kWarpSize phases: a thread owns a channel, holds its phases in 32 register slots
// (phase l in slot l, zero beyond num_phases) loaded with constant indices, and folds them with the same tree: no
// barrier and no shared memory. Launched only for num_phases <= kWarpSize; a larger count stores nothing rather than a
// short sum.
template <typename T, typename T_ACC>
__global__ __launch_bounds__(256) void SumPartialsTreeKernel(int64_t num_phases, int64_t num_channels,
                                                            const T_ACC* __restrict__ partial, T* __restrict__ dgamma,
                                                            T* __restrict__ dbeta) {
  const bool plane = blockIdx.y != 0;
  const T_ACC* __restrict__ base = partial + (plane ? num_phases * num_channels : 0);
  T* __restrict__ out = plane ? dbeta : dgamma;
  const int64_t c = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= num_channels) return;
  if (num_phases > cuda_utils::kWarpSize) return;
  T_ACC v[cuda_utils::kWarpSize];
#pragma unroll
  for (int l = 0; l < cuda_utils::kWarpSize; ++l) {
    v[l] = (l < num_phases) ? base[(int64_t)l * num_channels + c] : T_ACC(0);
  }
#pragma unroll
  for (int off = cuda_utils::kWarpSize / 2; off > 0; off >>= 1) {
#pragma unroll
    for (int l = 0; l < off; ++l) v[l] += v[l + off];
  }
  out[c] = static_cast<T>(v[0]);
}

// The register tree for num_phases <= 16 at four channels a thread: one 16-byte load a phase, 16 live slots and one
// packed store. The 32-slot tree's first step, v[l] += v[l + 16], is kept as an add of +0.0, which turns a -0.0
// partial into +0.0 exactly as that step does. Launched only for num_phases <= 16 and num_channels % 4 == 0 (the
// outputs are fresh allocations, so a multiple-of-4 offset is aligned for the packed store); any other call stores
// nothing. kPlanes: 1 (dgamma) or 2 (blockIdx.y picks the plane).
template <typename T, typename T_ACC, int kPlanes>
__global__ __launch_bounds__(256) void SumPartialsTreeVec4Kernel(int64_t num_phases, int64_t num_channels,
                                                                const T_ACC* __restrict__ partial,
                                                                T* __restrict__ dgamma, T* __restrict__ dbeta) {
  cudaGridDependencySynchronize();
  static_assert(kPlanes == 1 || kPlanes == 2, "this fold writes one or two planes");
  const bool plane = kPlanes == 2 && blockIdx.y != 0;
  const T_ACC* __restrict__ base = partial + (plane ? num_phases * num_channels : 0);
  T* out = plane ? dbeta : dgamma;
  const int64_t c0 = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
  if (c0 >= num_channels) return;
  if (num_phases > 16) return;
  if ((num_channels & 3) != 0) return;
  T_ACC v[16][4];
#pragma unroll
  for (int l = 0; l < 16; ++l) {
    if (l < num_phases) {
      // The fold is the only reader of every partial: a last-use load.
      const float4 d = __ldlu(reinterpret_cast<const float4*>(base + (int64_t)l * num_channels + c0));
      v[l][0] = d.x; v[l][1] = d.y; v[l][2] = d.z; v[l][3] = d.w;
    } else {
#pragma unroll
      for (int k = 0; k < 4; ++k) v[l][k] = T_ACC(0);
    }
  }
#pragma unroll
  for (int l = 0; l < 16; ++l) {
#pragma unroll
    for (int k = 0; k < 4; ++k) v[l][k] += T_ACC(0);  // the 32-slot tree's v[l] += v[l + 16], +0.0 here
  }
#pragma unroll
  for (int off = 8; off > 0; off >>= 1) {
#pragma unroll
    for (int l = 0; l < off; ++l) {
#pragma unroll
      for (int k = 0; k < 4; ++k) v[l][k] += v[l + off][k];
    }
  }
  static_assert(sizeof(T) == 2 || sizeof(T) == 4, "the packed store is written for 2- and 4-byte outputs only");
  T o[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) o[k] = static_cast<T>(v[0][k]);
  if constexpr (sizeof(T) == 4) {
    float4 p;
    p.x = o[0];
    p.y = o[1];
    p.z = o[2];
    p.w = o[3];
    *reinterpret_cast<float4*>(out + c0) = p;
  } else {
    struct alignas(8) Pack4 {
      T v[4];
    } p;
#pragma unroll
    for (int k = 0; k < 4; ++k) p.v[k] = o[k];
    *reinterpret_cast<uint2*>(out + c0) = *reinterpret_cast<const uint2*>(&p);
  }
}

// Both planes in one launch, (kWarpSize, kWarpSize) blocks of 32 channels and blockIdx.y the plane; the planes are
// contiguous, so one base pointer addresses both.
template <typename T, typename T_ACC>
__global__ void SumPartials2Kernel(int64_t num_phases, int64_t num_channels,
                                   const T_ACC* __restrict__ partial, T* __restrict__ dgamma,
                                   T* __restrict__ dbeta) {
  __shared__ T_ACC shm[cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];
  const bool plane = blockIdx.y != 0;
  const T_ACC* __restrict__ base = partial + (plane ? num_phases * num_channels : 0);
  T* __restrict__ out = plane ? dbeta : dgamma;
  cudaGridDependencySynchronize();
  const int64_t c = blockIdx.x * cuda_utils::kWarpSize + threadIdx.x;
  T_ACC sum = T_ACC(0);
  if (c < num_channels) {
    for (int64_t i = threadIdx.y; i < num_phases; i += blockDim.y) {
      sum += base[i * num_channels + c];
    }
  }
  shm[threadIdx.y][threadIdx.x] = sum;
  __syncthreads();
  sum = reduce::WarpReduce(shm[threadIdx.x][threadIdx.y], reduce::SumOp<T_ACC>());
  const int64_t c_out = blockIdx.x * cuda_utils::kWarpSize + threadIdx.y;
  if (threadIdx.x == 0 && c_out < num_channels) {
    out[c_out] = static_cast<T>(sum);
  }
}

// Both planes, (kWarpSize, kWarpSize) blocks of 32 channels: a thread reads the dgamma and the dbeta plane of its
// channel with two independent loads, on half SumPartials2Kernel's grid.
template <typename T, typename T_ACC>
__global__ void SumPartials2DualKernel(int64_t num_phases, int64_t num_channels,
                                       const T_ACC* __restrict__ partial, T* __restrict__ dgamma,
                                       T* __restrict__ dbeta) {
  __shared__ T_ACC shm[2][cuda_utils::kWarpSize][cuda_utils::kWarpSize + 1];
  const T_ACC* __restrict__ baseg = partial;
  const T_ACC* __restrict__ baseb = partial + num_phases * num_channels;
  const int64_t c = blockIdx.x * cuda_utils::kWarpSize + threadIdx.x;
  T_ACC sumg = T_ACC(0);
  T_ACC sumb = T_ACC(0);
  if (c < num_channels) {
    for (int64_t i = threadIdx.y; i < num_phases; i += blockDim.y) {
      const int64_t off = i * num_channels + c;
      sumg += baseg[off];
      sumb += baseb[off];
    }
  }
  shm[0][threadIdx.y][threadIdx.x] = sumg;
  shm[1][threadIdx.y][threadIdx.x] = sumb;
  __syncthreads();
  sumg = reduce::WarpReduce(shm[0][threadIdx.x][threadIdx.y], reduce::SumOp<T_ACC>());
  sumb = reduce::WarpReduce(shm[1][threadIdx.x][threadIdx.y], reduce::SumOp<T_ACC>());
  const int64_t c_out = blockIdx.x * cuda_utils::kWarpSize + threadIdx.y;
  if (threadIdx.x == 0 && c_out < num_channels) {
    dgamma[c_out] = static_cast<T>(sumg);
    dbeta[c_out] = static_cast<T>(sumb);
  }
}

// Both planes with two loads a thread, in SumPartialsWarpKernel's blocks of 512 threads over 16 channels.
template <typename T, typename T_ACC>
__global__ __launch_bounds__(512) void SumPartials2Dual16Kernel(int64_t num_phases, int64_t num_channels,
                                                               const T_ACC* __restrict__ partial,
                                                               T* __restrict__ dgamma, T* __restrict__ dbeta) {
  __shared__ T_ACC shm[2][cuda_utils::kWarpSize][16 + 1];
  const T_ACC* __restrict__ baseg = partial;
  const T_ACC* __restrict__ baseb = partial + num_phases * num_channels;
  const int txc = threadIdx.x & 15;
  const int tyr = threadIdx.x >> 4;
  const int64_t c = (int64_t)blockIdx.x * 16 + txc;
  T_ACC sumg = T_ACC(0);
  T_ACC sumb = T_ACC(0);
  if (c < num_channels) {
    // A dependent launch waits for its producer here; an ordinary one returns at once.
    cudaGridDependencySynchronize();
    for (int64_t i = tyr; i < num_phases; i += cuda_utils::kWarpSize) {
      const int64_t off = i * num_channels + c;
      sumg += baseg[off];
      sumb += baseb[off];
    }
  }
  shm[0][tyr][txc] = sumg;
  shm[1][tyr][txc] = sumb;
  __syncthreads();
  const int lane = threadIdx.x & (cuda_utils::kWarpSize - 1);
  const int w = threadIdx.x / cuda_utils::kWarpSize;
  T_ACC vg = reduce::WarpReduce(shm[0][lane][w], reduce::SumOp<T_ACC>());
  T_ACC vb = reduce::WarpReduce(shm[1][lane][w], reduce::SumOp<T_ACC>());
  const int64_t c_out = (int64_t)blockIdx.x * 16 + w;
  if (lane == 0 && c_out < num_channels) {
    dgamma[c_out] = static_cast<T>(vg);
    dbeta[c_out] = static_cast<T>(vb);
  }
}

// Both planes, blocks of 512 threads over 8 channels: 8 channels x 2 planes x 32 residues, so a warp is 8 channels of
// one plane at two residues, and the block's sixteen warps fold its sixteen (channel, plane) pairs.
template <typename T, typename T_ACC>
__global__ __launch_bounds__(512) void SumPartials2Dual8Kernel(int64_t num_phases, int64_t num_channels,
                                                              const T_ACC* __restrict__ partial,
                                                              T* __restrict__ dgamma, T* __restrict__ dbeta) {
  __shared__ T_ACC shm[cuda_utils::kWarpSize][16 + 1];
  const int txc = threadIdx.x & 7;               // channel inside the block
  const int tpl = (threadIdx.x >> 3) & 1;        // 0 = dgamma's plane, 1 = dbeta's plane
  const int tyr = threadIdx.x >> 4;              // phase residue, 0..31
  const int64_t c = (int64_t)blockIdx.x * 8 + txc;
  const T_ACC* __restrict__ base = partial + (tpl ? num_phases * num_channels : 0);
  T_ACC sum = T_ACC(0);
  if (c < num_channels) {
    cudaGridDependencySynchronize();
    for (int64_t i = tyr; i < num_phases; i += cuda_utils::kWarpSize) {
      sum += base[i * num_channels + c];
    }
  }
  shm[tyr][txc + 8 * tpl] = sum;
  __syncthreads();
  const int lane = threadIdx.x & (cuda_utils::kWarpSize - 1);
  const int w = threadIdx.x / cuda_utils::kWarpSize;
  const T_ACC v = reduce::WarpReduce(shm[lane][w], reduce::SumOp<T_ACC>());
  const int64_t c_out = (int64_t)blockIdx.x * 8 + (w & 7);
  if (lane == 0 && c_out < num_channels) {
    if (w & 8) {
      dbeta[c_out] = static_cast<T>(v);
    } else {
      dgamma[c_out] = static_cast<T>(v);
    }
  }
}

// The launch shape of a call.
struct Layout {
  int64_t vec;              // elements an access
  TeamKind kind;
  int64_t team_size;        // threads a segment (a cluster team's: set by FitLayout)
  int64_t block_size;       // (a cluster team's: set by FitLayout)
  int64_t teams_per_block;  // > 1 for kSubWarp and for the backward's multi-team block
  int64_t cluster_size;     // blocks a team for kCluster (set by FitLayout), else 1
  int64_t threads;          // threads that hold elements of a segment
  int64_t elems;            // elements a thread
  int64_t combine;          // the backward's multi-team block: teams that combine their partial sums, else 1
};

inline int64_t NextPow2(int64_t x) {
  int64_t p = 1;
  while (p < x) {
    p <<= 1;
  }
  return p;
}

// The layout of a segment of channels_per_group channels at `elems` elements a thread: vectors of `vec_bytes` when the
// width is a whole number of them and every pointer is aligned to them, else single elements; then a sub-warp team (a
// power of two up to a warp), a block team (a power of two up to kMaxBlockSize threads) or a cluster team.
template <typename T>
Layout MakeLayout(int64_t channels_per_group, std::initializer_list<const void*> ptrs, int64_t vec_bytes = kVecBytes,
                  int64_t elems = kElemsPerThread) {
  const int64_t kMaxVec = vec_bytes / sizeof(T);
  bool vectorize = channels_per_group % kMaxVec == 0;
  for (const void* ptr : ptrs) {
    vectorize &= reinterpret_cast<std::uintptr_t>(ptr) % vec_bytes == 0;
  }
  Layout layout;
  layout.vec = vectorize ? kMaxVec : 1;
  layout.elems = elems;
  layout.combine = 1;
  const int64_t vecs_per_thread = elems / layout.vec;
  layout.threads = utils::DivUp(channels_per_group / layout.vec, vecs_per_thread);
  if (layout.threads <= kMaxBlockSize) {
    layout.team_size = NextPow2(layout.threads);
    layout.kind = layout.team_size > cuda_utils::kWarpSize ? TeamKind::kBlock : TeamKind::kSubWarp;
    layout.block_size = layout.kind == TeamKind::kBlock ? layout.team_size : kSubWarpBlockSize;
    layout.teams_per_block = layout.block_size / layout.team_size;
    layout.cluster_size = 1;
  } else {
    // The block and cluster sizes depend on the kernel; see FitLayout.
    layout.kind = TeamKind::kCluster;
    layout.team_size = 0;
    layout.block_size = 0;
    layout.teams_per_block = 1;
    layout.cluster_size = 0;
  }
  return layout;
}

// Calls f(std::integral_constant<int, kVec>, std::integral_constant<TeamKind, kTeam>) for a layout of single elements
// or 16-byte vectors.
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

// An ordinary launch made dependent on the previous kernel of the stream (programmatic dependent launch): the kernel
// may start before its predecessor has finished, and its cudaGridDependencySynchronize() waits for it. A driver that
// refuses the attribute gets the ordinary launch: the same work in the same order, without the overlap.
template <class KernelFunc, class... Args>
void LaunchKernelPdl(KernelFunc kernel, dim3 grid, dim3 block, int64_t smem, cudaStream_t stream, Args... args) {
  cudaLaunchAttribute attr;
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config = {};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = smem;
  config.stream = stream;
  config.attrs = &attr;
  config.numAttrs = 1;
  if (cudaLaunchKernelEx(&config, kernel, args...) == cudaSuccess) {
    return;
  }
  (void)cudaGetLastError();
  cuda_utils::LaunchKernel(kernel, grid, block, smem, stream, args...);
}

// A launch of num_clusters clusters of cluster_size blocks, with `smem` bytes of dynamic shared memory a block, and
// with `pdl` dependent on the previous kernel of the stream. `attrs` must outlive the launch.
inline cudaLaunchConfig_t ClusterLaunchConfig(int64_t cluster_size, int64_t block_size, int64_t num_clusters,
                                              cudaStream_t stream, cudaLaunchAttribute (&attrs)[2], int64_t smem,
                                              bool pdl = false) {
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = cluster_size;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(num_clusters * cluster_size);
  config.blockDim = dim3(block_size);
  config.dynamicSmemBytes = static_cast<size_t>(smem);
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = pdl ? 2 : 1;
  return config;
}

// Occupancy queries and dynamic shared memory opt-ins cost microseconds of host time each and depend only on the
// kernel, the device and the launch shape, so each is made once per process; a refused opt-in stays refused. A kernel
// is keyed by its host stub: every loaded copy of this module has its own stubs, function attributes and entries.
enum class LaunchQueryKind {
  kBlocksPerSm,       // blocks of `block` threads resident on an SM, without clusters
  kClusters,          // clusters of `cluster` blocks of `block` threads resident on the device
  kDynamicSmemOptIn,  // 1 if the kernel may take `smem` bytes of dynamic shared memory, else 0
};

struct LaunchQuery {
  LaunchQueryKind kind;
  const void* kernel;
  int device;
  int64_t cluster;
  int64_t block;
  int64_t smem;
  bool operator==(const LaunchQuery&) const = default;
};

struct LaunchQueryHash {
  size_t operator()(const LaunchQuery& q) const {
    size_t h = std::hash<const void*>{}(q.kernel);
    for (const int64_t v : {static_cast<int64_t>(q.kind), static_cast<int64_t>(q.device), q.cluster, q.block, q.smem}) {
      h = h * 1000003 ^ std::hash<int64_t>{}(v);
    }
    return h;
  }
};

// The cache has internal linkage, so that two different builds of this file loaded into one process never share it.
static int AskLaunchQuery(const LaunchQuery& q) {
  int result = 0;
  switch (q.kind) {
    case LaunchQueryKind::kBlocksPerSm:
      AT_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&result, q.kernel, static_cast<int>(q.block),
                                                                  static_cast<size_t>(q.smem)));
      break;
    case LaunchQueryKind::kClusters: {
      cudaLaunchAttribute attrs[2];
      const cudaLaunchConfig_t config = ClusterLaunchConfig(q.cluster, q.block, 1, nullptr, attrs, q.smem);
      AT_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&result, q.kernel, &config));
      break;
    }
    case LaunchQueryKind::kDynamicSmemOptIn:
      if (cudaFuncSetAttribute(q.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(q.smem)) ==
          cudaSuccess) {
        result = 1;
      } else {
        (void)cudaGetLastError();  // a refusal selects another body; it is not an error of the call
      }
      break;
  }
  return result;
}

static int CachedLaunchQuery(const LaunchQuery& query) {
  static std::mutex mutex;
  static std::unordered_map<LaunchQuery, int, LaunchQueryHash> results;
  {
    const std::lock_guard<std::mutex> lock(mutex);
    const auto it = results.find(query);
    if (it != results.end()) {
      return it->second;
    }
  }
  const int result = AskLaunchQuery(query);
  const std::lock_guard<std::mutex> lock(mutex);
  return results.emplace(query, result).first->second;
}

// Opts `kernel` into `smem` bytes of dynamic shared memory on `device`; false when the device refuses.
template <typename Kernel>
bool RaiseDynamicSmem(Kernel kernel, int device, int64_t smem) {
  const void* const stub = reinterpret_cast<const void*>(kernel);
  return CachedLaunchQuery({LaunchQueryKind::kDynamicSmemOptIn, stub, device, 0, 0, smem}) != 0;
}

// Completes the layout for `kernel` and returns it with the number of launch units, blocks or for kCluster whole
// clusters, that fit on the device at once. A cluster team spreads its threads over blocks of at most 1024 or 512
// threads (at most max_block); each cluster normalizes one segment at a time, so the shape with the most resident
// clusters is selected, preferring smaller blocks on a tie. When no cluster shape fits, the call fails, or with
// `may_not_fit` returns 0 units.
template <typename Kernel>
std::pair<Layout, int64_t> FitLayout(Kernel kernel, const Layout& layout, int device, int num_sms, int64_t smem = 0,
                                     int64_t max_block = kMaxBlockSize, bool may_not_fit = false) {
  const void* const stub = reinterpret_cast<const void*>(kernel);
  if (layout.kind != TeamKind::kCluster) {
    const int blocks_per_sm =
        CachedLaunchQuery({LaunchQueryKind::kBlocksPerSm, stub, device, 0, layout.block_size, 0});
    return {layout, std::max(blocks_per_sm, 1) * static_cast<int64_t>(num_sms)};
  }
  Layout best = layout;
  int best_clusters = -1;
  for (const int64_t block_limit : {kMaxBlockSize, kMaxBlockSize / 2}) {
    if (block_limit > max_block) {
      continue;
    }
    Layout fit = layout;
    fit.cluster_size = utils::DivUp(layout.threads, block_limit);
    if (fit.cluster_size > kMaxClusterSize) {
      continue;
    }
    fit.block_size =
        utils::DivUp(utils::DivUp(layout.threads, fit.cluster_size), cuda_utils::kWarpSize) * cuda_utils::kWarpSize;
    fit.team_size = fit.block_size * fit.cluster_size;
    const int num_clusters =
        CachedLaunchQuery({LaunchQueryKind::kClusters, stub, device, fit.cluster_size, fit.block_size, smem});
    if (num_clusters >= best_clusters) {
      best = fit;
      best_clusters = num_clusters;
    }
  }
  if (best_clusters <= 0 && may_not_fit) {
    return {layout, 0};
  }
  TORCH_CHECK(best_clusters > 0, "no cluster of ", best.cluster_size, " blocks of ", best.block_size, " threads and ",
              smem, " bytes of dynamic shared memory fits");
  return {best, best_clusters};
}

// Launches num_units units (blocks, or whole clusters for kCluster) of the fitted layout, with `smem` bytes of dynamic
// shared memory a block, and with `pdl` dependent on the previous kernel of the stream (an ordinary launch if the
// driver refuses that).
template <typename Kernel, typename Params>
void LaunchUnits(Kernel kernel, const Layout& layout, int64_t num_units, cudaStream_t stream, const Params& params,
                 int64_t smem = 0, bool pdl = false) {
  if (layout.kind == TeamKind::kCluster) {
    cudaLaunchAttribute attrs[2];
    if (pdl) {
      const cudaLaunchConfig_t config =
          ClusterLaunchConfig(layout.cluster_size, layout.block_size, num_units, stream, attrs, smem, true);
      if (cudaLaunchKernelEx(&config, kernel, params) == cudaSuccess) {
        return;
      }
      (void)cudaGetLastError();
    }
    const cudaLaunchConfig_t config =
        ClusterLaunchConfig(layout.cluster_size, layout.block_size, num_units, stream, attrs, smem);
    AT_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, params));
  } else if (pdl) {
    LaunchKernelPdl(kernel, dim3(num_units), dim3(layout.block_size), smem, stream, params);
  } else {
    cuda_utils::LaunchKernel(kernel, dim3(num_units), dim3(layout.block_size), smem, stream, params);
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
  Layout layout = MakeLayout<T>(channels_per_group, {x, gamma, beta, y});
  // A segment too wide for one block takes kFwdWideElems elements a thread: at most 2048 threads, as
  // channels_per_group is at most 65536.
  if (layout.kind == TeamKind::kCluster) {
    layout = MakeLayout<T>(channels_per_group, {x, gamma, beta, y}, kVecBytes, kFwdWideElems);
  }
  const FwdParams<T, T_ACC> params{num_segments, num_groups, channels_per_group, layout.team_size,
                                   static_cast<T_ACC>(eps), x, gamma, beta, y, mean, rstd};
  const c10::DeviceIndex device = c10::cuda::current_device();
  const int num_sms = at::cuda::getDeviceProperties(device)->multiProcessorCount;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);
  // Launches `kernel` with the `shape` layout on as many units as are resident, at most one team a segment.
  auto launch = [&](auto kernel, const Layout& shape, int64_t smem = 0, int64_t max_block = kMaxBlockSize) {
    const auto [fit, resident_units] = FitLayout(kernel, shape, device, num_sms, smem, max_block);
    const int64_t num_units = std::min(resident_units, utils::DivUp(num_segments, fit.teams_per_block));
    LaunchUnits(kernel, fit, num_units, stream, params, smem);
  };
  DispatchLayout<T>(layout, [&](auto vec, auto team) {
    constexpr int kVec = decltype(vec)::value;
    constexpr TeamKind kTeam = decltype(team)::value;
    constexpr int kElems = static_cast<int>(kElemsPerThread);
    constexpr int kWideElems = static_cast<int>(kFwdWideElems);
    auto with_affine = [&](auto affine) {
      constexpr bool kAffine = decltype(affine)::value;
      if constexpr (kTeam == TeamKind::kSubWarp) {
        launch(GroupNormFwdKernel<T, T_ACC, kVec, kElems, kTeam, kRMS, kAffine>, layout);
      } else if constexpr (kTeam == TeamKind::kBlock) {
        if (layout.elems == kFwdWideElems) {
          launch(GroupNormFwdKernel<T, T_ACC, kVec, kWideElems, kTeam, kRMS, kAffine>, layout);
        } else {
          launch(GroupNormFwdKernel<T, T_ACC, kVec, kElems, kTeam, kRMS, kAffine>, layout);
        }
      } else if constexpr (UseSharedWideInput<T, kVec, kTeam, kWideElems, kMaxBlockSize / 2>()) {
        launch(GroupNormFwdKernel<T, T_ACC, kVec, kWideElems, kTeam, kRMS, kAffine, kMaxBlockSize / 2>, layout, 0,
               kMaxBlockSize / 2);
      } else if constexpr (UseWidePrefetch<T, kAffine, kVec, kTeam, kWideElems, kMaxBlockSize>()) {
        constexpr int64_t kSmem = WidePrefetchSmemBytes<T, kVec, kWideElems, static_cast<int>(kMaxBlockSize)>();
        const auto kernel = GroupNormFwdKernel<T, T_ACC, kVec, kWideElems, kTeam, kRMS, true>;
        if (RaiseDynamicSmem(kernel, device, kSmem)) {
          launch(kernel, layout, kSmem);
        } else {
          // A device that refuses the slots runs the kElemsPerThread layout, which needs none.
          launch(GroupNormFwdKernel<T, T_ACC, kVec, kElems, kTeam, kRMS, true>,
                 MakeLayout<T>(channels_per_group, {x, gamma, beta, y}));
        }
      } else {
        launch(GroupNormFwdKernel<T, T_ACC, kVec, kWideElems, kTeam, kRMS, kAffine>, layout);
      }
    };
    if (gamma != nullptr) {
      with_affine(std::true_type{});
    } else {
      with_affine(std::false_type{});
    }
  });
}

// Launches one single-team backward body with kHoist fixed: a row-staged body where a stage applies, else the plain
// body. launch(kernel, kernel_pdl, smem) is GroupNormBwd's.
template <typename T, bool kRMS, int kVec, TeamKind kTeam, bool kAffine, bool kME, bool kHoist, typename Launch>
void LaunchBwdBody(int64_t rows, int64_t channels_per_group, int64_t num_elements, Launch&& launch) {
  using T_ACC = at::acc_type<T, true>;
  constexpr bool kFullVec = kVec == kVecBytes / sizeof(T);
  // The bulk-copy stages need kLongRows rows and a segment that fills its power-of-two team exactly, as the device
  // tests; the per-lane stage runs only on the hoisted body, which has kLongRows rows.
  constexpr int64_t kVecsPerThread = kElemsPerThread / kVec;
  const int64_t vecs = channels_per_group / kVec;
  const bool fills_team = vecs == NextPow2(utils::DivUp(vecs, kVecsPerThread)) * kVecsPerThread;
  if constexpr (std::is_same_v<T, at::BFloat16> && kAffine && kFullVec && kTeam != TeamKind::kSubWarp) {
    // bf16 bulk copies: block teams of 2048 to 8192 channels, cluster teams from 32768.
    const bool band = kTeam == TeamKind::kCluster ? channels_per_group >= 32768
                                                  : channels_per_group >= 2048 && channels_per_group <= 8192;
    if (rows >= kLongRows && band && fills_team) {
      launch(GroupNormBwdKernel<T, T_ACC, kVec, kTeam, kRMS, kAffine, kME, kHoist, true, 1, false>,
             GroupNormBwdKernel<T, T_ACC, kVec, kTeam, kRMS, kAffine, kME, kHoist, true, 1, true>);
      return;
    }
  }
  if constexpr (std::is_same_v<T, at::BFloat16> && kAffine && kFullVec && kTeam == TeamKind::kSubWarp && !kRMS &&
                kHoist) {
    // bf16 LayerNorm from y in sub-warp teams: per-lane copies, from 12M elements (smaller calls measured no gain).
    if (num_elements >= 12000000) {
      launch(GroupNormBwdKernel<T, T_ACC, kVec, kTeam, kRMS, kAffine, kME, kHoist, true, 1, false>,
             GroupNormBwdKernel<T, T_ACC, kVec, kTeam, kRMS, kAffine, kME, kHoist, true, 1, true>);
      return;
    }
  }
  if constexpr (std::is_same_v<T, float> && kAffine && kFullVec && kTeam == TeamKind::kCluster) {
    // fp32 bulk copies into a dynamic frame of one row of dy and x_or_y for a 1024-thread block, from 32768 channels;
    // never launched dependent. A device that refuses the frame, or a fit that cannot hold it, takes the plain body.
    constexpr int64_t kFrameBytes = 2 * kVecsPerThread * kMaxBlockSize * kVec * static_cast<int64_t>(sizeof(T));
    if (rows >= kLongRows && channels_per_group >= 32768 && fills_team &&
        launch(GroupNormBwdKernel<T, T_ACC, kVec, kTeam, kRMS, kAffine, kME, kHoist, true, 1, false>,
               GroupNormBwdKernel<T, T_ACC, kVec, kTeam, kRMS, kAffine, kME, kHoist, true, 1, false>, kFrameBytes)) {
      return;
    }
  }
  launch(GroupNormBwdKernel<T, T_ACC, kVec, kTeam, kRMS, kAffine, kME, kHoist, false, 1, false>,
         GroupNormBwdKernel<T, T_ACC, kVec, kTeam, kRMS, kAffine, kME, kHoist, false, 1, true>);
}

// Launches one single-team backward form: the hoisted body where HoistableForm allows it, from kLongRows rows or at
// 65536 channels; below that the unhoisted body, which then never stages and is always launched dependent.
template <typename T, bool kRMS, int kVec, TeamKind kTeam, bool kAffine, bool kME, typename Launch>
void LaunchBwdForm(int64_t rows, int64_t channels_per_group, int64_t num_elements, Launch&& launch) {
  using T_ACC = at::acc_type<T, true>;
  if constexpr (kAffine && kME && HoistableForm<T, kRMS, kTeam, kVec>()) {
    if (rows >= kLongRows || channels_per_group >= 65536) {
      LaunchBwdBody<T, kRMS, kVec, kTeam, kAffine, kME, true>(rows, channels_per_group, num_elements, launch);
    } else {
      launch(GroupNormBwdKernel<T, T_ACC, kVec, kTeam, kRMS, kAffine, kME, false, false, 1, true>,
             GroupNormBwdKernel<T, T_ACC, kVec, kTeam, kRMS, kAffine, kME, false, false, 1, true>);
    }
  } else {
    LaunchBwdBody<T, kRMS, kVec, kTeam, kAffine, kME, false>(rows, channels_per_group, num_elements, launch);
  }
}

// Folds the partial rows into dgamma (and dbeta for LayerNorm): `partial` holds num_phases rows of num_channels sums
// for dgamma, then as many for dbeta. The kernel is picked by the plane's size and by how many of its blocks cover the
// device. A fold is launched dependent on the backward kernel only where that measured faster: on short calls an early
// fold measured slower.
template <typename T, bool kRMS>
void FoldPartials(const float* partial, int64_t num_phases, int64_t num_channels, T* dgamma, T* dbeta, int num_sms,
                  cudaStream_t stream) {
  using T_ACC = float;
  const int64_t num_cells = num_phases * num_channels;
  // One plane: four channels a thread while 128-channel blocks cover the device, else 16 channels a block.
  auto fold_plane = [&](const T_ACC* plane, T* out) {
    if (utils::DivUp<int64_t>(num_channels, 4 * cuda_utils::kWarpSize) >= num_sms) {
      const dim3 grid(utils::DivUp<int64_t>(num_channels, 4 * cuda_utils::kWarpSize));
      cuda_utils::LaunchKernel(SumPartialsKernel<T, T_ACC>, grid, dim3(cuda_utils::kWarpSize, cuda_utils::kWarpSize),
                               0, stream, num_phases, num_channels, plane, out);
    } else {
      const std::uint32_t c16 = (std::uint32_t)utils::DivUp<int64_t>(num_channels, 16);
      cuda_utils::LaunchKernel(SumPartialsWarpKernel<T, T_ACC>, dim3(c16), dim3(512), 0, stream,
                               num_phases, num_channels, plane, out);
    }
  };
  // At most a warp's worth of phases above kFuseMaxChannels channels: the register trees.
  const bool tree = num_phases <= cuda_utils::kWarpSize && num_channels > kFuseMaxChannels;
  const bool tree_vec4 = tree && num_phases <= 16 && (num_channels & 3) == 0;
  const std::uint32_t vec4_blocks = (std::uint32_t)utils::DivUp<int64_t>(num_channels / 4, 256);
  if constexpr (kRMS) {
    if (tree_vec4) {
      LaunchKernelPdl(SumPartialsTreeVec4Kernel<T, T_ACC, 1>, dim3(vec4_blocks, 1u), dim3(256), 0, stream,
                      num_phases, num_channels, partial, dgamma, nullptr);
    } else {
      fold_plane(partial, dgamma);
    }
    return;
  }
  if (tree_vec4) {
    LaunchKernelPdl(SumPartialsTreeVec4Kernel<T, T_ACC, 2>, dim3(vec4_blocks, 2u), dim3(256), 0, stream, num_phases,
                    num_channels, partial, dgamma, dbeta);
  } else if (tree) {
    const std::uint32_t blocks = (std::uint32_t)utils::DivUp<int64_t>(num_channels, 256);
    cuda_utils::LaunchKernel(SumPartialsTreeKernel<T, T_ACC>, dim3(blocks, 2u), dim3(256), 0, stream, num_phases,
                             num_channels, partial, dgamma, dbeta);
  } else if (num_cells > kFuseMaxPartialCells || num_channels > kFuseMaxChannels) {
    fold_plane(partial, dgamma);
    fold_plane(partial + num_cells, dbeta);
  } else {
    // Both planes in one launch, in the block shape whose grid still covers the device.
    const dim3 grid32(utils::DivUp<int64_t>(num_channels, cuda_utils::kWarpSize));
    const dim3 block32(cuda_utils::kWarpSize, cuda_utils::kWarpSize);
    const std::uint32_t c16 = (std::uint32_t)utils::DivUp<int64_t>(num_channels, 16);
    const std::uint32_t c8 = (std::uint32_t)utils::DivUp<int64_t>(num_channels, 8);
    if (utils::DivUp<int64_t>(num_channels, cuda_utils::kWarpSize) >= num_sms) {
      if (utils::DivUp<int64_t>(num_channels, 16) >= 3 * num_sms) {
        LaunchKernelPdl(SumPartials2Dual16Kernel<T, T_ACC>, dim3(c16), dim3(512), 0, stream, num_phases, num_channels,
                        partial, dgamma, dbeta);
      } else {
        cuda_utils::LaunchKernel(SumPartials2DualKernel<T, T_ACC>, grid32, block32, 0, stream, num_phases,
                                 num_channels, partial, dgamma, dbeta);
      }
    } else if (utils::DivUp<int64_t>(num_channels, 16) >= num_sms && num_cells <= 524288) {
      cuda_utils::LaunchKernel(SumPartials2Dual16Kernel<T, T_ACC>, dim3(c16), dim3(512), 0, stream, num_phases,
                               num_channels, partial, dgamma, dbeta);
    } else if (utils::DivUp<int64_t>(num_channels, 8) >= 2 * num_sms) {
      LaunchKernelPdl(SumPartials2Dual8Kernel<T, T_ACC>, dim3(c8), dim3(512), 0, stream, num_phases, num_channels,
                      partial, dgamma, dbeta);
    } else {
      LaunchKernelPdl(SumPartials2Kernel<T, T_ACC>, dim3(grid32.x, 2), block32, 0, stream, num_phases, num_channels,
                      partial, dgamma, dbeta);
    }
  }
}

// gamma == nullptr means no affine parameters, and dgamma/dbeta are then not written. beta is only read for LayerNorm
// from y (memory_efficient); mean is unused for RMS norm.
template <typename T, bool kRMS>
void GroupNormBwd(int64_t outer_size, int64_t num_groups, int64_t channels_per_group, bool memory_efficient,
                  const T* dy, const T* x_or_y, const at::acc_type<T, true>* mean,
                  const at::acc_type<T, true>* rstd, const T* gamma, const T* beta,
                  T* dx, T* dgamma, T* dbeta) {
  using T_ACC = at::acc_type<T, true>;
  const int64_t num_segments = outer_size * num_groups;
  if (num_segments == 0) {
    return;
  }
  const int64_t num_channels = num_groups * channels_per_group;
  const bool affine = gamma != nullptr;
  // fp32 affine calls of at most 8192 channels a group take the 32-byte access, except where it measured slower: RMS
  // from y up to 128 channels, RMS from x from 256 channels, and the multi-team form below.
  const bool wide_vec = std::is_same_v<T, float> && affine && channels_per_group <= 8192 &&
                        channels_per_group % (kVecBytesWide / sizeof(T)) == 0 &&
                        !(kRMS && memory_efficient && channels_per_group <= 128) &&
                        !(kRMS && !memory_efficient && channels_per_group >= 256) &&
                        !(!kRMS && !memory_efficient && channels_per_group == kMultiTeamCpg);
  Layout layout = MakeLayout<T>(channels_per_group, {dy, x_or_y, gamma, beta, dx},
                                wide_vec ? kVecBytesWide : kVecBytes);
  // The multi-team block: fp32 affine LayerNorm from x at kMultiTeamCpg channels and 16-byte vectors, on a whole
  // number of kMultiTeams rows.
  if constexpr (std::is_same_v<T, float> && !kRMS) {
    if (affine && !memory_efficient && channels_per_group == kMultiTeamCpg && layout.vec == kMultiTeamVec &&
        outer_size % kMultiTeams == 0) {
      layout.block_size = kMultiTeams * kMultiTeamSize;
      layout.teams_per_block = kMultiTeams;
      layout.combine = kMultiTeams;
    }
  }
  const c10::DeviceIndex device = c10::cuda::current_device();
  const int num_sms = at::cuda::getDeviceProperties(device)->multiProcessorCount;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(device);
  BwdParams<T, T_ACC> params{num_segments, num_groups, channels_per_group, layout.team_size,
                             dy, x_or_y, mean, rstd, gamma, beta, dx, nullptr, nullptr};
  c10::DataPtr partial;
  int64_t num_phases = 0;

  // Launches `kernel`, or its twin `kernel_pdl` launched dependent below kPdlMaxRows rows, on as many units as are
  // resident, and allocates the partial rows of an affine call. A body with `smem` bytes of dynamic shared memory is
  // never launched dependent; it returns false, having launched nothing, when the device refuses the bytes or no
  // cluster holding them fits.
  auto launch = [&](auto kernel, auto kernel_pdl, int64_t smem = 0) -> bool {
    if (smem > 0 && !RaiseDynamicSmem(kernel, device, smem)) {
      return false;
    }
    const auto [fit, resident_units] = FitLayout(kernel, layout, device, num_sms, smem, kMaxBlockSize, smem > 0);
    if (resident_units == 0) {
      return false;
    }
    int64_t num_units = std::min(resident_units, utils::DivUp(num_segments, fit.teams_per_block));
    if (affine) {
      if (layout.combine > 1) {
        // A block's teams share a group: whole groups of blocks, one partial row a block of each group.
        num_units = std::max<int64_t>(utils::DivUp<int64_t>(num_units, num_groups), 1) * num_groups;
        num_phases = num_units / num_groups;
      } else {
        // A multiple of num_groups teams, so that a team only ever sees one group.
        const int64_t step = num_groups / std::gcd(num_groups, fit.teams_per_block);
        num_units = std::max<int64_t>(num_units / step, 1) * step;
        num_phases = num_units * fit.teams_per_block / num_groups;
      }
      // The kernel writes every partial row before the fold reads it.
      partial = at::cuda::getCUDADeviceAllocator()->allocate((kRMS ? 1 : 2) * num_phases * num_channels *
                                                             sizeof(T_ACC));
      params.dgamma_partial = static_cast<T_ACC*>(partial.get());
      params.dbeta_partial = kRMS ? nullptr : params.dgamma_partial + num_phases * num_channels;
    }
    if (smem == 0 && outer_size < kPdlMaxRows) {
      LaunchUnits(kernel_pdl, fit, num_units, stream, params, 0, true);
    } else {
      LaunchUnits(kernel, fit, num_units, stream, params, smem);
    }
    return true;
  };
  const int64_t num_elements = num_segments * channels_per_group;
  auto launch_form = [&](auto vec, auto team, auto affine_c) {
    constexpr int kVec = decltype(vec)::value;
    constexpr TeamKind kTeam = decltype(team)::value;
    constexpr bool kAffine = decltype(affine_c)::value;
    if (memory_efficient) {
      LaunchBwdForm<T, kRMS, kVec, kTeam, kAffine, true>(outer_size, channels_per_group, num_elements, launch);
    } else {
      LaunchBwdForm<T, kRMS, kVec, kTeam, kAffine, false>(outer_size, channels_per_group, num_elements, launch);
    }
  };
  auto launch_main = [&]() {
    if constexpr (std::is_same_v<T, float>) {
      if constexpr (!kRMS) {
        if (layout.combine > 1) {
          launch(GroupNormBwdKernel<T, T_ACC, kMultiTeamVec, TeamKind::kBlock, kRMS, true, false, false, false,
                                    kMultiTeams, false>,
                 GroupNormBwdKernel<T, T_ACC, kMultiTeamVec, TeamKind::kBlock, kRMS, true, false, false, false,
                                    kMultiTeams, true>);
          return;
        }
      }
      constexpr int kWideVec = static_cast<int>(kVecBytesWide / sizeof(T));
      if (layout.vec == kWideVec) {
        // The 32-byte access: affine and at most 8192 channels, so never a cluster team.
        if (layout.kind == TeamKind::kSubWarp) {
          launch_form(std::integral_constant<int, kWideVec>{}, std::integral_constant<TeamKind, TeamKind::kSubWarp>{},
                      std::true_type{});
        } else {
          launch_form(std::integral_constant<int, kWideVec>{}, std::integral_constant<TeamKind, TeamKind::kBlock>{},
                      std::true_type{});
        }
        return;
      }
    }
    DispatchLayout<T>(layout, [&](auto vec, auto team) {
      if (affine) {
        launch_form(vec, team, std::true_type{});
      } else {
        launch_form(vec, team, std::false_type{});
      }
    });
  };
  launch_main();
  if (affine) {
    FoldPartials<T, kRMS>(params.dgamma_partial, num_phases, num_channels, dgamma, dbeta, num_sms, stream);
  }
}

}  // namespace xllm::ops::blackwell::generic
