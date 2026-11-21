# Group norm kernels for data-center Blackwell

This directory holds the CUDA kernels behind `GroupRMSNorm` and `GroupLayerNorm`
(`xllm_extension.ops.group_{rms,layer}_norm_{fwd,bwd}[_affine]`) on GPUs of compute capability 10.x: B200 and B300.
On B300, over the 427 shapes of the benchmark suite (the group norms traced from model training plus a grid of group
counts, widths, row counts and dtypes), they make the public calls 1.26x faster than the portable kernels in
`../group_norm_kernel.cuh` (1.32x on device time) and no shape more than 1.3% slower, and every call is bitwise
deterministic.

## When these kernels run

- **Build.** `setup.py` compiles this directory and defines `XLLM_BLACKWELL` when every entry of
  `TORCH_CUDA_ARCH_LIST` is 10.x, or, without that variable, when every visible GPU has compute capability 10.x. Any
  other build holds only the portable kernels and behaves as before.
- **Call.** `../group_rms_norm.cc` and `../group_layer_norm.cc` send a call here when `blackwell::Supported(x)`
  holds: `x` is a non-empty CUDA tensor of float16, bfloat16 or float32 on a device of compute capability 10.x.
  Float64 and every other device take the portable kernels, which this directory leaves unchanged.

## Contract

The ops keep the portable kernels' signatures, checks and results:

- `x` is `[B, S, C]` with any strides (it is made contiguous), split into `G` groups of `W = C / G` channels,
  `1 <= W <= 65536`. A *segment* is one (row, group) pair; a call has `B * S * G` of them.
- Forward returns `y` in the dtype of `x` and float32 statistics `rstd` (and `mean` for the layer norm), each
  `[B * S, G]`. Affine calls take `gamma` (and `beta` for the layer norm) of `C` elements.
- Backward takes the forward's statistics and either `x` or, with `memory_efficient`, the forward's output `y`, from
  which it recovers the normalized input as `(y - beta) / gamma`. It returns `dx` and, for affine calls, `dgamma`
  (and `dbeta`).

## Determinism and numerics

- No kernel accumulates with atomics. Every sum (over a segment, and over the rows behind `dgamma` and `dbeta`) runs
  in a fixed order, so a call repeated on the same tensors returns the same bits, also under
  `torch.use_deterministic_algorithms(True)`.
- Which kernel and layout serve a call depends on its shape, so a row can round differently in calls of different
  batch sizes, and results differ from the portable kernels' in the last bits.
- Statistics accumulate in float32. The variance comes from centred values, or from raw moments behind a guard that
  recomputes it from centred values when cancellation would cost accuracy or the sum of squares overflows. Outputs
  are computed in float32 and rounded once to their dtype.
- Against a float64 reference these kernels are as accurate as the portable ones: `test_group_norm.py` holds both to
  the same tolerances, on random inputs and on inputs that are offset, nearly constant or overflow-prone in float32.

## Routing

`dispatch.cu` sends each call to one kernel through a fixed chain of rules. A rule names a class of calls and the
kernel measured fastest on it on B300 among those at most 5% slower than the portable kernels both on device time and
as a public call. A kernel can lead on one and trail on the other: device time repeats a call on the same tensors in a
CUDA graph, where they stay in L2 if they fit, while the public call cycles through four copies of them and includes
the host's dispatch and allocation. A specialized kernel may still decline a call it cannot take (a launch limit, say);
the call then goes on down the chain, which ends at the generic kernels, which take every call.

First, `portable()` names the classes where upstream's portable kernels are the fastest, and `native.cpp` sends them
to upstream's own entry points before it allocates anything, so that they cost what they cost before:

- Forward: the widest groups with too few segments to fill the GPU (more than 16384 channels in at most 16 segments;
  more than 8192 channels that are not a whole number of 16-byte vectors in fewer than 4096 segments).
- Backward: bf16 affine widths that are not a multiple of 8 on long inputs, fp32 affine layer norm of 385 to 768
  channels on many segments, and fp32 affine RMS from y above 128 channels on very many segments.

Every other forward call:

1. Inputs that are not 16-byte aligned, or rows of more than INT_MAX channels: the generic kernels.
2. Classes where the generic kernels are fastest (`prefer_generic_forward`): widths that are not a whole number of
   16-byte vectors on short inputs, fp32 groups of more than 1024 channels in an odd number of vectors on few
   segments, narrow groups without affine on short inputs, and a few wider classes.
3. Dense lanes for 2-byte affine groups of 40 channels, for groups without affine whose width is not a power of two,
   and for fp32 groups without affine whose grid fills the GPU.
4. The flat kernel (groups of up to 1024 vectors of 16 bytes, and every width that is not a whole number of vectors),
   then the wide kernels (fp32 above 4096 channels, 2-byte above 8192), dense lanes, and the generic kernels.

Every other backward call:

1. Inputs that are not 16-byte aligned, or rows of more than INT_MAX channels: the generic kernels.
2. Kernels for exact classes: wide rows (bf16 without affine, 65536-channel groups), cluster narrow (affine, at most
   256 rows; in bf16, groups whose width is a power of two) and the two one-cluster kernels (bf16 affine at 128 rows:
   RMS with 2048-channel groups, and eight groups of 512 channels).
3. Classes where the generic kernels are fastest (`prefer_generic_backward`): fp16 groups of up to 384 channels, fp32
   groups of up to 128 without affine, 2-byte groups without affine wider than 256 channels, bf16 groups without
   affine on short inputs, the bf16 layer norm without affine below 131072 segments, bf16 affine calls of at most 256
   rows whose group width is not a power of two, the bf16 affine layer norm from x on 257 to 2048 rows of at least 4M
   elements, and a few more bf16 classes.
4. The TMA kernel (bf16 affine, at most 8192 channels, groups of whole 16-byte vectors) when its plan keeps at least
   two thirds of its consumer warps busy (a short last wave of tiles, or warps without columns, leave the rest idle;
   the generic kernels take the call otherwise), then dense lanes (without affine), the by-width kernel, the one-pass
   kernel, dense lanes (affine), and the generic kernels.

## Kernels

Every kernel accumulates in float32 and sums in a fixed order.

| File | Calls it serves | Mapping | Parameter gradients |
|---|---|---|---|
| `fwd_flat.cu` | Forward: groups of up to 1024 vectors of 16 bytes (8192 channels in 16 bits, 4096 in fp32), and every width that is not a whole number of vectors | The call's segments form one flat list; a segment takes a few lanes, a warp or several warps, and stays in registers from its statistics to its output | |
| `fwd_wide.cu` | fp32 forward: groups of 4097 to 8192 channels, and of up to 65536 when the call has at most one segment an SM | CTAs own a slice of the rows of one group, or one CTA streams one segment | |
| `fwd_wide_resident.cu` | 16-bit forward: groups of 8193 to 65536 channels | One CTA holds a whole segment, in registers or (bf16, the widest groups) in shared memory | |
| `dense_lanes.cu` | Forward and backward: groups of 1 to 32 vectors of 16 bytes | A group takes as many lanes of a warp as it has vectors, so several groups share a warp and its statistics never leave it | A fold kernel sums each warp's partials in order |
| `bwd_tma.cu`, `tma.cuh` | bf16 affine backward of up to 8192 channels | A persistent, warp-specialized pipeline: one producer warp streams tiles of rows into shared memory with bulk copies, consumer warps own fixed columns | Kept in registers per consumer warp, then summed across CTAs in a fixed order |
| `bwd_one_pass.cu` | bf16 affine backward, groups of 8 to 256 channels | Column-stationary warps, one segmented shuffle a group | Per row chunk in a workspace, summed by a second kernel in a fixed order |
| `bwd_cluster_narrow.cu` | Affine backward of up to 256 rows (bf16: power-of-two group widths) | Clusters of 8 CTAs own slices of channels; each CTA owns a slice of rows | Folded across the cluster through distributed shared memory, in rank order |
| `bwd_one_cluster_grad.cu`, `bwd_one_cluster_grad512.cu` | bf16 affine backward at 128 rows: RMS with 2048-channel groups; eight groups of 512 channels | One 16-CTA cluster a group, one launch | Slabs in distributed shared memory with one writer each, summed in rank order |
| `bwd_wide_rows.cu` | bf16 backward without affine, 65536-channel groups | One 4-CTA cluster a segment | |
| `bwd_by_width.cu` | bf16 backward without affine, groups of 2049 to 8192 channels | One CTA keeps a group and walks its rows | |
| `generic.cuh`, `fallback.cu` | Every call the others do not take | A team of threads a segment: part of a warp, a block, or a cluster for the widest groups | Per-team partial planes, summed by a second kernel in a fixed order |

The other files hold the plumbing: `native.cpp` (the entry points the `.cc` files call: operand checks, the portable
classes and output allocation), `dispatch.cu` and `dispatch.h` (the routing, and the `Shape` and `Call` every launcher
takes), `pdl.h` (programmatic dependent launch for the forward kernels) and `width_common.cuh` (16-bit vector helpers).

## Performance

B300 SXM6 (sm_103a, 2032 MHz), CUDA 13.4, PyTorch 2.16. Speedup is the portable kernels' time over ours, as a
geometric mean: device time is a CUDA graph replay of the call; the public call is the op as Python calls it,
allocation included.

The benchmark suite is the set the kernels were tuned on: the 27 group norms traced from model training, a grid of
(groups, width) pairs over row counts and dtypes, and the narrow widths where earlier versions regressed. The official
comparison set adds more widths and row counts.

| Shapes | Cases | Device | Public call | Slowest device | Slowest public call |
|---|---|---|---|---|---|
| Traced model calls | 27 | 1.34x | 1.36x | 1.05x | 1.06x |
| Grid of (groups, width) pairs x rows x dtypes | 384 | 1.32x | 1.25x | 0.99x | 0.99x |
| Narrow-width witnesses | 16 | 1.44x | 1.34x | 1.16x | 1.01x |
| Forward | 157 | 1.38x | 1.22x | 1.03x | 0.99x |
| Backward | 270 | 1.29x | 1.29x | 0.99x | 1.00x |
| **Benchmark suite** | 427 | 1.32x | 1.26x | 0.99x | 0.99x |
| Official comparison set | 692 | 1.27x | 1.22x | 0.99x | 0.97x |

The grids are further shapes, measured on the final tree at five rounds a cell, and again at thirty rounds for the 20
cells that came out more than 5% slower than the portable kernels on either measure. Of their 17,016 cells, none is more
than 5% slower on either measure: the slowest is 4.9% slower on device and 5.0% as a public call.

| Grid | Cells | Device | Public call | Slowest device | Slowest public call | >5% slower |
|---|---|---|---|---|---|---|
| odd (odd widths 9-4097, 128-32768 rows) | 2544 | 1.21x | 1.15x | 0.96x | 0.96x | 0 |
| wide (widths 1024-8192, 128-32768 rows) | 2952 | 1.21x | 1.15x | 0.96x | 0.95x | 0 |
| narrow (widths 24-256, 8-128 groups, 128-65536 rows) | 4320 | 1.28x | 1.18x | 0.95x | 0.96x | 0 |
| holdout (widths 32-3072, 256-16384 rows) | 3780 | 1.28x | 1.18x | 0.96x | 0.96x | 0 |
| mid-groups (12-24 groups of 64-256, 128-4096 rows) | 720 | 1.38x | 1.29x | 0.97x | 0.97x | 0 |
| odd-wide (odd widths 2049-65535, 1-8192 rows) | 1224 | 1.17x | 1.17x | 0.97x | 0.96x | 0 |
| aligned-wide (widths 12288-49152, 1-8192 rows) | 1476 | 1.32x | 1.26x | 0.97x | 0.95x | 0 |
| **All grid cells** | 17016 | 1.26x | 1.18x | 0.95x | 0.95x | 0 |

The traced model calls, one by one:

| Traced model call | Upstream (us) | Ours (us) | Device | Public call |
|---|---|---|---|---|
| layer 8x8192, G=64, W=128, affine, backward | 786.5 | 496.7 | 1.58x | 1.60x |
| layer 8x8192, G=64, W=128, affine, forward | 618.3 | 340.2 | 1.82x | 1.86x |
| rms 32x8192, G=1, W=1536, affine, backward | 583.1 | 369.5 | 1.58x | 1.58x |
| rms 32x8192, G=1, W=1536, affine, forward | 385.0 | 239.6 | 1.61x | 1.63x |
| rms 32x8192, G=64, W=24, forward | 302.8 | 258.3 | 1.17x | 1.18x |
| rms 32x8192, G=64, W=24, backward from y | 466.4 | 372.7 | 1.25x | 1.26x |
| rms 32x8192, G=64, W=64, affine, backward | 1405.6 | 992.3 | 1.42x | 1.43x |
| rms 32x8192, G=64, W=64, affine, forward | 808.5 | 655.5 | 1.23x | 1.25x |
| rms 8x8192, G=1, W=4096, affine, backward | 370.8 | 257.1 | 1.44x | 1.44x |
| rms 8x8192, G=1, W=4096, affine, forward | 225.4 | 165.1 | 1.37x | 1.38x |
| rms 8x8192, G=2, W=1280, affine, backward | 280.5 | 163.1 | 1.72x | 1.71x |
| rms 8x8192, G=2, W=1280, affine, forward | 186.5 | 106.0 | 1.76x | 1.80x |
| rms 8x8192, G=32, W=128, affine, backward | 361.0 | 260.7 | 1.38x | 1.39x |
| rms 8x8192, G=32, W=128, affine, forward | 209.5 | 163.7 | 1.28x | 1.29x |
| rms 8x8192, G=32, W=80, forward | 146.4 | 107.0 | 1.37x | 1.40x |
| rms 8x8192, G=32, W=80, backward from y | 218.5 | 153.6 | 1.42x | 1.44x |
| rms 8x8192, G=4, W=1024, affine, backward | 313.1 | 252.1 | 1.24x | 1.24x |
| rms 8x8192, G=4, W=1024, affine, forward | 208.2 | 164.3 | 1.27x | 1.28x |
| rms 8x8192, G=4, W=1024, affine, backward from y | 345.8 | 310.0 | 1.12x | 1.12x |
| rms 8x8192, G=64, W=128, affine, backward | 698.5 | 502.1 | 1.39x | 1.40x |
| rms 8x8192, G=64, W=128, affine, forward | 412.9 | 323.2 | 1.28x | 1.29x |
| rms 8x8192, G=64, W=40, forward | 144.7 | 114.5 | 1.26x | 1.29x |
| rms 8x8192, G=64, W=40, backward from y | 219.0 | 161.8 | 1.35x | 1.37x |
| rms 8x8192, G=64, W=64, forward | 175.0 | 166.0 | 1.05x | 1.06x |
| rms 8x8192, G=64, W=64, backward from y | 270.7 | 237.2 | 1.14x | 1.15x |
| rms 8x8192, G=8, W=128, forward | 49.7 | 45.4 | 1.09x | 1.11x |
| rms 8x8192, G=8, W=128, backward from y | 72.3 | 65.6 | 1.10x | 1.12x |

## Tests and benchmarks

```bash
python -m pytest tests/normalization/test_group_norm.py -q            # accuracy, layouts, robustness, determinism
python -m tests.normalization.test_group_norm_small_groups --reference-device cuda
python -m tests.normalization.benchmark_group_norm_regression --baseline <old xllm_extension.so> \
    --output /tmp/group_norm.json --match close                       # device time against an earlier build
```

`test_group_norm.py` checks both norms, every dtype and both backward modes against float64 on shapes that reach every
kernel, on the shapes traced from model training, on strided and misaligned inputs, on inputs whose variance cancels
and float32 inputs whose sum of squares overflows, and checks that repeated calls return the same bits.

A change of a rule in `dispatch.cu` or of a kernel should keep `test_group_norm.py` passing and be measured with
`benchmark_group_norm_regression.py` against the build before it, on the shapes of the class it touches.
