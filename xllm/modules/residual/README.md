# Residual connections

`residual_func` selects how a sublayer's output `y` joins the residual stream. Every variant is a module
built by `build_residual` and owned by the sublayer (`self.residual`); the sublayer reads its input from
the residual state and writes its output back:

```
residual = x
x, rc = self.residual.read(residual)          # base / ortho: (x, None); hyper-connections: (sum_n a_n X_n, (R, b))
...                                           # pre-norm, attention / FFN, dropout
out = self.residual(y, residual, state=rc)    # base: x + y; ortho: x + rej(y, .); hyper-connections: R X + b y
```

| `residual_func` | state | update |
| --- | --- | --- |
| `base` / `add` | `x [B, L, D]` | `x' = x + y` |
| `ortho` | `x [B, L, D]` | `x' = x + rej(y, RMSNorm_g(x))`, `residual_heads` groups (arXiv:2505.11881) |
| `ihc` / `mhc` / `ohc` | `X [B, L, 4, D]` | hyper-connections, below |

## Hyper-connections (`hyper_connection.py`)

The residual state is `N = 4` streams. The controller of a sublayer emits FP32 coefficients from the
streams in `read`, which returns the write's share `(R, b)` as the state `forward` takes back:

```
a, R, b = coefficients(X)        # a, b [B, L, N]; R [B, L, N, N] (destination rows) or None
h  = read(X)   = sum_n a_n X_n   # branch input [B, L, D]
X' = write(X)  = R X + b y
```

Controller: `logits = RMSNorm(X.flatten(-2)) @ W`, `W [N*D, n_logits]`, `W ~ N(0, 1/(N*D))`, eps
`rmsnorm_eps`; read gates `a = sigmoid(s_a * l_a + bias_a)`, write gates `b = 2 sigmoid(s_b * l_b + bias_b)`.
Scales start at 0.01 and every bias at 0, so training starts close to a plain residual (`a ~ 0.5`,
`b ~ 1`, `R` near `I`), and weight decay, which the optimizer applies to every parameter, pulls the
biases back to that start rather than to another transport.

| variant | transport `R` | logits | reference |
| --- | --- | --- | --- |
| `ihc` | identity (`None`) | `2N` | oHC paper, sections 3 / 6.1 |
| `mhc` | `Sinkhorn(s_r * l_r + bias_r + 4 I)`, 20 iterations: rows sum to one, columns approximately | `2N + N^2` | mHC, arXiv:2512.24880 |
| `ohc` | `SO(4)` from 8 quaternion logits, `R = L_q R_conj(r)`, `alpha = 0.01` constant | `2N + 8` | oHC, arXiv:2609.02672, section 5 |

The mHC `4 I` is a constant offset (diagonal ~0.95 at the start); at 20 iterations around it the column
sums stay within ~2e-2 of one for logit noise of std 0.5.

The model expands the streams from the embedding (every stream = `x0`, `expand_streams`) and sums them,
with FP32 accumulation, before the final norm (`sum_streams`); the health collector records that sum.
Streams keep the model dtype; coefficient math and stream mixing run in FP32. Under FSDP2 mixed precision
`W`, the biases and the scales arrive in the compute dtype (BF16). Parameters:
`layers.*.{attention,nffn,...}.residual.{weight,bias,scale}`. Under Muon the projection `weight` is a Muon matrix
(`register_muon(..., layout="i o")`), `bias` and `scale` stay on AdamW, the split DeepSeek-V4 uses for mHC.

```json
{"residual_func": "ohc"}
```

No other config: `N = 4`, the Sinkhorn iterations and `alpha` are constants of the classes. Parameter
and FLOP counts (`num_parameters`, `tflops_per_token`) include the controllers.

### Kernels and limits

On CUDA BF16 / FP16 streams every stream op is a fused Triton kernel
(`xllm/modules/fused_ops/hyper_connection.py`) that reads and writes the streams directly and keeps FP32
in registers: read, write (with or without a transport), the iHC / oHC gates and the SO(4) transport, and the mHC
Sinkhorn iterations; each with its backward. The controller projection `rstd * (X @ W)` is a Triton rstd
kernel and a cuBLAS BF16 matmul with FP32 accumulate and output; its backward is one Triton kernel. The
torch code in `hyper_connection.py` is the reference path (CPU, FP32 / FP64 models); it keeps FP32 copies
of the `[B, L, N, D]` streams alive until backward, so on GPU it needs several times the memory.

Per sublayer the streams are read three times (write, controller statistics, read) and written once.

- `fused_block` works for dense transformer layers with `ihc` / `ohc`
  (`xllm/models/fused_blocks/transformer/hyper_connection.py`: the base block with the residual adds replaced by
  the stream ops). It saves the two branch outputs per layer but not the streams: a write is invertible
  (`X = R^T (X' - b y)`), so the backward recovers each layer's streams from the layer above. Between layers the
  streams travel as hi + lo (two BF16 tensors: hi is what the branches and controllers read, hi + lo holds the
  FP32 write result to ~2^-17), so the inversion reproduces the forward's streams and the gradients match the
  stored-streams path at the BF16 noise floor; the lo parts pass through a `StreamSlot`. `mhc` (Sinkhorn
  transport, not invertible), MoE / MoVA and Gekko fused blocks are asserted off in `ModelConf`. k2-horizon-0.9B,
  8 x H200, bsz 32 x 8k: base fused block 116 GB 48.5k wps/GPU; `ohc` fused block 135.4 GB 40.1k (the stream
  path's working set in the backward, not saved streams, sets the peak); `ohc` with `layerwise_ckpt` 121-125 GB 37k.
- A variant's coefficient kernels live on its class (`coef_fwd` / `coef_bwd`, raw FP32 ops on `[T, n_logits]`
  logits); the module path on CUDA (`fused_coefficients`) and the fused block both call them, so adding a variant
  with a fused block means one class with a torch reference (`coefficients_from_logits`) and those two methods, and
  its name in `config.FUSED_BLOCK_RESIDUALS`. The stream kernels tile `fused.MAX_STREAMS = 4` streams per program
  (fewer are masked); their launchers assert contiguous `[T, N <= 4, D]` and `supported()` raises for more.
- No model parallelism, and so no MoE expert parallelism (it runs over the model-parallel group): the
  controller normalizes over all `N * D` channels of a token. Supporting it would touch `logits` (reduce the
  sum of squares for `rstd` across ranks, row-shard `W [N * D / tp, n_logits]` and all-reduce the logits),
  `register_muon(..., tp_dim=0)`, `num_params_in_hyper_connection`, and the fused block's `controller_fwd/bwd`;
  read / write / unwrite are per channel and shard along `D` as they are. `layerwise_ckpt` works; context
  parallelism should, as every hyper-connection op is per token, but has not been run.
- Memory: the streams cost `N` times the plain residual, and `layerwise_ckpt` keeps each layer's input
  streams, `N * D * 2` bytes per token per layer (k2-horizon-0.9B, bsz 32 x 8k: about 3.2 GB per layer,
  ~121-125 GB per GPU in total).
- oHC: the quaternion map is singular where a rotation bias reaches `(1 + z0, z1, z2, z3) = 0`
  (`z0 = -1`), far from its zero start.

### Tests

```bash
python -m pytest -q tests/residual/test_hyper_connection.py   # CPU: coefficients, read / write, the sublayer pairing, config checks
```
