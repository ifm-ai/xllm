# Residual connections

`residual_func` selects how a sublayer's output `y` joins the residual stream.

| `residual_func` | state | update |
| --- | --- | --- |
| `base` / `add` | `x [B, L, D]` | `x' = x + y` |
| `ortho` | `x [B, L, D]` | `x' = x + rej(y, RMSNorm_g(x))`, `residual_heads` groups (arXiv:2505.11881) |
| `ihc` / `mhc` / `ohc` | `X [B, L, N, D]` | hyper-connections, below |

`base`, `add` and `ortho` live inside each sublayer module (`self.residual`). Hyper-connections
are owned by the block (`TransformerBlock`, `TransformerMoEBlock`, `TransformerMoVABlock`,
`GekkoBlock`): the sublayers then get `residual_func='none'` and add nothing.

## Hyper-connections (`hyper_connection.py`)

The residual state is `N = residual_streams` streams. Around every attention / FFN branch a
controller reads the streams and emits FP32 coefficients:

```
a, R, b = hc(X)                  # a, b [B, L, N]; R [B, L, N, N] (destination rows) or None
h  = hc_read(X, a)               # h  = sum_n a_n X_n          -> branch input [B, L, D]
y  = branch(h)                   # pre-norm + attention / FFN + dropout, no residual add
X' = hc_write(X, R, b, y)        # X' = R X + b y
```

Controller: `logits = RMSNorm(X.flatten(-2)) @ W`, `W [N*D, n_logits]`, `W ~ N(0, 1/(N*D))`, eps
`rmsnorm_eps`; read gates `a = sigmoid(s_a * l_a + bias_a)`, write gates `b = 2 sigmoid(s_b * l_b + bias_b)`.
Scales start at 0.01 and every bias at 0, so training starts close to a plain residual (`a ~ 0.5`,
`b ~ 1`, `R` near `I`), and weight decay, which the optimizer applies to every parameter, pulls the
biases back to that start rather than to another transport.

| variant | transport `R` | logits | reference |
| --- | --- | --- | --- |
| `ihc` | identity (`None`) | `2N` | oHC paper, sections 3 / 6.1 |
| `mhc` | `Sinkhorn(s_r * l_r + bias_r + 4 I)`, `residual_sinkhorn_iterations`: rows sum to one, columns approximately | `2N + N^2` | mHC, arXiv:2512.24880 |
| `ohc` | `SO(4)` from 8 quaternion logits, `R = L_q R_conj(r)`, `alpha = 0.01` constant | `2N + 8` | oHC, arXiv:2609.02672, section 5 |

The mHC `4 I` is a constant offset (diagonal ~0.95 at the start); at 20 iterations around it the column
sums stay within ~2e-2 of one for logit noise of std 0.5. `ohc` needs `residual_streams = 4`. With
`residual_groups = G > 1` the `D` channels split into `G` contiguous groups with their own rotation and
gate biases: the per-token logits are shared, group `g` turns at rate `residual_freq_base^(-g / (G - 1))`
(multi-frequency oHC). Coefficients then carry a group axis (`a, b [B, L, G, N]`, `R [B, L, G, N, N]`).
`alpha` and the group rates are constants of the config and are not checkpointed.

The model expands the streams from the embedding (every stream = `x0`) and sums them, with FP32
accumulation, before the final norm. Streams keep the model dtype; coefficient math and stream mixing
run in FP32. Under FSDP2 mixed precision `W`, the biases and the scales arrive in the compute dtype (BF16).

```json
{"residual_func": "ohc", "residual_streams": 4, "residual_groups": 1}
```

Config: `residual_streams` (4, from 2 to 4), `residual_groups` (1, `ohc` only), `residual_freq_base` (2.0, >= 1),
`residual_sinkhorn_iterations` (20, `mhc`). Parameter and FLOP counts (`num_parameters`,
`tflops_per_token`) include the controllers.

### Limits

The coefficients and the stream read / write are torch ops (`hyper_connection.py`) that keep FP32
copies of the `[B, L, N, D]` streams alive until backward.

- `fused_block` must be off (asserted in `ModelConf`); long sequences need `layerwise_ckpt`.
- No model parallelism, and so no MoE expert parallelism (it runs over the model-parallel group): the
  controller normalizes over all `N * D` channels of a token. `layerwise_ckpt` works; context parallelism
  should, as every hyper-connection op is per token, but has not been run.
- Memory: the streams cost `N` times the plain residual, and `layerwise_ckpt` keeps each layer's input
  streams, `N * D * 2` bytes per token per layer (k2-horizon-0.9B, bsz 32 x 8k: about 3.2 GB per layer,
  ~121-125 GB per GPU in total).
- oHC: the quaternion map is singular where a rotation bias reaches `(1 + z0, z1, z2, z3) = 0`
  (`z0 = -1`), far from its zero start.

### Tests

```bash
python -m pytest -q tests/residual/test_hyper_connection.py   # CPU: reference coefficients, read / write, gradients, config checks
python tests/residual/test_hyper_connection_model.py          # 1 GPU: Transformer (dense / MoE / MoVA) + Gekko smoke
```
