"""Hyper-connections: N = 4 residual streams around every attention / FFN branch, as a residual module.

The residual state is X [B, L, N, D] instead of x [B, L, D]. The sublayer calls

    h, rc = self.residual.read(X)       # h  = sum_n a_n X_n       (branch input [B, L, D])
    y  = branch(norm(h))                # pre-norm, attention / FFN, dropout: unchanged
    X' = self.residual(y, X, state=rc)  # X' = R X + b y

where the controller emits FP32 coefficients from X in `read`, which hands the write's share (R, b)
back as `state` for `forward`:

    a [B, L, N]       read gates
    R [B, L, N, N]    transport (destination rows); None = identity
    b [B, L, N]       write gates

Variants (``residual_func``):

    ihc   identity transport, R = I                          (oHC paper, sections 3 / 6.1)
    mhc   Sinkhorn-Knopp R: rows sum to one, columns approximately (mHC, arXiv:2512.24880, eq. 7-9)
    ohc   R in SO(4) from eight quaternion logits            (oHC, arXiv:2609.02672, section 5)

Controller: logits = RMSNorm(X.flatten(-2)) @ W with W [N * D, n_logits];
a = sigmoid(s_a * l_a + bias_a), b = 2 sigmoid(s_b * l_b + bias_b). W ~ N(0, 1 / (N * D)),
scales start at 0.01 and every bias at 0 (mHC adds a constant 4 I to its transport logits),
so the model starts close to a plain residual (a ~ 0.5, b ~ 1, R near I) and weight decay
pulls the biases back to that start. Coefficient math and stream mixing run in FP32; the
branch input and the streams keep the model dtype. On CUDA BF16 / FP16 streams every stream op
runs as a fused Triton kernel (xllm/modules/fused_ops/hyper_connection.py); the torch code here is
the reference path.

The model expands the streams from the embedding (every stream = x0, `expand_streams`) and
sums them before the final norm (`sum_streams`).
"""

import math
from typing import NamedTuple, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor, nn

from xllm.distributed import get_model_parallel_group
from xllm.modules.fused_ops import hyper_connection as fused
from xllm.optim.muon_spec import register_muon
from .base import BaseResidual


NUM_STREAMS = 4
OHC_EPS = 1e-12  # quaternion normalization of the oHC transport


class WriteState(NamedTuple):
    """What `HyperConnection.read` hands to `forward`: the transport (None = identity) and the write gates, FP32."""
    R: Optional[Tensor]
    b: Tensor


class _FusedCoefficients(torch.autograd.Function):
    """Autograd over a variant's raw coefficient kernels (`coef_fwd` / `coef_bwd`): the module path's coefficients on
    CUDA, the same kernels the fused block launches."""

    @staticmethod
    def forward(ctx, hc_cls, logits, scale, bias):
        ctx.hc_cls = hc_cls
        ctx.save_for_backward(logits, scale, bias)
        a, R, b = hc_cls.coef_fwd(logits, scale, bias)
        ctx.has_r = R is not None
        return (a, R, b) if ctx.has_r else (a, b)

    @staticmethod
    def backward(ctx, *grads):
        logits, scale, bias = ctx.saved_tensors
        da, dR, db = grads if ctx.has_r else (grads[0], None, grads[1])
        dlogits, dscale, dbias = ctx.hc_cls.coef_bwd(logits, scale, bias, da, dR, db)
        return None, dlogits, dscale, dbias


def fused_coefficients(hc_cls, logits: Tensor, scale: Tensor, bias: Tensor) -> Tuple[Tensor, Optional[Tensor], Tensor]:
    """(a, R, b) of `hc_cls` from logits [B, L, n_logits] (FP32), through its kernels; R None for the identity."""
    B, L, NL = logits.shape
    out = _FusedCoefficients.apply(hc_cls, logits.reshape(B * L, NL).contiguous(), scale.contiguous(), bias.contiguous())
    a, R, b = out if len(out) == 3 else (out[0], None, out[1])
    return a.view(B, L, -1), None if R is None else R.view(B, L, R.shape[-2], R.shape[-1]), b.view(B, L, -1)


def rms_normalize(x: Tensor, eps: float) -> Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


def sinkhorn_knopp(logits: Tensor, iterations: int = 20) -> Tensor:
    """Approximately doubly stochastic matrices from square logits [..., N, N].

    Column then row normalization in log space; rows sum to one up to roundoff,
    columns converge with the iterations.
    """
    log_m = logits
    for _ in range(iterations):
        log_m = log_m - log_m.logsumexp(dim=-2, keepdim=True)
        log_m = log_m - log_m.logsumexp(dim=-1, keepdim=True)
    return log_m.exp()


def _left(q: Tensor) -> Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([w, -x, -y, -z], -1),
        torch.stack([x, w, -z, y], -1),
        torch.stack([y, z, w, -x], -1),
        torch.stack([z, -y, x, w], -1),
    ], -2)


def _right_conj(r: Tensor) -> Tensor:
    a, b, c, d = r.unbind(-1)
    return torch.stack([
        torch.stack([a, b, c, d], -1),
        torch.stack([-b, a, -d, c], -1),
        torch.stack([-c, d, a, -b], -1),
        torch.stack([-d, -c, b, a], -1),
    ], -2)


def quaternion_transport(logits: Tensor, eps: float = OHC_EPS) -> Tensor:
    """logits [..., 8] -> H [..., 4, 4] in SO(4): x -> q x conj(r) for unit quaternions
    q = (1 + l0, l1, l2, l3) / |.| and r = (1 + l4, l5, l6, l7) / |.|. Zero logits give H = I."""
    one = torch.zeros_like(logits[..., :4])
    one[..., 0] = 1
    vq, vr = logits[..., :4] + one, logits[..., 4:] + one
    q = vq * torch.rsqrt(vq.square().sum(-1, keepdim=True) + eps)
    r = vr * torch.rsqrt(vr.square().sum(-1, keepdim=True) + eps)
    return _left(q) @ _right_conj(r)


def expand_streams(x: Tensor, num_streams: int) -> Tensor:
    """x [B, L, D] -> X [B, L, N, D], every stream a copy of x. N = 1 (no hyper-connections): x itself."""
    return x if num_streams == 1 else x.unsqueeze(-2).expand(-1, -1, num_streams, -1).contiguous()


def sum_streams(x: Tensor, num_streams: int) -> Tensor:
    """X [B, L, N, D] -> sum of the streams [B, L, D] in the streams' dtype, accumulated in FP32 without an
    FP32 copy. N = 1 (no hyper-connections): x itself."""
    return x if num_streams == 1 else x.sum(dim=-2, dtype=torch.float32).to(x.dtype)


def hc_read(streams: Tensor, a: Tensor) -> Tensor:
    """h [B, L, D] = sum_n a_n X_n in FP32, returned in the streams' dtype. a [B, L, N]."""
    if fused.supported(streams):
        return fused.hc_read(streams, a)
    h = (a.unsqueeze(-1) * streams.to(a.dtype)).sum(dim=-2)
    return h.to(streams.dtype)


def hc_write(streams: Tensor, R: Optional[Tensor], b: Tensor, y: Tensor) -> Tensor:
    """X' = R X + b y in FP32, returned in the streams' dtype. R None is the identity transport."""
    if fused.supported(streams):
        return fused.hc_write(streams, R, b, y)
    x = streams.to(b.dtype)
    res = x if R is None else R @ x
    out = res + b.unsqueeze(-1) * y.to(b.dtype).unsqueeze(-2)
    return out.to(streams.dtype)


class HyperConnection(BaseResidual):
    """A sublayer's hyper-connection: `read` forms the branch input from the streams and returns the
    transport and write gates as the state `forward` writes the branch output back with. Subclasses set
    the logit and scale counts, form the coefficients from the logits in torch (`coefficients_from_logits`, the
    reference path) and, where kernels exist, through `coef_fwd` / `coef_bwd`, the one place the fused path
    (module path on CUDA and the fused block alike) gets a variant's coefficient kernels from."""

    NAME: str  # the residual_func value (xllm.config.HYPER_CONNECTION_FUNCS lists them for the config check)
    NUM_SCALES = 2  # read (a), write (b)

    @staticmethod
    def num_logits_for(num_streams: int) -> int:
        raise NotImplementedError

    def __init__(self, rc_id: int, model_dim: int, eps: float = 1e-6, num_streams: int = NUM_STREAMS):
        super().__init__(rc_id, model_dim)
        # the controller normalizes over all N * D channels of a token, which a model-parallel shard does
        # not hold. The group, not the cached world size: a model built before initialize_model_parallel
        # would pin that cache to 1.
        group = get_model_parallel_group()
        assert group is None or dist.get_world_size(group) == 1, \
            "hyper-connections do not support model / expert parallelism yet"
        assert 1 < num_streams <= fused.MAX_STREAMS, f"the stream kernels support 2..{fused.MAX_STREAMS} streams"
        self.num_streams = num_streams
        self.num_logits = self.num_logits_for(num_streams)
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(num_streams * model_dim, self.num_logits))
        self.bias = nn.Parameter(torch.zeros(self.num_logits))
        self.scale = nn.Parameter(torch.full((self.NUM_SCALES,), 0.01))
        with torch.no_grad():
            nn.init.normal_(self.weight, std=1.0 / math.sqrt(num_streams * model_dim))
        # Under Muon the projection is a Muon matrix (an [in, out] weight: the logits are x @ W), the static bias and
        # the gating scales stay on AdamW, as DeepSeek-V4 trains its mHC modules (arXiv 2606.19348, section 2.4).
        register_muon(self, "weight", layout="i o")

    def logits(self, streams: Tensor) -> Tensor:
        """[B, L, n_logits] in FP32 from the RMS-normalized, flattened streams."""
        if fused.supported(streams):
            return fused.hc_logits(streams.flatten(-2), self.weight, self.eps)
        flat = rms_normalize(streams.float().flatten(-2), self.eps)
        return flat @ self.weight.float()

    def gates(self, l_a: Tensor, l_b: Tensor, bias_a: Tensor, bias_b: Tensor) -> Tuple[Tensor, Tensor]:
        scale = self.scale.float()
        a = torch.sigmoid(scale[0] * l_a + bias_a)
        b = 2.0 * torch.sigmoid(scale[1] * l_b + bias_b)
        return a, b

    def coefficients(self, streams: Tensor) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        """(a, R, b) in FP32; R is None for the identity transport."""
        return self.coefficients_from_logits(self.logits(streams))

    def coefficients_from_logits(self, logits: Tensor) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        """(a, R, b) from the FP32 logits [B, L, n_logits]: the torch reference, or the variant's kernels on CUDA."""
        raise NotImplementedError

    @classmethod
    def coef_fwd(cls, logits: Tensor, scale: Tensor, bias: Tensor) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        """Raw kernels: FP32 contiguous logits [T, n_logits], scale, bias -> (a [T, N], R [T, N, N] or None, b [T, N])."""
        raise NotImplementedError(f"{cls.NAME} has no coefficient kernels (and so no fused block)")

    @classmethod
    def coef_bwd(cls, logits: Tensor, scale: Tensor, bias: Tensor, da: Tensor, dR: Optional[Tensor], db: Tensor
                 ) -> Tuple[Tensor, Tensor, Tensor]:
        """-> (dlogits, dscale, dbias); a None gradient counts as zero."""
        raise NotImplementedError(f"{cls.NAME} has no coefficient kernels (and so no fused block)")

    def read(self, streams: Tensor) -> Tuple[Tensor, WriteState]:
        """-> (branch input h [B, L, D], the write's (R, b) for `forward`)."""
        assert streams.shape[-2:] == (self.num_streams, self.num_features), \
            f"expected [B, L, {self.num_streams}, {self.num_features}] streams, got {tuple(streams.shape)}"
        a, R, b = self.coefficients(streams)
        return hc_read(streams, a), WriteState(R, b)

    def forward(self, h: Tensor, residual: Tensor, *args, state: Optional[WriteState] = None, **kwargs) -> Tensor:
        assert state is not None, "HyperConnection.forward needs the (R, b) state its read returned"
        return hc_write(residual, state.R, state.b, h)

    def extra_repr(self) -> str:
        return 'dim={}, streams={}, logits={}'.format(self.num_features, self.num_streams, self.num_logits)


class IdentityHyperConnection(HyperConnection):
    """Sigmoid read gates, 2 sigmoid write gates, R = I (returned as None)."""

    NAME = 'ihc'

    @staticmethod
    def num_logits_for(num_streams: int) -> int:
        return 2 * num_streams

    @classmethod
    def coef_fwd(cls, logits, scale, bias):
        a, b = fused.ihc_coef_fwd(logits, scale, bias)
        return a, None, b

    @classmethod
    def coef_bwd(cls, logits, scale, bias, da, dR, db):
        return fused.ihc_coef_bwd(logits, scale, bias, da, db)

    def coefficients_from_logits(self, logits: Tensor) -> Tuple[Tensor, None, Tensor]:
        n = self.num_streams
        bias = self.bias.float()
        if logits.is_cuda and n == 4:
            return fused_coefficients(type(self), logits, self.scale.float(), bias)
        a, b = self.gates(logits[..., :n], logits[..., n:], bias[:n], bias[n:])
        return a, None, b


class ManifoldHyperConnection(HyperConnection):
    """mHC: gates as IHC, transport = Sinkhorn(s_r * l_r + bias_r + 4 I). The constant 4 I starts R near I
    (diagonal ~0.95 at N = 4) with a zero bias, so weight decay keeps the transport near identity rather than
    pulling it to uniform mixing. Rows of R sum to one, columns up to the Sinkhorn convergence."""

    NAME = 'mhc'
    NUM_SCALES = 3  # read, write, transport
    IDENTITY_LOGIT = 4.0
    SINKHORN_ITERATIONS = 20

    @staticmethod
    def num_logits_for(num_streams: int) -> int:
        return 2 * num_streams + num_streams * num_streams

    def __init__(self, rc_id: int, model_dim: int, eps: float = 1e-6, num_streams: int = NUM_STREAMS):
        super().__init__(rc_id, model_dim, eps, num_streams)
        self.register_buffer('identity_logits', self.IDENTITY_LOGIT * torch.eye(num_streams), persistent=False)

    def coefficients_from_logits(self, logits: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        n = self.num_streams
        bias = self.bias.float()
        a, b = self.gates(logits[..., :n], logits[..., n:2 * n], bias[:n], bias[n:2 * n])
        r_logits = self.scale.float()[2] * logits[..., 2 * n:] + bias[2 * n:]
        r_logits = r_logits.view(*r_logits.shape[:-1], n, n) + self.identity_logits.float()
        if logits.is_cuda and n == 4:  # the Sinkhorn kernel; the gates stay torch (no fused block for mhc)
            return a, fused.sinkhorn(r_logits, self.SINKHORN_ITERATIONS), b
        return a, sinkhorn_knopp(r_logits, self.SINKHORN_ITERATIONS), b


class OrthogonalHyperConnection(HyperConnection):
    """oHC: gates as IHC, transport R = SO(4)(alpha * l_r + bias_r) from eight logits; alpha is the constant
    ROTATION_SCALE, as the paper freezes it for orthogonal variants. Four streams only."""

    NAME = 'ohc'
    ROTATION_LOGITS = 8
    ROTATION_SCALE = 0.01
    ROTATION_EPS = OHC_EPS

    @staticmethod
    def num_logits_for(num_streams: int) -> int:
        return 2 * num_streams + OrthogonalHyperConnection.ROTATION_LOGITS

    def __init__(self, rc_id: int, model_dim: int, eps: float = 1e-6, num_streams: int = NUM_STREAMS):
        assert num_streams == 4, "ohc parameterizes SO(4): four streams"
        super().__init__(rc_id, model_dim, eps, num_streams)

    @classmethod
    def coef_fwd(cls, logits, scale, bias):
        return fused.ohc_coef_fwd(logits, scale, bias, cls.ROTATION_SCALE, cls.ROTATION_EPS)

    @classmethod
    def coef_bwd(cls, logits, scale, bias, da, dR, db):
        return fused.ohc_coef_bwd(logits, scale, bias, da, dR, db, cls.ROTATION_SCALE, cls.ROTATION_EPS)

    def coefficients_from_logits(self, logits: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        n = self.num_streams
        bias = self.bias.float()
        if logits.is_cuda:
            return fused_coefficients(type(self), logits, self.scale.float(), bias)
        a, b = self.gates(logits[..., :n], logits[..., n:2 * n], bias[:n], bias[n:2 * n])
        R = quaternion_transport(self.ROTATION_SCALE * logits[..., 2 * n:] + bias[2 * n:], self.ROTATION_EPS)
        return a, R, b


HYPER_CONNECTIONS = {hc.NAME: hc for hc in (IdentityHyperConnection, ManifoldHyperConnection, OrthogonalHyperConnection)}


def num_params_in_hyper_connection(residual_func: str, model_dim: int) -> int:
    """Parameters of one sublayer's controller (projection + biases + scales)."""
    hc = HYPER_CONNECTIONS[residual_func]
    num_logits = hc.num_logits_for(NUM_STREAMS)
    return NUM_STREAMS * model_dim * num_logits + num_logits + hc.NUM_SCALES


def flops_in_hyper_connection(residual_func: str, model_dim: int) -> int:
    """Multiply-adds of one sublayer per token: controller projection, read and write."""
    projection = NUM_STREAMS * model_dim * HYPER_CONNECTIONS[residual_func].num_logits_for(NUM_STREAMS)
    read = NUM_STREAMS * model_dim
    transport = 0 if residual_func == 'ihc' else NUM_STREAMS * NUM_STREAMS * model_dim
    write = transport + NUM_STREAMS * model_dim
    return projection + read + write
