"""Hyper-connections: N residual streams around every attention / FFN branch.

The residual state is X [B, L, N, D] instead of x [B, L, D]. Around each branch a
controller reads the streams and emits FP32 coefficients

    a [B, L, N]       read gates:  h  = sum_n a_n X_n          (branch input)
    R [B, L, N, N]    transport:   X' = R X + b y              (destination rows, source columns)
    b [B, L, N]       write gates

The branch keeps its own pre-norm and dropout and adds no residual. Variants
(``residual_func``):

    ihc   identity transport, R = I                          (oHC paper, sections 3 / 6.1)
    mhc   Sinkhorn-Knopp R: rows sum to one, columns approximately (mHC, arXiv:2512.24880, eq. 7-9)
    ohc   R in SO(4) from eight quaternion logits, N = 4    (oHC, arXiv:2609.02672, section 5)

oHC can split the D channels into G contiguous groups (``residual_groups``) with
their own rotation and gate biases: one set of per-token logits is shared, group g
turns at rate freq_base^(-g / (G - 1)) (multi-frequency oHC). Coefficients then
carry a group axis, a / b [B, L, G, N] and R [B, L, G, N, N].

Controller: logits = RMSNorm(X.flatten(-2)) @ W with W [N * D, n_logits];
a = sigmoid(s_a * l_a + bias_a), b = 2 sigmoid(s_b * l_b + bias_b). W ~ N(0, 1 / (N * D)),
scales start at 0.01 and every bias at 0 (mHC adds a constant 4 I to its transport logits),
so the model starts close to a plain residual (a ~ 0.5, b ~ 1, R near I) and weight decay
pulls the biases back to that start. Coefficient math and stream mixing run in FP32; the
branch input and the streams keep the model dtype.

The model expands the streams from the embedding (every stream = x0) and sums them
before the final norm.
"""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from xllm.distributed import get_model_parallel_world_size


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


def quaternion_transport(logits: Tensor, eps: float = 1e-12) -> Tensor:
    """logits [..., 8] -> H [..., 4, 4] in SO(4): x -> q x conj(r) for unit quaternions
    q = (1 + l0, l1, l2, l3) / |.| and r = (1 + l4, l5, l6, l7) / |.|. Zero logits give H = I."""
    one = torch.zeros_like(logits[..., :4])
    one[..., 0] = 1
    vq, vr = logits[..., :4] + one, logits[..., 4:] + one
    q = vq * torch.rsqrt(vq.square().sum(-1, keepdim=True) + eps)
    r = vr * torch.rsqrt(vr.square().sum(-1, keepdim=True) + eps)
    return _left(q) @ _right_conj(r)


def hc_read(streams: Tensor, a: Tensor) -> Tensor:
    """h [B, L, D] = sum_n a_n X_n in FP32, returned in the streams' dtype. a [B, L, N] or [B, L, G, N]."""
    x = streams.to(a.dtype)
    if a.ndim == 3:
        h = (a.unsqueeze(-1) * x).sum(dim=-2)
    else:
        bsz, slen, n, dim = x.shape
        groups = a.shape[-2]
        # [B, L, N, G, D/G] -> [B, L, G, D/G]
        h = torch.einsum('blngd,blgn->blgd', x.view(bsz, slen, n, groups, dim // groups), a).reshape(bsz, slen, dim)
    return h.to(streams.dtype)


def hc_write(streams: Tensor, R: Optional[Tensor], b: Tensor, y: Tensor) -> Tensor:
    """X' = R X + b y in FP32, returned in the streams' dtype. R None is the identity transport."""
    x = streams.to(b.dtype)
    yf = y.to(b.dtype)
    if b.ndim == 3:
        res = x if R is None else R @ x
        out = res + b.unsqueeze(-1) * yf.unsqueeze(-2)
    else:
        bsz, slen, n, dim = x.shape
        groups = b.shape[-2]
        # [B, L, G, N, D/G]
        xg = x.view(bsz, slen, n, groups, dim // groups).transpose(2, 3)
        res = xg if R is None else R @ xg
        out = res + b.unsqueeze(-1) * yf.view(bsz, slen, groups, 1, dim // groups)
        out = out.transpose(2, 3).reshape(bsz, slen, n, dim)
    return out.to(streams.dtype)


class HyperConnection(nn.Module):
    """Coefficient generator for one sublayer: (a, R, b) = hc(streams). Subclasses set the
    logit and scale counts and form the transport."""

    NUM_SCALES = 2  # read (a), write (b)

    @staticmethod
    def num_logits_for(num_streams: int) -> int:
        raise NotImplementedError

    def __init__(self, dim: int, num_streams: int, groups: int = 1, norm_eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_streams = num_streams
        self.num_logits = self.num_logits_for(num_streams)
        self.groups = groups
        self.norm_eps = norm_eps

        self.weight = nn.Parameter(torch.empty(num_streams * dim, self.num_logits))
        self.bias = nn.Parameter(torch.zeros(self.num_logits) if groups == 1 else torch.zeros(groups, self.num_logits))
        self.scale = nn.Parameter(torch.full((self.NUM_SCALES,), 0.01))
        with torch.no_grad():
            nn.init.normal_(self.weight, std=1.0 / math.sqrt(num_streams * dim))

    def logits(self, streams: Tensor) -> Tensor:
        """[B, L, n_logits] in FP32 from the RMS-normalized, flattened streams."""
        bsz, slen, n, dim = streams.shape
        assert n == self.num_streams and dim == self.dim, f"expected [B, L, {self.num_streams}, {self.dim}] streams, got {tuple(streams.shape)}"
        flat = rms_normalize(streams.float().flatten(-2), self.norm_eps)
        return flat @ self.weight.float()

    def gates(self, l_a: Tensor, l_b: Tensor, bias_a: Tensor, bias_b: Tensor) -> Tuple[Tensor, Tensor]:
        scale = self.scale.float()
        a = torch.sigmoid(scale[0] * l_a + bias_a)
        b = 2.0 * torch.sigmoid(scale[1] * l_b + bias_b)
        return a, b

    def forward(self, streams: Tensor) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        """(a, R, b) in FP32; R is None for the identity transport."""
        raise NotImplementedError

    def extra_repr(self) -> str:
        return 'dim={}, streams={}, logits={}, groups={}'.format(self.dim, self.num_streams, self.num_logits, self.groups)


class IdentityHyperConnection(HyperConnection):
    """Sigmoid read gates, 2 sigmoid write gates, R = I (returned as None)."""

    @staticmethod
    def num_logits_for(num_streams: int) -> int:
        return 2 * num_streams

    def __init__(self, dim: int, num_streams: int, norm_eps: float = 1e-6):
        super().__init__(dim, num_streams, norm_eps=norm_eps)

    def forward(self, streams: Tensor) -> Tuple[Tensor, None, Tensor]:
        n = self.num_streams
        logits = self.logits(streams)
        bias = self.bias.float()
        a, b = self.gates(logits[..., :n], logits[..., n:], bias[:n], bias[n:])
        return a, None, b


class ManifoldHyperConnection(HyperConnection):
    """mHC: gates as IHC, transport = Sinkhorn(s_r * l_r + bias_r + 4 I). The constant 4 I starts R near I
    (diagonal ~0.95 at N = 4) with a zero bias, so weight decay keeps the transport near identity rather than
    pulling it to uniform mixing. Rows of R sum to one, columns up to the Sinkhorn convergence."""

    NUM_SCALES = 3  # read, write, transport
    IDENTITY_LOGIT = 4.0

    @staticmethod
    def num_logits_for(num_streams: int) -> int:
        return 2 * num_streams + num_streams * num_streams

    def __init__(self, dim: int, num_streams: int, sinkhorn_iterations: int = 20, norm_eps: float = 1e-6):
        super().__init__(dim, num_streams, norm_eps=norm_eps)
        self.sinkhorn_iterations = sinkhorn_iterations

    def forward(self, streams: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        n = self.num_streams
        logits = self.logits(streams)
        bias = self.bias.float()
        a, b = self.gates(logits[..., :n], logits[..., n:2 * n], bias[:n], bias[n:2 * n])
        r_logits = self.scale.float()[2] * logits[..., 2 * n:] + bias[2 * n:]
        eye = torch.eye(n, device=r_logits.device, dtype=r_logits.dtype)
        r_logits = r_logits.view(*r_logits.shape[:-1], n, n) + self.IDENTITY_LOGIT * eye
        R = sinkhorn_knopp(r_logits, self.sinkhorn_iterations)
        return a, R, b

    def extra_repr(self) -> str:
        return super().extra_repr() + ', sinkhorn_iterations={}'.format(self.sinkhorn_iterations)


class OrthogonalHyperConnection(HyperConnection):
    """oHC: gates as IHC, transport R = SO(4)(alpha * l_r + bias_r) from eight logits; alpha is a constant
    (a non-persistent buffer), as the paper freezes it for orthogonal variants. Four streams only.

    groups > 1: the eight rotation logits are shared by all channel groups, group g uses them at rate
    freq_base^(-g / (G - 1)) (1 down to 1 / freq_base) with its own biases; the read / write gates share
    their logits too and differ per group by their biases only. freq_base = 1 gives equal rates."""

    ROTATION_LOGITS = 8

    @staticmethod
    def num_logits_for(num_streams: int) -> int:
        return 2 * num_streams + OrthogonalHyperConnection.ROTATION_LOGITS

    def __init__(
        self,
        dim: int,
        num_streams: int = 4,
        groups: int = 1,
        freq_base: float = 2.0,
        rotation_scale: float = 0.01,
        norm_eps: float = 1e-6,
    ):
        super().__init__(dim, num_streams, groups=groups, norm_eps=norm_eps)
        self.freq_base = float(freq_base)
        # constants of the config, rebuilt at construction rather than stored in checkpoints
        self.register_buffer('rotation_scale', torch.full((), float(rotation_scale)), persistent=False)
        rates = [self.freq_base ** (-g / (groups - 1)) for g in range(groups)] if groups > 1 else [1.0]
        self.register_buffer('rates', torch.tensor(rates), persistent=False)

    def forward(self, streams: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        n = self.num_streams
        logits = self.logits(streams)
        bias = self.bias.float()
        if self.groups > 1:
            # shared logits [B, L, 1, k] against per-group biases [G, k]
            logits = logits.unsqueeze(-2)
            rot_logits = logits[..., 2 * n:] * self.rates.float().view(-1, 1)
        else:
            rot_logits = logits[..., 2 * n:]
        a, b = self.gates(logits[..., :n], logits[..., n:2 * n], bias[..., :n], bias[..., n:2 * n])
        R = quaternion_transport(self.rotation_scale.float() * rot_logits + bias[..., 2 * n:])
        return a, R, b

    def extra_repr(self) -> str:
        return super().extra_repr() + ', freq_base={}, rotation_scale={}'.format(
            self.freq_base, float(self.rotation_scale))


HYPER_CONNECTIONS = {
    'ihc': IdentityHyperConnection,
    'mhc': ManifoldHyperConnection,
    'ohc': OrthogonalHyperConnection,
}


def build_hyper_connection(
    residual_func: str,
    model_dim: int,
    num_streams: int,
    groups: int = 1,
    freq_base: float = 2.0,
    sinkhorn_iterations: int = 20,
    norm_eps: float = 1e-6,
) -> HyperConnection:
    if residual_func == 'ihc':
        return IdentityHyperConnection(model_dim, num_streams, norm_eps=norm_eps)
    elif residual_func == 'mhc':
        return ManifoldHyperConnection(model_dim, num_streams, sinkhorn_iterations=sinkhorn_iterations, norm_eps=norm_eps)
    elif residual_func == 'ohc':
        return OrthogonalHyperConnection(model_dim, num_streams, groups=groups, freq_base=freq_base, norm_eps=norm_eps)
    else:
        raise ValueError(f"Unknown hyper-connection: {residual_func}")


def build_hyper_connections(cfg) -> Tuple[Optional[HyperConnection], Optional[HyperConnection]]:
    """The (attention, FFN) controllers of one block from a ModelConf; (None, None) without hyper-connections."""
    if cfg.residual_func not in HYPER_CONNECTIONS:
        return None, None
    # the controller normalizes over all N * D channels of a token, which a model-parallel shard does not hold
    assert get_model_parallel_world_size() == 1, "hyper-connections do not support model / expert parallelism yet"
    return tuple(
        build_hyper_connection(
            cfg.residual_func, cfg.model_dim, cfg.residual_streams,
            groups=cfg.residual_groups, freq_base=cfg.residual_freq_base,
            sinkhorn_iterations=cfg.residual_sinkhorn_iterations, norm_eps=cfg.rmsnorm_eps,
        )
        for _ in range(2)
    )


def num_params_in_hyper_connection(residual_func: str, model_dim: int, num_streams: int, groups: int = 1) -> int:
    """Parameters of one sublayer's controller (projection + biases + scales)."""
    if residual_func not in HYPER_CONNECTIONS:
        return 0
    hc = HYPER_CONNECTIONS[residual_func]
    num_logits = hc.num_logits_for(num_streams)
    return num_streams * model_dim * num_logits + groups * num_logits + hc.NUM_SCALES


def flops_in_hyper_connection(residual_func: str, model_dim: int, num_streams: int) -> int:
    """Multiply-adds of one sublayer per token: controller projection, read and write."""
    if residual_func not in HYPER_CONNECTIONS:
        return 0
    projection = num_streams * model_dim * HYPER_CONNECTIONS[residual_func].num_logits_for(num_streams)
    read = num_streams * model_dim
    transport = 0 if residual_func == 'ihc' else num_streams * num_streams * model_dim
    write = transport + num_streams * model_dim
    return projection + read + write
