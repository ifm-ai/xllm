"""Parameter counts for hybrid Gekko MoE drafts, following xLLM's per-block num_parameters formulas
(xllm/models/blocks/{gekko_ffn,mha_moe,mha_ffn}.py, LatentMoE from PR #33, iHC from PR #41).
Assumes Gekko layers >= num_dense_layers carry MoE (a GekkoMoE block; not in main yet)."""
from dataclasses import dataclass
from typing import Optional

HC_LOGITS = {'base': 0, 'ihc': 8, 'ohc': None, 'mhc': 24}


@dataclass
class Cfg:
    name: str
    arch: str = 'hybrid'
    num_layers: int = 64
    full_attn_layers: tuple = ()
    model_dim: int = 6144
    num_heads: int = 64
    num_kv_heads: int = 8
    head_dim: int = 128
    v_head_dim: int = 128
    qknorm: bool = True
    apply_attn_gate: bool = True
    apply_conv_in_causal_attn: bool = True
    conv_width: int = 4
    apply_bias_term: bool = False
    ffn_hidden_dim: int = 16384
    num_dense_layers: int = 3
    num_experts: int = 192
    num_activated_experts: int = 8
    num_shared_experts: int = 1
    expert_inter_dim: int = 1792
    moe_latent_alpha: Optional[int] = None
    residual_func: str = 'ihc'
    vocab: int = 250624


def hc(c):  # one sublayer's controller
    L = HC_LOGITS[c.residual_func]
    return 0 if not L else 4 * c.model_dim * L + L + 2


def gekko_attn(c):
    d, H, KV, hd, vd, w = c.model_dim, c.num_heads, c.num_kv_heads, c.head_dim, c.v_head_dim, c.conv_width
    p = d * (H + KV) * (hd + vd) + d * H * vd
    p += (KV + H) * vd if c.apply_bias_term else KV * vd
    p += w * (H * hd + KV * (hd + vd))
    p += d * 3 + hd * (H + KV)  # norms (rmsnorm)
    return p + hc(c)


def full_attn(c):
    d, H, KV, hd, w = c.model_dim, c.num_heads, c.num_kv_heads, c.head_dim, c.conv_width
    p = d * (H * 2 + KV * 2) * hd
    if c.apply_attn_gate:
        p += d * H * hd
    p += d * 2 + ((H + KV) * hd if c.qknorm else 0)
    if c.apply_conv_in_causal_attn:
        p += w * (H * hd + 2 * KV * hd)
    return p + hc(c)


def moe(c):
    d, I = c.model_dim, c.expert_inter_dim
    a = c.moe_latent_alpha or 1
    rd, E, k = d // a, c.num_experts * a, c.num_activated_experts * a
    shared = 3 * d * I * c.num_shared_experts
    routed, active = 3 * rd * I * E, 3 * rd * I * k
    router = d * E
    latent = 2 * d * rd if a > 1 else 0
    return shared + routed + router + latent + hc(c), shared + active + router + latent + hc(c)


def dense_ffn(c):
    return 3 * c.model_dim * c.ffn_hidden_dim + hc(c)


def count(c):
    tot = act = 0
    n_full = 0
    for i in range(c.num_layers):
        is_full = c.arch == 'transformer' or i in c.full_attn_layers
        n_full += is_full
        a = full_attn(c) if is_full else gekko_attn(c)
        if i < c.num_dense_layers:
            f_t = f_a = dense_ffn(c)
        else:
            f_t, f_a = moe(c)
        tot += a + f_t
        act += a + f_a
    emb = 2 * c.vocab * c.model_dim
    return dict(total=tot + emb, active=act + emb, total_ne=tot, active_ne=act, n_full=n_full)


def fmt(c):
    r = count(c)
    a = c.moe_latent_alpha or 1
    return (f"{c.name}: {c.num_layers}L ({r['n_full']} full), d{c.model_dim}, "
            f"E{c.num_experts * a} k{c.num_activated_experts * a} I{c.expert_inter_dim}"
            f"{' latent d/' + str(a) if a > 1 else ''} | total {r['total'] / 1e9:.1f}B "
            f"(non-emb {r['total_ne'] / 1e9:.1f}B), active {r['active'] / 1e9:.1f}B (non-emb {r['active_ne'] / 1e9:.1f}B)")


def every(n_layers, stride, last=True):
    """Full-attention layers every `stride`, ending on the last layer (as in the hybrid-gekko presets)."""
    return tuple(range(n_layers - 1, -1, -stride))[::-1]


if __name__ == '__main__':
    ref = Cfg('k2-horizon-375B-a23B (check)', arch='transformer', num_layers=61, model_dim=6144, num_heads=48,
              qknorm=False, apply_attn_gate=False, apply_conv_in_causal_attn=False, residual_func='base')
    print(fmt(ref))
