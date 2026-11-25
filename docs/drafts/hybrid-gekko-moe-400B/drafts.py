"""Draft 400B-class hybrid Gekko MoE presets (not runnable on main: assumes Gekko layers can carry MoE).
Validated against ModelConf on branch yash.akhauri/arch-draft; prints --model.* flag blocks and param counts."""
import sys
from params import Cfg, count, every

# Shared settings: hybrid-gekko-{0.9B,4B,7B} presets in xllm/models/hybrid.py + k2-horizon-375B-a23B (MoE recipe)
COMMON = dict(
    arch='hybrid', chunk_size=4096, head_dim=128, v_head_dim=128, num_kv_heads=8, qknorm=True,
    apply_attn_gate=True, attn_gate_func='softplus', apply_conv_in_causal_attn=True,
    rope_head_dim=128, rope_base=500000, apply_nope_in_causal_attn=True,
    causal_conv_width=4, causal_conv_weight_normalization=True, swiglu=True,
    apply_rmsnorm=True, layernorm_num_groups=1, timenorm_eps=1e-5, layernorm_eps=1e-5, rmsnorm_eps=1e-6,
    num_dense_layers=3, num_shared_experts=1,
    moe_router_type='dynamic_qb', moe_router_bias=False, moe_router_score_func='sigmoid',
    moe_router_dynamic_qb_update_method='average', moe_router_dynamic_qb_decay=0.3,
    moe_router_dynamic_qb_globalnorm=True,
    residual_func='ihc', fused_block=False,
)

def preset(num_layers, model_dim, num_heads, ffn_hidden_dim, E, k, I, alpha=None):
    full = every(num_layers, 8)
    p = dict(COMMON, num_layers=num_layers, full_attn_layer_list=','.join(map(str, full)), model_dim=model_dim,
             num_heads=num_heads, ffn_hidden_dim=ffn_hidden_dim, timenorm_num_groups=model_dim // 64,
             num_experts=E, num_activated_experts=k, expert_inter_dim=I)
    if alpha:
        p['moe_latent_alpha'] = alpha
    return p

DRAFTS = {
    # A: minimum = the features Max's team has validated (hybrid Gekko, iHC, QB-MoE) on a 375B-like body
    'hybrid-gekko-moe-400B-A': preset(64, 6144, 48, 16384, 288, 8, 1216),
    # B: A + LatentMoE (Alex's list): experts and top-k x4 in a d/4 latent space; same expert params and FLOPs
    'hybrid-gekko-moe-400B-B-latent': preset(64, 6144, 48, 16384, 288, 8, 1216, alpha=4),
    # C: deeper and narrower (80 layers), for looping added later rather than from scratch
    'hybrid-gekko-moe-400B-C-deep': preset(80, 5120, 40, 13824, 288, 8, 1152),
}

def to_cfg(name, p):
    return Cfg(name, num_layers=p['num_layers'], full_attn_layers=tuple(map(int, p['full_attn_layer_list'].split(','))),
               model_dim=p['model_dim'], num_heads=p['num_heads'], num_kv_heads=p['num_kv_heads'],
               head_dim=p['head_dim'], v_head_dim=p['v_head_dim'], ffn_hidden_dim=p['ffn_hidden_dim'],
               num_dense_layers=p['num_dense_layers'], num_experts=p['num_experts'],
               num_activated_experts=p['num_activated_experts'], num_shared_experts=p['num_shared_experts'],
               expert_inter_dim=p['expert_inter_dim'], moe_latent_alpha=p.get('moe_latent_alpha'),
               residual_func=p['residual_func'])

def flags(p):
    out = []
    for k, v in p.items():
        v = str(v).lower() if isinstance(v, bool) else v
        out.append(f'--model.{k} {v if not isinstance(v, str) or v.replace(".", "").replace("e-", "").isdigit() else repr(v).replace(chr(39), chr(34))}')
    return out

GROUPS = [
    ('Backbone: hybrid Gekko (as the hybrid-gekko presets)', ['arch', 'chunk_size', 'head_dim', 'v_head_dim', 'num_kv_heads', 'qknorm',
      'apply_attn_gate', 'attn_gate_func', 'apply_conv_in_causal_attn', 'rope_head_dim', 'rope_base',
      'apply_nope_in_causal_attn', 'causal_conv_width', 'causal_conv_weight_normalization', 'swiglu', 'apply_rmsnorm',
      'layernorm_num_groups', 'timenorm_eps', 'layernorm_eps', 'rmsnorm_eps']),
    ('MoE: QB router (PR #33); moe_router_bias must be false with QB', ['num_dense_layers', 'num_shared_experts',
      'moe_router_type', 'moe_router_bias', 'moe_router_score_func', 'moe_router_dynamic_qb_update_method',
      'moe_router_dynamic_qb_decay', 'moe_router_dynamic_qb_globalnorm']),
    ('Residual: iHC (PR #41); no fused block for iHC + MoE', ['residual_func', 'fused_block']),
]
SHAPE = ['num_layers', 'full_attn_layer_list', 'model_dim', 'num_heads', 'timenorm_num_groups', 'ffn_hidden_dim',
         'num_experts', 'num_activated_experts', 'expert_inter_dim', 'moe_latent_alpha']

def flag(k, v):
    if isinstance(v, bool):
        return f'--model.{k} "{str(v).lower()}"'
    if isinstance(v, str):
        return f'--model.{k} "{v}"'
    return f'--model.{k} {v}'

def render():
    out = ['# Common to all three drafts']
    any_p = next(iter(DRAFTS.values()))
    for title, keys in GROUPS:
        out.append(f'# {title}')
        out += [flag(k, any_p[k]) for k in keys]
    for n, p in DRAFTS.items():
        r = count(to_cfg(n, p))
        out.append('')
        out.append(f"# {n}: {r['total_ne']/1e9:.0f}B total, {r['active_ne']/1e9:.1f}B active (non-embedding, "
                   f"as in the k2-horizon-375B-a23B name); {r['n_full']} of {p['num_layers']} layers full attention")
        out += [flag(k, p[k]) for k in SHAPE if k in p]
    return '\n'.join(out)

if __name__ == '__main__':
    shared = set(k for _, ks in GROUPS for k in ks)
    for p in DRAFTS.values():
        assert set(p) == shared | {k for k in SHAPE if k in p}, set(p) ^ (shared | set(SHAPE))
        assert all(p[k] == next(iter(DRAFTS.values()))[k] for k in shared)
    if '--validate' in sys.argv:
        from xllm.config import ModelConf
        for n, p in DRAFTS.items():
            ModelConf(**p)
            print('ModelConf OK:', n)
    print(render())
