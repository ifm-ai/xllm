"""OptimConf for the Muon tests, from the flat names they were written with (lr, beta1, ..., muon_lr, muon_momentum, ...).

The tests' references were written against the earlier defaults (AdamW 1e-3 / beta2 0.98 / wd 0.1, Muon 0.01 with
Nesterov and the spectral_norm scale, cosine schedule, Muon decay = the AdamW decay), so unset fields keep those
values here rather than the library defaults (the production recipe).
"""
from xllm.config import AdamWConf, MuonConf, OptimConf

ADAMW_FIELDS = ("lr", "beta1", "beta2", "epsilon", "weight_decay")
ADAMW_TEST_DEFAULTS = dict(lr=1e-3, beta1=0.9, beta2=0.98, epsilon=1e-8, weight_decay=0.1)
MUON_TEST_DEFAULTS = dict(lr=0.01, momentum=0.95, nesterov=True, adjust_lr="spectral_norm", scale_shape="block")
OPTIM_TEST_DEFAULTS = dict(warmup=2000, scheduler="cosine", lr_end_ratio=0.01)


def optim_conf(**kw) -> OptimConf:
    adamw = dict(ADAMW_TEST_DEFAULTS, **{k: kw.pop(k) for k in ADAMW_FIELDS if k in kw})
    muon = dict(MUON_TEST_DEFAULTS, **{k[len("muon_"):]: kw.pop(k) for k in list(kw) if k.startswith("muon_")})
    muon.setdefault("weight_decay", adamw["weight_decay"])  # the references decay both groups alike
    return OptimConf(adamw=AdamWConf(**adamw), muon=MuonConf(**muon), **dict(OPTIM_TEST_DEFAULTS, **kw))
