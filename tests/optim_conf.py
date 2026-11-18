"""OptimConf from the flat names the Muon tests were written with (lr, beta1, ..., muon_lr, muon_momentum, ...)."""
from xllm.config import AdamWConf, MuonConf, OptimConf

ADAMW_FIELDS = ("lr", "beta1", "beta2", "epsilon", "weight_decay")


def optim_conf(**kw) -> OptimConf:
    adamw = {k: kw.pop(k) for k in ADAMW_FIELDS if k in kw}
    muon = {k[len("muon_"):]: kw.pop(k) for k in list(kw) if k.startswith("muon_")}
    if "weight_decay" in adamw:
        muon.setdefault("weight_decay", adamw["weight_decay"])  # the tests' references decay both groups alike
    return OptimConf(adamw=AdamWConf(**adamw), muon=MuonConf(**muon), **kw)
