from torch import nn

from xllm.config import OptimConf


def no_weight_decay_param_ids(model: nn.Module, cfg: OptimConf) -> set:
    """Ids of parameters excluded from weight decay by ``optim.disable_wd_{emb,out}``.

    Embedding: every parameter of ``model.embed``.
    Readout: the output projection ``model.output.output`` (the final norm keeps its decay).
    """
    if not (cfg.disable_wd_emb or cfg.disable_wd_out):
        return set()

    from torch.distributed.fsdp import FullyShardedDataParallel
    if isinstance(model, FullyShardedDataParallel):
        raise NotImplementedError(
            "optim.disable_wd_emb / optim.disable_wd_out require model.ddp_backend=fsdp2 "
            "(FSDP1 flattens parameters, so embedding/readout weights cannot be isolated)."
        )

    params = []
    if cfg.disable_wd_emb:
        params.extend(model.embed.parameters())
    if cfg.disable_wd_out:
        params.extend(model.output.output.parameters())
    return {id(p) for p in params}
