"""Multi-token prediction (MTP), as in DeepSeek-V3: at every position j, mtp_depth extra modules also learn the
tokens j+2, ..., j+1+mtp_depth.

Module k (1-based) takes the hidden state at j of the module before it (for k = 1, the main model's last hidden state,
before the final norm) and the embedding of the real token j+k, normalizes both, joins them and projects them back to
the model width (h'), runs one Transformer block, of the same kind as the main model's last layer, over the sequence
of h', and passes the result to the output layer, which predicts token j+1+k with the module's own final norm
(TransformerOutputLayer.mtp_norms[k - 1]) and the shared output weight.
Training only: the main model is unchanged, and eval, generation and inference never run the modules.

Under context parallelism the embedding of token j+k at the last k positions of a CP rank's slice is on the next rank;
those positions get no target of module k (k targets fewer per slice) instead of an exchange.
"""
import contextlib
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from xllm.distributed import get_context_parallel_world_size
from xllm.modules.context_parallel import gather_from_context_parallel_region
from xllm.modules.layer_norm import GroupLayerNorm
from xllm.modules.model_parallel import ColumnParallelLinear, gather_copy_model_parallel_region
from xllm.modules.rms_norm import GroupRMSNorm
from xllm.utils import get_init_fn

IGNORE_INDEX = -100


def mtp_targets(targets: Tensor, token_mask: Optional[Tensor], eos_id: int, k: int = 1) -> Tuple[Tensor, Tensor]:
    """The targets of MTP module k for one sequence slice (B x L, a CP rank's slice in training), and which count.

    The main target at j is token j+1 (targets[:, j]); the target of module k at j is token j+1+k, i.e.
    targets[:, j + k]. A position gets no target when that token is outside the slice (the last k positions), is
    masked or ignored as a main target, or is in another document: one of the tokens j+1, ..., j+k is an
    end-of-document token.
    Returns (targets with IGNORE_INDEX where none, bool mask of the positions that count).
    """
    assert k >= 1
    nxt = torch.full_like(targets, IGNORE_INDEX)
    nxt[:, :-k] = targets[:, k:]
    valid = nxt != IGNORE_INDEX
    is_eos = targets == eos_id
    for i in range(k):
        # token j+1+i (targets[:, j + i]) is an end of document
        eos_i = torch.zeros_like(is_eos)
        eos_i[:, :is_eos.shape[1] - i] = is_eos[:, i:]
        valid &= ~eos_i
    if token_mask is not None:
        nxt_mask = torch.zeros_like(token_mask, dtype=torch.bool)
        nxt_mask[:, :-k] = token_mask[:, k:].to(torch.bool)
        valid &= nxt_mask
    return nxt.masked_fill(~valid, IGNORE_INDEX), valid


class MTPLayer(nn.Module):
    """MTP module k: `block` is a Transformer block built like the main model's layers."""

    def __init__(self, cfg: Any, block: nn.Module, k: int):
        """cfg: the ModelConf (model_dim, norms, init); block: built with the main model's block class."""
        super().__init__()
        self.k = k
        norm_cls = GroupRMSNorm if cfg.apply_rmsnorm else GroupLayerNorm
        norm_eps = cfg.rmsnorm_eps if cfg.apply_rmsnorm else cfg.layernorm_eps
        # normalize the hidden state and the embedding, as the attention's input norm does
        self.hidden_norm, self.embed_norm = (
            norm_cls(cfg.model_dim, num_groups=cfg.layernorm_num_groups, elementwise_affine=cfg.norm_affine,
                     eps=norm_eps, memory_efficient=cfg.memory_efficient_norm)
            for _ in range(2)
        )
        # [hidden; embedding] (2 x model_dim, full on every TP rank) -> h' (model_dim / TP, as the layers' inputs)
        self.proj = ColumnParallelLinear(
            2 * cfg.model_dim,
            cfg.model_dim,
            bias=False,
            input_is_parallel=False,
            disable_input_reduce=True,
            gather_output=False,
            init_method=get_init_fn(cfg.init_mode, dim=2 * cfg.model_dim, std=cfg.init_std),
        )
        self.block = block

    def _normalize(self, norm: nn.Module, x: Tensor) -> Tensor:
        x = norm(x)
        if not norm.gather_input:
            x = gather_copy_model_parallel_region(x)
        return x

    def forward(self, hidden: Tensor, embeddings: Tensor, block_args: Tuple, layerwise_ckpt: bool):
        """hidden: the previous module's (or the main model's last) hidden states (B x L x D/TP); embeddings: the
        main model's token embeddings (B x L x D/TP, before dropout); block_args: the main layers' arguments after
        x, without the cache. Returns the block's output (B x L x D/TP) and its aux loss (or None)."""
        # the embedding of token j+k at position j; zero at the last k positions, which get no target
        nxt = torch.zeros_like(embeddings)
        nxt[:, :-self.k] = embeddings[:, self.k:]
        x = torch.cat([self._normalize(self.hidden_norm, hidden), self._normalize(self.embed_norm, nxt)], dim=-1)
        x = self.proj(x)
        args = (x, *block_args, None)
        if layerwise_ckpt:
            x, aux_loss, _ = checkpoint(self.block, *args, use_reentrant=False, preserve_rng_state=True)
        else:
            x, aux_loss, _ = self.block(*args)
        return x, aux_loss


class MultiTokenPrediction(nn.Module):
    """The mtp_depth MTP modules, each its own FSDP unit; their losses go through the main model's output layer."""

    def __init__(self, cfg: Any, block_cls: Callable[[Any, int], nn.Module], wrap: Callable[[nn.Module], nn.Module]):
        """block_cls: the main model's last-layer class, built with layer ids after the main layers'; wrap: the
        model's FSDP wrap."""
        super().__init__()
        self.layers = nn.ModuleList(
            wrap(MTPLayer(cfg, block_cls(cfg, cfg.num_layers + k), k + 1)) for k in range(cfg.mtp_depth)
        )
        self._outputs = None

    def forward(
        self,
        output: nn.Module,
        x: Tensor,
        embeddings: Tensor,
        targets: Tensor,
        token_mask: Optional[Tensor],
        eos_id: int,
        block_args: Tuple,
        layerwise_ckpt: bool,
    ) -> Tensor:
        """Runs the modules after the main model's last layer, in training, and calls the output layer for all
        losses. Returns the main token losses as the output layer does; the MTP ones are kept for take_outputs."""
        rows, aux_loss_sum = [], None
        hidden = x
        for k, layer in enumerate(self.layers, start=1):
            hidden, aux_loss = layer(hidden, embeddings, block_args, layerwise_ckpt)
            rows.append((hidden, *mtp_targets(targets, token_mask, eos_id, k)))
            if aux_loss is not None:
                aux_loss_sum = aux_loss if aux_loss_sum is None else aux_loss_sum + aux_loss
        loss, mtp_losses = output(x, targets, token_mask, mtp=rows)
        masks = [mask for _, _, mask in rows]
        if get_context_parallel_world_size() > 1:
            mtp_losses = [gather_from_context_parallel_region(t) for t in mtp_losses]
            masks = [gather_from_context_parallel_region(t) for t in masks]
        self._outputs = (mtp_losses, masks, aux_loss_sum)
        return loss

    def take_outputs(self) -> Optional[Tuple[List[Tensor], List[Tensor], Optional[Tensor]]]:
        """(token losses per module, masks of the tokens that count, summed aux loss or None) of the last training
        forward, or None; returns them once."""
        outputs, self._outputs = self._outputs, None
        return outputs

    @contextlib.contextmanager
    def _with_layers(self, model: nn.Module):
        """The model's counts with mtp_depth more layers of the last layer's kind: they are linear in the layers."""
        depth = len(self.layers)
        dense = model.num_dense_layers == model.num_layers
        model.num_layers += depth
        model.num_dense_layers += depth * int(dense)
        try:
            yield
        finally:
            model.num_layers -= depth
            model.num_dense_layers -= depth * int(dense)

    def num_parameters(self, model: nn.Module) -> Tuple[int, int, int]:
        """model._num_parameters() with the modules: a block, the projection, two input norms and a final norm each."""
        with self._with_layers(model):
            total, activated, embed = model._num_parameters()
        dim = model.model_dim
        extra = len(self.layers) * (2 * dim * dim + 3 * dim * (1 if model.apply_rmsnorm else 2))
        return total + extra, activated + extra, embed

    def tflops_per_token(self, model: nn.Module, seq_len: int) -> float:
        """model._tflops_per_token() with the modules: a block, the projection and an output-layer pass each."""
        with self._with_layers(model):
            tflops = model._tflops_per_token(seq_len)
        expansion_factor = 6
        extra = len(self.layers) * (2 * model.model_dim * model.model_dim + model.model_dim * model.vocab_size)
        return tflops + extra * expansion_factor / 10**12


def mtp_train_loss(
    outputs: Tuple[List[Tensor], List[Tensor], Optional[Tensor]],
    loss_weight: float,
    aux_loss_coeff: float,
    num_moe_layers: int,
    data_parallel_group: Any,
    data_parallel_size: int,
    context_parallel_size: int,
) -> Tuple[Tensor, List[Tensor]]:
    """The MTP term of the training loss, loss_weight / mtp_depth x the sum of the modules' losses (DeepSeek-V3), each
    normalized as the main loss over its own target count, plus the modules' router aux losses weighted as the main
    layers' (aux_loss_coeff / num_moe_layers each). Returns it and the modules' losses for logging."""
    mtp_losses, masks, aux_loss = outputs
    num_tokens = torch.stack([mask.to(torch.int64).sum() for mask in masks])
    torch.distributed.all_reduce(num_tokens, group=data_parallel_group)
    num_tokens_per_rank = num_tokens / (data_parallel_size * context_parallel_size)
    losses = [tok_loss.sum() / (n + 1e-6).to(tok_loss) for tok_loss, n in zip(mtp_losses, num_tokens_per_rank)]
    total = (loss_weight / len(losses)) * sum(losses[1:], losses[0])
    if aux_loss is not None:
        total = total + aux_loss_coeff * (aux_loss / num_moe_layers)
    return total, [loss.detach() / context_parallel_size for loss in losses]


def mtp_log_values(losses: Sequence[Tensor]) -> Dict[str, float]:
    """avg_mtp_loss, the mean of the modules' losses, and with more than one module avg_mtp_loss_<k> for each."""
    values = [loss.item() for loss in losses]
    logged = {"avg_mtp_loss": sum(values) / len(values)}
    if len(values) > 1:
        logged.update({f"avg_mtp_loss_{k}": value for k, value in enumerate(values, start=1)})
    return logged
