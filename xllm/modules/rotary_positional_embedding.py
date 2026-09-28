import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, embed_dim, max_positions, base):
        super().__init__()
        assert embed_dim % 2 == 0
        self.embed_dim = embed_dim
        self.max_positions = max_positions
        self.base = base
        self.freqs: Optional[torch.Tensor] = None
        self.freqs_cis: Optional[torch.Tensor] = None

    def _precompute_freqs(self, device: torch.device):
        freqs = [self.base ** (j / self.embed_dim) for j in range(0, self.embed_dim, 2)]
        freqs = torch.tensor(freqs, dtype=torch.float32, device=device)
        freqs = 1.0 / freqs
        return freqs

    @torch.no_grad()
    def _precompute_until(self, max_positions: int, device: torch.device):
        assert self.max_positions <= max_positions
        self.max_positions = max_positions
        if self.freqs is None:
            self.freqs = self._precompute_freqs(device)
        # C
        t = torch.arange(max_positions, dtype=torch.float32, device=self.freqs.device)
        # C x D/2
        freqs = torch.outer(t, self.freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def get_freqs_cis(self, start: int, end: int, device: torch.device) -> torch.Tensor:
        if self.freqs_cis is None:
            self.freqs_cis = self._precompute_until(self.max_positions, device)
        if end > self.freqs_cis.shape[0]:
            warnings.warn('Extending rotary range from {} to {}'.format(self.max_positions, end))
            self.freqs_cis = self._precompute_until(end, device)
        return self.freqs_cis[start:end]  # type: ignore

    def forward(self, xq, xk, start: int):
        # B x C x N x  D
        seq_len = xq.shape[1]
        freqs_cis = self.get_freqs_cis(start, start + seq_len, xq.device)
        return apply_rotary_embeddings(xq, xk, freqs_cis=freqs_cis, backward=False)

    def extra_repr(self) -> str:
        return 'dim={}, max positions={}, base={:.1f}'.format(self.embed_dim, self.max_positions, self.base)


def apply_rotary_embedding(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    backward: bool,
) -> torch.Tensor:
    """
    If backward=False:
        - inputs: x
        - outputs: x_out
    If backward=True:
        - inputs: grad_x_out
        - outputs: grad_x
    """
    if backward:
        freqs_cis = freqs_cis.conj()

    # B x C x N x D/2
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # C x 1 x D/2
    freqs_cis = freqs_cis.unsqueeze(1)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
    return x_out.type_as(x)


def apply_rotary_embeddings(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    backward: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    If backward=False:
        - inputs: (xq, xk)
        - outputs: (xq_out, xk_out)
    If backward=True:
        - inputs: (grad_xq_out, grad_xk_out)
        - outputs: (grad_xq, grad_xk)
    """
    if backward:
        freqs_cis = freqs_cis.conj()

    # B x C x N x D/2
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # C x 1 x D/2
    freqs_cis = freqs_cis.unsqueeze(1)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
