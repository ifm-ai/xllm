import torch
from torch.nn import functional as F

import xllm.models.fused_blocks.cross_entropy as cross_entropy


def test_fused_logit_stats_are_masked_and_add_no_tp_collective(monkeypatch):
    monkeypatch.setattr(cross_entropy, "get_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(cross_entropy, "get_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(cross_entropy, "get_model_parallel_group", lambda: None)
    monkeypatch.setattr(
        cross_entropy,
        "all_reduce",
        lambda tensor, parallel_region, async_op=False: (tensor, None),
    )
    tp_collective_sizes = []
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, **kwargs: tp_collective_sizes.append(tensor.numel()),
    )

    input_ = torch.tensor([[[1.0, 2.0, -1.0], [0.5, -2.0, 3.0]]])
    weight = torch.tensor([
        [1.0, 0.0, 0.5],
        [0.0, 1.0, -0.5],
        [0.25, -1.0, 1.0],
        [-0.5, 0.5, 0.0],
    ])
    target = torch.tensor([[0, 1]])
    mask = torch.tensor([[True, False]])

    plain_output = cross_entropy.fused_linear_cross_entropy_fwd(
        input_, weight, target, mask=mask, compute_grad=False,
        gather_input=False,
    )
    assert plain_output.logits_stats is None
    plain_collective_sizes = list(tp_collective_sizes)
    tp_collective_sizes.clear()

    output = cross_entropy.fused_linear_cross_entropy_fwd(
        input_, weight, target, mask=mask, compute_grad=False,
        gather_input=False, collect_logits_stats=True,
    )

    expected = F.linear(input_[:, :1], weight).float()
    assert output.logits_stats is not None
    torch.testing.assert_close(output.logits_stats.sum, expected.sum())
    assert output.logits_stats.count.item() == expected.numel()
    torch.testing.assert_close(output.logits_stats.minimum, expected.min())
    torch.testing.assert_close(output.logits_stats.maximum, expected.max())
    assert tp_collective_sizes == plain_collective_sizes == [1, 1]
