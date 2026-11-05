"""DP row balancing (xllm/data/balance.py): attention cost per EOS-packed row and the greedy rank assignment."""
import torch

from xllm.data.balance import _assign, attention_row_cost


def test_attention_row_cost():
    g = torch.Generator().manual_seed(0)
    for C in (None, 64):
        tokens = torch.randint(3, 50, (3, 300), generator=g)
        tokens[torch.rand(3, 300, generator=g) < 0.02] = 2
        ref = []
        for row in tokens.tolist():
            start, n = 0, 0
            for t in range(len(row)):
                if t > 0 and row[t - 1] == 2:
                    start = t
                lo = max(start, (t // C - 1) * C) if C else start
                n += t - max(lo, 0) + 1
            ref.append(n)
        assert attention_row_cost(tokens, 2, C).tolist() == ref


def test_assign():
    cost = [9, 1, 7, 3, 5, 5, 2, 8]
    slots = _assign(cost, 4, 2)
    assert sorted(i for s in slots for i in s) == list(range(8)) and all(len(s) == 2 for s in slots)
    loads = [sum(cost[i] for i in s) for s in slots]
    assert max(loads) - min(loads) <= 2
