import math

import pytest

from xllm.optim.lr_scheduler import lr_cosine, lr_linear, lr_wsd

INIT, END, CYCLES = 1e-4, 0.1, 1.0


def schedule(name, warmup, total_steps):
    if name == "linear":
        return lambda step: lr_linear(step, warmup, total_steps, INIT, END)
    return lambda step: lr_cosine(step, warmup, total_steps, CYCLES, INIT, END)


def reference(name, step, warmup, total_steps):
    # the formulas before the fix, valid when warmup < total_steps
    if step < warmup:
        return INIT + float(step) * (1.0 - INIT) / warmup
    if step > total_steps:
        return END
    s = float(step - warmup) / (total_steps - warmup)
    if name == "linear":
        return s * END + (1 - s)
    return END + 0.5 * (1 - END) * (math.cos(math.pi * s * CYCLES) + 1)


@pytest.mark.parametrize("name", ["linear", "cosine"])
@pytest.mark.parametrize("warmup,total_steps", [(0, 10), (1, 10), (4, 10), (9, 10), (2000, 100000)])
def test_unchanged_with_decay_steps(name, warmup, total_steps):
    fn = schedule(name, warmup, total_steps)
    for step in sorted({0, 1, warmup - 1, warmup, warmup + 1, total_steps - 1, total_steps, total_steps + 1}):
        if step >= 0:
            assert fn(step) == reference(name, step, warmup, total_steps)


@pytest.mark.parametrize("name", ["linear", "cosine"])
@pytest.mark.parametrize("steps", [0, 1, 4])
def test_warmup_equals_total_steps(name, steps):
    # the training loop calls the schedule up to step == total_steps (after the last optimizer step)
    fn = schedule(name, steps, steps)
    lrs = [fn(step) for step in range(steps + 2)]
    assert lrs[:steps] == [INIT + float(step) * (1.0 - INIT) / steps for step in range(steps)]
    # end of warmup: peak LR, as lr_wsd gives with no stable and no decay steps
    assert lrs[steps] == 1.0 == lr_wsd(steps, steps, steps, 0, "cosine", CYCLES, INIT, END)
    assert lrs[steps + 1] == END
