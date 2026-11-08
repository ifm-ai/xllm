from xllm.modules.fused_ops.attention.flash import _varlen_fwd_extra_args


def _doc(n_args):
    args = ", ".join(f"arg{i}: torch.Tensor" for i in range(n_args))
    return f"varlen_fwd({args}) -> list[torch.Tensor]\n"


def test_signature_detection():
    assert _varlen_fwd_extra_args(_doc(21)) == ()  # before Dao-AILab/flash-attention#2448
    assert _varlen_fwd_extra_args(_doc(22)) == (0,)  # with num_splits
    assert _varlen_fwd_extra_args(_doc(23)) is None
    assert _varlen_fwd_extra_args(None) is None
    assert _varlen_fwd_extra_args("MagicMock docstring without a signature") is None
