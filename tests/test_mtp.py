"""Multi-token prediction targets (xllm/modules/mtp.py): which token each position learns, and which count."""
import torch

from xllm.modules.mtp import IGNORE_INDEX, mtp_targets

EOS = 2


def test_targets_are_the_token_after_next():
    # tokens:  t0 t1 t2 t3 t4 t5 ; main targets y[j] = t(j+1)
    y = torch.tensor([[11, 12, 13, 14, 15, 16]])
    mtp_y, valid = mtp_targets(y, None, EOS)
    assert mtp_y.tolist() == [[12, 13, 14, 15, 16, IGNORE_INDEX]]
    assert valid.tolist() == [[True, True, True, True, True, False]]


def test_no_target_across_an_end_of_document():
    # document A ends with EOS at token 3 (y[2] == EOS): position 2's token after next is in document B
    y = torch.tensor([[11, 12, EOS, 21, 22, 23]])
    mtp_y, valid = mtp_targets(y, None, EOS)
    assert valid.tolist() == [[True, True, False, True, True, False]]
    assert mtp_y[0, 2].item() == IGNORE_INDEX
    # position 1 learns the EOS itself, which is in document A
    assert mtp_y[0, 1].item() == EOS


def test_masked_main_targets_give_no_mtp_target():
    y = torch.tensor([[11, IGNORE_INDEX, 13, 14, 15]])
    mtp_y, valid = mtp_targets(y, None, EOS)
    # position 0's token after next is y[1], ignored as a main target
    assert valid.tolist() == [[False, True, True, True, False]]
    assert mtp_y[0, 0].item() == IGNORE_INDEX


def test_token_mask_of_the_target_decides():
    # an SFT-style mask: the prompt (main targets 0-1) does not count, the answer (2-4) does
    y = torch.tensor([[11, 12, 13, 14, 15]])
    mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool)
    mtp_y, valid = mtp_targets(y, mask, EOS)
    # the MTP target at j is the main target at j+1, so it counts where mask[j+1] does
    assert valid.tolist() == [[False, True, True, True, False]]
    assert (mtp_y[~valid] == IGNORE_INDEX).all()
    assert mtp_y[valid].tolist() == [13, 14, 15]


def test_batch_rows_are_independent():
    y = torch.tensor([[11, 12, 13], [EOS, 22, 23]])
    mtp_y, valid = mtp_targets(y, None, EOS)
    assert valid.tolist() == [[True, True, False], [False, True, False]]


def test_depth_two_targets_skip_two_tokens():
    # module 2 learns t(j+3) = y[j+2]; the last two positions have none
    y = torch.tensor([[11, 12, 13, 14, 15, 16]])
    mtp_y, valid = mtp_targets(y, None, EOS, k=2)
    assert mtp_y.tolist() == [[13, 14, 15, 16, IGNORE_INDEX, IGNORE_INDEX]]
    assert valid.tolist() == [[True, True, True, True, False, False]]


def test_depth_two_no_target_across_an_end_of_document():
    # EOS is token 3 (y[2]): positions 1 and 2 would learn a token of document B (tokens j+1..j+2 include the EOS)
    y = torch.tensor([[11, 12, EOS, 21, 22, 23, 24]])
    mtp_y, valid = mtp_targets(y, None, EOS, k=2)
    assert valid.tolist() == [[True, False, False, True, True, False, False]]
    # position 0 learns the EOS itself, which is in document A
    assert mtp_y[0, 0].item() == EOS


def test_depth_two_token_mask_of_the_target_decides():
    y = torch.tensor([[11, 12, 13, 14, 15]])
    mask = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.bool)
    mtp_y, valid = mtp_targets(y, mask, EOS, k=2)
    assert valid.tolist() == [[True, True, True, False, False]]
    assert mtp_y[valid].tolist() == [13, 14, 15]


def test_depth_one_is_the_default():
    y = torch.tensor([[11, EOS, 13, 14, 15]])
    mask = torch.tensor([[1, 1, 0, 1, 1]], dtype=torch.bool)
    for a, b in zip(mtp_targets(y, mask, EOS), mtp_targets(y, mask, EOS, k=1)):
        assert torch.equal(a, b)
