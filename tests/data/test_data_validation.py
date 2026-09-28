import numpy as np

from xllm.data.buffer_assembler.buffer_assembler import BufferAssembler
from xllm.data.buffer_assembler.packer.packer import SimpleConcatPacker
from xllm.data.data_types import Instance
from xllm.data.dataset_streamer.feature_builder.feature_builder import ChatTemplateFeatureBuilder
from xllm.data.dataset_streamer.templator.templator import SimpleJsonlTemplator


def _assert_raises(error_type, fn):
    try:
        fn()
    except error_type as error:
        return error
    except Exception as error:
        raise AssertionError(f"Expected {error_type.__name__}, got {type(error).__name__}") from error
    raise AssertionError(f"Expected {error_type.__name__}")


def test_text_templator_requires_configured_string_key():
    templator = SimpleJsonlTemplator("text")

    assert templator.render({"text": "hello"}) == "hello"
    _assert_raises(KeyError, lambda: templator.render({"content": "hello"}))
    _assert_raises(TypeError, lambda: templator.render({"text": [1, 2, 3]}))


class _FailingChatTokenizer:
    bos_id = 1
    eos_id = 2

    def apply_chat_template(self, conversation, **kwargs):
        raise ValueError("invalid conversation")


def test_chat_template_failure_skips_the_bad_record():
    builder = ChatTemplateFeatureBuilder("chat_assistant", conversation_key="conversation")
    instance = Instance(
        raw_data={"conversation": [{"role": "user", "content": "hello"}]},
        filename="chat.jsonl",
        file_pos=0,
        line_num=7,
    )

    assert builder.build(instance, _FailingChatTokenizer()) is None


def test_malformed_chat_record_is_skipped_before_tokenization():
    builder = ChatTemplateFeatureBuilder("chat_assistant", conversation_key="conversation")
    instance = Instance(
        raw_data={"conversation": [{"content": "missing role"}]},
        filename="chat.jsonl",
        file_pos=0,
        line_num=8,
    )

    assert builder.build(instance, _FailingChatTokenizer()) is None


class _AlwaysSkippingStreamer:
    def iter_raw(self):
        line_num = 0
        while True:
            yield Instance({}, "bad.jsonl", line_num, line_num)
            line_num += 1

    def build_features(self, instance):
        return None

    def close(self):
        pass


class _FixedTokenStreamer:
    def iter_raw(self):
        line_num = 0
        while True:
            yield Instance({}, "fixed.jsonl", line_num, line_num)
            line_num += 1

    def build_features(self, instance):
        instance.tokens = [10, 11, 12]
        return instance

    def close(self):
        pass


class _CapturePacker:
    def __init__(self):
        self.prefilled = []

    def prefill(self, ds, instance):
        self.prefilled.append((ds, instance.line_num))

    def pack(self):
        return []

    def get_state(self):
        return {}

    def set_state(self, state):
        pass


def test_refill_budget_counts_all_document_tokens():
    streamer = _FixedTokenStreamer()
    assembler = BufferAssembler(
        dataset_streamers={"fixed": streamer},
        dataset_weights={"fixed": 1.0},
        packer=SimpleConcatPacker(seq_len=4),
        seq_len=4,
        num_buffered_seq=1,
    )
    try:
        instances, tokens_pushed = assembler._tokenize_source_to_budget(
            "fixed", target_tokens=5
        )
        assert len(instances) == 2
        assert tokens_pushed == 6
    finally:
        assembler.close()


def test_refill_shuffles_documents_before_packing():
    packer = _CapturePacker()
    assembler = BufferAssembler(
        dataset_streamers={"a": _FixedTokenStreamer(), "b": _FixedTokenStreamer()},
        dataset_weights={"a": 0.5, "b": 0.5},
        packer=packer,
        seq_len=4,
        num_buffered_seq=2,
        world_rank=0,
        world_size=1,
    )
    try:
        assembler._refill()

        expected = [("a", 0), ("a", 1), ("b", 0), ("b", 1)]
        replay_rng = np.random.RandomState()
        replay_rng.set_state(assembler._inactive_rng_state)
        replay_rng.shuffle(expected)
        assert packer.prefilled == expected
    finally:
        assembler.close()


def test_refill_fails_after_consecutive_feature_skips():
    streamer = _AlwaysSkippingStreamer()
    assembler = BufferAssembler(
        dataset_streamers={"bad": streamer},
        dataset_weights={"bad": 1.0},
        packer=SimpleConcatPacker(seq_len=4),
        seq_len=4,
        num_buffered_seq=1,
        max_consecutive_skips=3,
    )
    try:
        error = _assert_raises(
            RuntimeError,
            lambda: assembler._tokenize_source_to_budget("bad", target_tokens=4),
        )
        assert "3 consecutive records" in str(error)
    finally:
        assembler.close()
