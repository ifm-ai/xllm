import copy
import json
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from xllm.config import DataLoaderConfig
from xllm.data.build import build_data_loader
from xllm.data.buffer_assembler.packer.packer import BestFitPacker, SimpleConcatPacker
from xllm.data.dataloader import MultiSourceDataLoader
from xllm.data.data_types import PackedSeq


class _Tokenizer:
    bos_id = 1
    eos_id = 2
    pad_id = 0

    def encode(self, text, bos, eos):
        tokens = [10 + ord(char) % 100 for char in text]
        if bos:
            tokens.insert(0, self.bos_id)
        if eos:
            tokens.append(self.eos_id)
        return tokens


def _build_loader(data_dir, packing_type="bestfit"):
    loader = MultiSourceDataLoader(
        tokenizer=_Tokenizer(),
        data_mix_str=f"{data_dir}:1.0:text:text",
        seq_len=4,
        batch_size=1,
        num_buffered_seq=2,
        world_rank=0,
        world_size=1,
        packing_type=packing_type,
        num_workers=1,
    )
    loader.assembler._sync_refill = True
    return loader


@pytest.mark.parametrize("packing_type, packer_class", [
    ("simple", SimpleConcatPacker),
    ("bestfit", BestFitPacker),
])
def test_build_data_loader_selects_packing_type(tmp_path, packing_type, packer_class):
    (tmp_path / "data.chunk0.jsonl").write_text(
        json.dumps({"text": "abcdef"}) + "\n", encoding="utf-8"
    )
    cfg = SimpleNamespace(
        data=f"{tmp_path}:1.0:text:text",
        seq_len=4,
        batch_size=1,
        data_parallel_rank=0,
        data_parallel_size=1,
        dataloader=DataLoaderConfig(buffer_size=2, packing_type=packing_type),
    )
    loader = build_data_loader(_Tokenizer(), cfg)
    try:
        assert isinstance(loader.packer, packer_class)
        batch = next(loader)
        assert batch.x.shape == batch.y.shape == batch.mask.shape == (1, 4)
    finally:
        loader.close()


@pytest.mark.parametrize("packing_type", ["best_fit", "unknown"])
def test_invalid_packing_type_reports_supported_names(tmp_path, packing_type):
    with pytest.raises(ValueError, match="Supported: 'simple', 'bestfit'"):
        _build_loader(tmp_path, packing_type=packing_type)


@pytest.mark.parametrize("overrides", [
    {},
    dict(buffer_size=7, num_workers=2, packing_type="bestfit",
         skip_long_docs=True, max_consecutive_skips=23),
])
def test_build_data_loader_preserves_config_values(monkeypatch, overrides):
    import xllm.data.build as build_module

    data_cfg = DataLoaderConfig(**overrides)
    cfg = SimpleNamespace(
        data="/data/text:1:text:text", seq_len=8, batch_size=2,
        data_parallel_rank=1, data_parallel_size=4, dataloader=data_cfg,
    )
    monkeypatch.setattr(build_module, "MultiSourceDataLoader", lambda **kwargs: kwargs)
    tokenizer = _Tokenizer()
    kwargs = build_data_loader(tokenizer, cfg)
    assert kwargs == dict(
        tokenizer=tokenizer, data_mix_str=cfg.data, seq_len=8, batch_size=2,
        num_buffered_seq=data_cfg.buffer_size, world_rank=1, world_size=4,
        packing_type=data_cfg.packing_type, num_workers=data_cfg.num_workers,
        max_consecutive_skips=data_cfg.max_consecutive_skips,
        skip_long_docs=data_cfg.skip_long_docs,
    )


@pytest.mark.parametrize("value, expected", [("false", False), ("true", True)])
def test_dataloader_cli_parses_boolean(value, expected):
    parser = DataLoaderConfig.to_cli()
    cfg = DataLoaderConfig.from_cli(vars(parser.parse_args([
        "--packing_type", "bestfit", "--skip_long_docs", value,
    ])))
    assert cfg.packing_type == "bestfit"
    assert cfg.skip_long_docs is expected


def _assert_batch_equal(expected, actual):
    assert np.array_equal(expected.x, actual.x)
    assert np.array_equal(expected.y, actual.y)
    if expected.mask is None or actual.mask is None:
        assert expected.mask is actual.mask
    else:
        assert np.array_equal(expected.mask, actual.mask)
    assert expected.src_infos == actual.src_infos


def test_pretraining_batch_materializes_all_true_token_mask():
    loader = MultiSourceDataLoader.__new__(MultiSourceDataLoader)
    loader.seq_len = 4
    loader.batch_size = 1
    loader._steps = 0
    packed = PackedSeq(
        x=np.array([1, 2, 3, 4], dtype=np.int64),
        y=np.array([2, 3, 4, 5], dtype=np.int64),
    )
    loader.assembler = type("Assembler", (), {"pop_packed": lambda self: packed})()

    batch = next(loader)

    assert batch.mask is not None
    assert batch.mask.dtype == np.bool_
    assert batch.mask.shape == batch.y.shape
    assert batch.mask.all()


def test_resume_replays_refill_with_tokenized_pending_chunks():
    with tempfile.TemporaryDirectory() as data_dir:
        data_path = os.path.join(data_dir, "data.chunk0.jsonl")
        with open(data_path, "w", encoding="utf-8") as handle:
            for _ in range(32):
                handle.write(json.dumps({"text": "abcdef"}) + "\n")

        loader = _build_loader(data_dir)
        restored = None
        try:
            for _ in range(3):
                next(loader)

            state = copy.deepcopy(loader.get_state())
            pending = state["assembler"]["active_packer_state"]["pending_chunks"]
            assert pending
            assert isinstance(pending[0][0], list)

            expected = [next(loader) for _ in range(4)]

            restored = _build_loader(data_dir)
            restored.set_state(state)
            actual = [next(restored) for _ in range(4)]

            for expected_batch, actual_batch in zip(expected, actual):
                _assert_batch_equal(expected_batch, actual_batch)
        finally:
            loader.close()
            if restored is not None:
                restored.close()


def test_sync_refill_does_not_overwrite_ready_buffer():
    with tempfile.TemporaryDirectory() as data_dir:
        data_path = os.path.join(data_dir, "data.chunk0.jsonl")
        with open(data_path, "w", encoding="utf-8") as handle:
            for idx in range(32):
                handle.write(json.dumps({"text": f"{idx:06d}"}) + "\n")

        sync_loader = _build_loader(data_dir)
        async_loader = _build_loader(data_dir)
        async_loader.assembler._sync_refill = False
        try:
            for _ in range(8):
                _assert_batch_equal(next(async_loader), next(sync_loader))
        finally:
            sync_loader.close()
            async_loader.close()


def test_only_first_refill_total_time_is_logged():
    old_enabled = os.environ.get("XLLM_LOG_DATALOADER_TIMING")
    old_dir = os.environ.get("XLLM_DATALOADER_TIMING_DIR")
    with tempfile.TemporaryDirectory() as root:
        data_dir = os.path.join(root, "data")
        timing_dir = os.path.join(root, "timing")
        os.makedirs(data_dir)
        data_path = os.path.join(data_dir, "data.chunk0.jsonl")
        with open(data_path, "w", encoding="utf-8") as handle:
            for idx in range(32):
                handle.write(json.dumps({"text": f"{idx:06d}"}) + "\n")

        os.environ["XLLM_LOG_DATALOADER_TIMING"] = "1"
        os.environ["XLLM_DATALOADER_TIMING_DIR"] = timing_dir
        loader = None
        try:
            loader = _build_loader(data_dir)
            for _ in range(8):
                next(loader)
        finally:
            if loader is not None:
                loader.close()
            if old_enabled is None:
                os.environ.pop("XLLM_LOG_DATALOADER_TIMING", None)
            else:
                os.environ["XLLM_LOG_DATALOADER_TIMING"] = old_enabled
            if old_dir is None:
                os.environ.pop("XLLM_DATALOADER_TIMING_DIR", None)
            else:
                os.environ["XLLM_DATALOADER_TIMING_DIR"] = old_dir

        with open(os.path.join(timing_dir, "rank_00000.jsonl"), encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle]

    assert len(records) == 1
    assert set(records[0]) == {
        "event",
        "loader",
        "total_seconds",
        "num_tokenizer_workers",
        "world_rank",
        "world_size",
    }
    assert records[0]["event"] == "first_refill"
    assert records[0]["total_seconds"] >= 0
