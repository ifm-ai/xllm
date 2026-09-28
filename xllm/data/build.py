from xllm.config import TrainerConf
from xllm.data.dataloader import MultiSourceDataLoader
from xllm.data.dataset_streamer.tokenizer.tokenizer import Tokenizer


def build_data_loader(tokenizer: Tokenizer, cfg: TrainerConf) -> MultiSourceDataLoader:
    """
    Build a MultiSourceDataLoader from TrainerConf.

    Config fields consumed
    ----------------------
    cfg.data                : weighted source string; each source is path:weight:json_key:source_format
    cfg.seq_len             : sequence length
    cfg.batch_size          : batch size per data-parallel rank, resolved by train.py
    cfg.data_parallel_rank  : current rank index
    cfg.data_parallel_size  : total number of data-parallel ranks
    cfg.dataloader.buffer_size   : refill token-budget multiplier; target tokens ~= buffer_size * seq_len
    cfg.dataloader.packing_type  : "simple" | "bestfit"
    cfg.dataloader.num_workers   : source reader/tokenizer worker threads
    cfg.dataloader.max_consecutive_skips : invalid records allowed before a source fails
    cfg.dataloader.skip_long_docs : skip overlong non-text records with bestfit

    Defaults come from DataLoaderConfig; this builder does not override them.
    """

    dataloader_cfg = cfg.dataloader
    num_buffered_seq: int = dataloader_cfg.buffer_size
    packing_type: str = dataloader_cfg.packing_type
    num_workers: int = dataloader_cfg.num_workers
    max_consecutive_skips: int = dataloader_cfg.max_consecutive_skips
    skip_long_docs: bool = dataloader_cfg.skip_long_docs
    return MultiSourceDataLoader(
        tokenizer=tokenizer,
        data_mix_str=cfg.data,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        num_buffered_seq=num_buffered_seq,
        world_rank=cfg.data_parallel_rank,
        world_size=cfg.data_parallel_size,
        packing_type=packing_type,
        num_workers=num_workers,
        max_consecutive_skips=max_consecutive_skips,
        skip_long_docs=skip_long_docs,
    )
