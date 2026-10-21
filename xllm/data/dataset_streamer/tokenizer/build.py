from xllm.config import TokenizerConf
from xllm.data.dataset_streamer.tokenizer.tokenizer import Tokenizer
from xllm.data.dataset_streamer.tokenizer.sentencepiece import SentencePieceTokenizer
from xllm.data.dataset_streamer.tokenizer.llama3 import Llama3Tokenizer
from xllm.data.dataset_streamer.tokenizer.huggingface import HuggingFaceTokenizer


def build_tokenizer(cfg: TokenizerConf) -> Tokenizer:
    if cfg.type in ['llama2', 'sentencepiece']:
        return SentencePieceTokenizer(cfg)
    elif cfg.type == 'llama3':
        return Llama3Tokenizer(cfg)
    elif cfg.type == 'huggingface':
        return HuggingFaceTokenizer(cfg)
    else:
        raise ValueError(f"Unknown Tokenizer type: {cfg.type}")
