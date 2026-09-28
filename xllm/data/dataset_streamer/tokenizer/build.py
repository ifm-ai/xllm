from xllm.config import TokenizerConf
from xllm.data.dataset_streamer.tokenizer.tokenizer import Tokenizer
from xllm.data.dataset_streamer.tokenizer.sentencepiece import SentencePieceTokenizer
from xllm.data.dataset_streamer.tokenizer.llama3 import Llama3Tokenizer
from xllm.data.dataset_streamer.tokenizer.huggingface import HuggingFaceTokenizer


def build_tokenizer(tokenizer_cfg: TokenizerConf) -> Tokenizer:
    if tokenizer_cfg.type in ['llama2', 'sentencepiece']:
        return SentencePieceTokenizer(tokenizer_cfg)
    elif tokenizer_cfg.type == 'llama3':
        return Llama3Tokenizer(tokenizer_cfg)
    elif tokenizer_cfg.type == 'huggingface':
        return HuggingFaceTokenizer(tokenizer_cfg)
    else:
        raise ValueError(f"Unknown Tokenizer type: {tokenizer_cfg.type}")
