from sentencepiece import SentencePieceProcessor
from logging import getLogger
from typing import List, Optional

from xllm.data.dataset_streamer.tokenizer.tokenizer import Tokenizer
from xllm.config import TokenizerConf
import os


logger = getLogger()


class SentencePieceTokenizer(Tokenizer):
    """
    Tokenizing and encoding/decoding text using sentencepiece.
    """

    def __init__(self, cfg: TokenizerConf):
        super().__init__(cfg)

        # reload tokenizer
        model_path = cfg.tokenizer_path
        assert os.path.isfile(model_path), model_path
        self.model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        self.num_reserved_special_tokens = cfg.num_reserved_special_tokens
        self.sp_model_vocab_size = self.model.vocab_size()
        self.used_special_tokens = 0

        assert cfg.num_reserved_special_tokens % 256 == 0, \
            f"num. reserved special tokens is not multiple of 256: {cfg.num_reserved_special_tokens}"

        # BOS / EOS token IDs
        assert self.model.vocab_size() == self.model.get_piece_size()
        self.vocab_size = self.model.vocab_size() + cfg.num_reserved_special_tokens
        logger.info(
            f"#words: {self.vocab_size} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id} - PAD ID: {self.pad_id}"
        )

    def _encode(self, text: str) -> List[int]:
        tokens = self.model.encode(text)
        return tokens

    def _decode(self, tokens: List[int]) -> str:
        return self.model.decode(tokens)

    @property
    def bos_id(self) -> int:
        if self._bos_id is not None:
            return self._bos_id

        if self.model.bos_id() != -1 or self.num_reserved_special_tokens == 0:
            self._bos_id = self.model.bos_id()
            return self._bos_id

        bos_id = self.sp_model_vocab_size + self.used_special_tokens
        self.used_special_tokens += 1
        self._bos_id = bos_id
        return bos_id

    @property
    def eos_id(self) -> int:
        if self._eos_id is not None:
            return self._eos_id

        if self.model.eos_id() != -1 or self.num_reserved_special_tokens == 0:
            self._eos_id = self.model.eos_id()
            return self._eos_id

        eos_id = self.sp_model_vocab_size + self.used_special_tokens
        self.used_special_tokens += 1
        self._eos_id = eos_id
        return eos_id

    @property
    def pad_id(self) -> int:
        if self._pad_id is not None:
            return self._pad_id

        if self.model.pad_id() != -1 or self.num_reserved_special_tokens == 0:
            self._pad_id = self.model.pad_id()
            return self._pad_id

        pad_id = self.sp_model_vocab_size + self.used_special_tokens
        self.used_special_tokens += 1
        self._pad_id = pad_id
        return pad_id
