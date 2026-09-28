import os
from logging import getLogger
from typing import List, Union, Dict, Iterator, cast
from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe

from xllm.data.dataset_streamer.tokenizer.tokenizer import Tokenizer
from xllm.config import TokenizerConf

logger = getLogger()

TIKTOKEN_MAX_ENCODE_CHARS = 400_000
MAX_NO_WHITESPACES_CHARS = 25_000


class Llama3Tokenizer(Tokenizer):
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, tokenizer_cfg: TokenizerConf):
        super().__init__(tokenizer_cfg)

        # reload tokenizer
        model_path = tokenizer_cfg.tokenizer_path
        assert os.path.isfile(model_path), model_path

        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
                             "<|begin_of_text|>",
                             "<|end_of_text|>",
                             "<|reserved_special_token_0|>",
                             "<|reserved_special_token_1|>",
                             "<|reserved_special_token_2|>",
                             "<|reserved_special_token_3|>",
                             "<|start_header_id|>",
                             "<|end_header_id|>",
                             "<|reserved_special_token_4|>",
                             "<|eot_id|>",  # end of turn
                         ] + [
                             f"<|reserved_special_token_{i}|>"
                             for i in range(5, self.num_reserved_special_tokens - 5)
                         ]

        self.special_tokens = {token: num_base_tokens + i for i, token in enumerate(special_tokens)}
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        logger.info(f"Reloaded tiktoken model from {model_path}")

        # BOS / EOS token IDs
        self.vocab_size = self.model.n_vocab
        self._bos_id = self.special_tokens["<|begin_of_text|>"]
        self._eos_id = self.special_tokens["<|end_of_text|>"]
        self._pad_id = -1
        assert self._bos_id is not None
        assert self._eos_id is not None

        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }
        logger.info(
            f"#words: {self.vocab_size} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id} - PAD ID: {self.pad_id}"
        )

    def encode(self, s: Union[str, List[int]], bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i: i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(self.model.encode(substr, allowed_special='all'))

        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)

        return t

    def decode(self, tokens: List[int], cut_at_eos: bool = True, remove_prefix_bos: bool = True) -> str:
        if cut_at_eos:
            tokens = self.cut_at_eos(tokens)
        if remove_prefix_bos:
            tokens = self.remove_prefix_bos(tokens)

        return self.model.decode(cast(List[int], tokens))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(s: str, max_consecutive_slice_len: int) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def pad_id(self) -> int:
        return self._pad_id
