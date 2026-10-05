from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple
from xllm.config import TokenizerConf

logger = getLogger()


class Tokenizer:
    def __init__(self, cfg: TokenizerConf):
        # BOS / EOS token IDs
        self._bos_id: Optional[int] = None
        self._eos_id: Optional[int] = None
        self._pad_id: Optional[int] = None
        self.vocab_size: Optional[int] = None
        self.use_bos = cfg.use_bos

    def _add_bos_eos(self, tokens: List[int], bos: Optional[bool], eos: bool):
        # add BOS
        bos = self.use_bos if bos is None else bos
        if bos:
            tokens.insert(0, self.bos_id)
        # add EOS
        if eos:
            tokens.append(self.eos_id)

        return tokens

    def _encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def _decode(self, tokens: List[int]) -> str:
        raise NotImplementedError

    def encode(self, text: str, bos: Optional[bool], eos: bool) -> List[int]:
        tokens = self._encode(text)
        tokens = self._add_bos_eos(tokens, bos, eos)
        return tokens

    def decode(self, tokens: List[int], cut_at_eos: bool = True, remove_prefix_bos: bool = True) -> str:
        if cut_at_eos:
            tokens = self.cut_at_eos(tokens)
        if remove_prefix_bos:
            tokens = self.remove_prefix_bos(tokens)
        return self._decode(tokens)

    def apply_chat_template(self, conversation, **kwargs):
        raise NotImplementedError

    def convert_ids_to_tokens(self, *args, **kwargs):
        raise NotImplementedError
    
    @property
    def bos_id(self) -> int:
        raise NotImplementedError

    @property
    def eos_id(self) -> int:
        raise NotImplementedError

    @property
    def pad_id(self) -> int:
        raise NotImplementedError

    def cut_at_eos(self, tokens: List[int]) -> List[int]:
        for k, t in enumerate(tokens):
            if t == self.eos_id:
                tokens = tokens[:k]
                break

        return tokens

    def remove_prefix_bos(self, tokens: List[int]) -> List[int]:
        s = 0
        for k, t in enumerate(tokens):
            s = k
            if t != self.bos_id:
                break

        return tokens[s:]

    def get_state(self) -> Dict[str, Any]:
        return {}

    def set_state(self, state: Dict[str, Any]) -> None:
        pass

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        raise NotImplementedError
