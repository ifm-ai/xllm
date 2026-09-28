from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, Union
from xllm.config import TokenizerConf

logger = getLogger()


class Tokenizer:
    def __init__(self, tokenizer_cfg: TokenizerConf):
        # BOS / EOS token IDs
        self._bos_id: Optional[int] = None
        self._eos_id: Optional[int] = None
        self._pad_id: Optional[int] = None
        self.vocab_size: Optional[int] = None

    def encode(self, s: Union[str, List[int]], bos: bool, eos: bool) -> List[int]:
        raise NotImplemented

    def decode(self, tokens: List[int], cut_at_eos: bool = True, remove_prefix_bos: bool = True) -> str:
        raise NotImplemented

    def apply_chat_template(self, conversation, **kwargs):
        raise NotImplemented

    def convert_ids_to_tokens(self, *args, **kwargs):
        raise NotImplemented
    
    @property
    def bos_id(self) -> int:
        raise NotImplemented

    @property
    def eos_id(self) -> int:
        raise NotImplemented

    @property
    def pad_id(self) -> int:
        raise NotImplemented

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
        raise NotImplemented
