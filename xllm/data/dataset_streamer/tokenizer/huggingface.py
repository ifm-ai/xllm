import os
import json
from logging import getLogger
from typing import Any, Dict, List, Union

from xllm.data.dataset_streamer.tokenizer.tokenizer import Tokenizer
from xllm.config import TokenizerConf

logger = getLogger()


class HuggingFaceTokenizer(Tokenizer):
    def __init__(self, tokenizer_cfg: TokenizerConf):
        super().__init__(tokenizer_cfg)
        try:
            import transformers
        except ImportError:
            raise EnvironmentError(
                f"The transformers library must be installed to use huggingface tokenizer"
            )

        model_path = tokenizer_cfg.tokenizer_path
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
        if tokenizer_cfg.template_dict_json:
            template_dict = json.loads(tokenizer_cfg.template_dict_json)
            self._tokenizer.chat_template = {k: open(os.path.join(model_path, v)).read() for k, v in template_dict.items()}
        logger.info(f"Reloaded huggingface tokenizer from {model_path}")

        # BOS / EOS token IDs
        self.vocab_size = len(self._tokenizer)
        self._bos_id = self._tokenizer.bos_token_id
        self._eos_id = self._tokenizer.eos_token_id
        self._pad_id = -1 if self._tokenizer.pad_token_id is None else self._tokenizer.pad_token_id

        assert self._bos_id is not None
        assert self._eos_id is not None

        logger.info(
            f"#words: {self.vocab_size} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id} - PAD ID: {self.pad_id}"
        )

    def encode(self, s: Union[str, List[int]], bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self._tokenizer(s, add_special_tokens=False).input_ids

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

        return self._tokenizer.decode(tokens)

    def apply_chat_template(self, conversation, **kwargs):
        return self._tokenizer.apply_chat_template(conversation, **kwargs)

    def convert_ids_to_tokens(self, *args, **kwargs):
        return self._tokenizer.convert_ids_to_tokens(*args, **kwargs)
        
    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def pad_id(self) -> int:
        return self._pad_id

    def get_state(self) -> Dict[str, Any]:
        return {}

    def set_state(self, state: Dict[str, Any]) -> None:
        pass
