import os
import json
from logging import getLogger
from typing import Any, Dict, List, Optional

from xllm.data.dataset_streamer.tokenizer.tokenizer import Tokenizer
from xllm.config import TokenizerConf

logger = getLogger()


class HuggingFaceTokenizer(Tokenizer):
    def __init__(self, cfg: TokenizerConf):
        super().__init__(cfg)
        try:
            import transformers
        except ImportError:
            raise EnvironmentError(
                f"The transformers library must be installed to use huggingface tokenizer"
            )

        model_path = cfg.tokenizer_path
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
        if cfg.template_dict_json:
            template_dict = json.loads(cfg.template_dict_json)
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

    def _encode(self, text: str) -> List[int]:
        tokens = self._tokenizer(text, add_special_tokens=False).input_ids
        return tokens

    def _decode(self, tokens: List[int]) -> str:
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
