import os
import json
import threading
from logging import getLogger
from typing import Any, Dict, List, Optional

from xllm.data.dataset_streamer.tokenizer.tokenizer import Tokenizer
from xllm.config import TokenizerConf

logger = getLogger()

# Used only when per-thread templates are not available; see
# HuggingFaceTokenizer.apply_chat_template for the full explanation.
_CHAT_TEMPLATE_LOCK = threading.Lock()


def _per_thread_templates_supported() -> bool:
    """Return True if transformers builds a separate jinja Environment (and therefore a
    separate AssistantTracker) for every distinct template string it compiles.

    apply_chat_template relies on this to avoid sharing one tracker across threads.
    """
    try:
        from transformers.utils.chat_template_utils import _compile_jinja_template
        a = _compile_jinja_template("{# probe-a #}")
        b = _compile_jinja_template("{# probe-b #}")
        return a.environment is not b.environment
    except Exception:
        return False


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

        self._per_thread_templates = _per_thread_templates_supported()
        logger.info(
            "Chat template rendering: "
            + ("per-thread compiled templates" if self._per_thread_templates else "global lock (fallback)")
        )

    def _encode(self, text: str) -> List[int]:
        tokens = self._tokenizer(text, add_special_tokens=False).input_ids
        return tokens

    def _decode(self, tokens: List[int]) -> str:
        return self._tokenizer.decode(tokens)

    def apply_chat_template(self, conversation, **kwargs):
        """Thread-safe wrapper around transformers' apply_chat_template.

        Why: transformers compiles chat templates in `_compile_jinja_template`, which is
        wrapped in lru_cache keyed by the template string. Each compilation creates its own
        jinja Environment with a stateful AssistantTracker (used when
        return_assistant_tokens_mask=True). Because of the cache, all threads rendering the
        same template share ONE tracker; concurrent rendering then raises
        "AssistantTracker should not be reused before closed" (or a NoneType error). The
        templator used to turn that into a silently skipped sample, so the skipped samples
        depended on thread timing and ranks in the same TP group got different batches.

        Fix: give each thread its own compiled template by appending a per-thread jinja
        comment. Comments render to nothing, so the output is unchanged, but the cache key
        differs, so each thread gets its own Environment/tracker and tokenization stays
        parallel. If transformers ever stops creating one Environment per compiled template
        (checked once in __init__), we fall back to a global lock: correct, but slower.
        """
        name = kwargs.get("chat_template")
        templates = self._tokenizer.chat_template
        if self._per_thread_templates and isinstance(templates, dict) and name in templates:
            kwargs["chat_template"] = templates[name] + "{# thread-%d #}" % threading.get_ident()
            return self._tokenizer.apply_chat_template(conversation, **kwargs)
        # Fallback: serialize rendering (template given by raw string / no template dict,
        # or per-thread templates unsupported by this transformers version).
        with _CHAT_TEMPLATE_LOCK:
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
