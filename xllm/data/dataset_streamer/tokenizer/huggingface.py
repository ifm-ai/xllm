import os
import json
import re
from logging import getLogger
from typing import Any, Dict, List, Optional

from xllm.data.dataset_streamer.tokenizer.tokenizer import Tokenizer
from xllm.config import TokenizerConf

logger = getLogger()

# A non-special added token that looks like a template or tool marker ("<think>", "</tool_call>", "<|python_tag|>",
# "[TOOL_CALLS]"). Raw text never yields those either. Non-special added tokens that are ordinary text pieces (runs
# of spaces, words) do not match and keep encoding as before.
_CONTROL_MARKER = re.compile(r"<[^<>\s]+>|\[/?[A-Z][A-Z0-9_]*\]")


def _raw_text_backend(hf_tokenizer):
    """A private copy of the Rust backend in which raw text cannot match a control token.

    In the copy every special added token, and every non-special added token that looks like a control marker, is
    marked special, and the copy encodes with encode_special_tokens=True, so those strings go through the normalizer,
    pre-tokenizer and model like any other text. Marking instead of removing keeps every added token's id. The shared
    backend is never touched: transformers' split_special_tokens=True sets encode_special_tokens on the shared
    backend for each call, which races with apply_chat_template when tokenizer worker threads share one tokenizer,
    and it leaves non-special markers matched.
    """
    from tokenizers import Tokenizer as Backend

    special = set(hf_tokenizer.all_special_tokens)
    special.update(t.content for t in hf_tokenizer.added_tokens_decoder.values() if t.special)
    spec = json.loads(hf_tokenizer.backend_tokenizer.to_str())
    controls, markers = {}, []
    for token in spec.get("added_tokens", []):
        if token["special"] or token["content"] in special or _CONTROL_MARKER.fullmatch(token["content"]):
            if not token["special"]:
                markers.append(token["content"])
            token["special"] = True
            controls[token["content"]] = token["id"]
    spec["truncation"] = None
    spec["padding"] = None
    backend = Backend.from_str(json.dumps(spec))
    backend.encode_special_tokens = True
    # tokenizers < 0.15.1 accepts the attribute and ignores it; check that the copy really splits the markers.
    # A marker that is also a base-vocabulary entry is skipped: the model itself may emit it, with or without the flag.
    base_vocab = backend.get_vocab(with_added_tokens=False)
    contents = [c for c in controls if c not in base_vocab]
    encoded = backend.encode_batch(contents, add_special_tokens=False)
    matched = [c for c, e in zip(contents, encoded) if e.ids == [controls[c]]]
    if matched:
        raise EnvironmentError(
            f"raw text still matches {len(matched)} control tokens (e.g. {matched[0]!r}) with the tokenizers "
            "library in this environment; encode_special_tokens needs tokenizers >= 0.15.1"
        )
    if markers:
        logger.info(f"raw text encodes {len(markers)} non-special added markers as plain text: {markers[:8]}")
    return backend


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

        # raw text gets its own backend; apply_chat_template keeps the shared one with its control tokens
        self._raw_backend = _raw_text_backend(self._tokenizer) if self._tokenizer.is_fast else None
        if self._raw_backend is None:
            logger.warning(
                "slow tokenizer: raw text splits special tokens, but non-special added markers still match"
            )

        logger.info(
            f"#words: {self.vocab_size} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id} - PAD ID: {self.pad_id}"
        )

    def _encode(self, text: str) -> List[int]:
        # raw text never yields control tokens: a quoted "<|...|>" or "<think>" is ordinary text here
        if self._raw_backend is not None:
            return self._raw_backend.encode(text, add_special_tokens=False).ids
        # slow tokenizers read split_special_tokens per call, without shared state
        return self._tokenizer(text, add_special_tokens=False, split_special_tokens=True).input_ids

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
