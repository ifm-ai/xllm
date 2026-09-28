import hashlib
from typing import Optional

import numpy as np

from xllm.data.data_types import Instance
from xllm.data.dataset_streamer.templator.templator import SimpleJsonlTemplator
from xllm.data.dataset_streamer.templator.templator import (
    ChatTemplateError,
    IGNORE_INDEX,
    tokenize_multiturn_template,
)


class FeatureBuilder:
    def build(
        self,
        instance: Instance,
        tokenizer: "Tokenizer",
    ) -> Optional[Instance]:
        raise NotImplementedError


class DefaultFeatureBuilder(FeatureBuilder):
    """Default feature builder for text/pretraining data."""

    def __init__(self, templator: SimpleJsonlTemplator):
        self.templator = templator

    def build(
        self,
        instance: Instance,
        tokenizer: "Tokenizer",
    ) -> Instance:
        text = self.templator.render(instance.raw_data)
        instance.tokens = tokenizer.encode(text, bos=True, eos=True)
        instance.target_mask = None
        return instance


class ChatTemplateFeatureBuilder(FeatureBuilder):
    """Use the tokenizer's HF chat template, matching the old dl1 SFT path."""

    def __init__(self, text_format: str = "chat_assistant", conversation_key: str = "conversation"):
        self.text_format = text_format
        self.conversation_key = conversation_key

    def build(
        self,
        instance: Instance,
        tokenizer: "Tokenizer",
    ) -> Optional[Instance]:
        try:
            result = tokenize_multiturn_template(
                instance.raw_data,
                tokenizer,
                text_format=self.text_format,
                conversation_key=self.conversation_key,
                rng=self._rng_for_instance(instance),
            )
        except ChatTemplateError:
            return None

        token_ids = list(result["input_ids"])
        labels = list(result["labels"])
        target_mask = [label != IGNORE_INDEX for label in labels]
        instance.tokens = token_ids
        instance.target_mask = None if all(target_mask) else target_mask
        return instance

    def _rng_for_instance(self, instance: Instance) -> np.random.RandomState:
        seed_material = (
            f"{self.text_format}\0{instance.filename}\0{instance.file_pos}\0"
            f"{instance.line_num}\0{instance.repetition}\0{self.conversation_key}"
        ).encode("utf-8", errors="surrogatepass")
        digest = hashlib.blake2b(seed_material, digest_size=16).digest()
        seed = np.frombuffer(digest, dtype=np.uint32)
        return np.random.RandomState(seed)
