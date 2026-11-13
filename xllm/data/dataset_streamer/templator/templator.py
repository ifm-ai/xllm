import copy
import json
import os
from typing import Any, Dict, List, Optional

from xllm.data.dataset_streamer.tokenizer import Tokenizer


class Templator:
    def render(self, raw_record: Any) -> str:
        """Render a raw record into a single text string."""
        raise NotImplementedError


class SimpleJsonlTemplator(Templator):
    """
    Pretraining templator: extract a text field from a JSON record.
    All tokens contribute to the loss.
    """

    def __init__(self, content_key: str):
        self.content_key = content_key

    def render(self, raw_record: Any) -> str:
        if not isinstance(raw_record, dict):
            raise TypeError(f"Expected a JSON object, got {type(raw_record).__name__}")
        if self.content_key not in raw_record:
            raise KeyError(f"Missing required text key {self.content_key!r}")
        text = raw_record[self.content_key]
        if not isinstance(text, str):
            raise TypeError(
                f"Expected {self.content_key!r} to contain a string, got {type(text).__name__}"
            )
        return text

IGNORE_INDEX = -100


SYSTEM = (
    "You are K2, a helpful assistant created by Mohamed bin Zayed University of Artificial Intelligence "
    "(MBZUAI) Institute of Foundation Models (IFM)."
)

def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"Environment variable {name} is not set")
    return value

TOOL_PRESENTATION_FORMATS = json.loads(_require_env("TOOL_PRESENTATION_FORMATS"))
TOOL_CALL_FORMATS = json.loads(_require_env("TOOL_CALL_FORMATS"))


def _field(obj: Any, name: str) -> Any:
    if hasattr(obj, name):
        return getattr(obj, name)
    return obj[name]


def _repair_tool_schema(conversation: List[Dict[str, Any]]) -> None:
    if not conversation:
        return
    for tool in conversation[0].get("tools", []):
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        params = fn.get("parameters") if isinstance(fn, dict) else None
        if isinstance(params, dict):
            if params.get("required", False) is None:
                params.pop("required", None)
            props = params.get("properties")
            if isinstance(props, dict):
                for key, value in list(props.items()):
                    if value is None:
                        props[key] = {}


def _detect_think_key(conversation: List[Dict[str, Any]]) -> str:
    for turn in conversation:
        if turn.get("role") != "assistant":
            continue
        for key in ("think", "think_fast", "think_faster"):
            if key in turn:
                return key
    return "think"


class ChatTemplateError(RuntimeError):
    """A record could not be rendered by the configured chat template."""


def apply_chat_template_randomized(
    conversation: List[Dict[str, Any]],
    tokenizer: Tokenizer,
    text_format: str,
    rng: Optional[Any] = None,
) -> Any:
    """
    dl1-compatible HF chat-template path for BBQ mid data.

    The old dataloader called tokenizer.apply_chat_template for chat_assistant
    records so tokenizer templates could handle fields such as think/tools.
    Keep that behavior here instead of reconstructing ChatML by hand.
    """
    if rng is None:
        import numpy as np
        rng = np.random.RandomState()
    if not isinstance(conversation, list):
        raise ChatTemplateError(
            f"conversation must be a list, got {type(conversation).__name__}"
        )
    conversation = copy.deepcopy(conversation)
    if not conversation:
        raise ChatTemplateError("conversation is empty")
    for turn_idx, turn in enumerate(conversation):
        if not isinstance(turn, dict):
            raise ChatTemplateError(
                f"conversation turn {turn_idx} must be an object, got {type(turn).__name__}"
            )
        if not isinstance(turn.get("role"), str) or not turn["role"]:
            raise ChatTemplateError(f"conversation turn {turn_idx} has no valid role")

    tools = conversation[0].get("tools")
    if tools:
        if not isinstance(tools, list):
            raise ChatTemplateError(
                f"conversation tools must be a list, got {type(tools).__name__}"
            )
        if any(not isinstance(tool, dict) for tool in tools):
            raise ChatTemplateError("every conversation tool must be an object")

    do_insert_system_prompt = rng.random() < 0.0001
    tools_to_insert = []
    do_insert_tools = len(tools_to_insert) > 0

    if conversation[0]["role"] == "system":
        if do_insert_system_prompt and not conversation[0].get("content", False):
            conversation[0]["content"] = SYSTEM
        if do_insert_tools and not conversation[0].get("tools", False):
            conversation[0]["tools"] = tools_to_insert
    elif do_insert_system_prompt or do_insert_tools:
        new_system_message: Dict[str, Any] = {"role": "system"}
        if do_insert_system_prompt:
            new_system_message["content"] = SYSTEM
        if do_insert_tools:
            new_system_message["tools"] = tools_to_insert
        conversation.insert(0, new_system_message)

    has_tools = bool(conversation[0].get("tools", False))
    if has_tools:
        if rng.random() < 0.0001:
            for i, tool in enumerate(conversation[0]["tools"]):
                if not ("type" in tool and tool["type"] == "function"):
                    conversation[0]["tools"][i] = {"type": "function", "function": tool}
        tool_presentation_format = TOOL_PRESENTATION_FORMATS[rng.randint(len(TOOL_PRESENTATION_FORMATS))]
        tool_call_format = TOOL_CALL_FORMATS[rng.randint(len(TOOL_CALL_FORMATS))]
        kwargs = {
            "tool_presentation_format": tool_presentation_format,
            "tool_call_format": tool_call_format,
        }
    else:
        kwargs = {}

    try:
        return tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
            chat_template=text_format,
            **kwargs,
        )
    except Exception as initial_error:
        if not has_tools:
            raise ChatTemplateError(f"chat template rendering failed: {initial_error}") from initial_error
        _repair_tool_schema(conversation)
        try:
            return tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=False,
                return_assistant_tokens_mask=True,
                return_dict=True,
                chat_template=text_format,
                **kwargs,
            )
        except Exception as retry_error:
            raise ChatTemplateError(
                f"chat template rendering failed after tool-schema repair: {initial_error}"
            ) from retry_error


def tokenize_multiturn_template(
    sample: Dict[str, Any],
    tokenizer: Tokenizer,
    text_format: Optional[str] = None,
    conversation_key: str = "conversation",
    rng: Optional[Any] = None,
) -> Dict[str, Any]:
    if not isinstance(sample, dict):
        raise ChatTemplateError(f"Expected a JSON object, got {type(sample).__name__}")
    if conversation_key not in sample:
        raise ChatTemplateError(f"Missing required conversation key {conversation_key!r}")

    if text_format is None:
        conversations = sample[conversation_key]
        if "system" in sample and sample["system"] is not None:
            conversations = [{"role": "system", "content": sample["system"]}] + conversations
        tokenized_output = tokenizer.apply_chat_template(
            conversations,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
        input_ids = _field(tokenized_output, "input_ids")
        assistant_masks = _field(tokenized_output, "assistant_masks")
        idx = 0
        for idx in range(len(assistant_masks) - 1, -1, -1):
            if assistant_masks[idx] == 0:
                break
        for j in range(idx):
            assistant_masks[j] = 0
    else:
        conversations = sample[conversation_key]
        tokenized_output = apply_chat_template_randomized(conversations, tokenizer, text_format, rng=rng)
        input_ids = _field(tokenized_output, "input_ids")
        assistant_masks = _field(tokenized_output, "assistant_masks")

    labels = [
        token if mask == 1 else IGNORE_INDEX
        for token, mask in zip(input_ids, assistant_masks)
    ]

    # add BOS & EOS
    input_ids = list(input_ids) + [tokenizer.eos_id]
    labels = labels + [tokenizer.eos_id]
    if tokenizer.use_bos:
        input_ids.insert(0, tokenizer.bos_id)
        labels.insert(0, IGNORE_INDEX)

    return {
        "input_ids": input_ids,
        "labels": labels,
    }
