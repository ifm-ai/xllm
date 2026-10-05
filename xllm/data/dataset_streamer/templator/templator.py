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
SEARCH_TOOL = [
    {
        "name": "search",
        "description": "Semantic search over a document corpus. Returns relevant passages with document IDs.",
        "parameters": {
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant documents",
                },
                "return_fulltext": {
                    "type": "boolean",
                    "description": "If true, return full document text instead of passages",
                },
            },
            "required": ["query"],
        },
    }
]

GREETING_PHRASES = [
    "Hello! How can I help you?",
    "Hi! How can I help you today?",
    "Hi there! How can I help you?",
    "Hello! How can I assist you?",
    "Hi! How can I assist you today?",
    "Hi there! How can I assist?",
    "Hello! How may I help you?",
    "Hi! How may I help you today?",
    "Hi there! How may I assist?",
    "Hello! How may I assist you today?",
    "Hi! What can I help you with?",
    "Hi there! What can I help you with?",
    "Hello! What can I help you with today?",
    "Hi! What can I do for you?",
    "Hi there! What can I do for you?",
    "Hello! What can I do for you today?",
    "Hello! What would you like help with?",
    "Hi! What would you like help with?",
    "Hi there! What would you like help with today?",
    "Hello! What would you like to do?",
    "Hi! What would you like to do?",
    "Hello! What's on your mind?",
    "Hi! What's on your mind?",
    "Hi there! What's on your mind today?",
    "Hello! Ready when you are.",
    "Hi! Ready when you are.",
    "Hi there! Ready when you are.",
    "Hello! I'm here to help.",
    "Hi! I'm here to help.",
    "Hi there! I'm here to help.",
    "Hello! I'm ready to help.",
    "Hi! I'm ready to help.",
    "Hi there! I'm ready to help.",
    "Hello! Just let me know what you need.",
    "Hi! Just let me know what you need.",
    "Hi there! Just let me know what you need.",
    "Hello! Let me know how I can help.",
    "Hi! Let me know how I can help.",
    "Hi there! Let me know how I can help.",
    "Hello! Please go ahead.",
    "Hi! Please go ahead.",
    "Hello! Go ahead whenever you're ready.",
    "Hi! Go ahead whenever you're ready.",
    "Hello! Ask away.",
    "Hi! Ask away.",
    "Hi there! Ask away.",
    "Hello! What's the question?",
    "Hi! What's the question?",
    "Hi there! What's the question?",
    "Hello! What's the task?",
    "Hi! What's the task?",
    "Hi there! What's the task?",
    "Hello! How may I be of help?",
    "Hi! How may I be of help?",
    "Hello! How may I be of assistance?",
    "Hi! How may I be of assistance?",
    "Hello! Happy to help.",
    "Hi! Happy to help.",
    "Hi there! Happy to help.",
    "Hello! Glad to help.",
    "Hi! Glad to help.",
    "Hello! Pleased to help.",
    "Hi! Pleased to help.",
    "Hello! What brings you here today?",
    "Hi! What brings you here today?",
    "Hello! What would you like to know?",
    "Hi! What would you like to know?",
    "Hi there! What would you like to know?",
    "Hello! Tell me how I can help.",
    "Hi! Tell me how I can help.",
    "Hello! Tell me what you need.",
    "Hi! Tell me what you need.",
    "Hello! Ready to help.",
    "Hi! Ready to help.",
    "Hi there! Ready to help.",
    "Hello! Ready to assist.",
    "Hi! Ready to assist.",
    "Hello! Whenever you're ready.",
    "Hi! Whenever you're ready.",
    "Hello there!",
    "Hi there!",
    "Hey there!",
    "Hey!",
    "Hey! How can I help?",
    "Hey! How can I help you?",
    "Hey! What can I help you with?",
    "Hey! What can I do for you?",
    "Hey! What's the task?",
    "Hey! What's the question?",
    "Hey there! How can I help?",
    "Hey there! What can I do for you?",
    "Hey, what's up?",
    "Hey! What's up?",
    "Hey, what's going on?",
    "Hey! Fire away.",
    "Hey! Shoot.",
    "Hi! Fire away.",
    "Hello! Fire away.",
    "Hi! Shoot.",
    "Hi! Shoot — what do you need?",
    "Greetings! How can I help you?",
    "Greetings! How may I assist?",
    "Welcome! How can I help?",
    "Welcome! How can I assist you?",
    "Welcome! What can I help you with?",
    "Welcome! What would you like to do?",
    "Good morning! How can I help?",
    "Good morning! What can I help you with?",
    "Good morning! What's on your mind?",
    "Good morning! What would you like to work on?",
    "Good afternoon! How can I help?",
    "Good afternoon! What can I do for you?",
    "Good afternoon! What's the task?",
    "Good afternoon! What would you like to do?",
    "Good evening! How can I help?",
    "Good evening! What's on your mind?",
    "Good evening! What would you like to work on?",
    "Welcome back! What are we working on?",
    "Welcome back! Where would you like to pick up?",
    "Welcome back! What's next?",
    "Welcome back! How can I help?",
    "Hello again! How can I help?",
    "Hello again! What can I do for you?",
    "Hi again! What would you like to tackle?",
    "Hi again! Where would you like to start?",
    "Hi! What are we working on?",
    "Hello! What are we working on today?",
    "Hi there! What are we working on?",
    "Hi! What are we tackling today?",
    "Hello! What are we tackling today?",
    "Hi! What are we building today?",
    "Hello! What are we building today?",
    "Hi! Where would you like to start?",
    "Hello! Where would you like to start?",
    "Hi! Where shall we begin?",
    "Hello! Where shall we begin?",
    "Hi! Where do you want to start?",
    "Hi! What are you thinking about?",
    "Hello! What are you curious about?",
    "Hi! What would you like to explore?",
    "Hello! What would you like to dig into?",
    "Hi! What's the puzzle?",
    "Hello! What's the puzzle?",
    "Hi there! What's the problem?",
    "Hi! What are you working through?",
    "Hi.",
    "Hello.",
    "Hey.",
    "Yes?",
    "Go ahead.",
    "I'm here.",
    "What's up?",
    "What's the question?",
    "What's the task?",
]

TOOL_PRESENTATION_FORMATS = json.loads(
    os.getenv("TOOL_PRESENTATION_FORMATS", '["json", "xml", "xml", "markdown", "markdown"]')
)
TOOL_CALL_FORMATS = json.loads(os.getenv("TOOL_CALL_FORMATS", '["json", "xml", "xml", "xml_typed"]'))


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

    do_insert_system_prompt = rng.random() < 0.0003
    do_insert_search_tools = rng.random() < 0.0001
    do_insert_greeting = rng.random() < 0.0001
    tools_to_insert = list(SEARCH_TOOL) if do_insert_search_tools else []
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

    if do_insert_greeting:
        if conversation[0]["role"] == "user":
            insert_at = 0
        elif (
            conversation[0]["role"] == "system"
            and len(conversation) > 1
            and conversation[1]["role"] == "user"
        ):
            insert_at = 1
        else:
            insert_at = None
        if insert_at is not None:
            think_key = _detect_think_key(conversation)
            conversation.insert(insert_at, {
                "role": "assistant",
                "content": GREETING_PHRASES[rng.randint(len(GREETING_PHRASES))],
                think_key: "",
            })

    has_tools = bool(conversation[0].get("tools", False))
    if has_tools:
        if rng.random() < 0.05:
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
