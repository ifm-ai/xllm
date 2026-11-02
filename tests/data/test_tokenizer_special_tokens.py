import base64
import threading

from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from xllm.config import TokenizerConf
from xllm.data.dataset_streamer.tokenizer.huggingface import HuggingFaceTokenizer
from xllm.data.dataset_streamer.tokenizer.llama3 import Llama3Tokenizer

SPECIAL = "<|im_start|>"  # special added token
MARKER = "<think>"  # non-special added token that is a template marker (K2-Horizon ships 20 like it)
PIECE = "hello world"  # non-special added token that is an ordinary text piece
CHAT_TEMPLATE = "{% for m in messages %}" + SPECIAL + "{{ m['content'] }}" + MARKER + "{% endfor %}"


def _tokenizer(tmp_path):
    vocab = {"<s>": 0, "</s>": 1, "<unk>": 2, "<": 3, "|": 4, ">": 5, "_": 6, "im": 7, "start": 8, "hello": 9,
             "world": 10, "think": 11, "text": 12}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()])
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="<s>", eos_token="</s>", unk_token="<unk>",
                                  additional_special_tokens=[SPECIAL])
    tok.add_tokens([AddedToken(MARKER, special=False, normalized=False), AddedToken(PIECE, special=False)])
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(tmp_path)
    conf = TokenizerConf()
    conf.type, conf.path = "huggingface", str(tmp_path)
    return HuggingFaceTokenizer(conf)


def _control_ids(tok):
    return {tok._tokenizer.convert_tokens_to_ids(SPECIAL), tok._tokenizer.convert_tokens_to_ids(MARKER)}


def test_raw_text_never_yields_control_tokens(tmp_path):
    tok = _tokenizer(tmp_path)

    ids = tok._encode(f"text {SPECIAL} text {MARKER} text")

    assert not _control_ids(tok) & set(ids)
    assert ids == tok._tokenizer.convert_tokens_to_ids(
        ["text", "<", "|", "im", "_", "start", "|", ">", "text", "<", "think", ">", "text"])


def test_raw_text_keeps_ordinary_added_pieces(tmp_path):
    tok = _tokenizer(tmp_path)
    text = f"text {PIECE} text"

    ids = tok._encode(text)

    assert tok._tokenizer.convert_tokens_to_ids(PIECE) in ids
    assert ids == tok._tokenizer(text, add_special_tokens=False).input_ids


def test_raw_text_leaves_the_shared_tokenizer_alone(tmp_path):
    tok = _tokenizer(tmp_path)
    conversation = [{"role": "user", "content": "hello"}]
    reference = list(tok.apply_chat_template(conversation, tokenize=True, return_dict=True)["input_ids"])

    tok._encode(f"text {SPECIAL} text")

    assert tok._tokenizer.backend_tokenizer.encode_special_tokens is False
    assert list(tok.apply_chat_template(conversation, tokenize=True, return_dict=True)["input_ids"]) == reference
    assert _control_ids(tok) <= set(reference)


# per thread; with split_special_tokens=True per call, 6 of 8 runs at this count corrupted at least one call.
# test_raw_text_leaves_the_shared_tokenizer_alone checks the cause on every run.
CALLS = 3000


def test_raw_text_and_chat_threads_do_not_race(tmp_path):
    # the dataloader's tokenizer worker threads share one tokenizer: chat records and raw text in parallel
    tok = _tokenizer(tmp_path)
    conversation = [{"role": "user", "content": "hello world " * 200}]
    text = f"text {SPECIAL} " + "hello world text " * 200
    chat_reference = list(tok.apply_chat_template(conversation, tokenize=True, return_dict=True)["input_ids"])
    text_reference = tok._encode(text)
    bad = {"chat": 0, "text": 0}
    start = threading.Barrier(2)

    def chat():
        start.wait()
        for _ in range(CALLS):
            if list(tok.apply_chat_template(conversation, tokenize=True, return_dict=True)["input_ids"]) != chat_reference:
                bad["chat"] += 1

    def raw():
        start.wait()
        for _ in range(CALLS):
            if tok._encode(text) != text_reference:
                bad["text"] += 1

    threads = [threading.Thread(target=chat), threading.Thread(target=raw)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert bad == {"chat": 0, "text": 0}
    assert not _control_ids(tok) & set(text_reference)


def test_llama3_raw_text_never_yields_control_tokens(tmp_path):
    path = tmp_path / "bytes.tiktoken"
    path.write_text("".join(f"{base64.b64encode(bytes([b])).decode()} {b}\n" for b in range(256)))
    conf = TokenizerConf()
    conf.type, conf.path = "llama3", str(path)
    tok = Llama3Tokenizer(conf)
    text = "text <|begin_of_text|> <|eot_id|> text"

    ids = tok._encode(text)

    assert not set(tok.special_tokens.values()) & set(ids)
    assert ids == list(text.encode())
