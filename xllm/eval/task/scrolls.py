import os
import string
import re
from typing import Dict, List, Tuple
from abc import abstractmethod

import nltk
from rouge_score import rouge_scorer

from xllm.eval.task.base import Example, GenerationTask, ChoiceTask
from xllm.eval.utils import exact_match_score, f1_score, normalize_answer

_USER_SCROLLS_MAX_LEN = int(os.environ.get("SCROLLS_MAX_LEN", 1e9))
_8K = 8192
_16K = _8K * 2
_32K = _16K * 2
_64K = _32K * 2
_SCROLLS_DEFAULT_LEN = _32K * 4


def compute_rouge(prediction: str, reference: str, use_stemmer: bool = False):
    rouge_types = ["rouge1", "rouge2", "rougeL"]
    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer)

    score = scorer.score(reference, prediction)
    result = {
        key: (value[0] if isinstance(value, list) else value).fmeasure * 100
        for key, value in score.items()
    }
    return result


def postprocess_text(text):
    # rougeLSum expects newline after each sentence
    return "\n".join(nltk.sent_tokenize(text))


class SCROLLSTask(GenerationTask):
    datasets: List[str]
    metrics: List

    # all scrolls datasets only have answers for validation.jsonl
    eval_file = "validation.jsonl"
    prompt_wrapper = "{input}"

    def postprocess(self, tokens: List[int]) -> str:
        # assert isinstance(self.tokenizer, Tokenizer)
        generation = self.tokenizer.decode(tokens, cut_at_eos=True)
        result = self.extract_result(generation)
        return result

    def get_examplar(self, example: Example) -> str:
        # this is used for fewshot examples
        prompt = self.get_prompt(example)
        target = self.get_target(example)
        # make sure this matches how the prompt + target is constructed
        return f"{prompt} {target}"

    def get_prompt(self, example: Example) -> str:
        # remove clean_space
        example["input"] = self.clean_space(example["input"])
        return self.prompt_wrapper.format_map(example)

    def get_target(self, example: Example) -> str:
        target = example['output']
        if isinstance(target, list):
            target = target[0]
        return target.strip()

    @abstractmethod
    def clean_space(self, text):
        pass

    @abstractmethod
    def extract_result(self, generation: str) -> str:
        pass


class SCROLLSQA(SCROLLSTask):
    sep = "\n\n"

    def evaluate(self, prediction: str, example: Example) -> Dict[str, float]:
        target = example['output']
        if isinstance(target, list):
            ground_truths = [t.strip() for t in target]
        else:
            ground_truths = [target.strip()]

        sample_metrics = {
            "em": 100 * exact_match_score(prediction, ground_truths, normalize_answer),
            "f1": 100 * f1_score(prediction, ground_truths, normalize_answer),
        }
        return sample_metrics


class SCROLLSSumm(SCROLLSTask):
    sep = "\n\n"

    def evaluate(self, prediction: str, example: Example) -> Dict[str, float]:
        ground_truth = self.get_target(example)
        sample_metrics = compute_rouge(
            postprocess_text(prediction), postprocess_text(ground_truth)
        )
        return sample_metrics


class ContractNLI(SCROLLSQA):
    metrics = ["em", "nll"]
    datasets = ["scrolls/contract_nli"]
    max_text_len = min(_USER_SCROLLS_MAX_LEN, _SCROLLS_DEFAULT_LEN)
    max_gen_len = 32

    prompt_wrapper = (
        "{input}\n"
        "Consider the following hypothesis: {input_prefix}.\n"
        'Q: Is the hypothesis a "Entailment", "Contradition", or "Not Mentioned"?\nA:'
    )

    allowable_answers = ["entailment", "contradiction", "not mentioned"]

    def clean_space(self, text):
        return re.sub(r"\s+", " ", text).strip()

    def extract_result(self, generation: str) -> str:
        generation = generation.split("\n")[0].lower()
        for answer in self.allowable_answers:
            if answer in generation:
                return answer
        return ""


class NarrativeQA(SCROLLSQA):
    metrics = ["f1", "nll"]
    datasets = ["scrolls/narrative_qa"]
    max_gen_len = 64
    max_text_len = min(_USER_SCROLLS_MAX_LEN, _64K - max_gen_len)
    n_fewshot = 0

    question_prefix = "Q:"
    target_prefix = "A:"
    prompt_wrapper = "".join([
        "{input}\n\n",
        question_prefix,
        " {input_prefix}\n",
        target_prefix
    ])

    def clean_space(self, text):
        text = re.sub(r"\n(?!\n)", " ", text).strip()
        text = re.sub(r"\n\s+", "\n", text).strip()
        text = re.sub(r"\s+\n", "\n", text).strip()
        text = re.sub(r" +", " ", text).strip()
        return text

    def extract_result(self, generation: str) -> str:
        generation = generation.split(self.question_prefix)[0].split(self.target_prefix)[0].strip()
        generation = generation.split("\n")[0]
        generation = re.split(r"(?<!(Mrs|\sMr|\sDr|\sJr|\sSr))\. +", " " + generation)[0].strip()
        generation = re.sub(r"\.+$", ".", generation).strip()
        return generation


class Qasper(SCROLLSQA):
    metrics = ["f1", "nll"]
    datasets = ["scrolls/qasper"]
    max_text_len = min(_USER_SCROLLS_MAX_LEN, _SCROLLS_DEFAULT_LEN)
    max_gen_len = 80

    fewshot_mode = "first"
    n_fewshot = 2
    examplar_file = "train.jsonl"

    question_prefix = "Q:"
    target_prefix = "A:"
    prompt_wrapper = "".join([
        "{input}\n\n",
        question_prefix,
        " {input_prefix}\n",
        target_prefix
    ])

    def clean_space(self, text):
        text = re.sub(r"\n\s+", "\n", text).strip()
        text = re.sub(r"\s+\n", "\n", text).strip()
        text = re.sub(r" +", " ", text).strip()
        return text

    def extract_result(self, generation: str) -> str:
        generation = generation.split(self.question_prefix)[0].split(self.target_prefix)[0].strip()
        generation = re.split(r"(?<!\d)\. +", generation)[0]
        generation = re.sub(r"\.+$", "", generation).strip()
        if generation.startswith("Yes"):
            generation = "Yes"
        elif generation.startswith("No"):
            generation = "No"

        return generation


# quality generative task for API evals
class QuALITY_Gen(SCROLLSQA):
    metrics = ["f1", "em", "acc"]
    datasets = ["scrolls/quality_gen"]
    max_text_len = min(_USER_SCROLLS_MAX_LEN, _SCROLLS_DEFAULT_LEN)
    max_gen_len = 50

    n_fewshot = 2
    examplar_file = "train.jsonl"
    fewshot_index = [1, 12]

    question_prefix = "Q:"
    target_prefix = "A:"
    prompt_wrapper = "".join([
        "{input}\n\n",
        question_prefix,
        " {input_prefix}\n",
        target_prefix
    ])

    def clean_space(self, text):
        return re.sub(r"\s+", " ", text).strip()

    def clean_prefix_target(self, text):
        # text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s+", "\n", text).strip()
        text = re.sub(r"\s+\n", "\n", text).strip()
        text = re.sub(r" +", " ", text).strip()
        return text.strip()

    def options_to_dict(self, option_string):
        pattern = r"\((.*?)\) (.*)"
        matches = re.findall(pattern, option_string)
        options_dict = {v.strip(): k for k, v in matches}
        if len(options_dict) != 4:
            print("Warning: QuALITY data has invalid number of choices.")
        return options_dict

    def get_prompt(self, example: Example) -> str:
        # remove clean_space
        example["input"] = self.clean_space(example["input"])
        example["input_prefix"] = self.clean_prefix_target(example["input_prefix"])
        return self.prompt_wrapper.format_map(example)

    def get_target(self, example: Example) -> str:
        target = self.clean_prefix_target(example['output'])
        question = example["input_prefix"]
        options_dict = self.options_to_dict(question)
        gold_option = options_dict.get(target, None)
        if gold_option is None:
            raise ValueError(f"Warning: Quality data cannot find gold option {question}/{target}.")
        else:
            # clean target
            target = f"({gold_option}) " + target
        return target

    def extract_result(self, generation: str) -> str:
        return generation.split("\n")[0]

    def evaluate(self, prediction: str, example: Example) -> Dict[str, float]:
        question = example["input_prefix"]
        target = self.clean_prefix_target(example["output"])
        options_dict = self.options_to_dict(question)
        gold_option = options_dict.get(target, "")

        def extract_option_id(prediction_string):
            pattern = re.compile(r"\b[A-D]\b")
            match = pattern.search(prediction_string)
            if match is None:
                return None
            return match.group()

        predicted_option = extract_option_id(prediction)
        predicted_string = prediction[prediction.find(")") + 1:].strip()

        sample_metrics = {
            "em": 100 * exact_match_score(predicted_string, [target], normalize_answer),
            "f1": 100 * f1_score(predicted_string, [target], normalize_answer),
            "acc": 100 * (predicted_option == gold_option),
        }
        return sample_metrics


class QMSum(SCROLLSSumm):
    metrics = ["rouge1", "rouge2", "rougeL", "nll"]
    datasets = ["scrolls/qmsum"]
    max_text_len = min(_USER_SCROLLS_MAX_LEN, _SCROLLS_DEFAULT_LEN)
    max_gen_len = 256

    punc = "".join(list(set(string.punctuation) - {'"', "'", '(', ')', '[', ']', '{', '}'}))
    n_fewshot = 1
    examplar_file = "train.jsonl"

    question_prefix = "Q:"
    target_prefix = "A:"
    prompt_wrapper = "".join([
        "{input}\n\n",
        question_prefix,
        " {input_prefix}\n",
        target_prefix
    ])

    def clean_space(self, text):
        # un tokenization
        text = re.sub("\\s(?=([" + re.escape(self.punc) + "]))", "", text)
        text = re.sub(r"\n\s+", "\n", text).strip()
        text = re.sub(r"\s+\n", "\n", text).strip()
        text = re.sub(r" +", " ", text).strip()
        return text

    def extract_result(self, generation: str) -> str:
        generation = generation.split(self.question_prefix)[0].split(self.target_prefix)[0].strip()
        # generation = generation.split("\n")[0]
        return generation


class GovReport(SCROLLSSumm):
    metrics = ["rouge1", "rouge2", "rougeL", "nll"]
    datasets = ["scrolls/gov_report"]
    max_text_len = min(_USER_SCROLLS_MAX_LEN, _SCROLLS_DEFAULT_LEN)
    max_gen_len = 512

    n_fewshot = 5
    examplar_file = "train.jsonl"

    prompt_wrapper = (
        "{input}\n\n"
        "Summary:"
    )

    def clean_space(self, text):
        return re.sub(r"\s+", " ", text).strip()

    def extract_result(self, generation: str) -> str:
        return generation.split("\n")[0]


class SummScreenFD(SCROLLSSumm):
    metrics = ["rouge1", "rouge2", "rougeL", "nll"]
    datasets = ["scrolls/summ_screen_fd"]
    max_text_len = min(_USER_SCROLLS_MAX_LEN, _SCROLLS_DEFAULT_LEN)
    max_gen_len = 350

    n_fewshot = 5
    examplar_file = "train.jsonl"

    prompt_wrapper = (
        "{input}\n\n"
        "Summary:"
    )

    def clean_space(self, text):
        return re.sub(r"\s+", " ", text).strip()

    def extract_result(self, generation: str) -> str:
        return generation.split("\n")[0]


class QuALITY(ChoiceTask):
    datasets = ["scrolls/quality"]
    max_text_len = min(_USER_SCROLLS_MAX_LEN, _SCROLLS_DEFAULT_LEN)
    eval_file = "validation.jsonl"

    n_fewshot = 2
    sep = "\n"
    examplar_file = "train.jsonl"
    context_format = "{input} Q: {question} A:"

    def get_examplar(self, example: Example) -> str:
        question = self.extract_question_options(example)[0]
        context = self.context_format.format(input=example["input"], question=question)
        answer = example["output"]
        return f"{context} {answer}"

    def get_text_completion(self, example: Example) -> List[Tuple[str, str]]:
        question = self.extract_question_options(example)[0]
        context = self.context_format.format(input=example["input"], question=question)
        return [(f"{context} {c}", c) for c in self.choices(example)]

    def choices(self, example: Example) -> List[str]:
        return self.extract_question_options(example)[1]

    def extract_question_options(self, example: Example) -> Tuple[List[str], List[str]]:
        question = example["input_prefix"]
        assert "(A)" in question
        sep = question.find("(A)")
        question_parts = question[:sep].strip()
        choices = question[sep:]
        choices = [
            choice[choice.find(")") + 1 :].strip()
            for choice in choices.split("\n")
            if len(choice.strip()) > 0
        ]
        assert len(choices) == 4, example
        return question_parts, choices

    def get_label(self, example: Example) -> int:
        choices = self.choices(example)
        assert example["output"].strip() in choices
        for idx, c in enumerate(choices):
            if example["output"].strip() == c:
                return idx
        return -1
