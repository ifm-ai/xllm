from typing import Dict, List
import re

from xllm.eval.task.base import BaseTask, Example, GenerationTask
from xllm.eval.utils import classproperty

# Change TOKENIZER and CONTEXT_LENGTHS to match your desired settings
TOKENIZER = "jais-250k" # TODO: change to build_task or pass in from config
CONTEXT_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]

MAX_LEN = 131072
_MAX_LEN_NIAH = 128
_MAX_LEN_VT = 30
_MAX_LEN_CWE = 120
_MAX_LEN_FWE = 50
_MAX_LEN_QA = 32

RULER_TASKS={
    "niah": [
        "niah_single_1",
        "niah_single_2",
        "niah_single_3",
        "niah_multikey_1",
        "niah_multikey_2",
        "niah_multikey_3",
        "niah_multivalue",
        "niah_multiquery"],
    "vt":["vt"],
    "cwe":["cwe"],
    "fwe":["fwe"],
    "qa":["qa_1","qa_2"]
}


def string_match_part(prediction, ground_truths):
    score = max([1.0 if gt.lower() in prediction.lower() else 0.0 for gt in ground_truths]) 
    return round(score, 2)


def string_match_all(prediction, ground_truths):
    score = sum([1.0 if gt.lower() in prediction.lower() else 0.0 for gt in ground_truths]) / len(ground_truths) 
    return round(score, 2)


def clean_space(text):
    return re.sub(r"\s+", " ", text)


class RulerTask(GenerationTask):
    datasets: List[str]
    metrics: List
    eval_file = "validation.jsonl"
    max_text_len: int = MAX_LEN
    sep = "\n\n"

    def postprocess(self, tokens: List[int]) -> str:
        generation = self.tokenizer.decode(tokens, cut_at_eos=True)
        generation = generation.strip()
        # remove all non-printable characters
        np_pattern = re.compile(r'[\x00-\x1f]')
        generation = np_pattern.sub('\n', generation).strip()
        
        return generation

    def get_examplar(self, example: Example) -> str:
        prompt = self.get_prompt(example)
        target = self.get_target(example)
        # make sure this matches how the prompt + target is constructed
        return f"{prompt} {target}"

    def get_prompt(self, example: Example) -> str:
        
        assert 'input' in example, "input must be in example"
        
        input_str = example["input"]
        return f"{input_str}"

    def get_target(self, example: Example) -> str:
        
        assert 'output' in example or 'outputs' in example, "output or outputs must be in example"
        
        if 'outputs' in example:
            target = example['outputs']
        else:
            target = example['output']

        if target is list:
            target = " ".join(target)

        return target

    @classproperty
    def get_sub_tasks(cls) -> Dict[str, "RulerTask"]:
        tasks = {
            k: v for k, v in BaseTask.tasks.items()
            if k.startswith("ruler") and v is not RulerTask
        }
        return tasks


class RulerNIAH(RulerTask):
    datasets = [f"ruler/{TOKENIZER}/{context_length}/{task}" \
                for context_length in CONTEXT_LENGTHS for task in RULER_TASKS["niah"]]
    max_gen_len: int = _MAX_LEN_NIAH
    metrics = ["em"]
    
    def evaluate(self, prediction, example):
        ground_truths = example['outputs'] if 'outputs' in example else example['output']
        metrics = {
            "em": 100 * string_match_all(prediction, ground_truths),
        }
        return metrics


class RulerVT(RulerTask):
    datasets = [f"ruler/{TOKENIZER}/{context_length}/{task}" \
                for context_length in CONTEXT_LENGTHS for task in RULER_TASKS["vt"]]
    max_gen_len: int = _MAX_LEN_VT
    metrics = ["em"]
    
    def evaluate(self, prediction, example):
        ground_truths = example['outputs'] if 'outputs' in example else example['output']
        metrics = {
            "em": 100 * string_match_all(prediction, ground_truths),
        }
        return metrics


class RulerCWE(RulerTask):
    datasets = [f"ruler/{TOKENIZER}/{context_length}/{task}" \
                for context_length in CONTEXT_LENGTHS for task in RULER_TASKS["cwe"]]
    max_gen_len: int = _MAX_LEN_CWE
    metrics = ["em"]
    
    def evaluate(self, prediction, example):
        ground_truths = example['outputs'] if 'outputs' in example else example['output']
        metrics = {
            "em": 100 * string_match_all(prediction, ground_truths),
        }
        return metrics


class RulerFWE(RulerTask):
    datasets = [f"ruler/{TOKENIZER}/{context_length}/{task}" \
                for context_length in CONTEXT_LENGTHS for task in RULER_TASKS["fwe"]]
    max_gen_len: int = _MAX_LEN_FWE
    metrics = ["em"]
    
    def evaluate(self, prediction, example):
        ground_truths = example['outputs'] if 'outputs' in example else example['output']
        metrics = {
            "em": 100 * string_match_all(prediction, ground_truths),
        }
        return metrics


class RulerQA(RulerTask):
    datasets = [f"ruler/{TOKENIZER}/{context_length}/{task}" \
                for context_length in CONTEXT_LENGTHS for task in RULER_TASKS["qa"]]
    max_gen_len: int = _MAX_LEN_QA
    metrics = ["pm"]
    
    def evaluate(self, prediction, example):
        ground_truths = example['outputs'] if 'outputs' in example else example['output']
        metrics = {
            "pm": 100 * string_match_part(prediction, ground_truths),
        }
        return metrics
