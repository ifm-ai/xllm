import json
import os
from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Type, Optional

import numpy as np
import torch

from xllm.data.dataset_streamer.tokenizer import Tokenizer
from xllm.data.dataset_streamer.templator.templator import tokenize_multiturn_template


Example = Dict[str, Any]


class BaseTask:
    tasks: Dict[str, Type["BaseTask"]] = {}
    datasets: List[str] = []
    metrics: List
    sep: str
    eval_file: str
    max_text_len: int
    description: str = ""
    examplar_file: str
    n_fewshot: int = 0
    fewshot_index: Optional[List[int]] = None
    add_template: bool = False

    def __init__(self, tokenizer: Tokenizer, task_dir: str):
        self.tokenizer = tokenizer
        self.task_dir = task_dir
        self._all_examples: List[Example] = []

        assert self.eval_file.endswith(".jsonl")
        if self.n_fewshot > 0:
            assert os.path.isfile(os.path.join(self.task_dir, self.examplar_file)), f"examplar file does not exist"
            if self.fewshot_index is None:
                self.fewshot_index = list(range(self.n_fewshot))
            else:
                assert len(self.fewshot_index) == self.n_fewshot, f"the num. of few-shot indexes does not match n_fewshot."

    @classmethod
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        for dirname in cls.datasets:
            cls.tasks[dirname] = cls

    def process(self, example: Example, rng: np.random.RandomState):
        raise NotImplementedError

    def encode_prompts(
        self,
        text: str,
        target: str,
    ) -> Tuple[List[int], List[int]]:
        if self.add_template:
            # text is the prompt (user turn), target is the assistant response.
            # Format as a single-turn conversation and apply the chat template.
            sample = {
                "conversation": [
                    {"role": "user", "content": text.split(target)[0]},
                    {"role": "assistant", "content": target},
                ]
            }
            result = tokenize_multiturn_template(sample, self.tokenizer, text_format=None)
            x = result["input_ids"]
            y = result["labels"]
            return x[:-1], y[1:]
        else:
            x = self.tokenizer.encode(text, bos=True, eos=False)
            len_completion = len(self.tokenizer.encode(target, bos=False, eos=False))
            y = [-100 if k < len(x) - len_completion else t for k, t in enumerate(x)]
            return x[:-1], y[1:]

    def _load_examplar_file(self) -> List[Example]:
        all_examples: List[Example] = []
        if self.n_fewshot > 0:
            with open(os.path.join(self.task_dir, self.examplar_file)) as fin:
                for line in fin:
                    if not line:
                        continue
                    all_examples.append(json.loads(line))
        return all_examples

    def get_n_examples(self, example: Example) -> List[Example]:
        if self.n_fewshot == 0:
            return []
        if len(self._all_examples) == 0 and self.n_fewshot > 0:
            self._all_examples = self._load_examplar_file()

        examples = [
            self._all_examples[k] for k in self.fewshot_index
        ]
        return examples

    @abstractmethod
    def get_examplar(self, example: Example) -> str:
        raise NotImplementedError


class GenerationTask(BaseTask):
    max_gen_len: int

    def process(self, example: Example, rng: np.random.RandomState):
        examplars = [self.get_examplar(ex) for ex in self.get_n_examples(example)]

        prompt = self.get_prompt(example)
        prompt = self.description + self.sep.join(examplars + [prompt])
        target = self.get_target(example)
        text = self.sep.join([prompt, target])

        input_tokens, target_tokens = self.encode_prompts(text, target)
        return {
            "raw": example,
            "text_x": [input_tokens],
            "text_y": [target_tokens],
            "prompt": prompt,
        }

    @abstractmethod
    def postprocess(self, tokens: List[int]) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_prompt(self, example: Example) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_target(self, example: Example) -> str:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, prediction: str, example: Example) -> Dict[str, float]:
        raise NotImplementedError


class ChoiceTask(BaseTask):
    metrics = [
        "acc",
        "acc_token",
        "acc_char",
        "acc_compl",
        "nll",
        "nll_token",
        "nll_char",
        "nll_compl",
    ]

    def process(self, example: Example, rng: np.random.RandomState):
        examplars = [self.get_examplar(ex) for ex in self.get_n_examples(example)]

        text_completion = self.get_text_completion(example)
        text_completion = [
            (self.description + self.sep.join(examplars + [t]), c)
            for t, c in text_completion
        ]
        input_tokens, output_tokens = [], []
        input_completion, output_completion = [], []
        for text, completion in text_completion:
            it, ot = self.encode_prompts(text, completion)
            ic, oc = self.encode_prompts("Answer: " + completion, completion)
            input_tokens.append(it)
            output_tokens.append(ot)
            input_completion.append(ic)
            output_completion.append(oc)

        return {
            "raw": example,
            "text_x": input_tokens,
            "text_y": output_tokens,
            "n_completion": len(input_tokens),
            "completion_x": input_completion,
            "completion_y": output_completion,
            "full_text": [t for t, _ in text_completion],
            "completion_text": [c for _, c in text_completion],
        }

    def evaluate(
        self,
        nll: torch.Tensor,
        example: Example,
        n_token: torch.Tensor,
        choices: List[str],
        nll_completion: torch.Tensor,
    ) -> Dict[str, float]:
        label = self.get_label(example)

        n_char = torch.tensor([len(c) for c in choices]).cuda().float()
        nll_val, nll_idx = nll.min(dim=-1)
        nll_char_val, nll_char_idx = (nll / n_char).min(dim=-1)
        nll_token_val, nll_token_idx = (nll / n_token).min(dim=-1)
        nll_compl_val, nll_compl_idx = (nll - nll_completion).min(dim=-1)

        def compute_acc(pred, label):
            return 100.0 * (pred.item() == label)

        sample_metrics = {
            "acc": compute_acc(nll_idx, label),
            "acc_char": compute_acc(nll_char_idx, label),
            "acc_token": compute_acc(nll_token_idx, label),
            "acc_compl": compute_acc(nll_compl_idx, label),
            "nll": nll_val.item(),
            "nll_char": nll_char_val.item(),
            "nll_token": nll_token_val.item(),
            "nll_compl": nll_compl_val.item(),
        }
        return sample_metrics


    @abstractmethod
    def get_text_completion(self, example: Example) -> List[Tuple[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def get_label(self, example: Example) -> int:
        raise NotImplementedError
