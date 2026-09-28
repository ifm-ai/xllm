from typing import List, Tuple, Dict
import string
import numpy as np

from xllm.eval.task.base import ChoiceTask, Example


class MMLUProTask(ChoiceTask):
    datasets = ["mmlu_pro"]
    n_fewshot = 5
    examplar_file = "prompt.jsonl"
    eval_file = "test.jsonl"
    sep = "\n\n"
    max_text_len = 4096

    def _letters(self, n: int) -> List[str]:
        return list(string.ascii_uppercase[:n])

    def context(self, example: Example) -> str:

        q = example["question"]
        opts: List[str] = example["options"]
        letters = self._letters(len(opts))
        choice_lines = [f"{l}. {opt}" for l, opt in zip(letters, opts)]
        choice_str = "\n".join(choice_lines)
        return f"{q}\n{choice_str}\nAnswer:"

    def get_examplar(self, example: Example) -> str:
        ctx = self.context(example)
        ans = example["answer"]  
        return f"{ctx} {ans}"

    def get_text_completion(self, example: Example) -> List[Tuple[str, str]]:
        ctx = self.context(example)
        opts: List[str] = example["options"]
        letters = self._letters(len(opts))
        return [(f"{ctx} {l}", l) for l in letters]

    def get_label(self, example: Example) -> int:
        if "answer_index" in example:
            return int(example["answer_index"])
        letters = self._letters(len(example["options"]))
        mapping = {l: i for i, l in enumerate(letters)}
        return example["answer"]

    def get_n_examples(self, example: Example):
        if self.n_fewshot == 0:
            return []
        if not self._all_examples:
            self._all_examples = self._load_examplar_file()
        cat = example.get("category")
        same_cat = [ex for ex in self._all_examples if ex.get("category") == cat]
        return same_cat[: self.n_fewshot]
    
    def process(self, example: Example, rng: np.random.RandomState):
        examplars = [self.get_examplar(ex) for ex in self.get_n_examples(example)]

        text_completion = self.get_text_completion(example)
        text_completion = [
            (self.sep.join([self.description.format(topic=example["category"])] + examplars + [t]), c) for t, c in text_completion
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
