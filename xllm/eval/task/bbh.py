from typing import List, Dict
import numpy as np

from xllm.eval.task.base import GenerationTask, Example
from xllm.eval.utils import exact_match_score

# BBH example
# {
#     "input": "not ( True ) and ( True ) is",
#     "target": "False",
#     "category": "boolean_expressions",
#     "explaination": "",
#     "question_id": 0,
#     "description":
#     "Evaluate the result of a random Boolean expression.\n\n"
# }


def eval_normalizer(s: str) -> str:
    return s.strip(" .") if s else ""


class BBHTask(GenerationTask):
    datasets = ["bbh"]
    n_fewshot = 3
    metrics = ["em", "nll"]
    sep = "\n\n"
    eval_file = "test.jsonl"
    examplar_file = "prompt.jsonl"

    question_prefix = "Q:"
    target_prefix = "A:"

    max_text_len = 4096
    max_gen_len = 32

    def get_n_examples(self, example: Example) -> List[Example]:
        if not self._all_examples:
            self._all_examples = self._load_examplar_file()

        if self.n_fewshot == 0:
            return []

        cat = example.get("category")
        if cat is not None:
            same_cat = [
                ex for ex in self._all_examples
                if ex.get("category") == cat
            ]
        else:
            same_cat = self._all_examples

        examples = [same_cat[k] for k in self.fewshot_index]
        examples = [fs for fs in examples if fs != example]
        return examples

    def get_examplar(self, example: Example) -> str:
        return f"{self.get_prompt(example)} {self.get_target(example)}"

    def get_prompt(self, example: Example) -> str:
        desc = example.get("description") or ""
        inp = example["input"]
        parts = []
        if desc:
            parts.append(desc.strip())
        parts.append(f"{self.question_prefix}\n{inp}")
        parts.append(self.target_prefix)
        return "\n".join(parts)

    def get_target(self, example: Example) -> str:
        final_answer = example["target"]
        return final_answer

    def postprocess(self, tokens: List[int]) -> str:
        generation = self.tokenizer.decode(tokens, cut_at_eos=True)
        generation = generation.split("Q:")[0]
        if "\n\n" in generation:
            generation = generation.split("\n\n")[0]
        return generation

    def evaluate(self, prediction: str, example: Example) -> Dict[str, float]:
        gold = example["target"]
        em = exact_match_score(prediction, [gold], eval_normalizer)
        return {
            "em": 100 * em,
        }
