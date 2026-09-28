from typing import List, Tuple

from xllm.eval.task.base import ChoiceTask, Example


class ARCTask(ChoiceTask):
    datasets = ["arc_challenge", "arc_easy"]
    sep = " "
    examplar_file = "train.jsonl"
    eval_file = "test.jsonl"
    max_text_len = 512

    def get_examplar(self, example: Example) -> str:
        context = self.get_context(example)
        answer = self.choices(example)[self.get_label(example)]
        return f"{context} {answer}"

    def get_text_completion(self, example: Example) -> List[Tuple[str, str]]:
        context = self.get_context(example)
        return [(f"{context} {c}", c) for c in self.choices(example)]

    def get_label(self, example: Example) -> int:
        mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
        return mapping[example["label"]]

    def choices(self, example: Example) -> List[str]:
        return [example["choices"][c] for c in ["A", "B", "C", "D"]]

    def get_context(self, example: Example) -> str:
        question = example["question"].strip()
        if question.endswith("?"):
            return f"Question: {question}\nAnswer:"
        else:
            return f"{question}"
