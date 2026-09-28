from typing import List, Tuple

from xllm.eval.task.base import ChoiceTask, Example


class RACETask(ChoiceTask):
    datasets = ["race_middle", "race_high"]
    sep = " "
    examplar_file = "test.jsonl"
    eval_file = "test.jsonl"
    context_format = "{question}"
    max_text_len = 1024

    def get_examplar(self, example: Example) -> str:
        context = self.context_format.format_map(example)
        answer = self.choices(example)[self.get_label(example)]
        return f"{context} {answer}"

    def get_text_completion(self, example: Example) -> List[Tuple[str, str]]:
        context = self.context_format.format_map(example)
        return [(f"{context} {c}", c) for c in self.choices(example)]

    def get_label(self, example: Example) -> int:
        mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
        return mapping[example["label"]]

    def choices(self, example: Example) -> List[str]:
        return [example["choices"][c] for c in ["A", "B", "C", "D"]]
