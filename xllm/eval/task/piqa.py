from typing import List, Tuple

from xllm.eval.task.base import ChoiceTask, Example


class PIQATask(ChoiceTask):
    datasets = ["piqa"]
    sep = " "
    examplar_file = "train.jsonl"
    eval_file = "valid.jsonl"
    context_format = "Question: {goal}\nAnswer:"
    max_text_len = 384

    def get_examplar(self, example: Example) -> str:
        context = self.context_format.format_map(example)
        answer = self.choices(example)[self.get_label(example)]
        return f"{context} {answer}"

    def get_text_completion(self, example: Example) -> List[Tuple[str, str]]:
        context = self.context_format.format_map(example)
        return [(f"{context} {c}", c) for c in self.choices(example)]

    def get_label(self, example: Example) -> int:
        return example["label"]

    def choices(self, example: Example) -> List[str]:
        choices = [example["sol1"], example["sol2"]]
        return [c.strip() for c in choices]
