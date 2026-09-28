from typing import List, Tuple

from xllm.eval.task.base import ChoiceTask, Example


class SIQATask(ChoiceTask):
    datasets = ["siqa"]
    sep = " "
    examplar_file = "train.jsonl"
    eval_file = "dev.jsonl"
    context_format = "{context} Q: {question} A:"
    max_text_len = 256

    def get_examplar(self, example: Example) -> str:
        context = self.context_format.format_map(example)
        answer = self.choices(example)[self.get_label(example)]
        return f"{context} {answer}"

    def get_text_completion(self, example: Example) -> List[Tuple[str, str]]:
        context = self.context_format.format_map(example)
        return [(f"{context} {c}", c) for c in self.choices(example)]

    def get_label(self, example: Example) -> int:
        return example["label"] - 1

    def choices(self, example: Example) -> List[str]:
        choices = [example["answerA"], example["answerB"], example["answerC"]]
        return [c.strip() for c in choices]
