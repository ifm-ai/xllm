import re
from typing import List, Tuple

from xllm.eval.task.base import ChoiceTask, Example


class HellaSwagTask(ChoiceTask):
    datasets = ["hellaswag"]
    sep = " "
    examplar_file = "train.jsonl"
    eval_file = "val.jsonl"
    context_format = "{activity_label}: {ctx}"
    max_text_len = 256

    def get_examplar(self, example: Example) -> str:
        context = self.preprocess(self.context_format.format_map(example))
        answer = self.choices(example)[self.get_label(example)]
        return f"{context} {answer}"

    def get_text_completion(self, example: Example) -> List[Tuple[str, str]]:
        context = self.preprocess(self.context_format.format_map(example))
        return [(f"{context} {e}", e) for e in self.choices(example)]

    def get_label(self, example: Example) -> int:
        return int(example["label"])

    def choices(self, example: Example) -> List[str]:
        return [self.preprocess(e) for e in example["endings"]]

    def preprocess(self, text):
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text
