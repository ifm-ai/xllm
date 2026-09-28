from typing import List, Tuple

from xllm.eval.task.base import ChoiceTask, Example


class BoolQTask(ChoiceTask):
    datasets = ["boolq"]
    sep = " "
    examplar_file = "train.jsonl"
    eval_file = "dev.jsonl"
    context_format = "{title}\n{passage}\nQuestion: {question}\nAnswer:"
    max_text_len = 512

    def get_examplar(self, example: Example) -> str:
        context = self.context_format.format_map(example)
        answer = self.choices(example)[self.get_label(example)]
        return f"{context} {answer}"

    def get_text_completion(self, example: Example) -> List[Tuple[str, str]]:
        context = self.context_format.format_map(example)
        return [(f"{context} {c}", c) for c in self.choices(example)]

    def get_label(self, example: Example) -> int:
        return int(example["answer"])

    def choices(self, example: Example) -> List[str]:
        return ["no", "yes"]
