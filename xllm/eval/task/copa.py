from typing import List, Tuple

from xllm.eval.task.base import ChoiceTask, Example


class COPATask(ChoiceTask):
    datasets = ["copa"]
    sep = " "
    examplar_file = "train.jsonl"
    eval_file = "val.jsonl"
    context_format = "{context} {ask_for}"
    max_text_len = 256

    def get_examplar(self, example: Example) -> str:
        context = self.context_format.format_map(example)
        answer = self.choices(example)[self.get_label(example)]
        return f"{context} {answer}"

    def get_text_completion(self, example: Example) -> List[Tuple[str, str]]:
        if example["question"] == "cause":
            ask_for = "because"
        elif example["question"] == "effect":
            ask_for = "so"
        else:
            raise ValueError("Unknown question type: {}".format(example["question"]))
        context = example["premise"].strip(".").strip()
        context = self.context_format.format(context=context, ask_for=ask_for)
        return [(f"{context} {c}", c) for c in self.choices(example)]

    def get_label(self, example: Example) -> int:
        return example["label"]

    def choices(self, example: Example) -> List[str]:
        choices = [example["choice1"], example["choice2"]]
        return [c.lower() for c in choices]
