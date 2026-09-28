from typing import List, Tuple

from xllm.eval.task.base import ChoiceTask, Example


class WinoGrandeTask(ChoiceTask):
    datasets = ["winogrande_1.1"]
    sep = " "
    examplar_file = "train_l.jsonl"
    eval_file = "dev.jsonl"
    max_text_len = 256

    def get_examplar(self, example: Example) -> str:
        prefix, suffix = self.prefix_suffix(example)
        answer = self.choices(example)[self.get_label(example)]
        return f"{prefix} {answer} {suffix}"

    def get_text_completion(self, example: Example) -> List[Tuple[str, str]]:
        prefix, suffix = self.prefix_suffix(example)
        return [(f"{prefix} {c} {suffix}", f"{suffix}") for c in self.choices(example)]

    def prefix_suffix(self, example: Example) -> Tuple[str, str]:
        sentence = example["sentence"]
        pronoun_loc = sentence.index("_")
        prefix = sentence[:pronoun_loc].rstrip()
        suffix = sentence[pronoun_loc + 1 :].strip()
        return prefix, suffix

    def get_label(self, example: Example) -> int:
        return eval(example["answer"]) - 1

    def choices(self, example: Example) -> List[str]:
        choices = [example["option1"], example["option2"]]
        return [c.strip() for c in choices]
