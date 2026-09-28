from typing import List, Dict

from xllm.eval.task.base import GenerationTask, Example
from xllm.eval.utils import exact_match_score, f1_score, normalize_answer


class TQATask(GenerationTask):
    datasets = ["tqa"]
    metrics = ["em", "f1", "nll"]
    n_fewshot = 5
    sep = "\n"
    examplar_file = "train.jsonl"
    eval_file = "test.jsonl"
    question_prefix = "Question:"
    target_prefix = "Answer:"
    prompt_format = "{question_prefix} {question}\n{target_prefix}"
    max_text_len = 512
    max_gen_len = 24

    def get_examplar(self, example: Example) -> str:
        return f"{self.get_prompt(example)} {self.get_target(example)}"

    def get_prompt(self, example: Example) -> str:
        return self.prompt_format.format(
            question_prefix=self.question_prefix,
            question=example["question"],
            target_prefix=self.target_prefix,
        )

    def get_target(self, example: Example) -> str:
        return example["target"]

    def postprocess(self, tokens: List[int]) -> str:
        generation = self.tokenizer.decode(tokens, cut_at_eos=True)
        generation = generation.split(self.question_prefix)[0].split(
            self.target_prefix
        )[0]
        return generation

    def evaluate(self, prediction: str, example: Example) -> Dict[str, float]:
        ground_truths = example["answers"]
        sample_metrics = {
            "em": 100 * exact_match_score(prediction, ground_truths, normalize_answer),
            "f1": 100 * f1_score(prediction, ground_truths, normalize_answer),
        }
        return sample_metrics
