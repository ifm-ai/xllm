from typing import Dict, List

from xllm.eval.utils import exact_match_score, f1_score, normalize_answer
from xllm.eval.task.base import GenerationTask, Example


class COQATask(GenerationTask):
    datasets = ["coqa"]
    metrics = ["em", "f1", "nll"]
    fewshot_strategy = "first"
    sep = "\n"
    train_file = ""
    eval_file = "valid_small.jsonl"
    prompt_format = "{context}"
    max_text_len = 1024
    max_gen_len = 32

    def get_examplar(self, example: Example) -> str:
        context = self.get_prompt(example)
        answer = self.get_target(example)
        return f"{context} {answer}"

    def get_answers(self, example: Example) -> List[str]:
        return example["answers"]

    def get_prompt(self, example: Example) -> str:
        return example["context"]

    def get_target(self, example: Example) -> str:
        answers = example["answers"]
        target = answers[0]
        return target

    def postprocess(self, tokens: List[int]) -> str:
        generation = self.tokenizer.decode(tokens, cut_at_eos=True)
        generation = generation.split("\n")[0]
        return generation

    def evaluate(self, prediction: str, example: Example) -> Dict[str, float]:
        ground_truths = example["answers"]
        sample_metrics = {
            "em": 100 * exact_match_score(prediction, ground_truths, normalize_answer),
            "f1": 100 * f1_score(prediction, ground_truths, normalize_answer),
        }
        return sample_metrics
