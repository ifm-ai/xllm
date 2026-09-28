from typing import Dict, List

from xllm.eval.task.base import GenerationTask, Example
from xllm.eval.utils import exact_match_score, f1_score, normalize_answer


class NQTask(GenerationTask):
    datasets = ["nq"]
    metrics = ["em", "f1", "nll"]
    n_fewshot = 5
    sep = "\n"
    examplar_file = "train.jsonl"
    eval_file = "test.jsonl"
    question_prefix = "Question:"
    target_prefix = "Answer:"
    prompt_format = "{question_prefix} {question}\n{target_prefix}"
    max_text_len = 256
    max_gen_len = 24

    def get_examplar(self, example: Example) -> str:
        context = self.get_prompt(example)
        answer = self.get_target(example)
        return f"{context} {answer}"

    def get_prompt(self, example: Example) -> str:
        question = example["question"].capitalize().strip().rstrip("?") + "?"
        prompt = self.prompt_format.format(
            question_prefix=self.question_prefix,
            question=question,
            target_prefix=self.target_prefix,
        )
        return prompt

    def get_target(self, example: Example) -> str:
        answers = example["answers"]
        target = answers[0]  # select first answer as target
        # target = random.choice(answers)
        return target

    def postprocess(self, tokens: List[int]) -> str:
        generation = self.tokenizer.decode(tokens, cut_at_eos=True)
        generation = generation.split(self.question_prefix)[0].split(
            self.target_prefix
        )[0]
        generation = generation.split("\n")[0].split("(")[0].strip()
        return generation

    def evaluate(self, prediction: str, example: Example) -> Dict[str, float]:
        ground_truths = example["answers"]
        sample_metrics = {
            "em": 100 * exact_match_score(prediction, ground_truths, normalize_answer),
            "f1": 100 * f1_score(prediction, ground_truths, normalize_answer),
        }
        return sample_metrics
