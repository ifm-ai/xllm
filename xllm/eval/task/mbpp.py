from typing import List, Dict
import json

from xllm.eval.task.base import Example, GenerationTask
from xllm.eval.utils import exact_match_score, f1_score, normalize_answer, code_pass_test
from collections import defaultdict
from pathlib import Path


class MBPPTask(GenerationTask):

    datasets = ["mbpp"]
    metrics = ["em", "f1", "nll", "pass_at_1"]
    n_fewshot = 3
    fewshot_index: List[int] = [1, 2, 3]
    sep = "\n"
    examplar_file = "mbpp_prompting.jsonl"
    eval_file = "mbpp_test.jsonl"
    max_text_len = 3096
    max_gen_len = 256
    prompt_format = "You are an expert Python programmer, and here is your task: {context} Your code should pass these tests:\n\n{tests}\n[BEGIN]\n"
    pass_at_k: int = 1

    def get_examplar(self, example: Example) -> str:
        context = example["text"]
        code = self.get_target(example)
        tests_str = "\n".join(self.get_prompt_tests(example))
        return self.clean_mbpp(
            (
                self.prompt_format.format(context=context, tests=tests_str)
                + f"{code}\n[DONE]"
            )
        )

    def get_prompt(self, example: Example) -> str:
        context = example["text"]
        tests_str = "\n".join(self.get_prompt_tests(example))
        return self.clean_mbpp(
            self.prompt_format.format(context=context, tests=tests_str)
        )

    def get_target(self, example: Example) -> str:
        target = self.clean_mbpp(example["code"])
        return target

    def get_tests(self, example: Example) -> List[str]:
        target = example["test_list"] + example["challenge_test_list"]
        return [self.clean_mbpp(t) for t in target]

    def get_prompt_tests(self, example: Example) -> List[str]:
        target = example["test_list"]
        return [self.clean_mbpp(t) for t in target]

    def clean_mbpp(self, t):
        return t.replace("\r", "")

    def postprocess(self, tokens: List[int]) -> str:
        generation = self.tokenizer.decode(tokens, cut_at_eos=True)
        return generation.split("[DONE]")[0]

    def evaluate(self, prediction: str, example: Example) -> Dict[str, float]:
        ground_truths = [self.clean_mbpp(example["code"])]
        tests = "\n".join(self.get_tests(example))
        sample_metrics = {}
        sample_metrics["em"] = 100 * exact_match_score(
            prediction, ground_truths, normalize_answer
        )
        sample_metrics["f1"] = 100 * f1_score(
            prediction, ground_truths, normalize_answer
        )
        for metric in [m for m in self.metrics if m.startswith("pass_at_")]:
            sample_metrics[metric] = 100 * code_pass_test(
                code=prediction, test=tests, timeout=5
            )
        return sample_metrics

    def pass_k_acc(self, batch_paths: List[Path]) -> Dict[str, float]:
        all_samples = []
        for p in batch_paths:
            with p.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    all_samples.append(json.loads(line))

        by_task = defaultdict(list)
        for s in all_samples:
            raw = s.get("raw", {})
            task_id = raw.get("task_id") or s.get("task_id")
            if task_id is None:
                task_id = raw.get("text") or s["prompt"]
            by_task[task_id].append(s)

        per_task_ok = []
        for task_id, samples in by_task.items():
            raw0 = samples[0].get("raw", {})
            test_list = raw0.get("test_list") or samples[0].get("test_list", [])
            chall_list = raw0.get("challenge_test_list") or samples[0].get("challenge_test_list", [])
            tests_str = "\n".join(self.clean_mbpp(t) for t in (test_list + chall_list))

            this_ok = False
            for s in samples:
                gen = s["generation"]
                gen = self.clean_mbpp(gen.split("[DONE]")[0])
                if code_pass_test(code=gen, test=tests_str, timeout=5):
                    this_ok = True
                    break

            per_task_ok.append(1.0 if this_ok else 0.0)

        avg = sum(per_task_ok) / len(per_task_ok) if per_task_ok else 0.0
        return {f"pass_at_{self.pass_at_k}": avg * 100.0}
