from typing import List, Tuple, Dict, Optional, Type
from collections import defaultdict
import os
from pathlib import Path

from xllm.data.dataset_streamer.tokenizer import Tokenizer
from xllm.eval.task.base import ChoiceTask, Example, BaseTask
from xllm.eval.utils import classproperty


class ArabicMMLUTask(ChoiceTask):
    datasets: List[str] = ["arabic_mmlu/"]
    examplar_file: str = "dev.jsonl"
    eval_file: str = "test.jsonl"
    sep: str = "\n\n"
    max_text_len: int = 2048
    _nb_samples: Optional[int] = None
    task_dir_root: Optional[str] = None

    def __init__(self, tokenizer: Tokenizer, task_dir: str):
        super().__init__(tokenizer=tokenizer, task_dir=task_dir)
        if type(self).task_dir_root is None and type(self) is not ArabicMMLUTask:
            ArabicMMLUTask.task_dir_root = str(Path(task_dir).parent.parent)

    @classproperty
    def arabic_mmlu_tasks(cls) -> Dict[str, "ArabicMMLUTask"]:
        tasks = {
            k: v
            for k, v in BaseTask.tasks.items()
            if k.startswith("arabic_mmlu/") and v is not ArabicMMLUTask
        }
        return tasks  # type: ignore

    @classproperty
    def nb_samples(cls) -> int:
        if cls._nb_samples is None:
            if cls.datasets[0] == "arabic_mmlu/":
                cls._nb_samples = 0
                for task, task_cls in ArabicMMLUTask.arabic_mmlu_tasks.items():
                    if task != "arabic_mmlu/" and task.startswith("arabic_mmlu/"):
                        assert issubclass(task_cls, ArabicMMLUTask)
                        cls._nb_samples += task_cls.nb_samples  # type: ignore
            else:
                assert cls.task_dir_root is not None
                with open(
                    os.path.join(cls.task_dir_root, cls.datasets[0], "test.jsonl")
                ) as f:
                    cls._nb_samples = len([l for l in f])
        nb_samples = cls._nb_samples
        assert nb_samples is not None
        return nb_samples

    def context(self, example: Example) -> str:
        question = example["Question"]
        choices = []
        for i, l in enumerate(["A", "B", "C", "D", "E"]):
            if example[f"Option {i+1}"] != "":
                choices.append(f"{l}. {example[f'Option {i+1}']}")
        choice_str = "\n".join(choices)
        return f"Question: {question}\n{choice_str}\nAnswer:"

    def get_examplar(self, example: Example) -> str:
        context = self.context(example)
        answer = example["Answer Key"]
        return f"{context} {answer}"

    def get_text_completion(self, example: Example) -> List[Tuple[str, str]]:
        context = self.context(example)
        return [(f"{context} {l}", l) for l in ["A", "B", "C", "D", "E"]]

    def get_label(self, example: Example) -> int:
        mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        return mapping[example["Answer Key"]]


ARABIC_MMLU_TASKS_DOMAINS = {
    "Accounting_(University)": "Social Science",
    "Arabic_Language_(General)": "Language",
    "Arabic_Language_(Grammar)": "Language",
    "Arabic_Language_(High_School)": "Language",
    "Arabic_Language_(Middle_School)": "Language",
    "Arabic_Language_(Primary_School)": "Language",
    "Biology_(High_School)": "STEM",
    "Civics_(High_School)": "Social Science",
    "Civics_(Middle_School)": "Social Science",
    "Computer_Science_(High_School)": "STEM",
    "Computer_Science_(Middle_School)": "STEM",
    "Computer_Science_(Primary_School)": "STEM",
    "Computer_Science_(University)": "Social Science",
    "Driving_Test": "Other",
    "Economics_(High_School)": "Social Science",
    "Economics_(Middle_School)": "Social Science",
    "Economics_(University)": "Social Science",
    "General_Knowledge_(Middle_School)": "Other",
    "General_Knowledge_(Primary_School)": "Other",
    "General_Knowledge": "Other",
    "Geography_(High_School)": "Social Science",
    "Geography_(Middle_School)": "Social Science",
    "Geography_(Primary_School)": "Social Science",
    "History_(High_School)": "Humanities",
    "History_(Middle_School)": "Humanities",
    "History_(Primary_School)": "Humanities",
    "Islamic_Studies_(High_School)": "Humanities",
    "Islamic_Studies_(Middle_School)": "Humanities",
    "Islamic_Studies_(Primary_School)": "Humanities",
    "Islamic_Studies": "Humanities",
    "Law_(Professional)": "Humanities",
    "Management_(University)": "Other",
    "Math_(Primary_School)": "STEM",
    "Natural_Science_(Middle_School)": "STEM",
    "Natural_Science_(Primary_School)": "STEM",
    "Philosophy_(High_School)": "Humanities",
    "Physics_(High_School)": "STEM",
    "Political_Science_(University)": "Social Science",
    "Social_Science_(Middle_School)": "Social Science",
    "Social_Science_(Primary_School)": "Social Science",
}
ARABIC_MMLU_TASKS: Dict[str, Type[ArabicMMLUTask]] = {}
for task_name in ARABIC_MMLU_TASKS_DOMAINS:
    cap_task_name = "".join([s.capitalize() for s in task_name.split("_")])
    task_cls = type(
        cap_task_name,
        (ArabicMMLUTask,),
        {"datasets": [f"arabic_mmlu/{task_name}"]},
    )
    assert issubclass(task_cls, ArabicMMLUTask)
    ARABIC_MMLU_TASKS[task_name] = task_cls


def get_arabic_mmlu_scores(scores: Dict[str, float]) -> Dict[str, float]:
    metrics_ls_all: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    metrics_ls_domain: Dict[str, Dict[str, List[Tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for k, v in scores.items():
        if k.split("/")[1] == "arabic_mmlu":
            task_name = k.split("/")[2]
            domain = ARABIC_MMLU_TASKS_DOMAINS[task_name]
            task_cls = ArabicMMLUTask.arabic_mmlu_tasks["arabic_mmlu/" + task_name]
            metrics_ls_all["/".join(k.split("/")[3:])].append((task_cls.nb_samples, v))
            metrics_ls_domain[domain]["/".join(k.split("/")[3:])].append(
                (task_cls.nb_samples, v)
            )
    assert all(len(ls) == len(ArabicMMLUTask.arabic_mmlu_tasks) for ls in metrics_ls_all.values())
    metrics: Dict[str, float] = {}
    for domain, metrics_ls in metrics_ls_domain.items():
        nb_samples = sum(
            [
                task_cls.nb_samples
                for task, task_cls in ArabicMMLUTask.arabic_mmlu_tasks.items()
                if ARABIC_MMLU_TASKS_DOMAINS[task[len("arabic_mmlu/") :]] == domain
            ]
        )
        for k, ls in metrics_ls.items():
            assert sum([nb_samples for nb_samples, _ in ls]) == nb_samples
            metrics[f"{domain}/macro_avg/{k}"] = sum([v for _, v in ls]) / len(ls)
            metrics[f"{domain}/micro_avg/{k}"] = (
                sum([nb_samples * v for nb_samples, v in ls]) / nb_samples
            )
    for k, ls in metrics_ls_all.items():
        nb_samples = ArabicMMLUTask.nb_samples
        assert sum([nb_samples for nb_samples, _ in ls]) == nb_samples
        metrics[f"macro_avg/{k}"] = sum([v for _, v in ls]) / len(ls)
        metrics[f"micro_avg/{k}"] = (
            sum([nb_samples * v for nb_samples, v in ls]) / nb_samples
        )
    return metrics

