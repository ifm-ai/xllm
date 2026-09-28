import os

from xllm.data.dataset_streamer.tokenizer import Tokenizer
from xllm.eval.task.base import BaseTask
from xllm.eval.task.nq import NQTask
from xllm.eval.task.tqa import TQATask
from xllm.eval.task.boolq import BoolQTask
from xllm.eval.task.piqa import PIQATask
from xllm.eval.task.siqa import SIQATask
from xllm.eval.task.hellaswag import HellaSwagTask
from xllm.eval.task.winogrande import WinoGrandeTask
from xllm.eval.task.copa import COPATask
from xllm.eval.task.obqa import OBQATask
from xllm.eval.task.arc import ARCTask
from xllm.eval.task.race import RACETask
from xllm.eval.task.gsm8k import GSM8KTask
from xllm.eval.task.mmlu import MMLUTask
from typing import Tuple

from xllm.eval.task.scrolls import (
    GovReport,
    NarrativeQA,
    Qasper,
    QMSum,
    QuALITY,
    SummScreenFD,
)
from xllm.eval.task.ruler import (
    RulerNIAH,
    RulerVT,
    RulerCWE,
    RulerFWE,
    RulerQA,
)
from xllm.eval.task.arabic_mmlu import ArabicMMLUTask

from xllm.eval.task.gpqa import GPQATask
from xllm.eval.task.mmlu_pro import MMLUProTask
from xllm.eval.task.bbh_cot import BBHCoTTask
from xllm.eval.task.bbh import BBHTask
from xllm.eval.task.drop import DROPTask
from xllm.eval.task.human_eval import HumanEvalTask
from xllm.eval.task.mbpp import MBPPTask


def get_task(
    tasks_root: str, task_name: str, tokenizer: Tokenizer
) -> Tuple[str, BaseTask]:

    if task_name not in BaseTask.tasks:
        raise ValueError(f"Unknown task: {task_name}")
    task_cls = BaseTask.tasks[task_name]
    task_dir = os.path.join(tasks_root, task_name)
    task = task_cls(tokenizer, task_dir)
    return task_name, task
