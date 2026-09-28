from typing import List, Dict

from xllm.eval.task.base import GenerationTask, Example
from xllm.eval.utils import (
    exact_match_score,
    f1_score,
    math_accuracy_score
)
from xllm.eval.math import (
    extract_answer
)


class GSM8KTask(GenerationTask):
    datasets = ["gsm8k"]
    metrics = ["acc", "em", "f1", "nll"]
    sep = "\n\n"
    eval_file = "test.jsonl"
    answer_format: str = "{solution} The answer is {result}."
    max_text_len: int = 2048
    max_gen_len: int = 512

    def get_n_examples(self, example: Example) -> List[Example]:
        return [
            {
                "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
                "answer": "Let's think step by step. There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.\n#### 6",
            },
            {
                "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
                "answer": "Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.\n#### 5",
            },
            {
                "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
                "answer": "Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.\n#### 39",
            },
            {
                "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
                "answer": "Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.\n#### 8",
            },
            {
                "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
                "answer": "Let's think step by step. Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.\n#### 9",
            },
            {
                "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
                "answer": "Let's think step by step. There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.\n#### 29",
            },
            {
                "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
                "answer": "Let's think step by step. Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.\n#### 33",
            },
            {
                "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
                "answer": "Let's think step by step. Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.\n#### 8",
            },
        ]

    def get_examplar(self, example: Example) -> str:
        return f"{self.get_prompt(example)} {self.get_target(example)}"

    def get_prompt(self, example: Example) -> str:
        return f"Q: {example['question']}\nA:"

    def get_target(self, example: Example) -> str:
        split = example["answer"].rsplit("\n#### ", maxsplit=1)
        if len(split) != 2:
            return ""
        solution, result = split
        return f"{solution} The answer is {result}."

    def postprocess(self, tokens: List[int]) -> str:
        generation = self.tokenizer.decode(tokens, cut_at_eos=True)
        generation = generation.split("Q: ")[0].split("A:")[0]
        return generation

    def evaluate(self, prediction: str, example: Example) -> Dict[str, float]:
        ground_truths = [self.get_target(example)]
        sample_metrics = {
            "acc": 100 * math_accuracy_score(prediction, ground_truths, self.eval_normalizer),
            "em": 100 * exact_match_score(prediction, ground_truths, self.eval_normalizer),
            "f1": 100 * f1_score(prediction, ground_truths, self.eval_normalizer),
        }
        return sample_metrics

    @staticmethod
    def eval_normalizer(answer: str) -> str:
        return extract_answer(answer)
