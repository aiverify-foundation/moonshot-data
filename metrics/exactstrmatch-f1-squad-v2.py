import ast
import collections
import re
import string
from pathlib import Path
from typing import Any, List

from moonshot.src.metrics.metric_interface import MetricInterface
from moonshot.src.utils.timeit import timeit

ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)


class ExactStrMatchF1SquadV2(MetricInterface):
    def __init__(self):
        self.id = Path(__file__).stem
        self.name = "ExactStrMatchF1SquadV2"
        self.description = (
            "ExactStrMatch will compare the output from language model with a single target"
            " or multiple expected target."
        )
        self.metric_config = self.get_metrics_configuration(self.id)
        self.endpoints = self.metric_config.get("endpoints", [])
        self.configurations = self.metric_config.get("configurations", {})

    def get_metadata(self) -> dict | None:
        """
        Retrieves and returns the metadata of the ExactStrMatch class.

        Returns:
            dict | None: A dictionary containing the 'id', 'name', 'description', 'endpoints' and 'configurations'
            of the ExactStrMatch class, or None if not applicable.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "endpoints": self.endpoints,
            "configurations": self.configurations,
        }

    @timeit
    async def get_results(
        self, prompts: Any, predicted_results: Any, targets: Any, *args, **kwargs
    ) -> dict:
        """
        Calculates the accuracy of the predicted results by comparing them to the target results.

        Args:
            prompts (Any): The prompts used for prediction.
            predicted_results (Any): The predicted results.
            targets (Any): The target results.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the accuracy of the predicted results.
        """
        em_scores = 0
        f1_scores = 0
        total = len(predicted_results)
        individual_scores = []

        for prompt, result, target in zip(prompts, predicted_results, targets):
            # Check if the target is a single or multiple targets
            if isinstance(target, str):
                target = ast.literal_eval(target)

            # Process result
            result = result.response.strip()
            if result == "unanswerable":
                result = ""

            # Process target
            target = [t for t in target["text"] if self.normalize_answer(t)]
            if not target:
                # For unanswerable questions, only correct answer is empty string
                target = [""]
            em = max(self.compute_exact(tg, result) for tg in target)
            em_scores += em

            f1 = max(self.compute_f1(tg, result) for tg in target)
            f1_scores += f1

            individual_scores.append(
                {
                    "prompt": prompt,
                    "predicted_value": result,
                    "target": target,
                    "exact_match": em,
                    "f1_score": f1,
                }
            )

        return {
            "exactstrmatch_f1_squad_v2": {
                "em": float(em_scores / total) * 100,
                "f1": float(f1_scores / total) * 100,
                "individual_scores": individual_scores,
            },
            "grading_criteria": {"f1": float(f1_scores / total) * 100},
        }

    def compute_exact(self, a_gold: str, a_pred: str) -> int:
        """
        Computes whether the predicted answer matches the gold answer exactly.

        Args:
            a_gold (str): The gold (correct) answer.
            a_pred (str): The predicted answer.

        Returns:
            int: 1 if the answers match exactly, 0 otherwise.
        """
        return int(self.normalize_answer(a_gold) == self.normalize_answer(a_pred))

    def compute_f1(self, a_gold: str, a_pred: str) -> float:
        """
        Computes the F1 score between the gold answer and the predicted answer.

        Args:
            a_gold (str): The gold (correct) answer.
            a_pred (str): The predicted answer.

        Returns:
            float: The F1 score.
        """
        gold_toks = self.get_tokens(a_gold)
        pred_toks = self.get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def get_tokens(self, s: str) -> List[str]:
        """
        Tokenizes the input string by normalizing it and splitting it into words.

        Args:
            s (str): The input string to tokenize.

        Returns:
            List[str]: A list of tokens.
        """
        if not s:
            return []
        return self.normalize_answer(s).split()

    @staticmethod
    def normalize_answer(s: str) -> str:
        """
        Lower text and remove punctuation, articles and extra whitespace.
        Args:
            s (str): The input string to be normalized.

        Returns:
            str: The normalized string, with lowercase text, no punctuation, no articles, and reduced extra whitespace.
        """

        def remove_articles(text: str) -> str:
            return ARTICLES_REGEX.sub(" ", text)

        def white_space_fix(text: str) -> str:
            return " ".join(text.split())

        def remove_punc(text: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text: str) -> str:
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))
