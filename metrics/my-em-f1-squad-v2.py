import re
import ast
import string
import collections
from pathlib import Path
from typing import Any

from moonshot.src.metrics.metric_interface import MetricInterface
from moonshot.src.utils.timeit import timeit

ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)
class MyF1Score(MetricInterface):
    def __init__(self):
        self.id = Path(__file__).stem
        self.name = "MyF1Score"
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
        em_results = []
        f1_scores = 0
        f1_results = []
        total = len(predicted_results)

        for idx, (result, target) in enumerate(zip(predicted_results, targets)):
            # Check if the target is a single or multiple targets
            if isinstance(target, str):
                target = ast.literal_eval(target)

            # Process result
            result = result.strip()
            if result == 'unanswerable':
                result = ''

            # Process target
            target = [t for t in target['text'] if self.normalize_answer(t)]
            if not target:
                # For unanswerable questions, only correct answer is empty string
                target = ['']
            em = max(self.compute_exact(tg, result) for tg in target)
            em_scores += em
            em_results.append(em)
            f1 = max(self.compute_f1(tg, result) for tg in target)
            f1_scores += f1
            f1_results.append(f1)

        return {
            "em": float(em_scores / total) * 100,
            "f1": float(f1_scores / total) * 100,
            "em_results": em_results,
            "f1_results": f1_results,
            "grading_criteria": {"f1": float(f1_scores / total) * 100},
        }

    def compute_exact(self, a_gold, a_pred):
        return int(self.normalize_answer(a_gold) == self.normalize_answer(a_pred))

    def compute_f1(self, a_gold, a_pred):
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

    def get_tokens(self, s):
        if not s:
            return []
        return self.normalize_answer(s).split()

    @staticmethod
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return ARTICLES_REGEX.sub(" ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))


