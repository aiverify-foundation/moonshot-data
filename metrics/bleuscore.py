import statistics
from pathlib import Path
from typing import Any

from moonshot.src.metrics.metric_interface import MetricInterface
from moonshot.src.utils.timeit import timeit
from nltk.translate.bleu_score import sentence_bleu


class BleuScore(MetricInterface):
    def __init__(self):
        self.id = Path(__file__).stem
        self.name = "BleuScore"
        self.description = "Bleuscore uses Bleu to return the various rouge scores."
        self.metric_config = self.get_metrics_configuration(self.id)
        self.endpoints = self.metric_config.get("endpoints", [])
        self.configurations = self.metric_config.get("configurations", {})

    def get_metadata(self) -> dict | None:
        """
        Retrieves and returns the metadata of the BleuScore class.
        The metadata includes the unique identifier, the name, and the description of the class.

        Returns:
            dict | None: A dictionary containing the 'id', 'name', 'description', 'endpoints' 'and configurations'
            of the BleuScore class, or None if not applicable.
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
        Calculate the BLEU score for a list of predicted results and their corresponding target results.

        Args:
            prompts (Any): The prompts used to generate the predicted results.
            predicted_results (Any): The list of predicted results.
            targets (Any): The list of target results.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the BLEU score.

        Raises:
            None
        """
        bleu_scores = []
        for idx, (result, target) in enumerate(zip(predicted_results, targets)):
            output_split = result.split()
            target_split = target.split()

            score = sentence_bleu(output_split, target_split)
            bleu_scores.append(score)

        return {
            "bleu_score": statistics.mean(bleu_scores),
            "grading_criteria": {"bleu_score": statistics.mean(bleu_scores)},
        }
