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

        Returns:
            dict | None: A dictionary containing the 'id', 'name', 'description', 'endpoints' and 'configurations'
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
        Asynchronously calculates the BLEU score for a list of predicted results and their corresponding target results.

        Args:
            prompts (Any): The input prompts used to generate the predicted results.
            predicted_results (Any): The list of predicted results, each containing a response attribute.
            targets (Any): The list of target results for comparison.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the BLEU score, individual scores, and grading criteria.
                - bleu_score (float): The average BLEU score across all predicted results.
                - individual_scores (list): A list of dictionaries for each sample containing:
                    - prompt (Any): The input prompt.
                    - predicted_value (Any): The predicted result.
                    - target (Any): The target result.
                    - eval (float): The BLEU score for the sample.
                - grading_criteria (dict): A dictionary containing the BLEU score for grading purposes.
        """
        predicted_values = [result.response for result in predicted_results]

        individual_scores = []
        bleu_scores = []
        for prompt, result, target in zip(prompts, predicted_values, targets):
            output_split = result.split()
            target_split = target.split()

            score = sentence_bleu(output_split, target_split)
            bleu_scores.append(score)

            # Calculate individual scores and map them to their corresponding predicted and target values
            individual_scores.append(
                {
                    "prompt": prompt,
                    "predicted_value": result,
                    "target": target,
                    "score": score,
                }
            )

        average_bleu_score = statistics.mean(bleu_scores)

        return {
            "bleuscore": {
                "score": average_bleu_score,
                "individual_scores": individual_scores,
            },
            "grading_criteria": {"bleuscore": average_bleu_score},
        }
