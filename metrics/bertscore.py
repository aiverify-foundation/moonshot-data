from pathlib import Path
from typing import Any

import bert_score
from moonshot.src.metrics.metric_interface import MetricInterface
from moonshot.src.utils.timeit import timeit


class BertScore(MetricInterface):
    """
    BertScore uses Bert to check for the similarity in embedding between two sentences.
    Code reference from:
    https://github.com/Tiiiger/bert_score/blob/master/bert_score_cli/score.py
    """

    def __init__(self):
        self.id = Path(__file__).stem
        self.name = "BertScore"
        self.description = "BertScore uses Bert to check for the similarity in embedding between two sentences."
        self.metric_config = self.get_metrics_configuration(self.id)
        self.endpoints = self.metric_config.get("endpoints", [])
        self.configurations = self.metric_config.get("configurations", {})

    def get_metadata(self) -> dict | None:
        """
        Retrieves and returns the metadata of the BertScore class.

        Returns:
            dict | None: A dictionary containing the 'id', 'name', 'description', 'endpoints' and 'configurations'
            of the BertScore class, or None if not applicable.
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
        Calculate the BERTScore precision, recall, and F1 score between the predicted results and the target results.

        Args:
            prompts (Any): The input prompts used for generating the predicted results.
            predicted_results (Any): The predicted results generated by the model.
            targets (Any): The target results for comparison.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the BERTScore precision, recall, F1 score, and individual scores.
                - precision (float): The average precision score across all samples.
                - recall (float): The average recall score across all samples.
                - f1 (float): The average F1 score across all samples.
                - individual_scores (list): A list of dictionaries for each sample containing:
                    - prompt (Any): The input prompt.
                    - predicted_value (Any): The predicted result.
                    - target (Any): The target result.
                    - eval (dict): A dictionary containing:
                        - precision (float): The precision score for the sample.
                        - recall (float): The recall score for the sample.
                        - f1 (float): The F1 score for the sample.
                - grading_criteria (dict): An empty dictionary for grading criteria.
        """
        predicted_values = [result.response for result in predicted_results]

        # use default roberta model
        score = bert_score.score(
            predicted_values, targets, lang="en", rescale_with_baseline=True
        )

        # Calculate individual scores and map them to their corresponding predicted and target values
        individual_scores = [
            {
                "prompt": prompt,
                "predicted_value": predicted,
                "target": target,
                "score": {
                    "precision": s[0].cpu().item(),
                    "recall": s[1].cpu().item(),
                    "f1": s[2].cpu().item(),
                },
            }
            for prompt, predicted, target, s in zip(
                prompts, predicted_values, targets, zip(*score)
            )
        ]

        avg_scores = [s.mean(dim=0) for s in score]
        precision_value = avg_scores[0].cpu().item()
        recall_value = avg_scores[1].cpu().item()
        f1_value = avg_scores[2].cpu().item()

        return {
            "bertscore": {
                "precision": precision_value,
                "recall": recall_value,
                "f1": f1_value,
                "individual_scores": individual_scores,
            },
            "grading_criteria": {},
        }
