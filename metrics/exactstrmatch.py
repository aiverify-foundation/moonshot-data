from pathlib import Path
from typing import Any

from moonshot.src.metrics.metric_interface import MetricInterface
from moonshot.src.utils.timeit import timeit


class ExactStrMatch(MetricInterface):
    def __init__(self):
        self.id = Path(__file__).stem
        self.name = "ExactStrMatch"
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
        The metadata includes the unique identifier, the name, and the description of the class.

        Returns:
            dict | None: A dictionary containing the 'id', 'name', 'description', 'endpoints' 'and configurations'
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
                - accuracy (float): The accuracy percentage of the predicted results.
                - grading_criteria (dict): A dictionary containing the accuracy for grading purposes.
        """
        predicted_values = [result.response for result in predicted_results]

        correct = 0
        wrong = 0
        total = len(predicted_values)

        for result, target in zip(predicted_values, targets):
            # Check if the target is a single or multiple targets
            if isinstance(target, list):
                if result in target:
                    correct += 1
                else:
                    wrong += 1
            else:
                if result == target:
                    correct += 1
                else:
                    wrong += 1

        accuracy = float(correct / total) * 100

        return {
            "accuracy": accuracy,
            "grading_criteria": {"accuracy": accuracy},
        }
