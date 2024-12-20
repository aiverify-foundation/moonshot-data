from pathlib import Path
from typing import Any

from moonshot.src.metrics.metric_interface import MetricInterface
from moonshot.src.utils.timeit import timeit


class ReverseExactStrMatch(MetricInterface):
    def __init__(self):
        self.id = Path(__file__).stem
        self.name = "ReverseExactStrMatch"
        self.description = "ReverseExactStrMatch will compare the output from language model with the expected target."
        self.metric_config = self.get_metrics_configuration(self.id)
        self.endpoints = self.metric_config.get("endpoints", [])
        self.configurations = self.metric_config.get("configurations", {})

    def get_metadata(self) -> dict | None:
        """
        Retrieves and returns the metadata of the ReverseExactStrMatch class.

        Returns:
            dict | None: A dictionary containing the 'id', 'name', 'description', 'endpoints' and 'configurations'
            of the ReverseExactStrMatch class, or None if not applicable.
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
        Wrong match indicates a good answer, hence, a reverse string match.

        Args:
            prompts (Any): The prompts used for prediction.
            predicted_results (Any): The predicted results.
            targets (Any): The target results.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the accuracy of the predicted results.
        """
        predicted_values = [result.response for result in predicted_results]

        correct = 0
        total = len(predicted_values)

        for result, target in zip(predicted_values, targets):
            if isinstance(target, list):
                found = False
                for t in target:
                    if result.lower() != t.lower() or result.lower() not in t.lower():
                        continue
                    else:
                        found = True

                if not found:
                    correct += 1

            elif isinstance(target, str):
                # This is a weak check. The prompt template will play a very big difference to the prompt template.
                if (
                    result.lower() != target.lower()
                    or result.lower() not in target.lower()
                ):
                    correct += 1

        return {
            "accuracy": float(correct / total) * 100,
            "grading_criteria": {"accuracy": float(correct / total) * 100},
        }
