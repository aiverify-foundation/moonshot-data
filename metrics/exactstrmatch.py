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
        Asynchronously calculates the accuracy of the predicted results by comparing them to the target results.

        This method evaluates each predicted result against the corresponding target(s) to determine if it is correct.
        It supports both single and multiple target values for comparison.

        Args:
            prompts (Any): The prompts used for prediction.
            predicted_results (Any): The predicted results, each containing a response attribute.
            targets (Any): The target results, which can be a single value or a list of values.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the accuracy of the predicted results.
                - exactstrmatch (dict): Contains the accuracy and individual scores.
                    - accuracy (float): The accuracy percentage of the predicted results.
                    - individual_scores (dict): Contains lists of successful and unsuccessful predictions.
                        - successful (list): A list of dictionaries with correct predictions, including prompt,
                        target, and predicted result.
                        - unsuccessful (list): A list of dictionaries with wrong predictions, including prompt,
                        target, and predicted result.
                - grading_criteria (dict): Contains the accuracy for grading purposes.
        """
        predicted_values = [result.response for result in predicted_results]

        correct = 0
        wrong = 0
        total = len(predicted_values)

        correct_prompts = []
        wrong_prompts = []

        for prompt, result, target in zip(prompts, predicted_values, targets):
            # Check if the target is a single or multiple targets
            if isinstance(target, list):
                if result in target:
                    correct += 1
                    correct_prompts.append(
                        {
                            "prompt": prompt,
                            "predicted_value": result,
                            "target": target,
                            "eval": "correct",
                        }
                    )
                else:
                    wrong += 1
                    wrong_prompts.append(
                        {
                            "prompt": prompt,
                            "predicted_value": result,
                            "target": target,
                            "eval": "wrong",
                        }
                    )
            else:
                if result == target:
                    correct += 1
                    correct_prompts.append(
                        {
                            "prompt": prompt,
                            "predicted_value": result,
                            "target": target,
                            "eval": "correct",
                        }
                    )
                else:
                    wrong += 1
                    wrong_prompts.append(
                        {
                            "prompt": prompt,
                            "predicted_value": result,
                            "target": target,
                            "eval": "wrong",
                        }
                    )

        accuracy = float(correct / total) * 100

        return {
            "exactstrmatch": {
                "accuracy": accuracy,
                "individual_scores": {
                    "unsuccessful": wrong_prompts,
                    "successful": correct_prompts,
                },
            },
            "grading_criteria": {"accuracy": accuracy},
        }
