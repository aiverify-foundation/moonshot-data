import re
from pathlib import Path
from typing import Any

from moonshot.src.metrics.metric_interface import MetricInterface
from moonshot.src.utils.timeit import timeit


class RelaxStrMatch(MetricInterface):
    """
    RelaxStrMatch will remove symbols and spaces before comparing the output from language model with the expected
    target.
    """

    def __init__(self):
        self.id = Path(__file__).stem
        self.name = "RelaxStrMatch"
        self.description = (
            "RelaxStrMatch will remove symbols and spaces before comparing the output from language "
            "model with the expected target."
        )
        self.metric_config = self.get_metrics_configuration(self.id)
        self.endpoints = self.metric_config.get("endpoints", [])
        self.configurations = self.metric_config.get("configurations", {})

    def get_metadata(self) -> dict | None:
        """
        Retrieves and returns the metadata of the RelaxStrMatch class.

        Returns:
            dict | None: A dictionary containing the 'id', 'name', 'description', 'endpoints' and 'configurations'
            of the RelaxStrMatch class, or None if not applicable.
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
        Computes the accuracy of predicted results by comparing them to the target results after removing symbols
        and spaces.

        Args:
            prompts (Any): The prompts used for generating predictions.
            predicted_results (Any): The predicted results, each containing a response attribute.
            targets (Any): The expected target results for comparison.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing:
                - "relaxstrmatch": A dictionary with:
                    - "accuracy": The percentage of correct predictions.
                    - "individual_scores": A dictionary with lists of prompts categorized as "unsuccessful" (wrong)
                                           and "successful" (correct).
                - "grading_criteria": A dictionary with the accuracy percentage.
        """
        predicted_values = [result.response for result in predicted_results]

        correct = 0
        wrong = 0
        total = len(predicted_values)

        correct_prompts = []
        wrong_prompts = []

        for prompt, result, target in zip(prompts, predicted_values, targets):
            # Remove symbols and spaces
            result = re.sub(r"[^\w]", "", str(result).rstrip()).replace(" ", "")
            result = result.lower()

            # Support for benchmarks with multiple possible answers
            if isinstance(target, list):
                matched = False
                for each_item in target:
                    each_item = re.sub(r"[^\w]", "", str(each_item).rstrip()).replace(
                        " ", ""
                    )
                    each_item = each_item.lower()

                    if result == each_item:
                        correct += 1
                        correct_prompts.append(
                            {
                                "prompt": prompt,
                                "predicted_value": result,
                                "target": target,
                                "eval": "correct",
                            }
                        )
                        matched = True
                        break

                if not matched:
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
                target = re.sub(r"[^\w\s]", "", str(target).rstrip()).replace(" ", "")
                target = target.lower()

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
            "relaxstrmatch": {
                "accuracy": accuracy,
                "individual_scores": {
                    "unsuccessful": wrong_prompts,
                    "successful": correct_prompts,
                },
            },
            "grading_criteria": {"accuracy": accuracy},
        }
