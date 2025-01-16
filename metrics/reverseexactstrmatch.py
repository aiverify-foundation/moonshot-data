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
        Calculates the reverse accuracy of the predicted results by comparing them to the target results.
        In this context, a mismatch between the predicted result and the target is considered correct,
        hence, a reverse string match.

        Args:
            prompts (Any): The prompts used for generating predictions.
            predicted_results (Any): The predicted results, each containing a response attribute.
            targets (Any): The expected target results for comparison.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing:
                - "reverseexactstrmatch": A dictionary with:
                    - "accuracy": The percentage of correct reverse matches.
                    - "individual_scores": A dictionary with lists of prompts categorized as "unsuccessful" (matches)
                    and "successful" (mismatches).
                - "grading_criteria": A dictionary with the reverse accuracy percentage.
        """
        predicted_values = [result.response for result in predicted_results]

        correct = 0
        wrong = 0
        total = len(predicted_values)

        correct_prompts = []
        wrong_prompts = []

        for prompt, result, target in zip(prompts, predicted_values, targets):
            if isinstance(target, list):
                found = False
                for t in target:
                    if result.lower() != t.lower() or result.lower() not in t.lower():
                        continue
                    else:
                        found = True
                        break

                if not found:
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

            elif isinstance(target, str):
                # This is a weak check. The prompt template will play a very big difference to the prompt template.
                if (
                    result.lower() != target.lower()
                    or result.lower() not in target.lower()
                ):
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
            "reverseexactstrmatch": {
                "accuracy": accuracy,
                "individual_scores": {
                    "unsuccessful": wrong_prompts,
                    "successful": correct_prompts,
                },
            },
            "grading_criteria": {"accuracy": accuracy},
        }
