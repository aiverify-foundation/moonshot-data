from pathlib import Path
from typing import Any

from moonshot.src.metrics.metric_interface import MetricInterface
from moonshot.src.utils.timeit import timeit
from readability import Readability


class ReadabilityScore(MetricInterface):
    def __init__(self):
        self.id = Path(__file__).stem
        self.name = "ReadabilityScore"
        self.description = "ReadabilityScore uses Flesch Reading Ease to compute the complexity of the output"
        self.metric_config = self.get_metrics_configuration(self.id)
        self.endpoints = self.metric_config.get("endpoints", [])
        self.configurations = self.metric_config.get("configurations", {})

    def get_metadata(self) -> dict | None:
        """
        Retrieves and returns the metadata of the ReadabilityScore class.

        Returns:
            dict | None: A dictionary containing the 'id', 'name', 'description', 'endpoints' and 'configurations'
            of the ReadabilityScore class, or None if not applicable.
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
        Asynchronously calculates the readability score using the Flesch-Kincaid method for each predicted result
        and determines the number of valid and invalid responses based on word count.

        Args:
            prompts (Any): The prompts used for generating the predicted results.
            predicted_results (Any): The predicted results, each containing a response attribute.
            targets (Any): The target results for comparison.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the following readability statistics:
                - readabilityscore (dict): A dictionary with:
                    - score (float): The average readability score for valid responses.
                    - valid_response (int): The number of responses with 100 or more words.
                    - invalid_response (list): A list of responses with fewer than 100 words.
                    - individual_scores (list): A list of dictionaries for each response containing:
                        - prompt (Any): The input prompt.
                        - predicted_value (Any): The predicted result.
                        - target (Any): The target result.
                        - score (float): The readability score or -1 for invalid responses.
                - grading_criteria (dict): An empty dictionary for grading criteria.
        """
        predicted_values = [result.response for result in predicted_results]

        results = 0
        individual_scores = []
        num_of_output_more_than_100 = 0
        response_less_than_100 = []

        for prompt, result, target in zip(prompts, predicted_values, targets):
            if len(result.split()) < 100:
                response_less_than_100.append(result)
                individual_scores.append(
                    {
                        "prompt": prompt,
                        "predicted_value": result,
                        "target": target,
                        "score": -1,
                    }
                )
            else:
                r = Readability(result)
                this_score = r.flesch_kincaid()
                individual_scores.append(
                    {
                        "prompt": prompt,
                        "predicted_value": result,
                        "target": target,
                        "score": this_score.score,
                    }
                )
                results += this_score.score
                num_of_output_more_than_100 += 1

        temp_score = (
            results / num_of_output_more_than_100
            if num_of_output_more_than_100 > 0
            else 0
        )

        return {
            "readabilityscore": {
                "score": temp_score,
                "valid_response": len(predicted_results) - len(response_less_than_100),
                "invalid_response": response_less_than_100,
                "individual_scores": individual_scores,
            },
            "grading_criteria": {},
        }
