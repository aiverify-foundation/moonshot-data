import statistics
from pathlib import Path
from typing import Any

import nltk
from moonshot.src.metrics.metric_interface import MetricInterface
from moonshot.src.utils.timeit import timeit
from nltk.translate.meteor_score import meteor_score


class MeteorScore(MetricInterface):
    def __init__(self):
        self.id = Path(__file__).stem
        self.name = "MeteorScore"
        self.description = (
            "MeteorScore uses METEOR to score the similarity between the predicted "
            "response and the target, accounting for exact, stem, and synonym matches."
        )
        self.metric_config = self.get_metrics_configuration(self.id)
        self.endpoints = self.metric_config.get("endpoints", [])
        self.configurations = self.metric_config.get("configurations", {})

        # METEOR relies on WordNet for synonym matching. Ensure the corpora are
        # available without forcing the user to run a separate download step.
        for corpus in ("wordnet", "omw-1.4"):
            try:
                nltk.data.find(f"corpora/{corpus}")
            except LookupError:
                nltk.download(corpus, quiet=True)

    def get_metadata(self) -> dict | None:
        """
        Retrieves and returns the metadata of the MeteorScore class.

        Returns:
            dict | None: A dictionary containing the 'id', 'name', 'description',
            'endpoints' and 'configurations' of the MeteorScore class, or None if
            not applicable.
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
        Asynchronously calculates the METEOR score for a list of predicted results
        and their corresponding target results.

        Args:
            prompts (Any): The input prompts used to generate the predicted results.
            predicted_results (Any): The list of predicted results, each containing a
                response attribute.
            targets (Any): The list of target results for comparison.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the average METEOR score, individual scores,
                and grading criteria.
        """
        predicted_values = [result.response for result in predicted_results]

        individual_scores = []
        meteor_scores = []
        for prompt, result, target in zip(prompts, predicted_values, targets):
            # METEOR expects pre-tokenised input: a list of reference token lists
            # and a single hypothesis token list.
            hypothesis_tokens = result.split()
            reference_tokens = target.split()

            score = meteor_score([reference_tokens], hypothesis_tokens)
            meteor_scores.append(score)

            individual_scores.append(
                {
                    "prompt": prompt,
                    "predicted_value": result,
                    "target": target,
                    "score": score,
                }
            )

        average_meteor_score = (
            statistics.mean(meteor_scores) if meteor_scores else 0.0
        )

        return {
            "meteorscore": {
                "score": average_meteor_score,
                "individual_scores": individual_scores,
            },
            "grading_criteria": {"meteorscore": average_meteor_score},
        }
