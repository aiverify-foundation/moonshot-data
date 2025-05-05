from pathlib import Path
from typing import Any

from moonshot.src.metrics.metric_interface import MetricInterface
from moonshot.src.utils.timeit import timeit
from transformers import pipeline


class SpellingScore(MetricInterface):
    """
    SpellingScore uses Levenshetein Distance to find permutations within an edit distance of 2 form the original word
    before comparing to known words in a word frequency list.
    This code uses pyspellchecker (https://pypi.org/project/pyspellchecker/).
    """

    def __init__(self):
        self.id = Path(__file__).stem
        self.name = "SpellingScore"
        self.description = (
            "SpellingScore uses Levenshetein Distance to find permutations within an edit distance of 2 "
            "from the original word before comparing to known words in a word frequency list."
        )
        self.metric_config = self.get_metrics_configuration(self.id)
        self.endpoints = self.metric_config.get("endpoints", [])
        self.configurations = self.metric_config.get("configurations", {})

    def get_metadata(self) -> dict | None:
        """
        Retrieves and returns the metadata of the SpellingScore class.

        Returns:
            dict | None: A dictionary containing the 'id', 'name', 'description', 'endpoints' and 'configurations'
            of the SpellingScore class, or None if not applicable.
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
        Computes spelling correction results for given prompts, predicted results, and targets.

        Args:
            prompts (Any): The input prompts for the function.
            predicted_results (Any): The predicted results to be evaluated, each containing a response attribute.
            targets (Any): The target results for comparison.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the overall spelling score and individual scores for each input.
                  The 'spelling' key contains 'score' with corrected results and a spelling score, and
                  'individual_scores' with detailed information for each input, including the corrected text,
                  misspelled words, and counts of total and misspelled words.
        """
        predicted_values = [result.response for result in predicted_results]

        results = {}
        total_number_of_words = 0
        total_number_of_misspelled = 0

        fix_spelling = pipeline(
            "text2text-generation", model="oliverguhr/spelling-correction-english-base"
        )

        individual_scores = []  # Initialize the list to store individual results

        index = 0
        for prompt, result, target in zip(prompts, predicted_values, targets):
            this_result = {}

            corrected = fix_spelling(result, max_length=4096)[0]["generated_text"]
            corrected_split = corrected.split()
            difference = list(set(result.split()) - set(corrected_split))

            if len(corrected) != len(result):
                this_result["corrected"] = corrected
                this_result[
                    "error"
                ] = "Length of corrected text is not the same as given generated output."
            else:
                this_result["misspell"] = difference
                this_result["total_number_of_words"] = len(result)
                this_result["total_number_of_misspelled"] = len(difference)

            total_number_of_words += len(result)
            total_number_of_misspelled += len(difference)
            results[index] = this_result

            # Add individual score details
            individual_scores.append(
                {
                    "prompt": prompt,
                    "predicted_value": result,
                    "target": target,
                    "score": {
                        "results": this_result,
                        "spelling_score": (
                            total_number_of_words - total_number_of_misspelled
                        )
                        / total_number_of_words,
                    },
                }
            )

            index += 1

        scores = {
            "corrected": results,
            "spelling_score": (total_number_of_words - total_number_of_misspelled)
            / total_number_of_words,
        }
        return {
            "spelling": {
                "score": scores,
                "individual_scores": individual_scores,
            },
            "grading_criteria": {},
        }
