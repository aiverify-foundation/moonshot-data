from pathlib import Path
from typing import Any
import re

from moonshot.src.metrics.metric_interface import MetricInterface
from moonshot.src.utils.timeit import timeit


class AAMCQ(MetricInterface):
    def __init__(self):
        self.id = Path(__file__).stem
        self.name = "AAMCQ"
        self.description = (
            "Referenced from Artificial Analysis, this metric extracts multiple choice answers "
            "using a multi-stage approach to handle various answer formats. For single-letter "
            "responses, the letter is used directly. Otherwise, there are fallback attempts to "
            "match the primary pattern that looks for formal \"Answer: X\" format "
            "(accounting for optional markdown formatting)"
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
        Asynchronously calculates the accuracy of the predicted multiple-choice results 
        by comparing them to the target results. Uses a multi-stage regex approach 
        to robustly extract answers in different formats.
        
        Args:
            prompts (Any): The prompts used for prediction.
            predicted_results (Any): The predicted results, each containing a response attribute.
            targets (Any): The target results, which can be a single value or a list of values.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the accuracy of the predicted results.
                - aamcq (dict): Contains the accuracy and individual scores.
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

        # Define regex patterns in priority order
        patterns = [
            r'(?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])',  # Primary Answer: X
            r'\\boxed\{[^}]*([A-Z])[^}]*\}',                                               # LaTeX boxed
            r'answer is ([a-zA-Z])',                                                       # "answer is B"
            r'answer is \(([a-zA-Z])',                                                     # "answer is (C"
            r'([A-Z])\)\s*[^A-Z]*',                                                        # "D) some answer"
            r'([A-Z])\s+is\s+the\s+correct\s+answer',                                      # "E is the correct answer"
            r'([A-Z])\s*$',                                                                # Standalone letter
            r'([A-Z])\s*\.',                                                               # "F."
            r'([A-Z])\s*[^\w]',                                                            # Letter then non-word
        ]

        for prompt, result, target in zip(prompts, predicted_values, targets):
            extracted_answer = None

            for pattern in patterns:
                matches = re.findall(pattern, result)
                if matches:
                    # Always take last to handle self-corrections
                    extracted_answer = matches[-1].upper()
                    break
            
            # If none found, set placeholder
            if not extracted_answer:
                extracted_answer = "Not found"

            target = target.strip().upper()[0]

            if extracted_answer == target:
                correct += 1
                correct_prompts.append(
                    {
                        "prompt": prompt,
                        "predicted_value": result,
                        "extracted_value": extracted_answer,
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
                        "extracted_value": extracted_answer,
                        "target": target,
                        "eval": "wrong",
                    }
                )

        accuracy = float(correct / total) * 100 if total > 0 else 0.0

        return {
            "aamcq": {
                "accuracy": accuracy,
                "individual_scores": {
                    "unsuccessful": wrong_prompts,
                    "successful": correct_prompts,
                },
            },
            "grading_criteria": {"accuracy": accuracy},
        }
