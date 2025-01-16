import copy
from typing import Any

from moonshot.src.connectors.connector import Connector
from moonshot.src.connectors.connector_prompt_arguments import ConnectorPromptArguments
from moonshot.src.connectors_endpoints.connector_endpoint import ConnectorEndpoint
from moonshot.src.metrics.metric_interface import MetricInterface
from moonshot.src.utils.timeit import timeit

DEFAULT_EVALUATION_MODEL = "flageval-flagjudge"


class FlagJudgeAnnotator(MetricInterface):
    def __init__(self):
        self.id = "flagjudgeannotator"
        self.name = "FlagJudge Annotator"
        self.description = "Calculates the number of correct response using FlagJudge (A judge model from FlagEval Group)."  # noqa: E501
        self.metric_config = self.get_metrics_configuration(self.id)
        self.endpoints = self.metric_config.get("endpoints", [DEFAULT_EVALUATION_MODEL])
        self.configurations = self.metric_config.get("configurations", {})

    def get_metadata(self) -> dict | None:
        """
        Retrieves and returns the metadata of the FlagJudgeAnnotator class.

        Returns:
            dict | None: A dictionary containing the 'id', 'name', 'description', 'endpoints' and 'configurations'
            of the FlagJudgeAnnotator class, or None if not applicable.
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
        Asynchronously calculates the number of correct responses using the FlagJudge model.

        This method evaluates the predicted results against the target results to determine the correctness
        of each response, categorizing them as correct or wrong.

        Args:
            prompts (Any): The prompts used for prediction.
            predicted_results (Any): The predicted results.
            targets (Any): The target results.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the correct rate, number of correct and wrong responses,
                  total number of responses, and individual scores for each prompt.
        """
        evaluation_model = [
            Connector.create(ConnectorEndpoint.read(ep_id)) for ep_id in self.endpoints
        ][0]

        judge_results = []
        prompt_index = 0
        for prompt, result, target in zip(prompts, predicted_results, targets):
            sample_prompt_argument = ConnectorPromptArguments(
                prompt_index=prompt_index,
                prompt=prompt,
                target=target,
                predicted_results=result,
            )

            prompt_index += 1
            # We need to deepcopy because the connector will overwrite the prompt argument with the predicted results
            # and the duration taken.
            my_new_prompt = copy.deepcopy(sample_prompt_argument)

            judge_result = await evaluation_model.get_prediction(my_new_prompt)
            judge_results.append({"prompt": my_new_prompt, "result": judge_result})

        number_of_correct = 0
        number_of_wrong = 0

        correct_prompts = []
        wrong_prompts = []

        for temp in judge_results:
            prompt_info = temp.get("prompt")
            response = temp.get("result")
            if response == "10":
                number_of_correct += 1
                correct_prompts.append(
                    {
                        "prompt": prompt_info.prompt,
                        "predicted_value": prompt_info.predicted_results.response,
                        "target": prompt_info.target,
                        "eval_predicted_value": response,
                        "eval": "correct",
                    }
                )
            else:
                number_of_wrong += 1
                wrong_prompts.append(
                    {
                        "prompt": prompt_info.prompt,
                        "predicted_value": prompt_info.predicted_results.response,
                        "target": prompt_info.target,
                        "eval_predicted_value": response,
                        "eval": "wrong",
                    }
                )

        total = number_of_correct + number_of_wrong
        correct_rate = number_of_correct / total * 100

        return {
            "flagjudgeannotator": {
                "correct_rate": correct_rate,
                "number_of_correct": number_of_correct,
                "number_of_wrong": number_of_wrong,
                "total": total,
                "individual_scores": {
                    "unsuccessful": wrong_prompts,
                    "successful": correct_prompts,
                },
            },
            "grading_criteria": {"correct_rate": correct_rate},
        }
