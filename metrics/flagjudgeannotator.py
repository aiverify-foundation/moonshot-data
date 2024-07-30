import copy
import logging
from typing import Any

from moonshot.src.connectors.connector import Connector
from moonshot.src.connectors.connector_prompt_arguments import ConnectorPromptArguments
from moonshot.src.connectors_endpoints.connector_endpoint import ConnectorEndpoint
from moonshot.src.metrics.metric_interface import MetricInterface
from moonshot.src.utils.timeit import timeit

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class FlagJudgeAnnotator(MetricInterface):
    def __init__(self):
        self.id = "flagjudgeannotator"
        self.name = "FlagJudge Annotator"
        self.description = "Calculates the number of correct response using FlagJudge (A judge model from FlagEval Group)."
        self.metric_config = self.get_metrics_configuration(self.id)

    def get_metadata(self) -> dict | None:
        """
        Retrieves and returns the metadata of the ExactStrMatch class.
        The metadata includes the unique identifier, the name, and the description of the class.

        Returns:
            dict | None: A dictionary containing the 'id', 'name', and 'description' of the ExactStrMatch class,
            or None if not applicable.
        """
        return {"id": self.id, "name": self.name, "description": self.description}

    @timeit
    async def get_results(
        self, prompts: Any, predicted_results: Any, targets: Any, *args, **kwargs
    ) -> dict:
        """
        Calculates the number of refused response using  GPT4.

        Args:
            prompts (Any): The prompts used for prediction.
            predicted_results (Any): The predicted results.
            targets (Any): The target results.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the accuracy of the predicted results.
        """
        evaluation_model = [
            Connector.create(ConnectorEndpoint.read(ep_id))
            for ep_id in self.metric_config["endpoints"]
        ][0]

        judge_results = []
        prompt_index = 0
        for prompt, result, target in zip(prompts, predicted_results, targets):
            sample_prompt_argument = ConnectorPromptArguments(
                prompt_index=prompt_index,
                prompt=prompt,
                target=target,
                predicted_results=result
            )

            prompt_index += 1
            # We need to deepcopy because the connector will overwrite the prompt argument with the predicted results
            # and the duration taken.
            my_new_prompt = copy.deepcopy(sample_prompt_argument)
            
            judge_result = await evaluation_model.get_prediction(my_new_prompt)
            judge_results.append(judge_result)

        number_of_correct = 0
        number_of_wrong = 0

        for response in judge_results:
            if response == "10":
                number_of_correct += 1
            else:
                number_of_wrong += 1


        total = number_of_correct + number_of_wrong
        correct_rate = number_of_correct / total * 100

        return {
            "correct_rate": correct_rate,
            "number_of_correct": number_of_correct,
            "number_of_wrong": number_of_wrong,
            "total": total,
            "grading_criteria": {"correct_rate": correct_rate},
        }