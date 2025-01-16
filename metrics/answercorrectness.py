from pathlib import Path
from typing import Any

from moonshot.src.connectors.connector import Connector
from moonshot.src.connectors_endpoints.connector_endpoint import ConnectorEndpoint
from moonshot.src.metrics.metric_interface import MetricInterface
from moonshot.src.utils.timeit import timeit
from ragas import evaluate
from ragas.metrics import answer_correctness

from datasets import Dataset


class AnswerCorrectness(MetricInterface):
    def __init__(self):
        self.id = Path(__file__).stem
        self.name = "AnswerCorrectness"
        self.description = (
            "Answer correctness in the context of Ragas involves evaluating the accuracy of a generated answer "
            "compared to the ground truth. This process assesses both the semantic and factual similarities between "
            "the answer and the ground truth. Scores range from 0 to 1, where a higher score indicates a closer "
            "match, thus higher correctness."
        )
        self.metric_config = self.get_metrics_configuration(self.id)
        self.endpoints = self.metric_config.get("endpoints", [])
        self.configurations = self.metric_config.get("configurations", {})

    def get_metadata(self) -> dict | None:
        """
        Retrieves and returns the metadata of the AnswerCorrectness class.
        The metadata includes the unique identifier, the name, and the description of the class.

        Returns:
            dict | None: A dictionary containing the 'id', 'name', 'description', 'endpoints' 'and configurations'
            of the AnswerCorrectness class, or None if not applicable.
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
        Asynchronously retrieves the results of the answer correctness evaluation.

        This method evaluates the accuracy of generated answers compared to the ground truth
        using the Ragas framework. It utilizes both an evaluation model and an embeddings model
        to compute the answer correctness score.

        Args:
            prompts (Any): The input prompts or questions.
            predicted_results (Any): The generated answers to be evaluated.
            targets (Any): The ground truth answers for comparison.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the answer correctness scores and individual scores
            for each prompt, predicted result, and target.
        """
        predicted_values = [result.response for result in predicted_results]

        evaluation_model = [
            Connector.create(ConnectorEndpoint.read(ep_id)) for ep_id in self.endpoints
        ][0]

        embeddings_model = [
            Connector.create(ConnectorEndpoint.read(ep_id))
            for ep_id in self.configurations["embeddings"]
        ][0]

        data_samples = {
            "question": prompts,
            "answer": predicted_values,
            "ground_truth": targets,
        }
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(
            dataset,
            metrics=[answer_correctness],
            llm=evaluation_model.get_client(),  # type: ignore ; ducktyping
            embeddings=embeddings_model.get_client(),  # type: ignore ; ducktyping
        )
        df = score.to_pandas()
        answer_correctness_list = df["answer_correctness"].tolist()

        individual_scores = [
            {
                "prompt": prompt,
                "predicted_value": result,
                "target": target,
                "score": ans_score,
            }
            for prompt, result, target, ans_score in zip(
                prompts, predicted_values, targets, answer_correctness_list
            )
        ]

        return {
            "answercorrectness": {
                "score": answer_correctness_list,
                "individual_scores": individual_scores,
            },
            "grading_criteria": {},
        }
