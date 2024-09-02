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

    def get_metadata(self) -> dict | None:
        """
        Retrieves and returns the metadata of the AnswerCorrectness class.
        The metadata includes the unique identifier, the name, and the description of the class.

        Returns:
            dict | None: A dictionary containing the 'id', 'name', and 'description' of the AnswerCorrectness
            class, or None if not applicable.
        """
        return {"id": self.id, "name": self.name, "description": self.description}

    @timeit
    async def get_results(
        self, prompts: Any, predicted_results: Any, targets: Any, *args, **kwargs
    ) -> dict:
        """
        Asynchronously retrieves the results of the answer correctness evaluation.

        This method evaluates the accuracy of generated answers compared to the ground truth
        using the Ragas framework. It leverages both an evaluation model and an embeddings model
        to compute the answer correctness score.

        Args:
            prompts (Any): The input prompts/questions.
            predicted_results (Any): The generated answers to be evaluated.
            targets (Any): The ground truth answers for comparison.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the answer correctness scores and grading criteria.
        """
        evaluation_model = [
            Connector.create(ConnectorEndpoint.read(ep_id))
            for ep_id in self.metric_config["endpoints"]
        ][0]

        embeddings_model = [
            Connector.create(ConnectorEndpoint.read(ep_id))
            for ep_id in self.metric_config["embeddings"]
        ][0]

        data_samples = {
            "question": prompts,
            "answer": predicted_results,
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
        return {
            "answer_correctness": df["answer_correctness"].tolist(),
            "grading_criteria": {},
        }
