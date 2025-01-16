from pathlib import Path
from typing import Any

from moonshot.src.connectors.connector import Connector
from moonshot.src.connectors_endpoints.connector_endpoint import ConnectorEndpoint
from moonshot.src.metrics.metric_interface import MetricInterface
from moonshot.src.utils.timeit import timeit
from ragas import evaluate
from ragas.metrics import faithfulness

from datasets import Dataset


class Faithfulness(MetricInterface):
    def __init__(self):
        self.id = Path(__file__).stem
        self.name = "Faithfulness"
        self.description = (
            "This measures the factual consistency of the generated answer against the given context. It is "
            "calculated from answer and retrieved context. The answer is scaled to (0,1) range. Higher the better."
        )
        self.metric_config = self.get_metrics_configuration(self.id)
        self.endpoints = self.metric_config.get("endpoints", [])
        self.configurations = self.metric_config.get("configurations", {})

    def get_metadata(self) -> dict | None:
        """
        Retrieves and returns the metadata of the Faithfulness class.
        The metadata includes the unique identifier, the name, and the description of the class.

        Returns:
            dict | None: A dictionary containing the 'id', 'name', 'description', 'endpoints' 'and configurations'
            of the Faithfulness class, or None if not applicable.
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
        Asynchronously evaluates the faithfulness of generated answers.

        This method assesses the factual consistency of generated answers against the provided context
        using the Ragas framework. It utilizes both an evaluation model and an embeddings model
        to calculate the faithfulness score.

        Args:
            prompts (Any): The input prompts or questions.
            predicted_results (Any): The generated answers to be evaluated, each containing a response and context.
            targets (Any): The ground truth answers for comparison.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the faithfulness scores and individual evaluation details.
        """
        predicted_values = [result.response for result in predicted_results]
        predicted_contexts = [result.context for result in predicted_results]

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
            "contexts": predicted_contexts,
        }
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(
            dataset,
            metrics=[faithfulness],
            llm=evaluation_model.get_client(),  # type: ignore ; ducktyping
            embeddings=embeddings_model.get_client(),  # type: ignore ; ducktyping
        )
        df = score.to_pandas()
        faithfulness_list = df["faithfulness"].tolist()

        individual_scores = [
            {
                "prompt": prompt,
                "predicted_value": result,
                "predicted_context": context,
                "target": target,
                "score": ans_score,
            }
            for prompt, result, context, target, ans_score in zip(
                prompts,
                predicted_values,
                predicted_contexts,
                targets,
                faithfulness_list,
            )
        ]

        return {
            "faithfulness": {
                "score": faithfulness_list,
                "individual_scores": individual_scores,
            },
            "grading_criteria": {},
        }
