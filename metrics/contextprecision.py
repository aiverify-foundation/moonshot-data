from pathlib import Path
from typing import Any

from moonshot.src.connectors.connector import Connector
from moonshot.src.connectors_endpoints.connector_endpoint import ConnectorEndpoint
from moonshot.src.metrics.metric_interface import MetricInterface
from moonshot.src.utils.timeit import timeit
from ragas import evaluate
from ragas.metrics import context_precision

from datasets import Dataset


class ContextPrecision(MetricInterface):
    def __init__(self):
        self.id = Path(__file__).stem
        self.name = "ContextPrecision"
        self.description = (
            "Context Precision is a metric that evaluates whether all of the ground-truth relevant items present in "
            "the contexts are ranked higher or not. Ideally, all the relevant chunks must appear at the top ranks. "
            "This metric is computed using the question, ground_truth, and the contexts, with values ranging between "
            "0 and 1, where higher scores indicate better precision."
        )
        self.metric_config = self.get_metrics_configuration(self.id)
        self.endpoints = self.metric_config.get("endpoints", [])
        self.configurations = self.metric_config.get("configurations", {})

    def get_metadata(self) -> dict | None:
        """
        Retrieves and returns the metadata of the ContextPrecision class.
        The metadata includes the unique identifier, the name, and the description of the class.

        Returns:
            dict | None: A dictionary containing the 'id', 'name', 'description', 'endpoints' 'and configurations'
            of the ContextPrecision class, or None if not applicable.
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
        Asynchronously evaluates the context precision of generated answers.

        This method assesses the precision of retrieved contexts against the ground truth
        using the Ragas framework. It utilizes both an evaluation model and an embeddings model
        to calculate the context precision score.

        Args:
            prompts (Any): The input prompts or questions.
            predicted_results (Any): The generated answers to be evaluated, each containing a response
                                     and context attribute.
            targets (Any): The ground truth answers for comparison.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the context precision scores and individual evaluation details
                  for each prompt, predicted result, and target.
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
            "ground_truth": targets,
        }
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(
            dataset,
            metrics=[context_precision],
            llm=evaluation_model.get_client(),  # type: ignore ; ducktyping
            embeddings=embeddings_model.get_client(),  # type: ignore ; ducktyping
        )
        df = score.to_pandas()
        context_precision_list = df["context_precision"].tolist()

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
                context_precision_list,
            )
        ]

        return {
            "contextprecision": {
                "score": context_precision_list,
                "individual_scores": individual_scores,
            },
            "grading_criteria": {},
        }
