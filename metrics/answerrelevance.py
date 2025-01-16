from pathlib import Path
from typing import Any

from moonshot.src.connectors.connector import Connector
from moonshot.src.connectors_endpoints.connector_endpoint import ConnectorEndpoint
from moonshot.src.metrics.metric_interface import MetricInterface
from moonshot.src.utils.timeit import timeit
from ragas import evaluate
from ragas.metrics import answer_relevancy

from datasets import Dataset


class AnswerRelevance(MetricInterface):
    def __init__(self):
        self.id = Path(__file__).stem
        self.name = "AnswerRelevance"
        self.description = (
            "The evaluation metric, Answer Relevancy, focuses on assessing how pertinent the generated answer is to "
            "the given prompt. A lower score is assigned to answers that are incomplete or contain redundant "
            "information and higher scores indicate better relevancy. This metric is computed using the question, "
            "the context and the answer."
        )
        self.metric_config = self.get_metrics_configuration(self.id)
        self.endpoints = self.metric_config.get("endpoints", [])
        self.configurations = self.metric_config.get("configurations", {})

    def get_metadata(self) -> dict | None:
        """
        Retrieves and returns the metadata of the AnswerRelevance class.
        The metadata includes the unique identifier, the name, and the description of the class.

        Returns:
            dict | None: A dictionary containing the 'id', 'name', 'description', 'endpoints' 'and configurations'
            of the AnswerRelevance class, or None if not applicable.
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
        Asynchronously retrieves the results of the answer relevancy evaluation.

        This method evaluates the relevancy of the generated answers to the given prompts
        using the Ragas framework. It utilizes both an evaluation model and an embeddings model
        to compute the answer relevancy score.

        Args:
            prompts (Any): The input prompts/questions.
            predicted_results (Any): The generated answers to be evaluated, each containing a response
            and context attribute.
            targets (Any): The ground truth answers for comparison.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the answer relevancy scores and individual scores
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
        }
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(
            dataset,
            metrics=[answer_relevancy],
            llm=evaluation_model.get_client(),  # type: ignore ; ducktyping
            embeddings=embeddings_model.get_client(),  # type: ignore ; ducktyping
        )
        df = score.to_pandas()
        answer_relevancy_list = df["answer_relevancy"].tolist()

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
                answer_relevancy_list,
            )
        ]

        return {
            "answerrelevance": {
                "score": answer_relevancy_list,
                "individual_scores": individual_scores,
            },
            "grading_criteria": {},
        }
