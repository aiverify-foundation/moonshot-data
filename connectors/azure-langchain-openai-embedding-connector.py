import logging
from typing import Any

from langchain_openai.embeddings import AzureOpenAIEmbeddings
from moonshot.src.connectors.connector import Connector, perform_retry
from moonshot.src.connectors_endpoints.connector_endpoint_arguments import (
    ConnectorEndpointArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureLangchainOpenAIEmbeddingConnector(Connector):
    def __init__(self, ep_arguments: ConnectorEndpointArguments):
        # Initialize super class
        super().__init__(ep_arguments)

        # Azure OpenAI has additional parameters
        self.api_version = self.optional_params.get("api_version", "2024-02-01")

        # Set the model to use and remove it from optional_params if it exists
        self.model = self.optional_params.get("model", "")

        # Set OpenAI Key
        self._client = AzureOpenAIEmbeddings(
            azure_endpoint=self.endpoint,
            azure_deployment=self.model,
            model=self.model,
            api_key=self.token,
            api_version=self.api_version,
        )

    def get_client(self) -> Any:
        """
        Retrieve the Azure OpenAI client instance.

        This method returns the Azure OpenAI client instance that is used to interact with the Azure OpenAI API.

        Returns:
            Any: The Azure OpenAI client instance.
        """
        return self._client

    @Connector.rate_limited
    @perform_retry
    async def get_response(self, prompt: str) -> str:
        """
        This method is not implemented and will raise a NotImplementedError.

        Args:
            prompt (str): The input prompt to send to the OpenAI API.

        Returns:
            str: This method does not return a response as it is not implemented.

        Raises:
            NotImplementedError: Always raised to indicate this method is not implemented.
        """
        raise NotImplementedError(
            "This connector is not supposed to generate response."
        )
