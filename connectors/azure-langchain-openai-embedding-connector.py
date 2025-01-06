import logging
import os
from typing import Any

from langchain_openai.embeddings import AzureOpenAIEmbeddings
from moonshot.src.connectors.connector import Connector, perform_retry
from moonshot.src.connectors.connector_response import ConnectorResponse
from moonshot.src.connectors_endpoints.connector_endpoint_arguments import (
    ConnectorEndpointArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureLangchainOpenAIEmbeddingConnector(Connector):
    def __init__(self, ep_arguments: ConnectorEndpointArguments):
        # Initialize super class
        super().__init__(ep_arguments)

        # Initialize Azure OpenAI client with additional parameters
        # Select API key from token attribute or environment variable 'AZURE_OPENAI_API_KEY'
        api_key = self.token or os.getenv("AZURE_OPENAI_API_KEY") or ""

        # Select API version from optional parameters or environment variable 'AZURE_OPENAI_VERSION'
        # Default to '2024-02-01' if neither is provided
        api_version = (
            self.optional_params.get("api_version", "")
            or os.getenv("AZURE_OPENAI_VERSION")
            or "2024-02-01"
        )

        # Select API endpoint from endpoint attribute or environment variable 'AZURE_OPENAI_ENDPOINT'
        # Use an empty string if neither is provided
        api_endpoint = self.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT") or ""

        # Set OpenAI Key
        self._client = AzureOpenAIEmbeddings(
            azure_endpoint=api_endpoint,
            azure_deployment=self.model,
            model=self.model,
            api_key=api_key,
            api_version=api_version,
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
    async def get_response(self, prompt: str) -> ConnectorResponse:
        """
        This method is not implemented and will raise a NotImplementedError.

        Args:
            prompt (str): The input prompt to send to the OpenAI API.

        Returns:
            ConnectorResponse: This method does not return a response as it is not implemented.

        Raises:
            NotImplementedError: Always raised to indicate this method is not implemented.
        """
        raise NotImplementedError(
            "This connector is not supposed to generate response."
        )
