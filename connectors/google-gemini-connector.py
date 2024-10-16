import logging
from typing import Any

import google.generativeai as genai
from moonshot.src.connectors.connector import Connector, perform_retry
from moonshot.src.connectors_endpoints.connector_endpoint_arguments import (
    ConnectorEndpointArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleGeminiConnector(Connector):
    def __init__(self, ep_arguments: ConnectorEndpointArguments):
        # Initialize super class
        super().__init__(ep_arguments)

        # Set Google Gemini Key
        self._client = genai
        self._client.configure(api_key=self.token)

        # Set the model to use and remove it from optional_params if it exists
        self.model = self.optional_params.get("model", "")

    @Connector.rate_limited
    @perform_retry
    async def get_response(self, prompt: str) -> str:
        """
        Asynchronously sends a prompt to the Google Gemini API and returns the generated response.

        This method constructs a request with the given prompt, optionally prepended and appended with
        predefined strings, and sends it to the Google Gemini API. If a system prompt is set, it is included in the
        request. The method then awaits the response from the API, processes it, and returns the resulting message
        content as a string.

        Args:
            prompt (str): The input prompt to send to the Google Gemini API.

        Returns:
            str: The text response generated by the Google Gemini model.
        """
        connector_prompt = f"{self.pre_prompt}{prompt}{self.post_prompt}"

        model = self._client.GenerativeModel(
            model_name=self.model,
            system_instruction=self.system_prompt if self.system_prompt else None,
        )

        filtered_optional_params = {
            k: v for k, v in self.optional_params.items() if k != "model"
        }

        generation_config = genai.GenerationConfig(**filtered_optional_params)

        response = model.generate_content(
            connector_prompt, generation_config=generation_config
        )
        return await self._process_response(response)

    async def _process_response(self, response: Any) -> str:
        """
        Process the response from Google Gemini's API and return the message content as a string.

        This method processes the response received from Google Gemini's API call, specifically targeting
        the chat completion response structure. It extracts the message content from the first choice
        provided in the response, which is expected to contain the relevant information or answer.

        Args:
            response (Any): The response object received from a Google Gemini API call. It is expected to
            follow the structure of Google Gemini's chat completion response.

        Returns:
            str: A string containing the message content from the first choice in the response. This
            content represents the AI-generated text based on the input prompt.
        """
        return response.text
