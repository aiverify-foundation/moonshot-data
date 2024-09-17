from typing import Any

# import aiohttp
# import torch
# from aiohttp import ClientResponse
from moonshot.src.connectors.connector import Connector, perform_retry
from moonshot.src.connectors_endpoints.connector_endpoint_arguments import (
    ConnectorEndpointArguments,
)
from moonshot.src.utils.log import configure_logger

# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import pipeline

# Create a logger for this module
logger = configure_logger(__name__)


class HuggingFaceTransformerConnector(Connector):
    def __init__(self, ep_arguments: ConnectorEndpointArguments):
        # Initialize super class
        super().__init__(ep_arguments)

    @Connector.rate_limited
    @perform_retry
    async def get_response(self, prompt: str) -> str:
        """
        Retrieve and return a response.
        This method is used to retrieve a response, typically from an object or service represented by
        the current instance.

        Returns:
            str: retrieved response data
        """
        connector_prompt = f"{self.pre_prompt}{prompt}{self.post_prompt}"

        model_type = self.optional_params["model"]

        pipe = pipeline("text-generation", model=model_type, max_new_tokens=256)
        output_tokens = pipe(connector_prompt, pad_token_id=pipe.tokenizer.eos_token_id)
        if len(output_tokens) == 1:
            output = pipe(connector_prompt, pad_token_id=pipe.tokenizer.eos_token_id)[
                0
            ]["generated_text"]
            # this is complete the sentence, so we need to remove the prompt that we have sent in
            output = output.replace(connector_prompt, "")
            return output
        else:
            return ""

        # # Merge self.optional_params with additional parameters
        # new_params = {**self.optional_params, "inputs": connector_prompt}
        # async with aiohttp.ClientSession() as session:
        #     async with session.post(
        #         self.endpoint,
        #         headers=self._prepare_headers(),
        #         json=new_params,
        #     ) as response:
        #         return await self._process_response(response)

    def _prepare_headers(self) -> dict:
        """
        Prepare HTTP headers for authentication using a bearer token.

        This function takes a bearer token as input and prepares a dictionary of HTTP headers
        commonly used for authentication in API requests.

        Returns:
            dict: A dictionary containing HTTP headers with the 'Authorization' header set to
            'Bearer <bearer_token>'. This dictionary can be used in API requests for authentication
            purposes.
        """
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    async def _process_response(self, response: Any) -> str:
        """
        Process an HTTP response and extract relevant information as a string.

        This function takes an HTTP response object as input and processes it to extract relevant information
        as a string. The extracted information may include data from the response body, headers, or other attributes.

        Args:
            response (ClientResponse): An HTTP response object containing the response data.

        Returns:
            str: A string representing the relevant information extracted from the response.
        """
        try:
            output = response[0]["generated_text"]
            print(output)
            return output
        except Exception as exception:
            logger.error(
                f"[HuggingFaceConnector] An exception has occurred: {str(exception)}, {await response.json()}"
            )
            raise exception
