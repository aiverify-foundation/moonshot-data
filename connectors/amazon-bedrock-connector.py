import asyncio

import boto3
from botocore.config import Config
from moonshot.src.connectors.connector import Connector, perform_retry
from moonshot.src.connectors.connector_response import ConnectorResponse
from moonshot.src.connectors_endpoints.connector_endpoint_arguments import (
    ConnectorEndpointArguments,
)
from moonshot.src.utils.log import configure_logger

# Create a logger for this module
logger = configure_logger(__name__)


class AmazonBedrockConnector(Connector):
    """Amazon Bedrock connector for AI Verify Moonshot

    Although you *could* provide a `token` in the configuration for this connector (which would be
    treated as an AWS Session Token), it's recommended to leave this field as a short placeholder
    e.g. 'NONE' and instead configure your AWS credentials via the environment - as standard for
    boto3 and AWS CLI. Any `token` under 30 characters will be ignored as a placeholder.
    For info see: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    In normal usage, you can also leave `uri` as a short placeholder value (like "DEFAULT") because
    this will be automatically inferred based on your AWS environment setup. If you provide a `uri`
    of 8 characters or more, it'll be treated like specifying a boto3 `client.endpoint_url`.

    You can override boto3 Session arguments (like `region_name`, `profile_name` by adding a
    `session` dictionary in the connector's `params`. For available options, see:
    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/core/session.html

    Likewise you can override boto3 Client arguments (like `endpoint_url`) by adding them to a
    `client` dictionary. Within this you can provide `config` options as shown below. See:
    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/core/session.html#boto3.session.Session.client
    https://botocore.amazonaws.com/v1/documentation/api/latest/reference/config.html

    Examples
    --------

    The following configuration JSON example shows customising AWS/boto3 session, client,
    client-config, inference guardrail, and other inference configuration parameters:

    >>> {
    ...     "name": "Amazon Bedrock Claude 3",
    ...     "connector_type": "aws-bedrock-connector",
    ...     "uri": "DEFAULT",
    ...     "token": "NONE",
    ...     "max_calls_per_second": 1,
    ...     "max_concurrency": 1,
    ...     "model": "anthropic.claude-3-sonnet-20240229-v1:0",
    ...     "params": {
    ...         "timeout": 300,
    ...         "max_attempts": 3,
    ...         "temperature": 0.5,
    ...         "client": {
    ...             "endpoint_url": "https://...",
    ...             "config": {
    ...                 "connect_timeout": 10,
    ...                 "read_timeout": 60,
    ...             }
    ...         },
    ...         "session": {
    ...             "region_name": "us-west-2",
    ...         },
    ...         "guardrailConfig": {
    ...             "guardrailIdentifier": "...",
    ...             "guardrailVersion": "DRAFT",
    ...             "trace": "enabled"
    ...         },
    ...         "inferenceConfig": {
    ...             "topP": 0.9
    ...         }
    ...     }
    ... }
    """

    def __init__(self, ep_arguments: ConnectorEndpointArguments):
        # Initialize super class
        super().__init__(ep_arguments)

        # Initialise AWS session:
        session_kwargs = self.optional_params.get("session", {})
        # Provide an option to set AWS access token via moonshot standard, but ignore placeholders
        # like 'NONE' since moonshot currently makes this field mandatory:
        if self.token:
            if len(self.token) < 30:
                logger.info(
                    "Ignoring `token` with %s characters (doesn't look like an aws_session_token)",
                    len(self.token),
                )
            else:
                session_kwargs["aws_session_token"] = self.token
        self._session = boto3.Session(**session_kwargs)

        # Optional advanced configurations for AWS service client:
        client_kwargs = self.optional_params.get("client", {})
        if "config" in client_kwargs:
            # Convert from JSON configuration dictionary to boto3 Python class:
            client_kwargs["config"] = Config(**client_kwargs["config"])
        # Provide an option to set endpoint_url via moonshot standard, but ignore placeholders
        # like 'DEFAULT' since moonshot currently makes this field mandatory:
        if self.endpoint:
            if len(self.endpoint) < 8:
                logger.info(
                    "Ignoring placeholder `endpoint` (doesn't look like an AWS endpoint). Got: %s",
                    self.endpoint,
                )
            elif "endpoint_url" in client_kwargs:
                logger.info(
                    "Configured `client.endpoint_url` %s override configured `endpoint` %s",
                    client_kwargs["endpoint_url"],
                    self.endpoint,
                )
            else:
                client_kwargs["endpoint_url"] = self.endpoint
        self._client = self._session.client("bedrock-runtime", **client_kwargs)

    @Connector.rate_limited
    @perform_retry
    async def get_response(self, prompt: str) -> ConnectorResponse:
        """Asynchronously send a prompt to the Amazon Bedrock API and return the generated response

        This method uses the Bedrock Converse API, which provides more cross-model standardization
        than the basic InvokeModel API:
        https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
        https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html

        Your `prompt` (plus this connector's configured `pre_prompt` prefix and `post_prompt`
        suffix) will be sent to the model as a single user message. If a `system_message` has been
        configured in the connector params, it will be passed as a `system` message in the Converse
        API. Likewise `guardrailConfig` and `inferenceConfig` can be specified in connector params
        and will be passed through.

        In the unexpected event that the model returns anything other than a single text message
        (e.g. multiple messages or multi-media responses), this function will ignore any non-text
        content and concatenate multiple text messages with double-newlines.

        Args:
            prompt (str): The input prompt to send to the model.

        Returns:
            ConnectorResponse: An object containing the text response generated by the selected model.
        """
        connector_prompt = f"{self.pre_prompt}{prompt}{self.post_prompt}"
        req_params = {
            "modelId": self.model,
            "messages": [
                {"role": "user", "content": [{"text": connector_prompt}]},
            ],
        }
        for key in ["inferenceConfig", "guardrailConfig"]:
            if key in self.optional_params:
                req_params[key] = self.optional_params[key]

        # aioboto3 requires clients to be used as async context managers (so would either need to
        # recreate the client for every request or otherwise hack around to work in Moonshot's API)
        # - so we'll use the official boto3 SDK (synchronous) client and just wrap it with asyncio:
        response = await asyncio.to_thread(lambda: self._client.converse(**req_params))
        message = response["output"]["message"]
        if (
            (not message)
            or message["role"] != "assistant"
            or len(message["content"]) < 1
        ):
            raise ValueError(
                "Bedrock response did not include an assistant message with content. Got: %s",
                message,
            )
        # Ignore any non-text contents, and join together with '\n\n' if multiple are returned:
        return ConnectorResponse(
            response="\n\n".join(
                map(
                    lambda m: m["text"],
                    filter(lambda m: "text" in m, message["content"]),
                )
            )
        )
