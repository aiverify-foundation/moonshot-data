import json

import httpx

from moonshot.src.connectors.connector import Connector, perform_retry
from moonshot.src.connectors.connector_response import ConnectorResponse
from moonshot.src.connectors_endpoints.connector_endpoint_arguments import (
    ConnectorEndpointArguments,
)


class ConstraintLayerConnector(Connector):
    def __init__(self, ep_arguments: ConnectorEndpointArguments):
        super().__init__(ep_arguments)

    @Connector.rate_limited
    @perform_retry
    async def get_response(self, prompt: str) -> ConnectorResponse:
        connector_prompt = prompt
        if self.pre_prompt:
            connector_prompt = f"{self.pre_prompt}{prompt}"
        if self.post_prompt:
            connector_prompt = f"{connector_prompt}{self.post_prompt}"

        payload = {"user_input": connector_prompt}

        headers = {
            "Content-Type": "application/json",
            "Cookie": "humans_21909=1",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
        }

        timeout = httpx.Timeout(
            connect=10.0,
            read=120.0,
            write=10.0,
            pool=10.0,
        )

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                self.endpoint,
                json=payload,
                headers=headers,
            ) as response:
                response.raise_for_status()
                full_text = ""
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data)
                        if "content" in parsed:
                            full_text += parsed["content"]
                        elif "error" in parsed:
                            raise RuntimeError(parsed["error"])
                    except json.JSONDecodeError:
                        continue

        return ConnectorResponse(response=full_text)
