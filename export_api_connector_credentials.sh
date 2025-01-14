# This script is responsible for exporting environment variables that store API keys and endpoints
# for various connectors used in the project. These connectors include services like OpenAI, Together,
# Azure OpenAI, Flageval, Anthropic, and Google Gemini. The script ensures that the necessary credentials
# and configurations are available for the application to interact with these external services.

# Note: Values specified in connector-endpoints configurations take precedence over these environment variables.
# To run this script in the terminal and export the environment variables, use the following command:
# . export_api_connector_credentials.sh

# openai-connector
# openai-t2i-connector
export OPENAI_API_KEY=

# together-connector
export TOGETHER_API_KEY=

# azure-openai-connector
# azure-openai-t2i-connector
# azure-langchain-openai-chatopenai-connector
# azure-langchain-openai-embedding-connector
export AZURE_OPENAI_ENDPOINT=
export AZURE_OPENAI_API_KEY=
export AZURE_OPENAI_VERSION=

# flageval-connector
export FLAG_EVAL_API_KEY=

# anthropic-connector
export ANTHROPIC_API_KEY=

# huggingface-connector
export HUGGINGFACE_API_URL=
export HUGGINGFACE_API_KEY=

# google-gemini-connector
export GOOGLE_API_KEY=

# amazon-bedrock-connector
export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
export AWS_DEFAULT_REGION=
