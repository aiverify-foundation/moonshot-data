#!/bin/bash

# Install moonshot package with all dependencies
echo "Installing moonshot..."
pip install "aiverify-moonshot[all]"

# Clone the forked repository and install its requirements
echo "Cloning forked repository and installing requirements..."
git clone https://github.com/$GITHUB_MOONSHOT_CI_REPO_LOC
pip install -r moonshot-data/requirements.txt

# Set logging information
echo "Setting log info..."
export MS_LOG_NAME=moonshot
export MS_LOG_LEVEL=debug
export MS_LOG_TO_FILE=true

# Create environment variables file
echo "Creating env file..."
cat <<EOF >> .env
ATTACK_MODULES="./moonshot-data/attack-modules"
BOOKMARKS="./moonshot-data/generated-outputs/bookmarks"
CONNECTORS="./moonshot-data/connectors"
CONNECTORS_ENDPOINTS="./moonshot-data/connectors-endpoints"
CONTEXT_STRATEGY="./moonshot-data/context-strategy"
COOKBOOKS="./moonshot-data/cookbooks"
DATABASES="./moonshot-data/generated-outputs/databases"
DATABASES_MODULES="./moonshot-data/databases-modules"
DATASETS="./moonshot-data/datasets"
IO_MODULES="./moonshot-data/io-modules"
METRICS="./moonshot-data/metrics"
PROMPT_TEMPLATES="./moonshot-data/prompt-templates"
RECIPES="./moonshot-data/recipes"
RESULTS="./moonshot-data/generated-outputs/results"
RESULTS_MODULES="./moonshot-data/results-modules"
RUNNERS="./moonshot-data/generated-outputs/runners"
RUNNERS_MODULES="./moonshot-data/runners-modules"
TOKENIZERS_PARALLELISM=false
HOST_ADDRESS=127.0.0.1
HOST_PORT=5000
MOONSHOT_UI_CALLBACK_URL=http://localhost:3000/api/v1/benchmarks/status
EOF