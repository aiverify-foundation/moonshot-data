#!/bin/bash

# Function to replace placeholders with actual environment variable values
replace_placeholders() {
  local cmd=$1
  cmd=${cmd//\$COOKBOOK_RUN_NAME/$COOKBOOK_RUN_NAME}
  echo "$cmd"
}

# Update endpoints based on the provided list
echo "Updating endpoints..."
IFS=';' read -ra ENDPOINTS <<< $MOONSHOT_UPDATE_ENDPOINTS_LIST
for cmd in "${ENDPOINTS[@]}"; do
  echo "Executing: $cmd"
  eval "$cmd"
done

# Trigger the cookbook run, which may take some time
echo "Triggering cookbook run... This may take some time."
IFS=';' read -ra COOKBOOKS <<< $MOONSHOT_RUN_COOKBOOKS_LIST
for cmd in "${COOKBOOKS[@]}"; do
  cmd=$(replace_placeholders "$cmd")
  echo "Executing: $cmd"
  eval "$cmd"
done

echo "Run completed"