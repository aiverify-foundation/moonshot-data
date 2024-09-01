#!/bin/bash

# Define the path to the JSON file containing the results
json_file="moonshot-data/generated-outputs/results/$COOKBOOK_RUN_NAME.json"

# Check if the JSON file exists
if [ ! -f "$json_file" ]; then
  echo "File $json_file not found!"
  exit 1
fi

# Install jq package for display
echo "Installing jq package"
apt-get update && apt-get install jq -y

# Print the table header
printf "+------------------+----------------------------------+------------------+-----------------+-------------------+-------+\n"
printf "| %-16s | %-32s | %-16s | %-15s | %-17s | %-5s |\n" "cookbook_id" "recipe_id" "model_id" "num_of_prompts" "avg_grade_value" "grade"
printf "+==================+==================================+==================+=================+===================+=======+\n"

# Parse the JSON file and print the results in a formatted table
jq -r '
  .results.cookbooks[] |
  .id as $cookbook_id |
  .recipes[] |
  .id as $recipe_id |
  .evaluation_summary[] |
  .model_id as $model_id |
  .num_of_prompts as $num_of_prompts |
  .avg_grade_value as $avg_grade_value |
  .grade as $grade |
  "\($cookbook_id[0:16]) \($recipe_id[0:32]) \($model_id[0:16]) \($num_of_prompts) \($avg_grade_value | tonumber | . * 100 | round / 100) \($grade)"
' "$json_file" | while read -r cookbook_id recipe_id model_id num_of_prompts avg_grade_value grade; do
  printf "| %-16s | %-32s | %-16s | %-15s | %-17s | %-5s |\n" "$cookbook_id" "$recipe_id" "$model_id" "$num_of_prompts" "$avg_grade_value" "$grade"
  printf "+------------------+----------------------------------+------------------+-----------------+-------------------+-------+\n"
done