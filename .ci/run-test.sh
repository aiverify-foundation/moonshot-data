#!/bin/bash

# Specify a list of source dirs
src_dir_list=( "attack-modules" "connectors" "context-strategy" "databases-modules"
               "io-modules" "metrics" "results-modules" "runners-modules" )

# Generate a string with multiple --cov=src_dir
cov_args=""
for src_dir in "${src_dir_list[@]}"; do
  cov_args+="--cov=$src_dir "
done

if [ "$1" == "-m" ]; then
  test_cmd="python3 -m pytest"
else
  test_cmd=pytest
fi

set +e
$test_cmd $cov_args --cov-branch --html=test-report.html --json=test-report.json
exit_code=$?
coverage html
coverage json --pretty-print
set -e
if [ $exit_code -ne 0 ]; then
  exit $exit_code
fi
