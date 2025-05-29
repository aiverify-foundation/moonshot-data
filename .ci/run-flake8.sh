#!/bin/bash

# Specify a list of source dirs
src_dir_list=("attack-modules" "connectors" "context-strategy" "databases-modules"
              "io-modules" "metrics" "results-modules" "runners-modules" )

# Generate a string with multiple src dirs
src_dirs=""
for src_dir in "${src_dir_list[@]}"; do
  src_dirs+="$src_dir "
done

set +e
flake8 --count  $src_dirs > flake8-report.txt
cat flake8-report.txt
exit_code=$?

if [ $exit_code -ne 0 ]; then

  exit $exit_code
fi
