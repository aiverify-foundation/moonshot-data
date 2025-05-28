#!/bin/bash

pip uninstall pytest pytest-mock pytest-html pytest-json pytest-cov coverage httpx anybadge -y
pip uninstall flake8 flake8-html -y

# license check
pip install pip-licenses
pip-licenses --format markdown --output-file licenses-found.md
pip uninstall pip-licenses prettytable wcwidth -y

# dependency check
pip install pip-audit
pip uninstall setuptools -y
set +e
pip-audit --format markdown --desc on -o pip-audit-report.md &> pip-audit-count.txt
exit_code=$?

if [ -f pip-audit-report.md ]; then
  echo "============ Vulnerabilities Found ============"
  cat pip-audit-report.md
fi

if [ -f licenses-found.md ]; then
  strongCopyleftLic=("GPL" "AGPL" "EUPL" "OSL")
  weakCopyleftLic=("LGPL" "MPL" "CCDL" "EPL" "CC-BY-SA" "CPL")

  # Array to store packages with weak copyleft licenses
  declare -a weakCopyleftFound=()
  echo "============ Weak Copyleft Licenses Found ============"
  head -n 2 licenses-found.md
  while IFS= read -r line; do
    # Special case for text-unidecode
    if [[ $line == *"text-unidecode"* && $line == *"Artistic License"* ]]; then
      echo "$line (Reclassified as weak copyleft)"
      # Extract package name to exclude later
      package_name=$(echo "$line" | awk -F '|' '{print $2}' | xargs)
      weakCopyleftFound+=("$package_name")
      continue
    fi

    for lic in "${weakCopyleftLic[@]}"; do
      if [[ $line == *"$lic"* ]]; then
        echo "$line"
        # Extract package name to exclude later
        package_name=$(echo "$line" | awk -F '|' '{print $2}' | xargs)
        weakCopyleftFound+=("$package_name")
        break
      fi
    done
  done < licenses-found.md

  echo "============ Strong Copyleft Licenses Found ============"
  head -n 2 licenses-found.md
  while IFS= read -r line; do
    # Skip text-unidecode with Artistic Licenses
    if [[ $line == *"text-unidecode"* && $line == *"Artistic License"* ]]; then
      continue
    fi

    # Extract package name to check if it's already in weak copyleft list
    package_name=$(echo "$line" | awk -F '|' '{print $2}' | xargs)

    # Skip if package was already found in weak copyleft
    if [[ " ${weakCopyleftFound[*]} " == *" $package_name "* ]]; then
      continue
    fi

    for lic in "${strongCopyleftLic[@]}"; do
      if [[ $line == *"$lic"* ]]; then
        echo "$line"
        break
      fi
    done
  done < licenses-found.md
fi

set -e
if [ $exit_code -ne 0 ]; then
  exit $exit_code
fi
