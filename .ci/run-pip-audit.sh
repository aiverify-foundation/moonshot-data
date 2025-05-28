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

  echo "============ Strong Copyleft Licenses Found ============"
  head -n 2 licenses-found.md
  while IFS= read -r line; do
    # Skip text-unidecode with Artistic Licenses
    if [[ $line == *"text-unidecode"* && $line == *"Artistic License"* ]]; then
      continue
    fi
    for lic in "${strongCopyleftLic[@]}"; do
      if [[ $line == *"$lic"* ]]; then
        echo "$line"
        break
      fi
    done
  done < licenses-found.md

  echo "============ Weak Copyleft Licenses Found ============"
  head -n 2 licenses-found.md
  while IFS= read -r line; do
    # Special case for text-unidecode
    if [[ $line == *"text-unidecode"* && $line == *"Artistic License"* ]]; then
      echo "$line (Reclassified as weak copyleft)"
      continue
    fi
    for lic in "${weakCopyleftLic[@]}"; do
      if [[ $line == *"$lic"* ]]; then
        echo "$line"
        break
      fi
    done
  done < licenses-found.md
fi

set -e
if [ $exit_code -ne 0 ]; then
#  echo "pip-audit failed, exiting..."
  exit $exit_code
fi
