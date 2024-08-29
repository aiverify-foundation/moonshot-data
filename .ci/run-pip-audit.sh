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
pip install mdtree
mdtree pip-audit-report.md > pip-audit-report.html
mdtree licenses-found.md > license-report.html

# Create badges
pip install anybadge
python3 .ci/createBadges.py dependency
python3 .ci/createBadges.py license

set -e
if [ $exit_code -ne 0 ]; then
  echo "pip-audit failed, exiting..."
  exit $exit_code
fi
