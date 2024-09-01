#!/bin/bash

# Change directory to moonshot-data
cd moonshot-data
# Configure git user for committing changes
git config --global user.email "ci@example.com"
git config --global user.name "CI Bot"
# Add generated output files to the commit
git add -f ./generated-outputs/*
# Commit the changes with a message including the pipeline ID
git commit -m "Update generated files [ci skip] - Pipeline ID ($GITHUB_RUN_ID)"
# Push the changes to the repository, skipping CI for this push
# git push -o ci.skip