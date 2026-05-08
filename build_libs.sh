#!/usr/bin/env bash
set -e

# This script should ALWAYS be run from inside a `pixi shell` session

for pkg in libs/*; do
    python -m build "$pkg" --outdir dist/ --wheel
done
