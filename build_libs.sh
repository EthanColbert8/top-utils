#!/usr/bin/env bash
set -e

# Command-line argument to override check for Pixi environment
OVERRIDE=""

while getopts "p" opt; do
    case "$opt" in
        p) OVERRIDE=true ;;
        ?) echo "Usage: $0 [-p]" >&2; exit 1 ;;
    esac
done

shift $((OPTIND - 1))

if [ $# -gt 0 ]; then
  echo "Error: Unexpected arguments: $@" >&2
  echo "Usage: $0 [-p]" >&2
  exit 1
fi

# Check for Pixi environment
if [ -z $PIXI_ENVIRONMENT_NAME ] && [ -z "$OVERRIDE" ]; then
    echo "This build script should always run inside the repo's default Pixi environment."
    echo "Please run \"pixi shell\" before this script, or precede it with \"pixi run\"."
    echo "This check can be overridden by passing the \"-p\" option."
    exit 1
fi

# The meat: build all the packages
for pkg in libs/*; do
    python -m build "$pkg" --outdir dist/ --wheel
done
