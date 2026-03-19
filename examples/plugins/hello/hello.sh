#!/usr/bin/env bash
NAME="${GRAPHIRM_ARG_NAME:-World}"
echo "Hello, ${NAME}! This is a Graphirm plugin."
echo "Working directory: $(pwd)"
echo "All args (JSON): ${GRAPHIRM_ARGS}"
