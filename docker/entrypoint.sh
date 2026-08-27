#!/bin/bash
# Make the Docker image compatible with Nextflow
set -e

case "$1" in
    bash|/bin/bash|sh|/bin/sh)
        exec "$@"
        ;;
    *)
        exec python /app/run_seq2pocket.py "$@"
        ;;
esac
