#!/usr/bin/env bash
# OPTIONAL from-scratch path (GPU, ~4-5 h): reruns the full grid and the
# instance-memory protocol live, then validates against the run of record.
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/run_all.sh
uv run --no-sync zerolinc knn
./scripts/reproduce.sh
