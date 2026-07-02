#!/usr/bin/env bash
# Minimal end-to-end test: baselines + one small model on 20 incidents (~2 min).
set -euo pipefail
cd "$(dirname "$0")/.."
uv run zerolinc baselines
uv run zerolinc run --model MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 --config en-name --limit 20
