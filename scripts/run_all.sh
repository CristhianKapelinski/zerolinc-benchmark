#!/usr/bin/env bash
# Full grid: 5 models x 6 prompt configs over all 185 incidents + baselines + report.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run zerolinc baselines
uv run zerolinc grid --skip-existing
uv run zerolinc report
