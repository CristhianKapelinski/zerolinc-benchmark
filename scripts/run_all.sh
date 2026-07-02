#!/usr/bin/env bash
# Full grid: (5 NLI + 4 modern) models x 8 prompt configs x 4 text views + baselines + report.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --no-sync zerolinc baselines
for view in full subject deboiler subject-deboiler; do
  uv run --no-sync zerolinc --view "$view" grid --skip-existing
  uv run --no-sync zerolinc --view "$view" grid --v2 --skip-existing
done
uv run --no-sync zerolinc report
