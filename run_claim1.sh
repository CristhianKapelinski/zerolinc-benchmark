#!/usr/bin/env bash
# Claim #1 (main): the instance-memory engine reaches 90.8% mean test accuracy with 89
# labeled reference tickets (range 89.2-92.5%, McNemar p<0.001 in every split).
#
# The incident corpus belongs to the reference study and is not ours to redistribute, so
# this claim has two paths and always verifies:
#   - with data/185_incidentes_anon.csv in place, the five splits are recomputed LIVE
#     (GPU ~3 min, CPU ~15 min);
#   - without it, the same five per-split records are read from the committed run of
#     record, which carries one row per test ticket with its true and predicted category
#     and no ticket text.
# The result block states which path it took. Same command either way.
set -euo pipefail
cd "$(dirname "$0")"

RECORD="results/runs/memory-knn-5seed"

if [ -f data/185_incidentes_anon.csv ]; then
  echo "Corpus found: recomputing the five splits on this machine..."
  uv run --no-sync zerolinc knn --out results/report >/dev/null 2>&1
  uv run --no-sync python scripts/show_claim1.py results/report live
else
  echo "No corpus here (it is not redistributable, see data/README.md);"
  echo "verifying against the committed run of record instead."
  uv run --no-sync python scripts/show_claim1.py "$RECORD" record
fi
