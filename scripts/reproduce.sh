#!/usr/bin/env bash
# Reproduce every paper number from the committed run of record. No GPU, no
# network: recomputes all metrics from stored per-ticket predictions, rebuilds
# the report and the selection protocol, and asserts the headline numbers.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run --no-sync python scripts/regen_metrics.py results/runs
uv run --no-sync zerolinc report
for s in 42 7 123 2024 99; do uv run --no-sync zerolinc protocol --seed "$s" >/dev/null; done
uv run --no-sync python - <<'PY'
import json
from pathlib import Path
best = max((json.loads(p.read_text()) for p in Path("results/runs").glob("*.json")
            if "baseline" not in p.name), key=lambda r: r["metrics"]["accuracy"])
assert abs(best["metrics"]["accuracy"] - 0.7088) < 1e-4, best["metrics"]["accuracy"]
knn = [json.loads((Path("results/report") / f"knn_seed{s}.json").read_text())
       for s in (42, 7, 123, 2024, 99)]
knn_mean = sum(r["test"]["accuracy"] for r in knn) / 5
assert abs(knn_mean - 0.9054) < 1e-3, knn_mean
ens = [json.loads((Path("results/report") / f"protocol_seed{s}.json").read_text())
       ["ensemble_rank"]["test"]["accuracy"] for s in (42, 7, 123, 2024, 99)]
ens_mean = sum(ens) / 5
assert abs(ens_mean - 0.6903) < 1e-3, ens_mean
print(f"OK: best zero-shot {best['metrics']['accuracy']:.4f}, "
      f"instance-memory mean {knn_mean:.4f}, ensemble mean {ens_mean:.4f} "
      f"— all match the paper.")
PY
echo "REPRODUCE OK"
