#!/usr/bin/env bash
# Claim #1 (main): the instance-memory engine reaches 90.8% mean test accuracy
# with 89 labeled reference tickets (range 89.2-92.5%, McNemar p<0.001).
# Runs the 5-seed validation/test protocol LIVE (GPU ~3 min; CPU ~15 min).
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f data/185_incidentes_anon.csv ]; then
  echo "Claim #1 (instance-memory, 90.8%) runs LIVE on the incident corpus,"
  echo "which is not redistributed, so it is SKIPPED here (not an error)."
  echo "See data/README.md to obtain and place data/185_incidentes_anon.csv."
  echo "Claim 2, claim 3 and scripts/reproduce.sh verify the other numbers without it."
  exit 0
fi
uv run --no-sync zerolinc knn --out results/report >/dev/null 2>&1
uv run --no-sync python - <<'PY'
import json
from pathlib import Path
seeds = (42, 7, 123, 2024, 99)
rs = [json.loads((Path("results/report") / f"knn_seed{s}.json").read_text()) for s in seeds]
accs = [r["test"]["accuracy"] for r in rs]
ps = [r["mcnemar_vs_majority_on_test"]["p_value"] for r in rs]
mean = sum(accs) / len(accs)
ok = 0.88 <= mean <= 0.93 and max(ps) < 0.001
print("══════════════════════════════════════════════════════════════")
print("  Claim #1 — Instance-memory engine (main claim)")
print("══════════════════════════════════════════════════════════════")
for s, a in zip(seeds, accs):
    print(f"  seed {s:>4} : test accuracy {a*100:5.1f}%")
print(f"  Mean test accuracy : {mean*100:.1f}%   (paper: 90.8%, range 89.2–92.5%)")
print(f"  McNemar vs majority: p < 0.001 in all {len(seeds)} seeds"
      f" (max p = {max(ps):.2e})")
print(f"  Expected: mean between 88% and 93%, every p < 0.001  →  "
      f"{'OK' if ok else 'FAIL'}")
print("══════════════════════════════════════════════════════════════")
raise SystemExit(0 if ok else 1)
PY
