#!/usr/bin/env bash
# Claim #2: zero-shot engines reach up to 70.9% accuracy on the full corpus
# (NLI mean 68.8% under the validation/test protocol). No GPU, no network:
# recomputed from the committed run of record (~2 min).
set -euo pipefail
cd "$(dirname "$0")"
uv run --no-sync python scripts/regen_metrics.py results/runs >/dev/null
for s in 42 7 123 2024 99; do uv run --no-sync zerolinc protocol --seed "$s" >/dev/null; done
uv run --no-sync python - <<'PY'
import json
from pathlib import Path
runs = [json.loads(p.read_text()) for p in Path("results/runs").glob("*.json")
        if "baseline" not in p.name]
best = max(runs, key=lambda r: r["metrics"]["accuracy"])
seeds = (42, 7, 123, 2024, 99)
protos = [json.loads((Path("results/report") / f"protocol_seed{s}.json").read_text())
          for s in seeds]
nli = [p["families"]["nli"]["test"]["accuracy"] for p in protos]
maj = protos[0]["majority_test_accuracy"]
ok = abs(best["metrics"]["accuracy"] - 0.7088) < 1e-3 and abs(sum(nli)/5 - 0.6882) < 1e-3
print("══════════════════════════════════════════════════════════════")
print("  Claim #2 — Zero-shot engines (from the committed run of record)")
print("══════════════════════════════════════════════════════════════")
print(f"  Best single run     : {best['metrics']['accuracy']*100:.1f}%  "
      f"({best['run_id']})")
print(f"  NLI protocol mean   : {sum(nli)/5*100:.1f}%  "
      f"(range {min(nli)*100:.1f}–{max(nli)*100:.1f}%) over {len(seeds)} splits")
print(f"  Majority-class floor: {maj*100:.1f}%")
print(f"  Expected: best 70.9%, NLI mean 68.8%  →  {'OK' if ok else 'FAIL'}")
print("══════════════════════════════════════════════════════════════")
raise SystemExit(0 if ok else 1)
PY
