"""Regenerate every stored run's metrics from its stored predictions.

Applies the current metric definition (fixed label set = classes present in
gold) and deduplicates repeated incident ids (the source CSV carries 3 exact
duplicate tickets). Predictions themselves are never altered; `n` keeps the
processed count and `metrics.n` reflects the deduplicated evaluation count.
"""

import json
import sys
from pathlib import Path

from zerolinc.metrics import evaluate


def dedup(preds: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for p in preds:
        if p["incident_id"] not in seen:
            seen.add(p["incident_id"])
            out.append(p)
    return out


def main() -> int:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/runs")
    changed = 0
    for f in sorted(results_dir.glob("*.json")):
        r = json.loads(f.read_text())
        preds = dedup(r["predictions"])
        r["predictions"] = preds
        r["metrics"] = evaluate([p["true"] for p in preds], [p["pred"] for p in preds])
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(r, ensure_ascii=False, indent=1))
        tmp.replace(f)
        changed += 1
    print(f"regenerated metrics for {changed} runs (n_eval={r['metrics']['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
