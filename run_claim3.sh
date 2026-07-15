#!/usr/bin/env bash
# Claim #3: the default zero-shot engine classifies the whole corpus in
# seconds and under 3 Wh on one GPU. Reads the committed run record (~5 s);
# with a GPU present, also re-times the run live (~1 min, first run
# downloads the 0.6 GB model).
set -euo pipefail
cd "$(dirname "$0")"
LIVE=0
if command -v nvidia-smi >/dev/null 2>&1 && [ "${SKIP_LIVE:-0}" != "1" ]; then
  LIVE=1
  uv run --no-sync zerolinc run --model gliclass:knowledgator/gliclass-modern-base-v3.0 --config en-event > /tmp/claim3_live.log 2>&1 || LIVE=0
fi
uv run --no-sync python - "$LIVE" <<'PY'
import json, sys
from pathlib import Path
r = json.loads(Path("results/runs/gliclass-modern-base-v3.0__en-event.json").read_text())
wall, wh = r["wall_seconds"], r.get("gpu_energy_wh")
vram = r.get("peak_vram_mb", 0) / 1024
acc = r["metrics"]["accuracy"]
live = sys.argv[1] == "1"
ok = wall < 60 and (wh is None or wh < 3) and acc > 0.6
print("══════════════════════════════════════════════════════════════")
print("  Claim #3 — Cost of the default zero-shot engine (182 tickets)")
print("══════════════════════════════════════════════════════════════")
print(f"  Wall-clock time : {wall:.1f} s   {'(measured live now)' if live else '(committed run record)'}")
print(f"  GPU energy      : {wh} Wh")
print(f"  Peak VRAM       : {vram:.1f} GB")
print(f"  Accuracy        : {acc*100:.1f}%")
print(f"  Reference (paper): LLM APIs needed minutes and US$ fees per pass")
print(f"  Expected: < 60 s, < 3 Wh, accuracy > 60%  →  {'OK' if ok else 'FAIL'}")
print("══════════════════════════════════════════════════════════════")
raise SystemExit(0 if ok else 1)
PY
