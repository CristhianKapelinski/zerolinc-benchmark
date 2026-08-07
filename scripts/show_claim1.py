#!/usr/bin/env python3
"""Report Claim #1: the instance-memory engine's mean test accuracy over five splits.

Two paths reach this script, and the block says which one it was. With the incident
corpus in place the five splits are recomputed live on this machine. Without it, and the
corpus is not ours to redistribute, the same five per-split records are read from the
committed run of record: they carry one row per test ticket with its true and predicted
category and no ticket text, which is how every other number in this artifact is
verifiable offline.

Both paths check the same thing: the mean over the five seeds, and that McNemar against
the majority-class baseline is significant in every one of them.

Usage: show_claim1.py REPORT_DIR (live|record)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SEEDS = (42, 7, 123, 2024, 99)
PAPER_MEAN = 0.908
LO, HI = 0.88, 0.93
BAR = "═" * 66
SEP = "─" * 66


def main() -> int:
    report = Path(sys.argv[1])
    origin = sys.argv[2]

    missing = [s for s in SEEDS if not (report / f"knn_seed{s}.json").exists()]
    if missing:
        print(f"missing per-seed reports for seeds {missing} in {report}", file=sys.stderr)
        return 1
    rs = [json.loads((report / f"knn_seed{s}.json").read_text()) for s in SEEDS]

    accs = [r["test"]["accuracy"] for r in rs]
    ps = [r["mcnemar_vs_majority_on_test"]["p_value"] for r in rs]
    mean = sum(accs) / len(accs)
    mean_ok = LO <= mean <= HI
    p_ok = max(ps) < 0.001

    print()
    print(BAR)
    print("  Claim #1: the instance-memory engine classifies incident tickets at")
    print("            90.8% mean test accuracy, with no gradient training")
    print(SEP)
    for s, a, p in zip(SEEDS, accs, ps):
        print(f"  seed {s:>4}   test accuracy {a * 100:5.1f}%     McNemar p = {p:.2e}")
    print(SEP)
    rows = [("mean test accuracy", f"{mean * 100:.1f}%", f"paper {PAPER_MEAN * 100:.1f}%", mean_ok),
            ("McNemar significant in all five", "yes" if p_ok else "no",
             f"max p = {max(ps):.1e}", p_ok)]
    for label, got, ref, good in rows:
        print(f"  {label:<34}: {got:<8}{'(' + ref + ')':<21}" + ("OK" if good else "FAIL"))
    print(SEP)
    if origin == "live":
        print(f"  {'source of these numbers':<38}: recomputed on this machine just now")
    else:
        print(f"  {'source of these numbers':<38}: the committed run of record; the corpus")
        print(f"  {'':<38}  is not redistributable, so run the live")
        print(f"  {'':<38}  path only if you obtained it (data/README.md)")
    print(f"  {'gate':<38}: mean within {LO * 100:.0f}-{HI * 100:.0f}%, every p < 0.001")
    print(SEP)
    ok = mean_ok and p_ok
    print(f"  RESULT: {'OK' if ok else 'FAIL'}   "
          f"({'the paper' + chr(39) + 's headline holds here' if ok else 'the headline did NOT hold'})")
    print(BAR)
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
