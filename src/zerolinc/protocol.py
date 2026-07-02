"""Honest selection protocol, computed offline from the stored run files.

Reporting the maximum of hundreds of configurations evaluated on the same
items inflates the headline (selection on the test set). This module
implements a seeded, stratified dev/test split of the corpus: configurations
are RANKED on the dev half only; the winner per backend family is REPORTED on
the untouched test half, with Wilson CI and a paired McNemar test against the
majority baseline on that same test half. Score calibration, when applied,
uses per-label statistics estimated on the dev half only.
"""

import json
import math
import random
from pathlib import Path

from .combine import _EPS, argmax_preds
from .labels import CODES
from .metrics import evaluate, mcnemar_vs


def stratified_split(
    ids: list[str], labels: list[str], seed: int = 42, test_frac: float = 0.5
) -> tuple[set[str], set[str]]:
    """Per-class seeded shuffle; first ceil(n*frac) of each class go to test."""
    rng = random.Random(seed)
    by_class: dict[str, list[str]] = {}
    for i, lab in zip(ids, labels):
        by_class.setdefault(lab, []).append(i)
    dev, test = set(), set()
    for lab in sorted(by_class):
        members = sorted(by_class[lab])
        rng.shuffle(members)
        k = math.ceil(len(members) * test_frac)
        test.update(members[:k])
        dev.update(members[k:])
    return dev, test


def _load_runs(results_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]


def _subset(preds: list[dict], keep: set[str]) -> tuple[list[str], list[str]]:
    kept = [p for p in preds if p["incident_id"] in keep]
    return [p["true"] for p in kept], [p["pred"] for p in kept]


def _dedup(preds: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for p in preds:
        if p["incident_id"] not in seen:
            seen.add(p["incident_id"])
            out.append(p)
    return out


def protocol_report(results_dir: str | Path, seed: int = 42) -> dict:
    """Per-family dev-selection -> test-report, plus a cross-family ensemble."""
    runs = [r for r in _load_runs(Path(results_dir)) if not r["run_id"].startswith("baseline")]
    if not runs:
        raise ValueError(f"no runs in {results_dir}")
    ref = _dedup(runs[0]["predictions"])
    ids = [p["incident_id"] for p in ref]
    labels = [p["true"] for p in ref]
    dev_ids, test_ids = stratified_split(ids, labels, seed=seed)

    majority = max(set(labels), key=[p["true"] for p in ref if p["incident_id"] in dev_ids].count)
    test_truth = [p["true"] for p in ref if p["incident_id"] in test_ids]
    majority_test_preds = [majority] * len(test_truth)

    out: dict = {
        "seed": seed,
        "n_total": len(ids),
        "n_dev": len(dev_ids),
        "n_test": len(test_ids),
        "majority_class_from_dev": majority,
        "majority_test_accuracy": round(
            sum(t == majority for t in test_truth) / len(test_truth), 4
        ),
        "families": {},
    }

    best_by_family: dict[str, dict] = {}
    for r in runs:
        fam = r.get("backend", "nli")
        preds = _dedup(r["predictions"])
        dev_true, dev_pred = _subset(preds, dev_ids)
        dev_acc = sum(t == p for t, p in zip(dev_true, dev_pred)) / len(dev_true)
        cur = best_by_family.get(fam)
        if cur is None or dev_acc > cur["dev_acc"]:
            best_by_family[fam] = {"run": r, "dev_acc": dev_acc}

    ensemble_members, member_ids = [], []
    for fam, sel in sorted(best_by_family.items()):
        r = sel["run"]
        preds = _dedup(r["predictions"])
        test_true, test_pred = _subset(preds, test_ids)
        pred_by_id = {p["incident_id"]: p for p in preds}
        aligned_pred = [pred_by_id[i]["pred"] for i in sorted(test_ids)]
        aligned_true = [pred_by_id[i]["true"] for i in sorted(test_ids)]
        m = evaluate(aligned_true, aligned_pred)
        out["families"][fam] = {
            "selected_run": r["run_id"],
            "dev_accuracy": round(sel["dev_acc"], 4),
            "test": {k: m[k] for k in
                     ("n", "accuracy", "accuracy_ci95", "macro_f1", "weighted_f1")},
            "mcnemar_vs_majority_on_test": mcnemar_vs(
                aligned_true, aligned_pred, [majority] * len(aligned_true)
            ),
        }
        if len(preds[0].get("scores", {})) == len(CODES):
            ensemble_members.append(preds)
            member_ids.append(r["run_id"])

    if len(ensemble_members) >= 2:
        ens_true, ens_pred = _ensemble_dev_calibrated(ensemble_members, dev_ids, test_ids)
        m = evaluate(ens_true, ens_pred)
        out["ensemble_dev_calibrated"] = {
            "members": member_ids,
            "test": {k: m[k] for k in
                     ("n", "accuracy", "accuracy_ci95", "macro_f1", "weighted_f1")},
            "mcnemar_vs_majority_on_test": mcnemar_vs(
                ens_true, ens_pred, [majority] * len(ens_true)
            ),
        }
    return out


def _ensemble_dev_calibrated(
    members: list[list[dict]], dev_ids: set[str], test_ids: set[str]
) -> tuple[list[str], list[str]]:
    """Mean of per-member log-scores calibrated with DEV-estimated label stats."""
    order = sorted(test_ids)
    acc = [[0.0] * len(CODES) for _ in order]
    truth: list[str] = []
    for m_i, preds in enumerate(members):
        by_id = {p["incident_id"]: p for p in preds}
        dev_rows = [
            [math.log(max(by_id[i]["scores"][c], _EPS)) for c in CODES]
            for i in sorted(dev_ids)
        ]
        n_dev = len(dev_rows)
        mu = [sum(r[j] for r in dev_rows) / n_dev for j in range(len(CODES))]
        sd = [
            math.sqrt(sum((r[j] - mu[j]) ** 2 for r in dev_rows) / n_dev) or 1.0
            for j in range(len(CODES))
        ]
        for row_i, ident in enumerate(order):
            p = by_id[ident]
            for j, c in enumerate(CODES):
                v = (math.log(max(p["scores"][c], _EPS)) - mu[j]) / sd[j]
                acc[row_i][j] += v
        if m_i == 0:
            truth = [by_id[i]["true"] for i in order]
    return truth, argmax_preds(acc)
