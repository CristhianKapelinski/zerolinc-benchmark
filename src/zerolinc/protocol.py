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

from .combine import argmax_preds
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
    best_scored_by_family: dict[str, dict] = {}
    for r in runs:
        fam = r.get("backend", "nli")
        preds = _dedup(r["predictions"])
        dev_true, dev_pred = _subset(preds, dev_ids)
        dev_acc = sum(t == p for t, p in zip(dev_true, dev_pred)) / len(dev_true)
        cur = best_by_family.get(fam)
        if cur is None or dev_acc > cur["dev_acc"]:
            best_by_family[fam] = {"run": r, "dev_acc": dev_acc}
        if len(preds[0].get("scores", {})) == len(CODES):
            cur_s = best_scored_by_family.get(fam)
            if cur_s is None or dev_acc > cur_s["dev_acc"]:
                best_scored_by_family[fam] = {"run": r, "dev_acc": dev_acc, "preds": preds}

    for fam, sel in sorted(best_by_family.items()):
        r = sel["run"]
        preds = _dedup(r["predictions"])
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

    # Ensemble rule, fixed a priori: one member per family (its dev-best run
    # among those carrying full score vectors), included only if its dev
    # accuracy is at least the dev majority-class accuracy.
    dev_majority_acc = sum(1 for i, lab in zip(ids, labels)
                           if i in dev_ids and lab == majority) / max(len(dev_ids), 1)
    ensemble_members, member_ids = [], []
    for fam, sel in sorted(best_scored_by_family.items()):
        if sel["dev_acc"] >= dev_majority_acc:
            ensemble_members.append(sel["preds"])
            member_ids.append(sel["run"]["run_id"])

    if len(ensemble_members) >= 2:
        ens_true, ens_pred = _rank_ensemble(ensemble_members, test_ids)
        m = evaluate(ens_true, ens_pred)
        out["ensemble_rank"] = {
            "members": member_ids,
            "test": {k: m[k] for k in
                     ("n", "accuracy", "accuracy_ci95", "macro_f1", "weighted_f1")},
            "mcnemar_vs_majority_on_test": mcnemar_vs(
                ens_true, ens_pred, [majority] * len(ens_true)
            ),
        }
    return out


def _rank_ensemble(
    members: list[list[dict]], test_ids: set[str]
) -> tuple[list[str], list[str]]:
    """Scale-free rank ensemble: average within-item score ranks across members.

    Raw scores are kept (no per-label centering): batch calibration assumes a
    uniform true label distribution, which is false on this skewed corpus and
    strips the genuine class prior out of every member. Ranks (0..k-1 within
    each item) equalize members with different score scales; a single-member
    "ensemble" reproduces that member's argmax exactly.
    """
    order = sorted(test_ids)
    k = len(CODES)
    acc = [[0.0] * k for _ in order]
    truth: list[str] = []
    for m_i, preds in enumerate(members):
        by_id = {p["incident_id"]: p for p in preds}
        for row_i, ident in enumerate(order):
            sc = [by_id[ident]["scores"][c] for c in CODES]
            for rank, j in enumerate(sorted(range(k), key=lambda j: sc[j])):
                acc[row_i][j] += rank
        if m_i == 0:
            truth = [by_id[i]["true"] for i in order]
    return truth, argmax_preds(acc)
