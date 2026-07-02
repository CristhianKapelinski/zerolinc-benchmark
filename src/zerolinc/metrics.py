"""Evaluation metrics: accuracy with Wilson CI, macro/per-class PRF, confusion."""

import math
from collections import Counter
from statistics import NormalDist

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .labels import CODES


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_vs(y_true: list[str], y_pred_a: list[str], y_pred_b: list[str]) -> dict:
    """Paired McNemar test (normal approx. with continuity correction).

    b = items A got right and B got wrong; c = the reverse. Two-sided p.
    """
    b = sum(1 for t, a, bb in zip(y_true, y_pred_a, y_pred_b) if a == t and bb != t)
    c = sum(1 for t, a, bb in zip(y_true, y_pred_a, y_pred_b) if a != t and bb == t)
    if b + c == 0:
        return {"b": 0, "c": 0, "p_value": 1.0}
    z = (abs(b - c) - 1) / math.sqrt(b + c)
    p = 2 * (1 - NormalDist().cdf(max(z, 0.0)))
    return {"b": b, "c": c, "p_value": round(min(p, 1.0), 4)}


def evaluate(y_true: list[str], y_pred: list[str]) -> dict:
    """Full evaluation dict for one run. Unknown predictions count as wrong.

    Macro/per-class metrics are computed over the FIXED set of classes present
    in the gold labels (identical for every run on the same corpus), so macro
    averages are comparable across runs; predictions into absent classes still
    penalize the true class's recall.
    """
    n = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    acc = accuracy_score(y_true, y_pred)
    lo, hi = wilson_ci(correct, n)
    labels = [c for c in CODES if c in set(y_true)]
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0.0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0.0
    )
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0.0
    )
    return {
        "n": n,
        "correct": correct,
        "accuracy": round(acc, 4),
        "accuracy_ci95": [round(lo, 4), round(hi, 4)],
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(w_f1), 4),
        "per_class": {
            lab: {
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1": round(float(f), 4),
                "support": int(s),
            }
            for lab, p, r, f, s in zip(labels, prec, rec, f1, support)
        },
        "confusion": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
        "pred_distribution": dict(Counter(y_pred)),
    }
