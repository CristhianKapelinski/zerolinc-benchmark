"""Training-free score post-processing: batch calibration and ensembling.

Both operate offline on stored run files that carry per-item score vectors.

- Batch calibration (contextual/batch calibration, Zhao et al. 2021; Zhou et
  al. 2023): zero-shot scorers carry per-label bias (some hypotheses are
  a-priori more entailable). Subtracting each label's mean log-score over the
  batch removes the bias without any labeled data.
- Ensemble: z-score each run's log-score matrix per label, average across
  runs, argmax. Combining runs from different backend families (NLI, GLiClass,
  embeddings, reranker) averages out family-specific failure modes.
"""

import json
import math
from pathlib import Path

from .labels import CODES
from .metrics import evaluate

_EPS = 1e-6


def load_score_matrix(run_file: str | Path) -> tuple[list[str], list[str], list[list[float]]]:
    """Return (incident_ids, y_true, matrix[n][12]) in CODES column order."""
    r = json.loads(Path(run_file).read_text())
    preds = r["predictions"]
    if "scores" not in preds[0]:
        raise ValueError(f"{run_file} has no per-item score vectors")
    ids = [p["incident_id"] for p in preds]
    y_true = [p["true"] for p in preds]
    matrix = [[p["scores"][c] for c in CODES] for p in preds]
    return ids, y_true, matrix


def _log(matrix: list[list[float]]) -> list[list[float]]:
    return [[math.log(max(v, _EPS)) for v in row] for row in matrix]


def calibrate(matrix: list[list[float]]) -> list[list[float]]:
    """Batch calibration: subtract each label's mean log-score."""
    logm = _log(matrix)
    n, k = len(logm), len(logm[0])
    mean = [sum(row[j] for row in logm) / n for j in range(k)]
    return [[row[j] - mean[j] for j in range(k)] for row in logm]


def _zscore(matrix: list[list[float]]) -> list[list[float]]:
    logm = _log(matrix)
    n, k = len(logm), len(logm[0])
    out = [[0.0] * k for _ in range(n)]
    for j in range(k):
        col = [row[j] for row in logm]
        mu = sum(col) / n
        sd = math.sqrt(sum((v - mu) ** 2 for v in col) / n) or 1.0
        for i in range(n):
            out[i][j] = (logm[i][j] - mu) / sd
    return out


def argmax_preds(matrix: list[list[float]]) -> list[str]:
    return [CODES[max(range(len(CODES)), key=lambda j: row[j])] for row in matrix]


def evaluate_calibrated(run_file: str | Path) -> dict:
    _, y_true, matrix = load_score_matrix(run_file)
    return evaluate(y_true, argmax_preds(calibrate(matrix)))


def evaluate_ensemble(run_files: list[str | Path], calibrated: bool = True) -> dict:
    """Average z-scored (optionally calibrated) matrices across runs, argmax."""
    ref_ids: list[str] | None = None
    y_true: list[str] = []
    acc: list[list[float]] | None = None
    for f in run_files:
        ids, truth, matrix = load_score_matrix(f)
        if ref_ids is None:
            ref_ids, y_true = ids, truth
        elif ids != ref_ids:
            raise ValueError(f"{f} covers different incidents than the first run")
        m = _zscore_of(calibrate(matrix)) if calibrated else _zscore(matrix)
        if acc is None:
            acc = m
        else:
            acc = [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(acc, m)]
    assert acc is not None
    return evaluate(y_true, argmax_preds(acc))


def _zscore_of(matrix: list[list[float]]) -> list[list[float]]:
    """Z-score an already-log-space matrix per label column."""
    n, k = len(matrix), len(matrix[0])
    out = [[0.0] * k for _ in range(n)]
    for j in range(k):
        col = [row[j] for row in matrix]
        mu = sum(col) / n
        sd = math.sqrt(sum((v - mu) ** 2 for v in col) / n) or 1.0
        for i in range(n):
            out[i][j] = (matrix[i][j] - mu) / sd
    return out
