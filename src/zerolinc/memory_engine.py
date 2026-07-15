"""Training-free instance-based classification: k-NN over sentence embeddings.

This is NOT zero-shot: the dev half of the corpus serves as a labeled
reference memory (no gradient updates, no fine-tuning). Each test ticket is
assigned the similarity-weighted majority label of its k nearest dev tickets.
k and the text view are selected by leave-one-out accuracy on the dev half
only; the test half is untouched until the final report.

Rationale: CSIRT corpora are template-heavy (near-duplicate coordination
shells and recurring alert formats), which instance-based methods exploit
directly, including on tickets whose informative content is boilerplate.
"""

import json
import math
from collections import defaultdict
from pathlib import Path

from .normalizer import Incident, apply_view
from .metrics import evaluate, mcnemar_vs
from .protocol import stratified_split

DEFAULT_KS = (1, 3, 5, 7)


def embed_texts(model_id: str, texts: list[str], batch_size: int = 8,
                max_seq_length: int = 2048):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        model_id, device=device,
        model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else None,
    )
    # corpus tickets are ~1k tokens; capping the window bounds activation memory
    model.max_seq_length = min(getattr(model, "max_seq_length", max_seq_length),
                               max_seq_length)
    emb = model.encode(texts, normalize_embeddings=True, batch_size=batch_size,
                       show_progress_bar=False)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return emb


def _vote(sims_row, labels: list[str], candidate_idx: list[int], k: int) -> str:
    """Similarity-weighted vote among the k most similar candidates."""
    top = sorted(candidate_idx, key=lambda j: -sims_row[j])[:k]
    weight: dict[str, float] = defaultdict(float)
    for j in top:
        weight[labels[j]] += float(sims_row[j])
    return max(sorted(weight), key=lambda lab: weight[lab])


def knn_report(
    incidents: list[Incident],
    embeddings_by_view: dict[str, "object"],
    seed: int = 42,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict:
    """Select (view, k) by leave-one-out on dev; report on the test half."""
    ids = [i.incident_id for i in incidents]
    labels = [i.label for i in incidents]
    dev_ids, test_ids = stratified_split(ids, labels, seed=seed)
    dev_idx = [j for j, i in enumerate(ids) if i in dev_ids]
    test_idx = [j for j, i in enumerate(ids) if i in test_ids]

    majority = max(sorted(set(labels)), key=[labels[j] for j in dev_idx].count)

    best = None
    for view, emb in embeddings_by_view.items():
        sims = emb @ emb.T
        for k in ks:
            correct = 0
            for j in dev_idx:
                others = [m for m in dev_idx if m != j]
                if _vote(sims[j], labels, others, k) == labels[j]:
                    correct += 1
            loo = correct / len(dev_idx)
            if best is None or loo > best["loo"]:
                best = {"view": view, "k": k, "loo": loo}

    emb = embeddings_by_view[best["view"]]
    sims = emb @ emb.T
    y_true = [labels[j] for j in test_idx]
    y_pred = [_vote(sims[j], labels, dev_idx, best["k"]) for j in test_idx]
    m = evaluate(y_true, y_pred)
    return {
        "seed": seed,
        "method": "knn-instance-memory",
        "selected_view": best["view"],
        "selected_k": best["k"],
        "dev_loo_accuracy": round(best["loo"], 4),
        "n_dev": len(dev_idx),
        "n_test": len(test_idx),
        "majority_class_from_dev": majority,
        "test": {key: m[key] for key in
                 ("n", "accuracy", "accuracy_ci95", "macro_f1", "weighted_f1")},
        "per_class": m["per_class"],
        "mcnemar_vs_majority_on_test": mcnemar_vs(
            y_true, y_pred, [majority] * len(y_true)),
        "predictions": [
            {"incident_id": ids[j], "true": t, "pred": p}
            for j, t, p in zip(test_idx, y_true, y_pred)
        ],
    }


def run_knn(
    data_path: str | Path,
    model_id: str,
    views: tuple[str, ...] = ("full", "subject"),
    seeds: tuple[int, ...] = (42, 7, 123, 2024, 99),
    out_dir: str | Path = "results/report",
) -> list[dict]:
    from .normalizer import load_incidents

    incidents = load_incidents(data_path)
    embeddings_by_view = {
        v: embed_texts(model_id, [i.text for i in apply_view(incidents, v)])
        for v in views
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for seed in seeds:
        r = knn_report(incidents, embeddings_by_view, seed=seed)
        r["embedding_model"] = model_id
        (out / f"knn_seed{seed}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
        reports.append(r)
    return reports
