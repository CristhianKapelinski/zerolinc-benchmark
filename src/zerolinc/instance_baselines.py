"""Competitors of the instance-memory engine, under the identical protocol.

The memory engine's claim ("with a few dozen labeled tickets, no training")
needs peers that consume the same labeled dev half: a nearest-class-centroid
classifier and a multinomial logistic regression over the same frozen
embeddings (isolating the decision rule from the representation), and a full
fine-tune of a small multilingual encoder (what actual training buys).
Per seed: hyperparameters and views are selected inside the dev half only;
the report is on the untouched test half with McNemar vs the dev majority.
"""

import json
import math
import time
from pathlib import Path

import numpy as np

from .metrics import evaluate, mcnemar_vs
from .protocol import stratified_split
from .verbalizer import CODES


def _report(seed, method, extra, ids, test_idx, y_true, y_pred, majority, wall):
    m = evaluate(y_true, y_pred)
    return {
        "seed": seed,
        "method": method,
        **extra,
        "n_test": len(test_idx),
        "majority_class_from_dev": majority,
        "wall_seconds": round(wall, 2),
        "test": {key: m[key] for key in
                 ("n", "accuracy", "accuracy_ci95", "macro_f1", "weighted_f1")},
        "mcnemar_vs_majority_on_test": mcnemar_vs(
            y_true, y_pred, [majority] * len(y_true)),
        "predictions": [
            {"incident_id": ids[j], "true": t, "pred": p}
            for j, t, p in zip(test_idx, y_true, y_pred)
        ],
    }


def _split(incidents, seed):
    ids = [i.incident_id for i in incidents]
    labels = [i.label for i in incidents]
    dev_ids, _ = stratified_split(ids, labels, seed=seed)
    dev_idx = [j for j, i in enumerate(ids) if i in dev_ids]
    test_idx = [j for j, i in enumerate(ids) if i not in dev_ids]
    majority = max(sorted(set(labels)), key=[labels[j] for j in dev_idx].count)
    return ids, labels, dev_idx, test_idx, majority


def _centroid_predict(emb, labels, dev_idx, query_idx, exclude_self=False):
    classes = sorted({labels[j] for j in dev_idx})
    sums = {c: np.zeros(emb.shape[1]) for c in classes}
    counts = {c: 0 for c in classes}
    for j in dev_idx:
        sums[labels[j]] += emb[j]
        counts[labels[j]] += 1
    preds = []
    for q in query_idx:
        best_c, best_s = None, -np.inf
        for c in classes:
            s, n = sums[c], counts[c]
            if exclude_self and q in dev_idx and labels[q] == c:
                if n == 1:
                    continue
                s, n = s - emb[q], n - 1
            cen = s / n
            cen = cen / (np.linalg.norm(cen) + 1e-12)
            sim = float(emb[q] @ cen)
            if sim > best_s:
                best_c, best_s = c, sim
        preds.append(best_c)
    return preds


def centroid_report(incidents, embeddings_by_view, seed=42):
    """Nearest class centroid; view selected by leave-one-out on dev."""
    t0 = time.time()
    ids, labels, dev_idx, test_idx, majority = _split(incidents, seed)
    best = None
    for view, emb in embeddings_by_view.items():
        preds = _centroid_predict(emb, labels, dev_idx, dev_idx, exclude_self=True)
        loo = sum(p == labels[j] for p, j in zip(preds, dev_idx)) / len(dev_idx)
        if best is None or loo > best["loo"]:
            best = {"view": view, "loo": loo}
    emb = embeddings_by_view[best["view"]]
    y_pred = _centroid_predict(emb, labels, dev_idx, test_idx)
    y_true = [labels[j] for j in test_idx]
    return _report(seed, "nearest-centroid",
                   {"selected_view": best["view"],
                    "dev_loo_accuracy": round(best["loo"], 4)},
                   ids, test_idx, y_true, y_pred, majority, time.time() - t0)


def _fit_lr(X, y, n_classes, l2, iters=200):
    import torch

    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    W = torch.zeros(X.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xt @ W + b, yt) \
            + l2 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def probe_report(incidents, embeddings_by_view, seed=42,
                 l2_grid=(1e-3, 1e-2, 1e-1)):
    """Multinomial logistic regression on frozen embeddings; (view, l2) by
    5-fold cross-validation on dev."""
    import torch

    t0 = time.time()
    ids, labels, dev_idx, test_idx, majority = _split(incidents, seed)
    classes = sorted({labels[j] for j in dev_idx})
    cls_id = {c: i for i, c in enumerate(classes)}
    y_dev = [cls_id[labels[j]] for j in dev_idx]

    folds = [dev_idx[i::5] for i in range(5)]
    best = None
    for view, emb in embeddings_by_view.items():
        for l2 in l2_grid:
            correct = 0
            for f, held in enumerate(folds):
                tr = [j for g, fold in enumerate(folds) if g != f for j in fold]
                W, b = _fit_lr(emb[tr], [cls_id[labels[j]] for j in tr],
                               len(classes), l2)
                logits = torch.tensor(emb[held], dtype=torch.float32) @ W + b
                pred = logits.argmax(1).tolist()
                correct += sum(classes[p] == labels[j] for p, j in zip(pred, held))
            cv = correct / len(dev_idx)
            if best is None or cv > best["cv"]:
                best = {"view": view, "l2": l2, "cv": cv}

    emb = embeddings_by_view[best["view"]]
    W, b = _fit_lr(emb[dev_idx], y_dev, len(classes), best["l2"])
    logits = torch.tensor(emb[test_idx], dtype=torch.float32) @ W + b
    y_pred = [classes[p] for p in logits.argmax(1).tolist()]
    y_true = [labels[j] for j in test_idx]
    return _report(seed, "linear-probe",
                   {"selected_view": best["view"], "selected_l2": best["l2"],
                    "dev_cv_accuracy": round(best["cv"], 4)},
                   ids, test_idx, y_true, y_pred, majority, time.time() - t0)


def finetune_report(incidents, seed=42,
                    model_id="xlm-roberta-base",
                    view="subject", epochs=15, lr=2e-5, batch_size=8,
                    max_length=512, warmup_frac=0.1):
    """Full fine-tune of a small multilingual encoder on the dev half.

    Fixed standard hyperparameters with linear warmup/decay (no inner
    selection: with 12 classes the dev half is too small to also carve out a
    tuning split, which is itself part of the comparison)."""
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer)

    from .normalizer import apply_view

    t0 = time.time()
    ids, labels, dev_idx, test_idx, majority = _split(incidents, seed)
    texts = [i.text for i in apply_view(incidents, view)]
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(CODES)).to(device)
    cls_id = {c: i for i, c in enumerate(CODES)}

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    steps_total = epochs * math.ceil(len(dev_idx) / batch_size)
    warmup = max(1, int(warmup_frac * steps_total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup if s < warmup
        else max(0.0, (steps_total - s) / (steps_total - warmup)))
    model.train()
    order = list(dev_idx)
    rng = np.random.RandomState(seed)
    epoch_losses = []
    for _ in range(epochs):
        rng.shuffle(order)
        running = []
        for s in range(0, len(order), batch_size):
            chunk = order[s:s + batch_size]
            enc = tok([texts[j] for j in chunk], truncation=True,
                      max_length=max_length, padding=True, return_tensors="pt"
                      ).to(device)
            y = torch.tensor([cls_id[labels[j]] for j in chunk], device=device)
            loss = model(**enc, labels=y).loss
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            running.append(float(loss))
        epoch_losses.append(round(sum(running) / len(running), 4))

    model.eval()
    y_pred = []
    with torch.no_grad():
        for s in range(0, len(test_idx), batch_size):
            chunk = test_idx[s:s + batch_size]
            enc = tok([texts[j] for j in chunk], truncation=True,
                      max_length=max_length, padding=True, return_tensors="pt"
                      ).to(device)
            pred = model(**enc).logits.argmax(1).tolist()
            y_pred.extend(CODES[p] for p in pred)
    y_true = [labels[j] for j in test_idx]
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return _report(seed, "finetuned-encoder",
                   {"model": model_id, "view": view, "epochs": epochs,
                    "train_loss_per_epoch": epoch_losses},
                   ids, test_idx, y_true, y_pred, majority, time.time() - t0)


def setfit_report(incidents, seed=42,
                  model_id="Qwen/Qwen3-Embedding-0.6B", view="subject",
                  num_iterations=5, lr=2e-5, batch_size=4, max_length=512,
                  l2_grid=(1e-3, 1e-2, 1e-1)):
    """SetFit-style few-shot learning [Tunstall et al. 2022]: contrastive
    fine-tune of the sentence encoder on same/different-class pairs, then a
    logistic head on the tuned embeddings. Implemented with the pinned
    sentence-transformers (no extra dependency), on the same backbone as the
    other instance-based methods so only the method varies.
    num_iterations=5 (SetFit's default is 20) bounds the pair budget; the
    corpus tickets are ~1k tokens, far above SetFit's sentence-length regime.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    ids, labels, dev_idx, test_idx, majority = _split(incidents, seed)
    from .normalizer import apply_view
    texts = [i.text for i in apply_view(incidents, view)]

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    pairs = []
    for j in dev_idx:
        same = [m for m in dev_idx if m != j and labels[m] == labels[j]]
        diff = [m for m in dev_idx if labels[m] != labels[j]]
        for _ in range(num_iterations):
            if same:
                pairs.append((j, same[rng.randint(len(same))], 1.0))
            pairs.append((j, diff[rng.randint(len(diff))], 0.0))
    rng.shuffle(pairs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_id, device=device)
    model.max_seq_length = max_length
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for s in range(0, len(pairs), batch_size):
        chunk = pairs[s:s + batch_size]
        embs = []
        for side in (0, 1):
            feats = model.preprocess([texts[p[side]] for p in chunk])
            feats = {k: (v.to(device) if hasattr(v, "to") else v)
                     for k, v in feats.items()}
            e = model(feats)["sentence_embedding"]
            embs.append(torch.nn.functional.normalize(e, dim=1))
        sim = (embs[0] * embs[1]).sum(1).float()
        y = torch.tensor([p[2] for p in chunk], device=device)
        loss = torch.nn.functional.mse_loss(sim, y)
        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        emb = model.encode(texts, normalize_embeddings=True,
                           batch_size=batch_size, show_progress_bar=False)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    classes = sorted({labels[j] for j in dev_idx})
    cls_id = {c: i for i, c in enumerate(classes)}
    folds = [dev_idx[i::5] for i in range(5)]
    best = None
    for l2 in l2_grid:
        correct = 0
        for f, held in enumerate(folds):
            tr = [j for g, fold in enumerate(folds) if g != f for j in fold]
            W, b = _fit_lr(emb[tr], [cls_id[labels[j]] for j in tr],
                           len(classes), l2)
            logits = torch.tensor(emb[held], dtype=torch.float32) @ W + b
            pred = logits.argmax(1).tolist()
            correct += sum(classes[p] == labels[j] for p, j in zip(pred, held))
        cv = correct / len(dev_idx)
        if best is None or cv > best["cv"]:
            best = {"l2": l2, "cv": cv}
    W, b = _fit_lr(emb[dev_idx], [cls_id[labels[j]] for j in dev_idx],
                   len(classes), best["l2"])
    logits = torch.tensor(emb[test_idx], dtype=torch.float32) @ W + b
    y_pred = [classes[p] for p in logits.argmax(1).tolist()]
    y_true = [labels[j] for j in test_idx]
    return _report(seed, "setfit-style",
                   {"model": model_id, "view": view,
                    "num_iterations": num_iterations,
                    "selected_l2": best["l2"],
                    "dev_cv_accuracy": round(best["cv"], 4)},
                   ids, test_idx, y_true, y_pred, majority, time.time() - t0)


def run_all(data_path="data/185_incidentes_anon.csv",
            embed_model="Qwen/Qwen3-Embedding-0.6B",
            views=("full", "subject"),
            seeds=(42, 7, 123, 2024, 99),
            out_dir="results/report"):
    from .memory_engine import embed_texts
    from .normalizer import apply_view, load_incidents

    incidents = load_incidents(data_path)
    embeddings_by_view = {
        v: embed_texts(embed_model, [i.text for i in apply_view(incidents, v)])
        for v in views
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for name, fn in (("centroid", centroid_report), ("probe", probe_report)):
            r = fn(incidents, embeddings_by_view, seed=seed)
            r["embedding_model"] = embed_model
            (out / f"{name}_seed{seed}.json").write_text(
                json.dumps(r, ensure_ascii=False, indent=1))
            print(f"{name} seed {seed}: acc={r['test']['accuracy']:.4f} "
                  f"mF1={r['test']['macro_f1']:.4f} "
                  f"p={r['mcnemar_vs_majority_on_test']['p_value']:.2e}")
        r = finetune_report(incidents, seed=seed)
        (out / f"finetune_seed{seed}.json").write_text(
            json.dumps(r, ensure_ascii=False, indent=1))
        print(f"finetune seed {seed}: acc={r['test']['accuracy']:.4f} "
              f"mF1={r['test']['macro_f1']:.4f} "
              f"p={r['mcnemar_vs_majority_on_test']['p_value']:.2e} "
              f"wall={r['wall_seconds']}s")


if __name__ == "__main__":
    run_all()
