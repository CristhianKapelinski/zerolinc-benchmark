"""Zero-shot classification backends beyond NLI cross-encoders.

Model specs are prefixed with the backend family:

- ``nli:<hf-id>``      entailment cross-encoder (Yin et al. 2019), via classifier.py
- ``gliclass:<hf-id>`` GLiClass generalist classifier (all labels in one pass)
- ``embed:<hf-id>``    instruction bi-encoder; cosine similarity between the
                       incident embedding and each category-description embedding

All families consume the same PromptConfig label verbalizations, so the
comparison isolates the scoring mechanism. GLiClass and embedding models do
not use the NLI hypothesis template; the embedding family uses a fixed task
instruction instead.
"""

import time

import torch

from .classifier import RunResult, classify as nli_classify
from .labels import PromptConfig

EMBED_INSTRUCTION = (
    "Given a security incident report, retrieve the category description "
    "that best matches the incident"
)


def parse_spec(spec: str) -> tuple[str, str]:
    """Split 'backend:model_id'; a bare model id defaults to the nli backend."""
    if ":" in spec and spec.split(":", 1)[0] in ("nli", "gliclass", "embed"):
        backend, model_id = spec.split(":", 1)
        return backend, model_id
    return "nli", spec


def classify_gliclass(
    model_id: str, texts: list[str], config: PromptConfig, batch_size: int = 8
) -> RunResult:
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    from transformers import AutoTokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device != "cpu":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(0)
    model = GLiClassModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, add_prefix_space=True)
    pipe = ZeroShotClassificationPipeline(
        model, tokenizer, classification_type="single-label", device=device
    )
    candidate_labels = list(config.labels.keys())

    start = time.perf_counter()
    outputs = pipe(texts, candidate_labels, threshold=0.0, batch_size=batch_size)
    wall = time.perf_counter() - start

    predictions, top_scores = [], []
    for result in outputs:
        best = max(result, key=lambda r: r["score"])
        predictions.append(config.labels[best["label"]])
        top_scores.append(float(best["score"]))
    peak = torch.cuda.max_memory_allocated(0) / 2**20 if device != "cpu" else 0.0
    name = torch.cuda.get_device_name(0) if device != "cpu" else "cpu"
    del pipe, model
    if device != "cpu":
        torch.cuda.empty_cache()
    return RunResult(predictions, top_scores, round(wall, 2), round(peak, 1), name)


def classify_embed(
    model_id: str, texts: list[str], config: PromptConfig, batch_size: int = 16
) -> RunResult:
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(0)
    model = SentenceTransformer(model_id, device=device)
    candidate_labels = list(config.labels.keys())

    start = time.perf_counter()
    doc_emb = model.encode(candidate_labels, normalize_embeddings=True)
    query_emb = model.encode(
        texts,
        prompt=f"Instruct: {EMBED_INSTRUCTION}\nQuery: ",
        normalize_embeddings=True,
        batch_size=batch_size,
    )
    sims = query_emb @ doc_emb.T
    wall = time.perf_counter() - start

    best_idx = sims.argmax(axis=1)
    predictions = [config.labels[candidate_labels[i]] for i in best_idx]
    top_scores = [float(sims[r, i]) for r, i in enumerate(best_idx)]
    peak = torch.cuda.max_memory_allocated(0) / 2**20 if device == "cuda" else 0.0
    name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return RunResult(predictions, top_scores, round(wall, 2), round(peak, 1), name)


def classify_any(
    spec: str, texts: list[str], config: PromptConfig, batch_size: int = 8
) -> RunResult:
    backend, model_id = parse_spec(spec)
    if backend == "gliclass":
        return classify_gliclass(model_id, texts, config, batch_size)
    if backend == "embed":
        return classify_embed(model_id, texts, config, batch_size)
    return nli_classify(model_id, texts, config, batch_size=batch_size)
