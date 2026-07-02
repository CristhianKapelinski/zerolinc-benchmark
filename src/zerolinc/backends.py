"""Zero-shot classification backends beyond NLI cross-encoders.

Model specs are prefixed with the backend family:

- ``nli:<hf-id>``      entailment cross-encoder (Yin et al. 2019), via classifier.py
- ``gliclass:<hf-id>`` GLiClass generalist classifier (all labels in one pass)
- ``embed:<hf-id>``    instruction bi-encoder; cosine similarity between the
                       incident embedding and each category-description embedding
- ``rerank:<hf-id>``   generative reranker (Qwen3-Reranker family): P("yes") that
                       the category description matches the incident

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
    if ":" in spec and spec.split(":", 1)[0] in ("nli", "gliclass", "embed", "rerank"):
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
    # multi-label + threshold 0 returns a score for EVERY candidate label; the
    # argmax is identical to single-label mode (same logits, monotonic activation)
    pipe = ZeroShotClassificationPipeline(
        model, tokenizer, classification_type="multi-label", device=device
    )
    candidate_labels = list(config.labels.keys())

    start = time.perf_counter()
    outputs = pipe(texts, candidate_labels, threshold=0.0, batch_size=batch_size)
    wall = time.perf_counter() - start

    predictions, top_scores, all_scores = [], [], []
    for result in outputs:
        best = max(result, key=lambda r: r["score"])
        predictions.append(config.labels[best["label"]])
        top_scores.append(float(best["score"]))
        all_scores.append(
            {config.labels[r["label"]]: round(float(r["score"]), 5) for r in result}
        )
    peak = torch.cuda.max_memory_allocated(0) / 2**20 if device != "cpu" else 0.0
    name = torch.cuda.get_device_name(0) if device != "cpu" else "cpu"
    max_length = getattr(tokenizer, "model_max_length", None)
    if max_length and max_length > 100_000:
        max_length = None
    del pipe, model
    if device != "cpu":
        torch.cuda.empty_cache()
    return RunResult(predictions, top_scores, round(wall, 2), round(peak, 1), name,
                     max_length, all_scores)


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
    codes = [config.labels[lab] for lab in candidate_labels]
    all_scores = [
        {c: round(float(s), 5) for c, s in zip(codes, row)} for row in sims
    ]
    peak = torch.cuda.max_memory_allocated(0) / 2**20 if device == "cuda" else 0.0
    name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    max_length = getattr(model, "max_seq_length", None)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return RunResult(predictions, top_scores, round(wall, 2), round(peak, 1), name,
                     max_length, all_scores)


RERANK_INSTRUCTION = (
    "Given a security incident report as the query, judge whether the candidate "
    "incident category matches the incident"
)
_RERANK_PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on '
    'the Query and the Instruct provided. Note that the answer can only be "yes" or '
    '"no".<|im_end|>\n<|im_start|>user\n'
)
_RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def classify_rerank(
    model_id: str, texts: list[str], config: PromptConfig, batch_size: int = 8
) -> RunResult:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(0)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16 if device == "cuda" else None
    ).to(device).eval()
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    candidate_labels = list(config.labels.keys())
    codes = [config.labels[lab] for lab in candidate_labels]

    pairs = [
        _RERANK_PREFIX
        + f"<Instruct>: {RERANK_INSTRUCTION}\n<Query>: {t[:6000]}\n<Document>: {lab}"
        + _RERANK_SUFFIX
        for t in texts
        for lab in candidate_labels
    ]

    start = time.perf_counter()
    p_yes: list[float] = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = tokenizer(
                pairs[i: i + batch_size], return_tensors="pt",
                padding=True, truncation=True, max_length=4096,
            ).to(device)
            logits = model(**batch).logits[:, -1, :]
            pair = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1)
            p_yes.extend(torch.softmax(pair.float(), dim=1)[:, 1].tolist())
    wall = time.perf_counter() - start

    k = len(candidate_labels)
    predictions, top_scores, all_scores = [], [], []
    for row_start in range(0, len(p_yes), k):
        row = p_yes[row_start: row_start + k]
        best = max(range(k), key=lambda j: row[j])
        predictions.append(codes[best])
        top_scores.append(float(row[best]))
        all_scores.append({c: round(float(s), 5) for c, s in zip(codes, row)})
    peak = torch.cuda.max_memory_allocated(0) / 2**20 if device == "cuda" else 0.0
    name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return RunResult(predictions, top_scores, round(wall, 2), round(peak, 1), name,
                     4096, all_scores)


def classify_any(
    spec: str, texts: list[str], config: PromptConfig, batch_size: int = 8
) -> RunResult:
    backend, model_id = parse_spec(spec)
    if backend == "gliclass":
        return classify_gliclass(model_id, texts, config, batch_size)
    if backend == "embed":
        return classify_embed(model_id, texts, config, batch_size)
    if backend == "rerank":
        return classify_rerank(model_id, texts, config, batch_size)
    return nli_classify(model_id, texts, config, batch_size=batch_size)
