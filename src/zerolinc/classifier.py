"""Zero-shot NLI classification over Hugging Face cross-encoders.

Each incident (premise) is paired with one hypothesis per category; the
model's entailment scores are normalized over the 12 candidates and the
argmax is the prediction (single-label mode, as in Yin et al. 2019).
No model is fine-tuned; inference only.
"""

import time
from dataclasses import dataclass

import torch
from transformers import pipeline

from .labels import PromptConfig


@dataclass(frozen=True)
class RunResult:
    predictions: list[str]  # category codes, aligned with input order
    top_scores: list[float]
    wall_seconds: float
    peak_vram_mb: float
    device: str
    max_length: int | None = None  # model context limit actually in effect
    all_scores: list[dict[str, float]] | None = None  # per item: category -> score


def classify(
    model_id: str,
    texts: list[str],
    config: PromptConfig,
    batch_size: int = 8,
    device: int | None = None,
) -> RunResult:
    """Run one model x prompt-config pass over all texts."""
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    if device >= 0:
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)

    clf = pipeline(
        "zero-shot-classification",
        model=model_id,
        device=device,
        torch_dtype=torch.float16 if device >= 0 else None,
    )
    candidate_labels = list(config.labels.keys())
    max_length = getattr(clf.tokenizer, "model_max_length", None)
    if max_length and max_length > 100_000:  # sentinel for "unset"
        max_length = None

    start = time.perf_counter()
    outputs = clf(
        texts,
        candidate_labels=candidate_labels,
        hypothesis_template=config.template,
        multi_label=False,
        batch_size=batch_size,
        truncation=True,
    )
    wall = time.perf_counter() - start

    if isinstance(outputs, dict):
        outputs = [outputs]
    predictions = [config.labels[o["labels"][0]] for o in outputs]
    top_scores = [float(o["scores"][0]) for o in outputs]
    all_scores = [
        {config.labels[lab]: round(float(s), 5) for lab, s in zip(o["labels"], o["scores"])}
        for o in outputs
    ]
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device >= 0 else 0.0

    del clf
    if device >= 0:
        torch.cuda.empty_cache()

    return RunResult(
        predictions=predictions,
        top_scores=top_scores,
        wall_seconds=round(wall, 2),
        peak_vram_mb=round(peak, 1),
        device=torch.cuda.get_device_name(device) if device >= 0 else "cpu",
        max_length=max_length,
        all_scores=all_scores,
    )
