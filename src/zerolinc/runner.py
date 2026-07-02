"""Experiment runner: model x prompt-config grid over the incident corpus.

Each run writes one JSON file under the results directory containing the
full configuration, per-incident predictions (the run of record), metrics,
and cost figures (wall-clock, throughput, peak VRAM), so every reported
number can be regenerated offline from the stored predictions.
"""

import json
import platform
from pathlib import Path

import torch

from . import baselines
from .classifier import classify
from .data import Incident
from .labels import PROMPT_CONFIGS
from .metrics import evaluate

DEFAULT_MODELS: tuple[str, ...] = (
    "facebook/bart-large-mnli",
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "joeddav/xlm-roberta-large-xnli",
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    "MoritzLaurer/bge-m3-zeroshot-v2.0",
)


def _machine() -> dict:
    info = {
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    return info


def run_id(model_id: str, config_name: str) -> str:
    return f"{model_id.split('/')[-1]}__{config_name}"


def run_one(
    model_id: str,
    config_name: str,
    incidents: list[Incident],
    results_dir: str | Path,
    batch_size: int = 8,
) -> dict:
    config = PROMPT_CONFIGS[config_name]
    texts = [i.text for i in incidents]
    y_true = [i.label for i in incidents]

    result = classify(model_id, texts, config, batch_size=batch_size)
    record = {
        "run_id": run_id(model_id, config_name),
        "model": model_id,
        "prompt_config": config_name,
        "hypothesis_template": config.template,
        "candidate_labels": config.labels,
        "machine": _machine(),
        "n": len(incidents),
        "wall_seconds": result.wall_seconds,
        "incidents_per_second": round(len(incidents) / result.wall_seconds, 2),
        "peak_vram_mb": result.peak_vram_mb,
        "device": result.device,
        "metrics": evaluate(y_true, result.predictions),
        "predictions": [
            {"incident_id": i.incident_id, "true": i.label, "pred": p, "score": round(s, 4)}
            for i, p, s in zip(incidents, result.predictions, result.top_scores)
        ],
    }
    out = Path(results_dir) / f"{record['run_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1))
    return record


def run_baselines(incidents: list[Incident], results_dir: str | Path) -> list[dict]:
    texts = [i.text for i in incidents]
    y_true = [i.label for i in incidents]
    records = []
    for name, fn in (("majority", baselines.majority_baseline),
                     ("keyword", baselines.keyword_baseline)):
        preds = fn(y_true, texts)
        record = {
            "run_id": f"baseline__{name}",
            "model": f"baseline/{name}",
            "prompt_config": name,
            "n": len(incidents),
            "wall_seconds": 0.0,
            "metrics": evaluate(y_true, preds),
            "predictions": [
                {"incident_id": i.incident_id, "true": i.label, "pred": p}
                for i, p in zip(incidents, preds)
            ],
        }
        out = Path(results_dir) / f"{record['run_id']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, ensure_ascii=False, indent=1))
        records.append(record)
    return records
