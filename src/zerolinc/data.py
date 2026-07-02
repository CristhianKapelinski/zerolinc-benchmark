"""Dataset loading and text normalization.

The corpus is a CSV of anonymized CSIRT tickets (columns: registro_id,
incidente_id, conteudo, categoria). Anonymization replaced every sensitive
span with a tag of the form ``[TYPE_hash]`` (e.g. ``[EMAIL_ADDRESS_f6f7086365]``).
Those 10-hex-digit hashes carry no signal and waste a large share of the
512-token window of NLI cross-encoders, so normalization compresses each tag
to a short placeholder (``<EMAIL>``); everything else is preserved.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_TAG = re.compile(r"\[([A-Z_]+?)_[a-f0-9]{6,}\]")

_PLACEHOLDER = {
    "EMAIL_ADDRESS": "<EMAIL>",
    "IP_ADDRESS": "<IP>",
    "URL": "<URL>",
    "ORGANIZATION": "<ORG>",
    "DATE_TIME": "<DATE>",
    "PERSON": "<PERSON>",
    "LOCATION": "<LOCATION>",
    "PHONE_NUMBER": "<PHONE>",
}


@dataclass(frozen=True)
class Incident:
    incident_id: str
    text: str
    label: str


def normalize_text(text: str) -> str:
    """Compress anonymization tags and collapse runs of whitespace."""
    text = _TAG.sub(lambda m: _PLACEHOLDER.get(m.group(1), "<REDACTED>"), text)
    return re.sub(r"[ \t]+", " ", text).strip()


def load_incidents(path: str | Path, normalize: bool = True) -> list[Incident]:
    df = pd.read_csv(path)
    required = {"incidente_id", "conteudo", "categoria"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset at {path} is missing columns: {sorted(missing)}")
    out = []
    for _, row in df.iterrows():
        text = str(row["conteudo"])
        out.append(
            Incident(
                incident_id=str(row["incidente_id"]),
                text=normalize_text(text) if normalize else text,
                label=str(row["categoria"]).strip().upper(),
            )
        )
    return out
