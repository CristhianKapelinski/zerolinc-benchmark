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


_SUBJECT = re.compile(r"(?:Assunto|Subject):\s*(.+)", re.I)


def subject_view(text: str) -> str:
    """Subject-first view: the e-mail subject field(s), then the full text.

    The corpus items are e-mail threads; Subject is a structural RFC 5322
    field (rendered "Assunto:" in this corpus). Encoders truncate at 512
    tokens, and the subject, the most condensed statement of what the ticket
    is, can be diluted by boilerplate inside the window. This view prepends a
    copy of the subject line(s) to the unmodified text (the subject therefore
    appears twice: emphasis by position, with nothing removed).
    """
    subjects = [s.strip() for s in _SUBJECT.findall(text)]
    seen: set[str] = set()
    uniq = [s for s in subjects if not (s in seen or seen.add(s))]
    if not uniq:
        return text
    return f"Assunto: {' | '.join(uniq)}. {text}"


def load_incidents(path: str | Path, normalize: bool = True) -> list[Incident]:
    df = pd.read_csv(path)
    required = {"incidente_id", "conteudo", "categoria"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset at {path} is missing columns: {sorted(missing)}")
    from .labels import CODES

    out, seen = [], set()
    for _, row in df.iterrows():
        incident_id = str(row["incidente_id"])
        if incident_id in seen:  # exact duplicate tickets exist in the source CSV
            continue
        label = str(row["categoria"]).strip().upper()
        if label not in CODES:
            raise ValueError(f"invalid category {label!r} for incident {incident_id}")
        if not isinstance(row["conteudo"], str) or not row["conteudo"].strip():
            raise ValueError(f"empty content for incident {incident_id}")
        seen.add(incident_id)
        text = normalize_text(row["conteudo"]) if normalize else row["conteudo"]
        out.append(Incident(incident_id=incident_id, text=text, label=label))
    return out


def boilerplate_lines(texts: list[str], threshold: float = 0.15, min_len: int = 16) -> set[str]:
    """Corpus-level template detection: lines occurring in >= threshold of docs.

    A purely statistical rule (line document frequency), no hand-written
    keyword lists: coordination disclaimers, header labels, and signatures
    shared across tickets are identified by their repetition across the
    corpus, whatever their language or wording.
    """
    from collections import Counter

    df: Counter[str] = Counter()
    for t in texts:
        df.update({ln.strip() for ln in t.splitlines() if len(ln.strip()) >= min_len})
    cut = max(2, int(threshold * len(texts)))
    return {ln for ln, c in df.items() if c >= cut}


def deboiler_view(incidents: list[Incident]) -> list[Incident]:
    boiler = boilerplate_lines([i.text for i in incidents])
    out = []
    for i in incidents:
        kept = "\n".join(
            ln for ln in i.text.splitlines() if ln.strip() not in boiler
        ).strip()
        out.append(Incident(i.incident_id, kept or i.text, i.label))
    return out


VIEWS = ("full", "subject", "deboiler", "subject-deboiler")


def apply_view(incidents: list[Incident], view: str) -> list[Incident]:
    """Apply a text view; see VIEWS. Views are corpus-level, deterministic."""
    if view == "full":
        return incidents
    if view == "subject":
        return [Incident(i.incident_id, subject_view(i.text), i.label) for i in incidents]
    if view == "deboiler":
        return deboiler_view(incidents)
    if view == "subject-deboiler":
        cleaned = deboiler_view(incidents)
        return [Incident(i.incident_id, subject_view(i.text), i.label) for i in cleaned]
    raise ValueError(f"unknown text view: {view}")
