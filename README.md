# ZeroLINC Benchmark: measurement study and artifact

This repository is the **evaluation artifact** behind the ZeroLINC paper: the full measurement study (292 runs), the run of record, the selection protocol, and every script that regenerates the paper's numbers and figures. The end-user **tool** extracted from this study lives in its own lean repository: [zerolinc](https://github.com/CristhianKapelinski/zerolinc).

ZeroLINC classifies CSIRT/SOC incident reports into the 12 NIST SP 800-61r3-derived categories locally, with no model training, no external API, and no LLM-scale hardware. Two engines: a **zero-shot** engine for day-zero deployments (up to **70.9%** accuracy on the evaluation corpus) and an **instance-memory** engine that reuses previously labeled tickets (**90.5%** mean test accuracy with 89 labeled references), at seconds and under **3 Wh** per full corpus pass on a single GPU.

> Paper: *ZeroLINC: Training-Free Local Classification of Security Incident Reports* (SBSeg 2026, Salão de Ferramentas — under review). This README is the single self-contained guide for artifact evaluation; the other docs are complementary.

## README structure

1. [Considered badges](#considered-badges) 2. [Basic information](#basic-information) 3. [Dependencies](#dependencies) 4. [Security concerns](#security-concerns) 5. [Installation](#installation) 6. [Minimal test](#minimal-test) 7. [Experiments](#experiments) 8. [LICENSE](#license)

## Considered badges

- **Disponível (SeloD):** the repository is publicly archived with an open license (AGPL).
- **Funcional (SeloF):** the minimal test below exercises the full pipeline end to end in ~2 minutes.
- **Sustentável (SeloS):** small typed modules (one responsibility each), offline unit tests (`uv run pytest`, ~6 s), no hardcoded paths; all behavior via CLI flags and environment variables.
- **Reprodutível (SeloR):** every number in the paper regenerates offline from the committed run of record (`results/runs/`) with one command per claim; live re-runs are deterministic (argmax inference, seeded splits).

## Basic information

| Component | Requirement |
|---|---|
| OS | Linux x86-64 |
| Runtime | Python ≥ 3.11, managed by `uv` |
| RAM | 16 GB |
| Disk | 15 GB free (model downloads) |
| GPU | one NVIDIA GPU with ≥ 6 GB VRAM (reference machine: RTX 5060 Ti 16 GB; CPU-only also works, slower) |

## Dependencies

Pinned via `pyproject.toml` + committed `uv.lock` (`torch` CUDA 12.8 wheels, `transformers`, `sentence-transformers`, `gliclass`, `scikit-learn`, `pandas`). Models are fetched automatically from Hugging Face on first use; override the cache location with `HF_HUB_CACHE` if the default disk is small.

## Security concerns

The tool runs entirely locally: no telemetry, no external API calls (only Hugging Face model downloads on first run), no credentials required. The bundled corpus is anonymized (all sensitive spans replaced by placeholder tags) and contains no personal data.

## Installation

```bash
git clone https://github.com/CristhianKapelinski/zerolinc-benchmark && cd zerolinc-benchmark
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed
uv sync --extra dev                                # (~3 min)
```

## Minimal test

One command; expected: 8+ baseline/model lines and a written run record (~2 min on GPU, first run downloads ~1 GB):

```bash
./scripts/smoke.sh
```

Expected output ends with a line like `mDeBERTa-...__en-name: acc=... wall=...s`.

## Experiments

**Main claim (all paper numbers, from the committed run of record):** one command, no GPU, no network (~1 min). Recomputes every metric from the stored per-ticket predictions, rebuilds the report and the selection protocol, and asserts the three headline numbers (70.9% best zero-shot, 90.5% instance-memory mean, 69.0% ensemble mean):

```bash
./scripts/reproduce.sh
```

Expected final line: `REPRODUCE OK`.

**Claim 2 (instance-memory engine, live re-run):** GPU, ~3 min (first run downloads the embedding model):

```bash
uv run zerolinc knn
```

Expected: five `seed N: ... test_acc=0.88-0.93 ... mcnemar_p=0.0` lines (mean 0.905).

**Claim 3 (figures):** regenerated offline from the run records (~10 s):

```bash
uv run python scripts/make_figures.py results/runs figures/subject subject
```

**Optional from-scratch path** (GPU, ~4-5 h): reruns the full grid live and then validates it against the run of record:

```bash
./scripts/reproduce_full.sh
```

**End-user tool demo:** classify a CSV (`conteudo` column) with or without a labeled memory:

```bash
uv run zerolinc classify --input data/185_incidentes_anon.csv --engine zeroshot --output predictions.csv
uv run zerolinc classify --input tickets.csv --memory labeled.csv --output predictions.csv
```

## LICENSE

[GNU AGPL-3.0-or-later](LICENSE).
