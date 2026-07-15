# ZeroLINC Benchmark — evaluation artifact

This repository is the evaluation artifact of the paper **"ZeroLINC: Training-Free Local Classification of Security Incident Reports"** (SBSeg 2026, Salão de Ferramentas). The paper presents ZeroLINC, an open-source tool that classifies SOC/CSIRT incident tickets into the 12 NIST SP 800-61r3-derived categories locally, with no training and no external API, through two engines: an instance-memory engine that reaches **90.5%** mean test accuracy with 89 labeled reference tickets, and a zero-shot engine that reaches up to **70.9%** with no labeled data, classifying the whole corpus in seconds and under 3 Wh on one GPU. This artifact contains the full measurement study behind the tool (292 evaluation runs), the committed run of record, the labeled corpus, and one script per paper claim. The end-user tool is packaged separately at [zerolinc](https://github.com/CristhianKapelinski/zerolinc).

# README structure

[Considered seals](#considered-seals) · [Basic information](#basic-information) · [Dependencies](#dependencies) · [Security concerns](#security-concerns) · [Installation](#installation) · [Minimal test](#minimal-test) · [Experiments](#experiments) (Claim #1, #2, #3) · [LICENSE](#license). Repository layout: `src/zerolinc/` (the modules, named as in the paper), `data/` (labeled corpus), `results/runs/` (run of record), `results/report/` (protocol outputs), `run_claim{1,2,3}.sh` (one script per claim), `scripts/` (harness utilities), `docs/` (design details), `tests/`.

# Considered seals

Os selos considerados são: **Disponíveis (SeloD), Funcionais (SeloF), Sustentáveis (SeloS) e Reprodutíveis (SeloR)**.

# Basic information

| Component | Requirement |
|---|---|
| OS | Linux x86-64 (tested: Ubuntu-based, kernel 6.17) |
| Runtime | Python ≥ 3.11, managed by `uv` |
| RAM | 16 GB |
| Disk | 15 GB free (model downloads) |
| GPU | optional but recommended: any NVIDIA GPU with ≥ 4 GB VRAM (CUDA 12.8 wheels). CPU-only works, slower |

Paper experiments ran on: AMD Ryzen 5 8600G (6 cores), 30 GB RAM, NVIDIA GeForce RTX 5060 Ti (16 GB VRAM), Linux kernel 6.17, Python 3.13, PyTorch 2.11 (cu128), Transformers 5.12.

# Dependencies

All Python dependencies are version-frozen in the committed `uv.lock` (PyTorch 2.11 cu128, Transformers 5.12, Sentence-Transformers 5.6, GLiClass 0.1.18, scikit-learn, pandas). No system packages are required beyond `git`, `curl`, and `uv`. Model checkpoints download automatically from Hugging Face on first use (~4 GB total for all claims); set `HF_HUB_CACHE` to control where. The labeled corpus (182 unique anonymized CSIRT tickets) is included in `data/`.

# Security concerns

The artifact runs entirely locally: no telemetry, no external API calls (only Hugging Face model downloads on first run), no credentials, no ports opened. The bundled corpus is anonymized (every sensitive span replaced by placeholder tags) and contains no personal data.

# Installation

```bash
git clone https://github.com/CristhianKapelinski/zerolinc-benchmark
cd zerolinc-benchmark
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is not installed
uv sync --extra dev                                # ~3 min
```

# Minimal test

Offline unit tests plus one small live run (~2 min on GPU; first run downloads ~1 GB):

```bash
uv run pytest -q          # expected: "15 passed" (~10 s, no network)
./scripts/smoke.sh        # expected: ends with a line "...__en-name: acc=... wall=...s"
```

# Experiments

The paper makes three claims; each has one self-contained script that prints a result box ending in `OK`. Reviewers with no GPU can still verify Claims #2 and #3 (both read the committed run of record) and run Claim #1 on CPU (slower).

## Claim #1 — Instance-memory engine reaches 90.5% mean test accuracy (main claim)

- **Description:** with 89 labeled reference tickets and no training, the k-NN instance-memory engine reaches 90.5% mean accuracy over 5 validation/test splits (range 88.2–92.5%), McNemar p<0.001 vs the majority baseline in every split. Runs the full protocol live.
- **Execution:** `./run_claim1.sh` (no flags)
- **Expected time:** ~3 min on GPU (first run +1 min model download); ~15 min CPU-only
- **Expected resources:** ~4 GB RAM, ~2 GB VRAM (GPU path), 1 GB disk (model cache)
- **Expected result** (GPU fp16 embedding introduces small run-to-run variation: per-seed accuracies may shift by 1–2 p.p. and the mean by ±0.5 p.p.; the assertion band 88–93% absorbs it):

```
══════════════════════════════════════════════════════════════
  Claim #1 — Instance-memory engine (main claim)
══════════════════════════════════════════════════════════════
  seed   42 : test accuracy  91.4%
  seed    7 : test accuracy  90.3%
  seed  123 : test accuracy  92.5%
  seed 2024 : test accuracy  88.2%
  seed   99 : test accuracy  90.3%
  Mean test accuracy : 90.5%   (paper: 90.5%, range 88.2–92.5%)
  McNemar vs majority: p < 0.001 in all 5 seeds (max p = ...)
  Expected: mean between 88% and 93%, every p < 0.001  →  OK
══════════════════════════════════════════════════════════════
```

## Claim #2 — Zero-shot engines reach up to 70.9% (protocol mean 68.8%)

- **Description:** recomputes every metric of the 292-run study from the committed per-ticket predictions (no GPU, no network) and re-runs the 5-seed selection protocol, asserting the paper's zero-shot numbers.
- **Execution:** `./run_claim2.sh` (no flags)
- **Expected time:** ~2 min, CPU only
- **Expected resources:** ~2 GB RAM, no GPU, no network
- **Expected result (deterministic):**

```
══════════════════════════════════════════════════════════════
  Claim #2 — Zero-shot engines (from the committed run of record)
══════════════════════════════════════════════════════════════
  Best single run     : 70.9%  (deberta-v3-large-zeroshot-v2.0__en-desc-kw__subject)
  NLI protocol mean   : 68.8%  (range 66.7–69.9%) over 5 splits
  Majority-class floor: 63.4%
  Expected: best 70.9%, NLI mean 68.8%  →  OK
══════════════════════════════════════════════════════════════
```

## Claim #3 — The default zero-shot engine costs seconds and under 3 Wh

- **Description:** verifies the cost claim for the default engine (GLiClass-modern) over the 182-ticket corpus. With a GPU present the run is re-timed live; without one, the committed run record is read (`SKIP_LIVE=1 ./run_claim3.sh` forces the no-GPU path). Wall-clock varies with hardware; the assertion is < 60 s and < 3 Wh.
- **Execution:** `./run_claim3.sh`
- **Expected time:** ~1 min on GPU (first run +1 min model download); ~5 s no-GPU path
- **Expected resources:** ~4 GB RAM, ~1 GB VRAM (live path), 0.8 GB disk (model cache)
- **Expected result:**

```
══════════════════════════════════════════════════════════════
  Claim #3 — Cost of the default zero-shot engine (182 tickets)
══════════════════════════════════════════════════════════════
  Wall-clock time : 7.0 s   (measured live now)
  GPU energy      : 0.29 Wh
  Peak VRAM       : 0.9 GB
  Accuracy        : 67.0%
  Reference (paper): LLM APIs needed minutes and US$ fees per pass
  Expected: < 60 s, < 3 Wh, accuracy > 60%  →  OK
══════════════════════════════════════════════════════════════
```

All figures and tables of the paper regenerate from the same records: `uv run zerolinc report` and `uv run python scripts/make_figures.py results/runs figures/subject subject`. Design details (verbalization sets, input views, protocol definition) are in [docs/DESIGN.md](docs/DESIGN.md).

# LICENSE

[GNU AGPL-3.0-or-later](LICENSE).
