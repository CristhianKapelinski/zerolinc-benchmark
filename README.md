# ZeroLINC: Zero-shot NLI classification of security incidents

Zero-shot classification of CSIRT incident reports into the 12 NIST SP 800-61r3
categories using off-the-shelf NLI cross-encoders (no fine-tuning, no API, no GPU
cloud). Counterpart to the LLM prompt-engineering studies on the same corpus:
instead of prompting a generative model, each incident is scored against one
natural-language hypothesis per category and the best-entailed category wins
(Yin et al., EMNLP 2019).

**Cost model:** everything runs locally on a consumer GPU (reference machine:
NVIDIA RTX 5060 Ti 16 GB); the marginal cost per classification is wall-clock
seconds, not API dollars.

## Layout

- `src/zerolinc/` - the tool: dataset loading/normalization, NIST label sets and
  hypothesis templates (`labels.py`), the NLI classifier wrapper, non-neural
  baselines (majority class, keyword matching), metrics (accuracy + Wilson 95% CI,
  macro-F1, per-class PRF), grid runner, report generator.
- `data/185_incidentes_anon.csv` - 185 anonymized, expert-labeled CSIRT tickets.
- `results/runs/` - one JSON per run: full config + per-incident predictions
  (the run of record) + metrics + cost figures.
- `tests/` - offline unit tests (no network, no models).

## Quickstart

```bash
uv sync                      # install (pinned, CUDA 12.8 wheels)
uv run pytest -q             # offline unit tests (~5 s)
./scripts/smoke.sh           # baselines + 1 small model on 20 incidents (~2 min)
./scripts/run_all.sh         # full grid: 5 models x 6 prompt configs (~1-2 h)
uv run zerolinc report       # aggregate tables -> results/report/summary.md
```

## Prompt configurations (the zero-shot analogue of prompt engineering)

| Config | Hypothesis template | Label verbalization |
|---|---|---|
| `en-name` | "This text is about {}." | category name |
| `en-desc` | "This text is about {}." | name: NIST description |
| `en-desc-domain` | "This security incident report describes {}." | name + description |
| `en-desc-kw` | "This security incident report describes {}." | name + description + search terms |
| `pt-name` | "Este texto é sobre {}." | Portuguese name |
| `pt-desc` | "Este relato de incidente de segurança descreve {}." | PT name + PT description |

Category names, descriptions, and search terms are the exact ones used by the
reference LLM prompts, so the comparison isolates the classification mechanism.
