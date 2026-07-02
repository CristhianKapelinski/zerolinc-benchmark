# ZeroLINC: Zero-shot NLI classification of security incidents

Zero-shot classification of CSIRT incident reports into the 12 NIST SP 800-61r3
categories using off-the-shelf encoder classifiers (no fine-tuning, no API, no GPU
cloud). Counterpart to the LLM prompt-engineering studies on the same corpus
lineage: four backend families score each incident against natural-language
category verbalizations and the best-scored category wins.

**Cost model:** everything runs locally on a consumer GPU (reference machine:
NVIDIA RTX 5060 Ti 16 GB, 30 GB RAM); the marginal cost per classification is
wall-clock seconds and watt-hours, not API dollars. Every run records time,
throughput, peak VRAM, GPU power, and integrated energy.

## Layout

- `src/zerolinc/` - the tool: dataset loading/normalization and text views
  (`data.py`), NIST label sets and verbalizations (`labels.py`), four zero-shot
  backends (`classifier.py` NLI; `backends.py` GLiClass, instruction embeddings,
  generative reranker), non-neural baselines, metrics (accuracy + Wilson 95% CI,
  fixed-label macro-F1, per-class PRF, McNemar), grid runner, report generator,
  score post-processing (`combine.py`), and the dev/test selection protocol
  (`protocol.py`).
- `data/185_incidentes_anon.csv` - anonymized, expert-labeled CSIRT tickets
  (182 unique after in-loader deduplication).
- `results/runs/` - the run of record: one JSON per run with full config,
  per-incident predictions (and score vectors where captured), metrics, and cost.
- `results/report/` - regenerable summary tables and protocol results.
- `scripts/` - one-command entry points (see below).
- `tests/` - offline unit tests (no network, no models).

## Quickstart

```bash
uv sync --extra dev              # install (pinned, CUDA 12.8 wheels)
uv run pytest -q                 # offline unit tests (~6 s)
./scripts/smoke.sh               # baselines + 1 small model on 20 incidents (~2 min)
./scripts/run_all.sh             # full grid: 10 models x 8 configs x 4 views (~4 h)
uv run zerolinc report           # summary tables -> results/report/summary.md
uv run zerolinc protocol         # dev/test selection + McNemar -> protocol_seed42.json
uv run python scripts/make_figures.py results/runs figures/subject subject
uv run python scripts/regen_metrics.py   # recompute all metrics from stored predictions
```

Model downloads default to the Hugging Face cache; override with
`HF_HOME`/`HF_HUB_CACHE` if the default disk is small.

## Experimental axes

- **Backends** (`backend:model` specs): `nli:` entailment cross-encoders,
  `gliclass:` GLiClass, `embed:` instruction bi-encoders, `rerank:` generative
  rerankers. Defaults in `runner.py` (`DEFAULT_MODELS`, `V2_MODELS`).
- **Verbalizations** (8, `labels.py`): the zero-shot analogue of prompt
  engineering; names, NIST descriptions, domain templates, the reference LLM
  prompts' search terms and examples, event-style hypotheses, PT variants.
- **Text views** (4, `--view`): `full`, `subject` (subject-first), `deboiler`
  (corpus-statistical boilerplate removal), `subject-deboiler`.
- **Protocol**: configurations are selected on a seeded stratified dev half and
  reported on the untouched test half, with a paired McNemar test against the
  majority baseline; a cross-family rank ensemble combines one dev-best
  score-carrying run per family.
