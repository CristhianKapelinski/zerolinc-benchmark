# ZeroLINC Benchmark

Measurement study behind [ZeroLINC](https://gitlab.com/cristhianavila.aluno/zerolinc), the training-free local classifier of security incident reports (SBSeg 2026, Salão de Ferramentas). **Artifact evaluation happens in the tool repository**; this one is the research companion: the full grid of 292 evaluation runs, the committed run of record, the labeled corpus, the selection protocol, and the scripts that regenerate every number and figure of the paper.

## What is here

- `src/zerolinc/`: the classifier modules (named as in the paper) plus the evaluation harness: grid `runner.py`, `baselines.py`, `metrics.py` (Wilson CI, fixed-label macro-F1, McNemar), `protocol.py` (seeded validation/test selection), `combine.py` (calibration/ensembles), `report.py`.
- `data/`: the corpus is **not redistributed** (see `data/README.md`: how to request it, sha256, and what runs without it). The committed run records carry per-ticket predictions and labels, no ticket text.
- `results/runs/`: the run of record: one JSON per run with per-ticket predictions, score vectors, metrics, and cost figures (wall-clock, VRAM, GPU energy).
- `results/report/`: protocol outputs per seed.
- `run_claim{1,2,3}.sh`: one script per paper claim (invoked automatically by the tool repository's wrappers).
- `scripts/`: `reproduce.sh` (no-GPU, asserts all headline numbers), `reproduce_full.sh` (from-scratch grid, ~4–5 h GPU), `make_figures.py`, `make_paper_macros.py`, `regen_metrics.py`, `run_all.sh`, `smoke.sh`.
- `docs/DESIGN.md`: corpus, backends, verbalizations, views, protocol, determinism notes.

## Quickstart

```bash
uv sync --extra dev
uv run pytest -q          # 15 offline tests
./scripts/reproduce.sh    # every paper number from the run of record, no GPU (~1 min)
./scripts/run_all.sh      # full 292-run grid from scratch (GPU, ~4-5 h)
```

## License

[GNU AGPL-3.0-or-later](LICENSE).
