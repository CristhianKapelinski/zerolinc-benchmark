# Evaluation corpus (not redistributed)

The labeled corpus (`185_incidentes_anon.csv`, 185 rows, 182 unique tickets after in-loader dedup) is **not publicly redistributable**: the reference studies publish only a five-ticket anonymized sample in their artifact ([SecLINC](https://github.com/AI-Horizon-Labs/SecLINC)). The full anonymized corpus is available on request from the authors of the reference study (Severo et al., SBSeg 2025, DOI [10.5753/sbseg_estendido.2025.12510](https://doi.org/10.5753/sbseg_estendido.2025.12510)).

Once obtained, place the file at:

```
data/185_incidentes_anon.csv
```

Integrity check (the exact file used in the paper):

```
sha256sum data/185_incidentes_anon.csv
dcb4ecec5f79482e6614983129eb82867fe21afae2ca4044d5c4299b3e627d74
```

## What works without the corpus

Every reported number regenerates from the committed run records, which contain per-ticket predictions and labels but **no ticket text**:

- `./scripts/reproduce.sh` (all headline numbers, no GPU)
- `run_claim2.sh` (zero-shot study recomputation)
- `SKIP_LIVE=1 ./run_claim3.sh` (cost claim from the committed record)

## What needs the corpus

- `run_claim1.sh` (live 5-seed protocol of the instance-memory engine)
- `run_claim3.sh` live re-timing path
- `./scripts/run_all.sh` / `reproduce_full.sh` (from-scratch grid)
