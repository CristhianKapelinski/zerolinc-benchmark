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

**Every number the paper reports**, including the 90.8% headline. They regenerate from the
committed run records, which carry one row per ticket with its true and predicted category
and **no ticket text**:

- `./run_claim1.sh`, `./run_claim2.sh` and `./run_claim3.sh`, each of which states in its
  result block whether it measured here or read the record
- `./scripts/reproduce.sh` (all headline numbers, no GPU)

The five per-split records behind the headline are in `results/runs/memory-knn-5seed/`.

## What the corpus adds

Re-measuring instead of reading. With the file in place, `./run_claim1.sh` recomputes the
five splits on your machine and `./run_claim3.sh` re-times the cost claim; the from-scratch
grid (`./scripts/run_all.sh`, `reproduce_full.sh`) also needs it. No command *requires* it.
