# Experimental design details

Complementary to the README (which is the single guide needed for artifact evaluation).

## Corpus

182 unique expert-labeled tickets (185 CSV rows minus 3 exact duplicates, deduplicated at load time) from real Brazilian CSIRT e-mail threads, labeled into the 12 NIST SP 800-61r3-derived categories by two security specialists (see the reference study, DOI 10.5753/sbseg_estendido.2025.12510). Heavily skewed: 118/182 (64.8%) are CAT5; four categories have zero support. Bilingual: 132 predominantly Portuguese, 50 predominantly English. Anonymization tags (`[EMAIL_ADDRESS_<hash>]`) are compressed to short placeholders (`<EMAIL>`) at load time.

## Classifier families (backends)

- `nli:` entailment cross-encoders (ticket = premise, category verbalization = hypothesis)
- `gliclass:` GLiClass single-pass label scoring
- `embed:` instruction bi-encoders (cosine similarity between ticket and category-description embeddings)
- `rerank:` generative rerankers (P("yes") that a category matches the ticket)

## Verbalization sets (8)

`en-name`, `pt-name` (bare category names); `en-desc`, `pt-desc` (NIST descriptions); `en-desc-domain` (domain-anchored hypothesis template); `en-desc-kw` (descriptions + the reference prompts' search terms); `en-desc-ex` (descriptions + their examples); `en-event` (event-style declarative hypotheses). All text is taken verbatim from the reference studies' prompt material.

## Input views (4)

`full` (normalized text); `subject` (a copy of the e-mail Subject line(s) prepended); `deboiler` (drop every line occurring in ≥ 15% of corpus documents — corpus-statistical, no hand-written patterns); `subject-deboiler` (both).

## Selection protocol

Seeded, class-stratified validation/test split (89/93 tickets). Configurations are selected on the validation half only; the selected configuration is evaluated on the untouched test half with a paired McNemar test against the majority baseline. Repeated over seeds 42, 7, 123, 2024, 99. For the instance-memory engine, the validation half acts as the labeled reference set and k plus the input view are chosen by leave-one-out inside it.

## Run of record

`results/runs/*.json` — one self-contained record per run: full configuration, per-ticket predictions (with 12-way score vectors where captured), metrics (accuracy + Wilson 95% CI, fixed-label macro-F1, per-class PRF, confusion matrix), and cost figures (wall-clock, throughput, peak VRAM, GPU power and integrated energy). Every number and figure in the paper regenerates from these records; `scripts/regen_metrics.py` recomputes all metrics from the stored predictions.

## Determinism

Inference is deterministic (argmax, no sampling); splits and k-selection are seeded. Metrics recomputed from the stored predictions reproduce exactly. Live re-runs of the zero-shot engines reproduced the committed accuracies exactly in our tests; live re-runs of the instance-memory engine vary slightly (per-seed accuracy ±1–2 p.p., mean ±0.5 p.p.) due to GPU fp16 embedding nondeterminism. Wall-clock and energy vary with hardware.
