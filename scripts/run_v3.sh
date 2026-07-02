#!/usr/bin/env bash
# Curated v3 subset: best run per family/view with full score vectors + reranker.
set -euo pipefail
cd "$(dirname "$0")/.."
Z="uv run --no-sync zerolinc"
$Z --view subject run --model nli:MoritzLaurer/deberta-v3-large-zeroshot-v2.0 --config en-desc-kw
$Z --view subject run --model nli:MoritzLaurer/deberta-v3-large-zeroshot-v2.0 --config en-desc-ex
$Z --view subject run --model nli:MoritzLaurer/deberta-v3-large-zeroshot-v2.0 --config en-event
$Z --view subject run --model nli:MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 --config en-desc
$Z run --model gliclass:knowledgator/gliclass-modern-base-v3.0 --config en-event
$Z run --model gliclass:knowledgator/gliclass-x-base --config pt-desc
$Z --view deboiler run --model gliclass:knowledgator/gliclass-modern-base-v3.0 --config en-name
$Z --view subject-deboiler run --model embed:Qwen/Qwen3-Embedding-0.6B --config pt-name
$Z --view deboiler run --model embed:Qwen/Qwen3-Embedding-0.6B --config en-desc-domain
$Z --view subject run --model embed:intfloat/multilingual-e5-large-instruct --config en-desc
$Z --view subject run --model rerank:Qwen/Qwen3-Reranker-0.6B --config en-desc --batch-size 4
$Z --view subject run --model rerank:Qwen/Qwen3-Reranker-0.6B --config en-event --batch-size 4
echo "V3 SUBSET DONE"
