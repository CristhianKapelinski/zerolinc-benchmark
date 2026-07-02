"""Non-neural reference baselines: majority class and keyword matching.

The keyword baseline scores each category by counting case-insensitive
occurrences of its "search terms" (the exact lists embedded in the reference
LLM prompts) and predicts the argmax; ties and zero-match texts fall back to
the training-free prior (the majority class), making the rule deterministic.
"""

from collections import Counter

from .labels import CATEGORIES


def majority_baseline(y_true: list[str], texts: list[str]) -> list[str]:
    majority = Counter(y_true).most_common(1)[0][0]
    return [majority] * len(texts)


def keyword_baseline(y_true: list[str], texts: list[str]) -> list[str]:
    majority = Counter(y_true).most_common(1)[0][0]
    preds = []
    for text in texts:
        low = text.lower()
        scores = {c.code: sum(low.count(k) for k in c.keywords) for c in CATEGORIES}
        best = max(scores.values())
        winners = [code for code, s in scores.items() if s == best]
        preds.append(winners[0] if best > 0 and len(winners) == 1 else majority)
    return preds
