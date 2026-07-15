"""Non-neural reference baselines: majority class and keyword matching.

The keyword baseline scores each category by counting case-insensitive
whole-word occurrences of its "search terms" (the exact lists embedded in the
reference LLM prompts) and predicts the argmax; ties and zero-match texts fall
back to the majority class. Note: the majority class is taken from the gold
labels of the evaluated corpus itself, the usual descriptive floor; both
baselines therefore see label-distribution information no zero-shot model gets.
"""

import re
from collections import Counter

from .verbalizer import CATEGORIES

_KW_RE = {
    c.code: [re.compile(r"(?<!\w)" + re.escape(k) + r"(?!\w)") for k in c.keywords]
    for c in CATEGORIES
}


def majority_baseline(y_true: list[str], texts: list[str]) -> list[str]:
    majority = Counter(y_true).most_common(1)[0][0]
    return [majority] * len(texts)


def keyword_baseline(y_true: list[str], texts: list[str]) -> list[str]:
    majority = Counter(y_true).most_common(1)[0][0]
    preds = []
    for text in texts:
        low = text.lower()
        scores = {
            code: sum(len(rx.findall(low)) for rx in rxs) for code, rxs in _KW_RE.items()
        }
        best = max(scores.values())
        winners = [code for code, s in scores.items() if s == best]
        preds.append(winners[0] if best > 0 and len(winners) == 1 else majority)
    return preds
