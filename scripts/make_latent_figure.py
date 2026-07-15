"""t-SNE projection of the ticket embeddings, colored by gold category.

Illustrates the design argument of the instance-memory engine: CSIRT traffic
is template-heavy, so recurring alert formats form tight clusters in the
embedding space and a handful of labeled instances resolves whole families.
Deterministic (fixed seed); requires the corpus and a GPU-less run works too.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zerolinc.memory_engine import embed_texts
from zerolinc.normalizer import apply_view, load_incidents

CAT_COLOR = {
    "CAT1": "#e15759", "CAT2": "#b07aa1", "CAT3": "#f28e2b", "CAT5": "#a7c7e7",
    "CAT7": "#76b7b2", "CAT9": "#59a14f", "CAT10": "#9c755f", "CAT12": "#1f3f77",
}


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("figures/subject")
    incidents = load_incidents("data/185_incidentes_anon.csv")
    texts = [i.text for i in apply_view(incidents, "subject")]
    labels = [i.label for i in incidents]
    emb = embed_texts("Qwen/Qwen3-Embedding-0.6B", texts)

    xy = TSNE(n_components=2, metric="cosine", perplexity=15, random_state=42,
              init="pca").fit_transform(np.asarray(emb))

    import json
    curve = json.loads(Path("results/report/learning_curve.json").read_text())

    plt.rcParams.update({"font.size": 8, "figure.dpi": 150})
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(9, 2.2), gridspec_kw={"width_ratios": [1.5, 1]})
    for cat in sorted(CAT_COLOR, key=lambda c: -labels.count(c)):
        idx = [j for j, lab in enumerate(labels) if lab == cat]
        if not idx:
            continue
        ax.scatter(xy[idx, 0], xy[idx, 1], s=14, color=CAT_COLOR[cat],
                   edgecolors="white", linewidths=0.4,
                   label=f"{cat} ({len(idx)})")
    ax.legend(fontsize=7, frameon=False, borderpad=0.2, handletextpad=0.2,
              loc="upper center", bbox_to_anchor=(0.5, -0.03), ncols=8, columnspacing=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
    ns = sorted(int(k) for k in curve)
    means = [curve[str(n)]["mean"] * 100 for n in ns]
    lo = [curve[str(n)]["min"] * 100 for n in ns]
    hi = [curve[str(n)]["max"] * 100 for n in ns]
    ax2.fill_between(ns, lo, hi, color="#a7c7e7", alpha=0.45, lw=0,
                     label="min-max over 5 splits")
    ax2.plot(ns, means, "o-", color="#1f3f77", ms=4, lw=1.4, label="mean")
    ax2.axhline(70.9, color="#f28e2b", ls="--", lw=1.1)
    ax2.text(ns[-1], 71.6, "best zero-shot (70.9)", ha="right", fontsize=7,
             color="#b36a10")
    ax2.set_xlabel("labeled reference tickets")
    ax2.set_ylabel("test accuracy (%)")
    ax2.set_ylim(60, 95)
    ax2.legend(fontsize=7, loc="lower right")
    ax2.spines[["top", "right"]].set_visible(False)
    ax.set_title("(a) ticket embedding space (t-SNE)", fontsize=8, loc="left")
    ax2.set_title("(b) accuracy vs. reference-set size", fontsize=8, loc="left")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_latent.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_latent.png", bbox_inches="tight")
    print(f"latent figure written to {out_dir}/fig_latent.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
