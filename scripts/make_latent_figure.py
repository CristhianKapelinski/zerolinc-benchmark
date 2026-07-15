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
    best = max((json.loads(f.read_text()) for f in Path("results/runs").glob("*.json")
                if "baseline" not in f.name), default=None,
               key=lambda r: r["metrics"]["accuracy"] if r.get("backend") != "memory" else -1)
    per = best["metrics"]["per_class"]

    fig, (ax, ax2, ax3) = plt.subplots(
        1, 3, figsize=(9.6, 2.3), gridspec_kw={"width_ratios": [1.3, 1, 1.2], "wspace": 0.25})
    for cat in sorted(CAT_COLOR, key=lambda c: -labels.count(c)):
        idx = [j for j, lab in enumerate(labels) if lab == cat]
        if not idx:
            continue
        ax.scatter(xy[idx, 0], xy[idx, 1], s=14, color=CAT_COLOR[cat],
                   edgecolors="white", linewidths=0.4,
                   label=f"{cat} ({len(idx)})")
    ax.legend(fontsize=7, frameon=False, borderpad=0.2, handletextpad=0.2,
              loc="upper center", bbox_to_anchor=(0.5, -0.04), ncols=4, columnspacing=0.7)
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
    cats = list(per)
    xx = np.arange(len(cats))
    f1v = [per[c]["f1"] for c in cats]
    rcv = [per[c]["recall"] for c in cats]
    ax3.bar(xx - 0.2, f1v, 0.38, color="#2a78d6", label="F1")
    ax3.bar(xx + 0.2, rcv, 0.38, color="#1baf7a", label="recall")
    for xi, v in zip(xx - 0.2, f1v):
        ax3.text(xi, v + 0.03, ("1" if v == 1 else f"{v:.1f}".replace("0.", ".")), ha="center",
                 va="bottom", fontsize=5.6, color="#2a78d6")
    for xi, v in zip(xx + 0.2, rcv):
        ax3.text(xi, v + 0.03, ("1" if v == 1 else f"{v:.1f}".replace("0.", ".")), ha="center",
                 va="bottom", fontsize=5.6, color="#118a5f")
    for xi, c in zip(xx, cats):
        ax3.text(xi, 1.26, str(per[c]["support"]), ha="center", fontsize=6,
                 color="#52514e")
    ax3.set_xticks(xx, [c.replace("CAT", "") for c in cats], fontsize=7)
    ax3.set_xlabel("category (top: $n$)", fontsize=8)
    ax3.set_yticks([])
    ax3.set_ylim(0, 1.38)
    ax3.legend(fontsize=6.2, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.55), borderpad=0.1, handlelength=1.2)
    ax3.spines[["top", "right", "left"]].set_visible(False)

    ax.set_title("(a) ticket embedding space (t-SNE)", fontsize=8, loc="left")
    ax2.set_title("(b) accuracy vs. reference-set size", fontsize=8, loc="left")
    ax3.set_title("(c) best zero-shot, per class", fontsize=8, loc="left")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "fig_latent.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_latent.png", bbox_inches="tight")
    print(f"latent figure written to {out_dir}/fig_latent.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
