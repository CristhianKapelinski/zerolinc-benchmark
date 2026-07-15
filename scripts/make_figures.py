"""Generate paper figures from stored runs. Reads results/runs, writes figures/.

Design: dataviz method — form first (heatmap for the model x config matrix,
scatter for cost x quality, bars for per-class), validated palette, in-cell
direct labels (relief rule), recessive grid, no chartjunk. Ultrawide panels.
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.8,
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "text.color": "#0b0b0b",
    "axes.labelcolor": "#0b0b0b",
    "figure.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# validated palette (dataviz reference instance, light mode)
CAT = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ)

MODEL_SHORT = {
    "bart-large-mnli": "BART-large",
    "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7": "mDeBERTa-base",
    "xlm-roberta-large-xnli": "XLM-R-large",
    "deberta-v3-large-zeroshot-v2.0": "DeBERTa-large-zs",
    "bge-m3-zeroshot-v2.0": "BGE-M3-zs",
    "gliclass-x-base": "GLiClass-x-base",
    "gliclass-modern-base-v3.0": "GLiClass-modern",
    "gliclass-large-v3.0": "GLiClass-large",
    "Qwen3-Reranker-0.6B": "Qwen3-Rerank-0.6B",
    "multilingual-e5-large-instruct": "mE5-large-inst",
    "Qwen3-Embedding-0.6B": "Qwen3-Emb-0.6B",
}


def load(results_dir: Path, view: str = "full") -> list[dict]:
    runs = [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]
    runs = [r for r in runs
            if not r["run_id"].startswith("baseline")
            and r.get("text_view", "full") == view]
    # drop models covering under half the configs in this view (sparse heatmap rows)
    by_model = {}
    for r in runs:
        by_model.setdefault(r["model"], []).append(r)
    n_cfg = max(len(v) for v in by_model.values())
    return [r for r in runs if len(by_model[r["model"]]) >= n_cfg / 2]


def fig_matrix(runs: list[dict], out: Path, metric: str, fname: str, title: str) -> None:
    models = sorted({r["model"].split("/")[-1] for r in runs})
    configs = sorted({r["prompt_config"] for r in runs})
    grid = np.full((len(models), len(configs)), np.nan)
    for r in runs:
        i = models.index(r["model"].split("/")[-1])
        j = configs.index(r["prompt_config"])
        grid[i, j] = r["metrics"][metric] * 100
    fig, ax = plt.subplots(figsize=(9, 2.6))
    vmax = np.nanmax(grid)
    ax.imshow(grid, cmap=SEQ_CMAP, aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(configs)), configs, fontsize=8)
    ax.set_yticks(range(len(models)), [MODEL_SHORT.get(m, m) for m in models], fontsize=8)
    best = np.unravel_index(np.nanargmax(grid), grid.shape)
    for i in range(len(models)):
        for j in range(len(configs)):
            if np.isnan(grid[i, j]):
                continue
            dark = grid[i, j] > 0.55 * vmax
            weight = "bold" if (i, j) == best else "normal"
            ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if dark else "#0b0b0b", fontweight=weight)
    ax.set_title(title, fontsize=9, loc="left")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(out / f"{fname}.pdf")
    fig.savefig(out / f"{fname}.png")
    plt.close(fig)


BACKEND_COLOR = {"nli": CAT[0], "gliclass": CAT[1], "embed": CAT[2], "rerank": CAT[4], "memory": CAT[3]}


def fig_cost(runs: list[dict], out: Path) -> None:
    """Cost x quality as a two-panel dot plot: no in-plot label collisions.

    One row per checkpoint (best run across all views/configs), sorted by
    accuracy; panel (a) accuracy with Wilson CI, panel (b) wall-clock (log).
    """
    best = {}
    for r in runs:
        key = (r.get("backend", "nli"), r.get("method", ""), r["model"].split("/")[-1])
        if key not in best or r["metrics"]["accuracy"] > best[key]["metrics"]["accuracy"]:
            best[key] = r
    items = sorted(best.items(), key=lambda kv: kv[1]["metrics"]["accuracy"])
    def display(k, r):
        if r.get("backend") == "memory":
            size = "4B" if "4B" in r["model"] else "0.6B"
            rule = "k-NN" if r.get("method", "knn") == "knn" else "centroid"
            return f"{rule} memory (Qwen3-Emb-{size})"
        return f"{MODEL_SHORT.get(k[2], k[2])} ({r['prompt_config']})"
    names = [display(k, r) for k, r in items]
    ys = np.arange(len(items))
    colors = [BACKEND_COLOR.get(r.get("backend", "nli"), CAT[5]) for _, r in items]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9, 3.4), sharey=True,
        gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.06})
    for y, (k, r), c in zip(ys, items, colors):
        acc = r["metrics"]["accuracy"] * 100
        lo, hi = (v * 100 for v in r["metrics"]["accuracy_ci95"])
        ax1.errorbar(acc, y, xerr=[[acc - lo], [hi - acc]], fmt="o", color=c,
                     markersize=6, capsize=2.5, lw=1)
        ax2.plot([r["wall_seconds"]], [y], "o", color=c, markersize=6)
        ax2.annotate(f"{r['wall_seconds']:.0f}s", (r["wall_seconds"], y),
                     textcoords="offset points", xytext=(7, -3), fontsize=7,
                     color="#52514e")
    ax1.set_yticks(ys, names, fontsize=8)
    ax1.set_xlabel("(a) accuracy (%), 95% CI")
    ax1.set_xlim(0, 100)
    ax2.set_xscale("log")
    ax2.set_xticks([2, 10, 60, 300])
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.minorticks_off()
    ax2.set_xlim(1.1, 900)
    ax2.set_xlabel("(b) wall-clock seconds, log")
    seen = {}
    for _, r in items:
        b = r.get("backend", "nli")
        seen[b] = BACKEND_COLOR.get(b, CAT[5])
    handles = [matplotlib.lines.Line2D([], [], marker="o", ls="", color=c, label=b)
               for b, c in sorted(seen.items())]
    ax1.legend(handles=handles, frameon=False, fontsize=7.5, loc="lower right")
    for ax in (ax1, ax2):
        ax.grid(True, axis="x", color="#e1e0d9", lw=0.6)
        ax.set_axisbelow(True)
    fig.savefig(out / "fig_cost.pdf")
    fig.savefig(out / "fig_cost.png")
    plt.close(fig)


def fig_perclass(runs: list[dict], out: Path) -> None:
    best = max((r for r in runs if r.get("backend") != "memory"),
               key=lambda r: r["metrics"]["accuracy"])
    per = best["metrics"]["per_class"]
    labels = list(per)
    f1 = [per[c]["f1"] for c in labels]
    rec = [per[c]["recall"] for c in labels]
    sup = [per[c]["support"] for c in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 2.1))
    ax.bar(x - 0.2, f1, 0.38, color=CAT[0], label="F1")
    ax.bar(x + 0.2, rec, 0.38, color=CAT[1], label="recall")
    for xi, s in zip(x, sup):
        ax.text(xi, 1.02, f"n={s}", ha="center", fontsize=7, color="#52514e")
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    ax.set_title(f"Per-class metrics, best run ({best['run_id']})", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.grid(True, axis="y", color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(out / "fig_perclass.pdf")
    fig.savefig(out / "fig_perclass.png")
    plt.close(fig)


def main() -> int:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/runs")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures")
    view = sys.argv[3] if len(sys.argv) > 3 else "full"
    out.mkdir(parents=True, exist_ok=True)
    runs = load(results_dir, view)
    if not runs:
        print("no model runs found", file=sys.stderr)
        return 1
    fig_matrix(runs, out, "accuracy", "fig_grid_acc", "Accuracy (%) by model x category verbalization")
    fig_matrix(runs, out, "macro_f1", "fig_grid_f1", "Macro-F1 (%) by model x category verbalization")
    all_runs = [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))
                if "baseline" not in p.name]
    fig_cost(all_runs, out)
    fig_perclass(runs, out)
    print(f"figures written to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
