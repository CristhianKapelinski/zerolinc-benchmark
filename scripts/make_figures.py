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
    "multilingual-e5-large-instruct": "mE5-large-inst",
    "Qwen3-Embedding-0.6B": "Qwen3-Emb-0.6B",
}


def load(results_dir: Path, view: str = "full") -> list[dict]:
    runs = [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]
    return [r for r in runs
            if not r["run_id"].startswith("baseline")
            and r.get("text_view", "full") == view]


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


BACKEND_COLOR = {"nli": CAT[0], "gliclass": CAT[1], "embed": CAT[2], "rerank": CAT[4]}


def fig_cost(runs: list[dict], out: Path) -> None:
    """Cost x quality: best run per model; color = backend family, direct labels."""
    best = {}
    for r in runs:
        key = r["model"].split("/")[-1]
        if key not in best or r["metrics"]["accuracy"] > best[key]["metrics"]["accuracy"]:
            best[key] = r
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    items = sorted(best.items(), key=lambda kv: kv[1]["wall_seconds"])
    seen_backends = {}
    for k, (name, r) in enumerate(items):
        x = r["wall_seconds"]
        y = r["metrics"]["accuracy"] * 100
        lo, hi = (v * 100 for v in r["metrics"]["accuracy_ci95"])
        backend = r.get("backend", "nli")
        color = BACKEND_COLOR.get(backend, CAT[5])
        seen_backends[backend] = color
        ax.errorbar(x, y, yerr=[[y - lo], [hi - y]], fmt="o", color=color,
                    markersize=7, capsize=3, lw=1)
        above = k % 2 == 0
        ax.annotate(f"{MODEL_SHORT.get(name, name)} ({r['prompt_config']})",
                    (x, y), textcoords="offset points",
                    xytext=(0, 26 if above else -32), ha="center", fontsize=7.5,
                    arrowprops={"arrowstyle": "-", "color": "#c3c2b7", "lw": 0.6})
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 60])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.minorticks_off()
    ax.set_xlabel("wall-clock seconds for 185 incidents (log scale)")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 90)
    handles = [matplotlib.lines.Line2D([], [], marker="o", ls="", color=c, label=b)
               for b, c in sorted(seen_backends.items())]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right",
              title="backend", title_fontsize=8)
    ax.grid(True, color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(out / "fig_cost.pdf")
    fig.savefig(out / "fig_cost.png")
    plt.close(fig)


def fig_perclass(runs: list[dict], out: Path) -> None:
    best = max(runs, key=lambda r: r["metrics"]["accuracy"])
    per = best["metrics"]["per_class"]
    labels = list(per)
    f1 = [per[c]["f1"] for c in labels]
    rec = [per[c]["recall"] for c in labels]
    sup = [per[c]["support"] for c in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 2.8))
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
    fig_matrix(runs, out, "accuracy", "fig_grid_acc", "Accuracy (%) by model x prompt configuration")
    fig_matrix(runs, out, "macro_f1", "fig_grid_f1", "Macro-F1 (%) by model x prompt configuration")
    fig_cost(runs, out)
    fig_perclass(runs, out)
    print(f"figures written to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
