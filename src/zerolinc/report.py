"""Aggregate stored run files into comparison tables (markdown + CSV)."""

import json
from pathlib import Path

import pandas as pd


def load_runs(results_dir: str | Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(results_dir).glob("*.json"))]


def summary_frame(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        m = r["metrics"]
        rows.append(
            {
                "model": r["model"].split("/")[-1],
                "backend": r.get("backend", "nli"),
                "prompt_config": r["prompt_config"],
                "text_view": r.get("text_view", "full"),
                "accuracy": m["accuracy"],
                "ci95_low": m["accuracy_ci95"][0],
                "ci95_high": m["accuracy_ci95"][1],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                "n": m["n"],
                "wall_s": r.get("wall_seconds", 0.0),
                "inc_per_s": r.get("incidents_per_second"),
                "peak_vram_mb": r.get("peak_vram_mb"),
                "gpu_energy_wh": r.get("gpu_energy_wh"),
            }
        )
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


def write_report(results_dir: str | Path, out_dir: str | Path) -> Path:
    runs = load_runs(results_dir)
    df = summary_frame(runs)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)

    lines = ["# ZeroLINC results", "", df.to_markdown(index=False), ""]
    best = next((r for r in runs if f"{r['model'].split('/')[-1]}__{r['prompt_config']}"
                 == f"{df.iloc[0]['model']}__{df.iloc[0]['prompt_config']}"), None)
    if best:
        lines.append(f"## Per-class metrics of the best run ({best['run_id']})")
        per = pd.DataFrame(best["metrics"]["per_class"]).T
        lines += ["", per.to_markdown(), ""]
    (out / "summary.md").write_text("\n".join(lines))
    return out / "summary.md"
