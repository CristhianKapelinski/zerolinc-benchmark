"""Command-line interface: run one config, the full grid, baselines, or the report."""

import argparse
import sys
from pathlib import Path

from .data import apply_view, load_incidents
from .labels import PROMPT_CONFIGS
from .runner import DEFAULT_MODELS, run_baselines, run_one  # noqa: F401 (V2_MODELS lazy)
from .report import write_report

DEFAULT_DATA = Path("data/185_incidentes_anon.csv")
DEFAULT_RESULTS = Path("results/runs")
DEFAULT_REPORT = Path("results/report")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="zerolinc", description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="incident CSV path")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="run output dir")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=8)
    parser.add_argument("--view", choices=("full", "subject", "deboiler", "subject-deboiler"),
                        default="full", help="text view fed to the models")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run one model x prompt-config pass")
    p_run.add_argument("--model", required=True)
    p_run.add_argument("--config", required=True, choices=sorted(PROMPT_CONFIGS))
    p_run.add_argument("--limit", type=int, default=0, help="use only the first N incidents")
    p_run.add_argument("--batch-size", dest="batch_size", type=int,
                       default=argparse.SUPPRESS)

    p_grid = sub.add_parser("grid", help="run all models x all prompt configs")
    p_grid.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    p_grid.add_argument("--v2", action="store_true",
                        help="use the v2 model set (GLiClass + embedding backends)")
    p_grid.add_argument("--configs", nargs="*", default=sorted(PROMPT_CONFIGS),
                        choices=sorted(PROMPT_CONFIGS))
    p_grid.add_argument("--limit", type=int, default=0)
    p_grid.add_argument("--skip-existing", action="store_true",
                        help="skip runs whose result file already exists")

    sub.add_parser("baselines", help="run majority and keyword baselines")

    p_rep = sub.add_parser("report", help="aggregate stored runs into summary tables")
    p_rep.add_argument("--out", type=Path, default=DEFAULT_REPORT)

    p_proto = sub.add_parser(
        "protocol", help="dev/test selection protocol + McNemar, from stored runs")
    p_proto.add_argument("--seed", type=int, default=42)
    p_proto.add_argument("--out", type=Path, default=DEFAULT_REPORT)

    args = parser.parse_args(argv)

    if args.command == "report":
        path = write_report(args.results, args.out)
        print(f"report written to {path}")
        return 0

    if args.command == "protocol":
        import json as _json

        from .protocol import protocol_report
        result = protocol_report(args.results, seed=args.seed)
        args.out.mkdir(parents=True, exist_ok=True)
        out_file = args.out / f"protocol_seed{args.seed}.json"
        out_file.write_text(_json.dumps(result, ensure_ascii=False, indent=1))
        print(_json.dumps(result, ensure_ascii=False, indent=1))
        print(f"\nprotocol written to {out_file}")
        return 0

    incidents = apply_view(load_incidents(args.data), args.view)
    if args.command in ("run", "grid") and args.limit:
        incidents = incidents[: args.limit]
    tag = args.view if args.view != "full" else ""

    if args.command == "baselines":
        for record in run_baselines(incidents, args.results, tag):
            m = record["metrics"]
            print(f"{record['run_id']}: acc={m['accuracy']} macro_f1={m['macro_f1']}")
        return 0

    if args.command == "run":
        record = run_one(args.model, args.config, incidents, args.results, args.batch_size, tag)
        m = record["metrics"]
        print(f"{record['run_id']}: acc={m['accuracy']} ci95={m['accuracy_ci95']} "
              f"macro_f1={m['macro_f1']} wall={record['wall_seconds']}s")
        return 0

    if args.command == "grid":
        from .runner import V2_MODELS, run_id as rid
        if args.v2:
            args.models = list(V2_MODELS)
        total = len(args.models) * len(args.configs)
        done = 0
        for model in args.models:
            for config in args.configs:
                done += 1
                out_file = args.results / f"{rid(model, config, tag)}.json"
                if args.skip_existing and out_file.exists():
                    print(f"[{done}/{total}] skip {out_file.name}")
                    continue
                print(f"[{done}/{total}] {model} x {config} ...", flush=True)
                try:
                    record = run_one(model, config, incidents, args.results, args.batch_size, tag)
                except Exception as exc:  # keep the grid going; a failed run is reported
                    print(f"  FAILED: {exc}", file=sys.stderr)
                    continue
                m = record["metrics"]
                print(f"  acc={m['accuracy']} macro_f1={m['macro_f1']} "
                      f"wall={record['wall_seconds']}s vram={record['peak_vram_mb']}MB")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
