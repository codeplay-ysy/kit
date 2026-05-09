from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from evaluate_subitem_metrics import build_confusion, compute_metrics, load_predictions, load_truth, match_beats


def _norm_label(label: str) -> str:
    return "AF" if label in {"AF", "Af"} else label


def _format_hms(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def make_strong_only_events(windows_csv: Path, out_json: Path, gap_ms: int = 5_000, min_duration_ms: int = 30_000) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(windows_csv.open(encoding="utf-8-sig")))
    strong = [row for row in rows if row["label"] == "strong_af"]
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for row in strong:
        start_ms = int(float(row["start_ms"]))
        end_ms = int(float(row["end_ms"]))
        if current is None:
            current = {"start_ms": start_ms, "end_ms": end_ms, "window_count": 1}
        elif start_ms - int(current["end_ms"]) <= gap_ms:
            current["end_ms"] = max(int(current["end_ms"]), end_ms)
            current["window_count"] = int(current["window_count"]) + 1
        else:
            segments.append(current)
            current = {"start_ms": start_ms, "end_ms": end_ms, "window_count": 1}
    if current is not None:
        segments.append(current)

    events: list[dict[str, Any]] = []
    for index, segment in enumerate(
        [item for item in segments if int(item["end_ms"]) - int(item["start_ms"]) >= min_duration_ms],
        start=1,
    ):
        start_ms = int(segment["start_ms"])
        end_ms = int(segment["end_ms"])
        events.append(
            {
                "type": "af_family",
                "subtype": "fibrillation",
                "layer": "rr_2d_filter_original_strong_only",
                "rule": "rr_2d_original_strong_only_5s_gap",
                "event_index": index,
                "t0_ms": start_ms,
                "t1_ms": end_ms,
                "time": f"{start_ms} ms ~ {end_ms} ms",
                "duration": _format_hms((end_ms - start_ms) / 1000.0),
                "stats": segment,
            }
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    return events


def run_subitem(edf: Path, merged_csv: Path, events_json: Path, out_csv: Path, out_summary: Path) -> None:
    cmd = [
        sys.executable,
        "subitem_experiment.py",
        "--edf",
        str(edf),
        "--merged-csv",
        str(merged_csv),
        "--af-events-json",
        str(events_json),
        "--out-csv",
        str(out_csv),
        "--out-summary",
        str(out_summary),
    ]
    subprocess.run(cmd, check=True)


def evaluate_case(truth_csv: Path, pred_csv: Path, out_figure: Path | None = None) -> dict[str, Any]:
    if out_figure is not None:
        subprocess.run(
            [
                sys.executable,
                "evaluate_subitem_metrics.py",
                "--truth-csv",
                str(truth_csv),
                "--pred-csv",
                str(pred_csv),
                "--tolerance-ms",
                "80",
                "--out-figure",
                str(out_figure),
            ],
            check=True,
        )

    truths = load_truth(truth_csv)
    preds = load_predictions(pred_csv, "merged_milliseconds", "symbol")
    matches, unmatched_truths, unmatched_preds = match_beats(truths, preds, tolerance_ms=80.0)
    pairs = [(_norm_label(match.truth.label), _norm_label(match.pred.label)) for match in matches]
    labels, matrix = build_confusion(pairs)
    metrics = compute_metrics(labels, matrix, total_truth=len(truths), total_pred=len(preds), matched_count=len(matches))
    metrics["af_row"] = dict(matrix.get("Af", {}))
    metrics["unmatched_truth_count"] = len(unmatched_truths)
    metrics["unmatched_prediction_count"] = len(unmatched_preds)
    return metrics


def default_paths(case_tag: str) -> dict[str, Path]:
    root = Path("out") / case_tag
    return {
        "merged_csv": root / "merged" / f"merged_rpeaks_by_quality_window_v5_priority_{case_tag}.csv",
        "windows_csv": root / "rr_2d_filter" / f"{case_tag}_rr_2d_windows.csv",
        "enhanced_events": root / "rr_2d_filter" / f"{case_tag}_rr_2d_events.json",
        "strong_events": root / "rr_2d_filter" / f"{case_tag}_rr_2d_events_strong_only_original.json",
        "subitem_dir": root / "subitem",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RR2D strong-only vs enhanced AF events on subitem metrics.")
    parser.add_argument("--case-tag", required=True, help="Case output folder/name, e.g. 李祥根 or 周大宝.")
    parser.add_argument("--edf", required=True, help="EDF path.")
    parser.add_argument("--truth-csv", required=True, help="Ground-truth annotation CSV path.")
    parser.add_argument("--merged-csv", default="", help="Merged R-peak CSV override.")
    parser.add_argument("--windows-csv", default="", help="RR2D windows CSV override.")
    parser.add_argument("--enhanced-events-json", default="", help="Enhanced RR2D events JSON override.")
    parser.add_argument("--out-json", default="", help="Comparison metrics JSON output path.")
    parser.add_argument("--skip-subitem", action="store_true", help="Only evaluate existing subitem outputs.")
    args = parser.parse_args()

    paths = default_paths(args.case_tag)
    edf = Path(args.edf)
    truth_csv = Path(args.truth_csv)
    merged_csv = Path(args.merged_csv) if args.merged_csv else paths["merged_csv"]
    windows_csv = Path(args.windows_csv) if args.windows_csv else paths["windows_csv"]
    enhanced_events = Path(args.enhanced_events_json) if args.enhanced_events_json else paths["enhanced_events"]
    strong_events = paths["strong_events"]
    subitem_dir = paths["subitem_dir"]
    subitem_dir.mkdir(parents=True, exist_ok=True)

    make_strong_only_events(windows_csv=windows_csv, out_json=strong_events)

    original_pred = subitem_dir / "subitem_experiment_beats_with_rr2d_original_strong_only.csv"
    original_summary = subitem_dir / "subitem_experiment_summary_with_rr2d_original_strong_only.json"
    enhanced_pred = subitem_dir / "subitem_experiment_beats_with_rr2d_af.csv"
    enhanced_summary = subitem_dir / "subitem_experiment_summary_with_rr2d_af.json"

    if not args.skip_subitem:
        run_subitem(edf=edf, merged_csv=merged_csv, events_json=strong_events, out_csv=original_pred, out_summary=original_summary)
        run_subitem(edf=edf, merged_csv=merged_csv, events_json=enhanced_events, out_csv=enhanced_pred, out_summary=enhanced_summary)

    original_metrics = evaluate_case(
        truth_csv=truth_csv,
        pred_csv=original_pred,
        out_figure=subitem_dir / "subitem_evaluation_with_rr2d_original_strong_only.pdf",
    )
    enhanced_metrics = evaluate_case(
        truth_csv=truth_csv,
        pred_csv=enhanced_pred,
        out_figure=subitem_dir / "subitem_evaluation_with_rr2d_af.pdf",
    )

    result = {
        "case_tag": args.case_tag,
        "paths": {
            "truth_csv": str(truth_csv),
            "edf": str(edf),
            "merged_csv": str(merged_csv),
            "windows_csv": str(windows_csv),
            "strong_events": str(strong_events),
            "enhanced_events": str(enhanced_events),
            "original_pred": str(original_pred),
            "enhanced_pred": str(enhanced_pred),
        },
        "rr2d_original_strong_only": original_metrics,
        "rr2d_enhanced": enhanced_metrics,
    }

    out_json = Path(args.out_json) if args.out_json else subitem_dir / "rr2d_original_vs_enhanced_metrics.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    for label, metrics in [("rr2d_original_strong_only", original_metrics), ("rr2d_enhanced", enhanced_metrics)]:
        af = metrics.get("per_label", {}).get("Af", {})
        print(f"--- {label}")
        print(f"matched_accuracy={metrics.get('matched_accuracy')} macro_f1={metrics.get('macro_f1')}")
        print(f"Af precision={af.get('precision')} recall={af.get('recall')} f1={af.get('f1')} support={af.get('support')}")
        print(f"Af row={metrics.get('af_row')}")
    print(f"metrics_json={out_json}")


if __name__ == "__main__":
    main()
