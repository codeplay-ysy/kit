from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

QUALITY_SCORES = {
    "Excellent": 2,
    "Barely acceptable": 1,
    "Unacceptable": 0,
}


@dataclass(slots=True)
class PeakRow:
    peak_order: str
    sample_index: str
    milliseconds_text: str
    milliseconds: float
    timestamp: str
    window_index: int


@dataclass(slots=True)
class QualityWindow:
    lead: str
    alias: str
    window_index: int
    window_start: str
    window_end: str
    quality_label: str
    quality_score: int


MERGED_FIELDS = [
    "beat_order",
    "merged_sample_index",
    "merged_milliseconds",
    "merged_timestamp",
    "source",
    "primary_lead",
    "primary_alias",
    "window_index",
    "window_start",
    "window_end",
    "selection_reason",
    "selected_quality_label",
    "selected_quality_score",
]

AUDIT_FIELDS = [
    "window_index",
    "window_start",
    "window_end",
    "window_status",
    "selection_reason",
    "selected_lead",
    "selected_alias",
    "selected_quality_label",
    "selected_quality_score",
    "v5_lead",
    "v5_quality_label",
    "v5_quality_score",
    "v1_lead",
    "v1_quality_label",
    "v1_quality_score",
    "v5_peak_count",
    "v1_peak_count",
    "selected_peak_count",
    "discarded_peak_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge CH1/CH3 R peaks by selecting one lead per quality window.")
    parser.add_argument("--quality-csv", required=True, help="Quality CSV path.")
    parser.add_argument("--ch1-csv", required=True, help="Raw CH1 R-peaks CSV.")
    parser.add_argument("--ch3-csv", required=True, help="Raw CH3 R-peaks CSV.")
    parser.add_argument("--window-seconds", type=int, default=30, help="Quality window length in seconds.")
    parser.add_argument("--v5-lead", default="ECG_CH1", help="Lead name corresponding to V5.")
    parser.add_argument("--v1-lead", default="ECG_CH3", help="Lead name corresponding to V1.")
    parser.add_argument("--case-tag", default="", help="Optional case tag used for output folder names.")
    parser.add_argument("--out-merged", default="", help="Output merged peaks CSV override.")
    parser.add_argument("--out-audit", default="", help="Output per-window audit CSV override.")
    parser.add_argument("--out-summary", default="", help="Output summary JSON override.")
    return parser.parse_args()


def case_tag_from_path(path: Path) -> str:
    stem = path.stem
    if "_ecg_quality_" in stem:
        return stem.split("_ecg_quality_", 1)[0]
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def output_paths(quality_csv: Path, case_tag: str, out_merged: str, out_audit: str, out_summary: str) -> tuple[Path, Path, Path]:
    tag = case_tag or case_tag_from_path(quality_csv)
    out_root = SCRIPT_DIR / "out" / tag / "merged"
    default_merged = out_root / f"merged_rpeaks_by_quality_window_v5_priority_{tag}.csv"
    default_audit = out_root / f"merged_rpeaks_by_quality_window_v5_priority_{tag}_window_audit.csv"
    default_summary = out_root / f"merged_rpeaks_by_quality_window_v5_priority_{tag}_summary.json"
    return (
        Path(out_merged).resolve() if out_merged else default_merged,
        Path(out_audit).resolve() if out_audit else default_audit,
        Path(out_summary).resolve() if out_summary else default_summary,
    )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_peaks(path: Path, window_ms: int) -> dict[int, list[PeakRow]]:
    required = {"peak_order", "sample_index", "milliseconds", "timestamp"}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = required - fieldnames
        if missing:
            raise SystemExit(f"Missing required columns in {path}: {sorted(missing)}")

        peaks_by_window: dict[int, list[PeakRow]] = {}
        for row in reader:
            milliseconds = float(row["milliseconds"])
            peak = PeakRow(
                peak_order=row["peak_order"],
                sample_index=row["sample_index"],
                milliseconds_text=row["milliseconds"],
                milliseconds=milliseconds,
                timestamp=row["timestamp"],
                window_index=int(milliseconds // window_ms),
            )
            peaks_by_window.setdefault(peak.window_index, []).append(peak)

    for peaks in peaks_by_window.values():
        peaks.sort(key=lambda item: item.milliseconds)
    return peaks_by_window


def load_quality_windows(path: Path, v5_lead: str, v1_lead: str) -> dict[int, dict[str, QualityWindow]]:
    required = {"lead", "window_index", "window_start", "window_end", "zhao2018"}
    target_leads = {v5_lead: "V5", v1_lead: "V1"}

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = required - fieldnames
        if missing:
            raise SystemExit(f"Missing required columns in {path}: {sorted(missing)}")

        windows: dict[int, dict[str, QualityWindow]] = {}
        for row in reader:
            lead = row["lead"]
            if lead not in target_leads:
                continue

            quality_label = (row.get("zhao2018") or "").strip()
            row_error = (row.get("error") or "").strip()
            if not quality_label and row_error:
                quality_label = "Unacceptable"
            if quality_label not in QUALITY_SCORES:
                raise SystemExit(f"Unknown quality label in {path}: {quality_label!r}")

            window_index = int(row["window_index"])
            windows.setdefault(window_index, {})[lead] = QualityWindow(
                lead=lead,
                alias=target_leads[lead],
                window_index=window_index,
                window_start=row["window_start"],
                window_end=row["window_end"],
                quality_label=quality_label,
                quality_score=QUALITY_SCORES[quality_label],
            )

    for window_index, lead_map in windows.items():
        missing_leads = {v5_lead, v1_lead} - set(lead_map)
        if missing_leads:
            raise SystemExit(f"Quality window {window_index} is missing leads: {sorted(missing_leads)}")
    return windows


def choose_window_lead(v5_window: QualityWindow, v1_window: QualityWindow) -> tuple[str | None, str, str | None, int | None]:
    highest_score = max(v5_window.quality_score, v1_window.quality_score)
    if highest_score == 0:
        return None, "invalid_highest_unacceptable", None, None

    if v5_window.quality_score >= v1_window.quality_score:
        reason = "tie_prefer_v5" if v5_window.quality_score == v1_window.quality_score else "higher_quality_v5"
        return v5_window.lead, reason, v5_window.quality_label, v5_window.quality_score

    return v1_window.lead, "higher_quality_v1", v1_window.quality_label, v1_window.quality_score


def merge_by_quality_window(
    quality_windows: dict[int, dict[str, QualityWindow]],
    v5_lead: str,
    v1_lead: str,
    v5_peaks_by_window: dict[int, list[PeakRow]],
    v1_peaks_by_window: dict[int, list[PeakRow]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    merged_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, str]] = []

    max_quality_window = max(quality_windows) if quality_windows else -1
    summary = {
        "total_windows": len(quality_windows),
        "valid_windows": 0,
        "invalid_windows": 0,
        "selected_window_counts": {v5_lead: 0, v1_lead: 0},
        "selected_peak_counts": {v5_lead: 0, v1_lead: 0},
        "discarded_peak_counts": {v5_lead: 0, v1_lead: 0},
        "tail_no_quality_peak_counts": {
            v5_lead: sum(len(peaks) for idx, peaks in v5_peaks_by_window.items() if idx > max_quality_window),
            v1_lead: sum(len(peaks) for idx, peaks in v1_peaks_by_window.items() if idx > max_quality_window),
        },
    }

    for window_index in sorted(quality_windows):
        lead_map = quality_windows[window_index]
        v5_window = lead_map[v5_lead]
        v1_window = lead_map[v1_lead]
        selected_lead, selection_reason, selected_quality_label, selected_quality_score = choose_window_lead(
            v5_window=v5_window,
            v1_window=v1_window,
        )

        v5_peaks = v5_peaks_by_window.get(window_index, [])
        v1_peaks = v1_peaks_by_window.get(window_index, [])

        if selected_lead is None:
            audit_rows.append(
                {
                    "window_index": str(window_index),
                    "window_start": v5_window.window_start,
                    "window_end": v5_window.window_end,
                    "window_status": "invalid",
                    "selection_reason": selection_reason,
                    "selected_lead": "",
                    "selected_alias": "",
                    "selected_quality_label": "",
                    "selected_quality_score": "",
                    "v5_lead": v5_lead,
                    "v5_quality_label": v5_window.quality_label,
                    "v5_quality_score": str(v5_window.quality_score),
                    "v1_lead": v1_lead,
                    "v1_quality_label": v1_window.quality_label,
                    "v1_quality_score": str(v1_window.quality_score),
                    "v5_peak_count": str(len(v5_peaks)),
                    "v1_peak_count": str(len(v1_peaks)),
                    "selected_peak_count": "0",
                    "discarded_peak_count": str(len(v5_peaks) + len(v1_peaks)),
                }
            )
            summary["invalid_windows"] += 1
            summary["discarded_peak_counts"][v5_lead] += len(v5_peaks)
            summary["discarded_peak_counts"][v1_lead] += len(v1_peaks)
            continue

        selected_alias = "V5" if selected_lead == v5_lead else "V1"
        selected_peaks = v5_peaks if selected_lead == v5_lead else v1_peaks
        discarded_peaks = v1_peaks if selected_lead == v5_lead else v5_peaks
        discarded_lead = v1_lead if selected_lead == v5_lead else v5_lead

        for peak in selected_peaks:
            merged_rows.append(
                {
                    "beat_order": "0",
                    "merged_sample_index": peak.sample_index,
                    "merged_milliseconds": peak.milliseconds_text,
                    "merged_timestamp": peak.timestamp,
                    "source": f"{selected_lead}-selected",
                    "primary_lead": selected_lead,
                    "primary_alias": selected_alias,
                    "window_index": str(window_index),
                    "window_start": v5_window.window_start,
                    "window_end": v5_window.window_end,
                    "selection_reason": selection_reason,
                    "selected_quality_label": selected_quality_label or "",
                    "selected_quality_score": str(selected_quality_score),
                }
            )

        audit_rows.append(
            {
                "window_index": str(window_index),
                "window_start": v5_window.window_start,
                "window_end": v5_window.window_end,
                "window_status": "valid",
                "selection_reason": selection_reason,
                "selected_lead": selected_lead,
                "selected_alias": selected_alias,
                "selected_quality_label": selected_quality_label or "",
                "selected_quality_score": str(selected_quality_score),
                "v5_lead": v5_lead,
                "v5_quality_label": v5_window.quality_label,
                "v5_quality_score": str(v5_window.quality_score),
                "v1_lead": v1_lead,
                "v1_quality_label": v1_window.quality_label,
                "v1_quality_score": str(v1_window.quality_score),
                "v5_peak_count": str(len(v5_peaks)),
                "v1_peak_count": str(len(v1_peaks)),
                "selected_peak_count": str(len(selected_peaks)),
                "discarded_peak_count": str(len(discarded_peaks)),
            }
        )
        summary["valid_windows"] += 1
        summary["selected_window_counts"][selected_lead] += 1
        summary["selected_peak_counts"][selected_lead] += len(selected_peaks)
        summary["discarded_peak_counts"][discarded_lead] += len(discarded_peaks)

    merged_rows.sort(key=lambda row: (float(row["merged_milliseconds"]), row["primary_lead"]))
    for beat_order, row in enumerate(merged_rows, start=1):
        row["beat_order"] = str(beat_order)

    summary["selected_peak_counts"]["total"] = len(merged_rows)
    summary["discarded_peak_counts"]["total"] = summary["discarded_peak_counts"][v5_lead] + summary["discarded_peak_counts"][v1_lead]
    summary["tail_no_quality_peak_counts"]["total"] = summary["tail_no_quality_peak_counts"][v5_lead] + summary["tail_no_quality_peak_counts"][v1_lead]
    return merged_rows, audit_rows, summary


def validate_rows(
    merged_rows: list[dict[str, str]],
    audit_rows: list[dict[str, str]],
    summary: dict[str, object],
    v5_lead: str,
    v1_lead: str,
) -> None:
    for expected_order, row in enumerate(merged_rows, start=1):
        if int(row["beat_order"]) != expected_order:
            raise SystemExit("Merged beat_order is not contiguous")

    previous_ms = None
    for row in merged_rows:
        current_ms = float(row["merged_milliseconds"])
        if previous_ms is not None and current_ms < previous_ms:
            raise SystemExit("Merged rows are not sorted by merged_milliseconds")
        previous_ms = current_ms

    if summary["total_windows"] != len(audit_rows):
        raise SystemExit("Audit row count does not match quality window count")

    valid_windows = sum(1 for row in audit_rows if row["window_status"] == "valid")
    invalid_windows = sum(1 for row in audit_rows if row["window_status"] == "invalid")
    if valid_windows != summary["valid_windows"] or invalid_windows != summary["invalid_windows"]:
        raise SystemExit("Summary window counts do not match audit rows")

    selected_peak_total = sum(int(row["selected_peak_count"]) for row in audit_rows)
    if selected_peak_total != len(merged_rows):
        raise SystemExit("Audit selected_peak_count total does not match merged row count")

    selected_windows = summary["selected_window_counts"]
    if selected_windows[v5_lead] + selected_windows[v1_lead] != summary["valid_windows"]:
        raise SystemExit("Selected window counts do not add up to valid windows")


def main() -> None:
    args = parse_args()
    quality_csv = Path(args.quality_csv).resolve()
    ch1_csv = Path(args.ch1_csv).resolve()
    ch3_csv = Path(args.ch3_csv).resolve()
    if not quality_csv.exists():
        raise SystemExit(f"Quality CSV not found: {quality_csv}")
    if not ch1_csv.exists():
        raise SystemExit(f"CH1 CSV not found: {ch1_csv}")
    if not ch3_csv.exists():
        raise SystemExit(f"CH3 CSV not found: {ch3_csv}")

    out_merged, out_audit, out_summary = output_paths(quality_csv, args.case_tag, args.out_merged, args.out_audit, args.out_summary)
    window_ms = args.window_seconds * 1000

    quality_windows = load_quality_windows(quality_csv, v5_lead=args.v5_lead, v1_lead=args.v1_lead)
    v5_peaks_by_window = load_peaks(ch1_csv, window_ms=window_ms)
    v1_peaks_by_window = load_peaks(ch3_csv, window_ms=window_ms)
    merged_rows, audit_rows, summary = merge_by_quality_window(
        quality_windows=quality_windows,
        v5_lead=args.v5_lead,
        v1_lead=args.v1_lead,
        v5_peaks_by_window=v5_peaks_by_window,
        v1_peaks_by_window=v1_peaks_by_window,
    )
    validate_rows(merged_rows=merged_rows, audit_rows=audit_rows, summary=summary, v5_lead=args.v5_lead, v1_lead=args.v1_lead)

    write_csv(out_merged, MERGED_FIELDS, merged_rows)
    write_csv(out_audit, AUDIT_FIELDS, audit_rows)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps({"quality_score_mapping": QUALITY_SCORES, "window_seconds": args.window_seconds, "v5_lead": args.v5_lead, "v1_lead": args.v1_lead, **summary}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Quality CSV: {quality_csv}")
    print(f"V5 lead: {args.v5_lead}")
    print(f"V1 lead: {args.v1_lead}")
    print(f"Valid windows: {summary['valid_windows']}")
    print(f"Invalid windows: {summary['invalid_windows']}")
    print(f"Selected windows ({args.v5_lead}): {summary['selected_window_counts'][args.v5_lead]}")
    print(f"Selected windows ({args.v1_lead}): {summary['selected_window_counts'][args.v1_lead]}")
    print(f"Merged beats total: {summary['selected_peak_counts']['total']}")
    print(f"Tail peaks without quality: {args.v5_lead}={summary['tail_no_quality_peak_counts'][args.v5_lead]}, {args.v1_lead}={summary['tail_no_quality_peak_counts'][args.v1_lead]}")
    print(f"Merged CSV: {out_merged}")
    print(f"Audit CSV: {out_audit}")
    print(f"Summary JSON: {out_summary}")


if __name__ == "__main__":
    main()
