from __future__ import annotations

import argparse
import csv
import json
from contextlib import suppress
from datetime import timedelta
from pathlib import Path

import numpy as np
import neurokit2 as nk

SCRIPT_DIR = Path(__file__).resolve().parent

from edf_reader import EdfReader


QUALITY_SCORES = {
    "Excellent": 2,
    "Barely acceptable": 1,
    "Unacceptable": 0,
}

CSV_FIELDS = [
    "lead",
    "window_index",
    "window_start",
    "window_end",
    "duration_seconds",
    "samples",
    "sampling_rate_hz",
    "quality_mean",
    "quality_median",
    "quality_min",
    "quality_max",
    "zhao2018",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run nk.ecg_quality() on fixed-size EDF windows.")
    parser.add_argument("--edf", required=True, help="Path to EDF file.")
    parser.add_argument("--window-seconds", type=int, default=15, help="Window length in seconds.")
    parser.add_argument("--leads", nargs="+", default=["ECG_CH1", "ECG_CH2", "ECG_CH3"], help="ECG leads.")
    parser.add_argument("--case-tag", default="", help="Optional case tag used for output folder names.")
    parser.add_argument("--out-csv", default="", help="Combined output CSV override.")
    parser.add_argument("--per-lead-dir", default="", help="Per-lead CSV output directory override.")
    return parser.parse_args()


def case_tag_from_edf(edf_path: Path) -> str:
    if "_" in edf_path.stem:
        return edf_path.stem.split("_", 1)[1]
    return edf_path.stem


def output_paths(edf_path: Path, case_tag: str, window_seconds: int, out_csv: str, per_lead_dir: str) -> tuple[Path, Path]:
    tag = case_tag or case_tag_from_edf(edf_path)
    out_root = SCRIPT_DIR / "out" / tag / "quality"
    default_combined = out_root / f"{tag}_ecg_quality_{window_seconds}s.csv"
    default_per_lead_dir = out_root / f"per_lead_window_{window_seconds}s"
    return (
        Path(out_csv).resolve() if out_csv else default_combined,
        Path(per_lead_dir).resolve() if per_lead_dir else default_per_lead_dir,
    )


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def iso_seconds(dt) -> str:
    return dt.isoformat(sep=" ", timespec="seconds")


def format_quality(value: float | None) -> str:
    if value is None or np.isnan(value):
        return ""
    return f"{value:.6f}"


def compute_lead_rows(reader: EdfReader, lead: str, window_seconds: int, num_windows: int) -> list[dict[str, str]]:
    window_ms = window_seconds * 1000
    end_ms = num_windows * window_ms
    full_window = reader.read_window(start_ms=0, end_ms=end_ms, labels=[lead])[0]
    sampling_rate_hz = float(full_window.fs_hz)
    rows: list[dict[str, str]] = []

    for window_index in range(num_windows):
        start_ms = window_index * window_ms
        end_ms = (window_index + 1) * window_ms
        start_sample = int(round(start_ms * sampling_rate_hz / 1000.0))
        end_sample = int(round(end_ms * sampling_rate_hz / 1000.0))
        segment = full_window.values[start_sample:end_sample]
        window_start = reader.header.start_time + timedelta(milliseconds=start_ms)
        window_end = reader.header.start_time + timedelta(milliseconds=end_ms)

        row = {
            "lead": lead,
            "window_index": str(window_index),
            "window_start": iso_seconds(window_start),
            "window_end": iso_seconds(window_end),
            "duration_seconds": str(window_seconds),
            "samples": str(int(segment.size)),
            "sampling_rate_hz": str(int(round(sampling_rate_hz))),
            "quality_mean": "",
            "quality_median": "",
            "quality_min": "",
            "quality_max": "",
            "zhao2018": "",
            "error": "",
        }

        try:
            cleaned = nk.ecg_clean(segment, sampling_rate=sampling_rate_hz)
            quality = nk.ecg_quality(cleaned, sampling_rate=sampling_rate_hz)
            zhao2018 = nk.ecg_quality(cleaned, sampling_rate=sampling_rate_hz, method="zhao2018")
            row["quality_mean"] = format_quality(float(np.mean(quality)))
            row["quality_median"] = format_quality(float(np.median(quality)))
            row["quality_min"] = format_quality(float(np.min(quality)))
            row["quality_max"] = format_quality(float(np.max(quality)))
            row["zhao2018"] = str(zhao2018)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"

        rows.append(row)

        if window_index == 0 or (window_index + 1) % 500 == 0 or window_index == num_windows - 1:
            with suppress(Exception):
                print(f"{lead}: {window_index + 1}/{num_windows} windows processed", flush=True)

    return rows


def main() -> None:
    args = parse_args()
    edf_path = Path(args.edf).resolve()
    if not edf_path.exists():
        raise SystemExit(f"EDF not found: {edf_path}")

    reader = EdfReader(edf_path)
    num_windows = int(reader.header.duration_s // args.window_seconds)
    if num_windows <= 0:
        raise SystemExit("No full windows fit inside the EDF duration.")

    out_csv, per_lead_dir = output_paths(edf_path, args.case_tag, args.window_seconds, args.out_csv, args.per_lead_dir)
    per_lead_dir.mkdir(parents=True, exist_ok=True)

    combined_rows: list[dict[str, str]] = []
    for lead in args.leads:
        print(f"Processing {lead} with {args.window_seconds}s windows...", flush=True)
        lead_rows = compute_lead_rows(reader=reader, lead=lead, window_seconds=args.window_seconds, num_windows=num_windows)
        combined_rows.extend(lead_rows)
        lead_path = per_lead_dir / f"{case_tag_from_edf(edf_path)}_{lead}_ecg_quality_{args.window_seconds}s.csv"
        write_csv(lead_path, lead_rows)
        print(f"Per-lead CSV: {lead_path}", flush=True)

    combined_rows.sort(key=lambda row: (int(row["window_index"]), row["lead"]))
    write_csv(out_csv, combined_rows)

    error_count = sum(1 for row in combined_rows if row["error"])
    print(f"EDF: {edf_path}")
    print(f"Window seconds: {args.window_seconds}")
    print(f"Leads: {', '.join(args.leads)}")
    print(f"Windows per lead: {num_windows}")
    print(f"Total rows: {len(combined_rows)}")
    print(f"Rows with errors: {error_count}")
    print(f"Combined CSV: {out_csv}")
    print(f"Per-lead dir: {per_lead_dir}")


if __name__ == "__main__":
    main()
