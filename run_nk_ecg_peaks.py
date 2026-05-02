from __future__ import annotations

import argparse
import csv
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import neurokit2 as nk

SCRIPT_DIR = Path(__file__).resolve().parent

from edf_reader import EdfReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NeuroKit2 R-peak detection on one ECG lead from an EDF file.")
    parser.add_argument("--edf", required=True, help="Path to EDF file.")
    parser.add_argument("--lead", default="", help="Legacy single ECG lead label, e.g. ECG_CH1.")
    parser.add_argument("--leads", nargs="+", default=[], help="One or more ECG lead labels, e.g. ECG_CH1 ECG_CH3.")
    parser.add_argument("--case-tag", default="", help="Optional case tag used for output folder names.")
    parser.add_argument("--out-dir", default="", help="Output directory override.")
    parser.add_argument("--chunk-seconds", type=int, default=300, help="Processing chunk size in seconds.")
    parser.add_argument("--overlap-seconds", type=int, default=2, help="Chunk overlap in seconds.")
    parser.add_argument("--method", default="neurokit", help="NeuroKit2 ecg_peaks method.")
    parser.add_argument("--correct-artifacts", action="store_true", help="Enable artifact correction in ecg_peaks().")
    return parser.parse_args()


def case_tag_from_edf(edf_path: Path) -> str:
    if "_" in edf_path.stem:
        return edf_path.stem.split("_", 1)[1]
    return edf_path.stem


def output_dir_for(edf_path: Path, case_tag: str, out_dir: str) -> Path:
    if out_dir:
        return Path(out_dir).resolve()
    tag = case_tag or case_tag_from_edf(edf_path)
    return SCRIPT_DIR / "out" / tag / "rpeaks"


def sample_from_ms(ms: int, fs_hz: float) -> int:
    return int(round(ms * fs_hz / 1000.0))


def detect_rpeaks_by_chunk(
    reader: EdfReader,
    lead: str,
    chunk_seconds: int,
    overlap_seconds: int,
    method: str,
    correct_artifacts: bool,
) -> tuple[np.ndarray, list[dict[str, str | int]]]:
    duration_ms = int(reader.header.duration_s * 1000)
    overlap_ms = max(0, overlap_seconds * 1000)
    step_ms = max(1, chunk_seconds * 1000)
    fs_hz = reader.header.signal(lead).samples_per_record / reader.header.duration_per_record_s

    collected_peaks: list[int] = []
    errors: list[dict[str, str | int]] = []

    for chunk_index, chunk_start_ms in enumerate(range(0, duration_ms, step_ms)):
        chunk_end_ms = min(duration_ms, chunk_start_ms + step_ms)
        read_start_ms = max(0, chunk_start_ms - overlap_ms)
        read_end_ms = min(duration_ms, chunk_end_ms + overlap_ms)

        window = reader.read_window(read_start_ms, read_end_ms, labels=[lead])[0]
        signal = np.asarray(window.values, dtype=float)

        if signal.size < max(10, int(fs_hz * 2)):
            errors.append(
                {
                    "chunk_index": chunk_index,
                    "chunk_start_ms": chunk_start_ms,
                    "chunk_end_ms": chunk_end_ms,
                    "error": "window_too_short",
                }
            )
            continue

        try:
            cleaned = nk.ecg_clean(signal, sampling_rate=fs_hz)
            _, info = nk.ecg_peaks(
                cleaned,
                sampling_rate=fs_hz,
                method=method,
                correct_artifacts=correct_artifacts,
            )
            local_peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        except Exception as exc:  # pragma: no cover
            errors.append(
                {
                    "chunk_index": chunk_index,
                    "chunk_start_ms": chunk_start_ms,
                    "chunk_end_ms": chunk_end_ms,
                    "error": str(exc),
                }
            )
            continue

        global_offset = sample_from_ms(read_start_ms, fs_hz)
        global_peaks = local_peaks + global_offset
        core_start_sample = sample_from_ms(chunk_start_ms, fs_hz)
        core_end_sample = sample_from_ms(chunk_end_ms, fs_hz)
        core_peaks = global_peaks[(global_peaks >= core_start_sample) & (global_peaks < core_end_sample)]
        collected_peaks.extend(int(value) for value in core_peaks)

    unique_peaks = np.asarray(sorted(set(collected_peaks)), dtype=int)
    return unique_peaks, errors


def build_rows(reader: EdfReader, lead: str, peaks: np.ndarray) -> list[dict[str, str | int | float]]:
    fs_hz = reader.header.signal(lead).samples_per_record / reader.header.duration_per_record_s
    start_time = reader.header.start_time

    rows: list[dict[str, str | int | float]] = []
    for order, sample_index in enumerate(peaks, start=1):
        milliseconds = sample_index / fs_hz * 1000.0
        rows.append(
            {
                "peak_order": order,
                "sample_index": int(sample_index),
                "milliseconds": round(float(milliseconds), 3),
                "timestamp": (start_time + timedelta(milliseconds=float(milliseconds))).isoformat(sep=" "),
            }
        )
    return rows


def build_summary(
    reader: EdfReader,
    lead: str,
    peaks: np.ndarray,
    errors: list[dict[str, str | int]],
    args: argparse.Namespace,
) -> dict[str, str | int | float | bool]:
    fs_hz = reader.header.signal(lead).samples_per_record / reader.header.duration_per_record_s
    rr_ms = np.diff(peaks) / fs_hz * 1000.0 if peaks.size > 1 else np.array([], dtype=float)
    duration_s = float(reader.header.duration_s)

    return {
        "edf_path": str(Path(args.edf).resolve()),
        "lead": lead,
        "start_time": reader.header.start_time.isoformat(sep=" "),
        "duration_s": round(duration_s, 3),
        "sampling_rate_hz": round(fs_hz, 4),
        "chunk_seconds": args.chunk_seconds,
        "overlap_seconds": args.overlap_seconds,
        "method": args.method,
        "correct_artifacts": bool(args.correct_artifacts),
        "rpeak_count": int(peaks.size),
        "mean_hr_bpm": round(float(peaks.size * 60.0 / duration_s), 3) if duration_s else 0.0,
        "rr_mean_ms": round(float(rr_ms.mean()), 3) if rr_ms.size else None,
        "rr_median_ms": round(float(np.median(rr_ms)), 3) if rr_ms.size else None,
        "rr_min_ms": round(float(rr_ms.min()), 3) if rr_ms.size else None,
        "rr_max_ms": round(float(rr_ms.max()), 3) if rr_ms.size else None,
        "error_chunk_count": int(len(errors)),
    }


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    fieldnames = ["peak_order", "sample_index", "milliseconds", "timestamp"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_error_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    fieldnames = ["chunk_index", "chunk_start_ms", "chunk_end_ms", "error"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    edf_path = Path(args.edf).resolve()
    if not edf_path.exists():
        raise SystemExit(f"EDF not found: {edf_path}")

    leads = args.leads or ([args.lead] if args.lead else ["ECG_CH1"])

    reader = EdfReader(edf_path)

    out_dir = output_dir_for(edf_path, args.case_tag, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"EDF: {edf_path}")
    print(f"Leads: {', '.join(leads)}")

    for lead in leads:
        signal = reader.header.signal(lead)
        fs_hz = signal.samples_per_record / reader.header.duration_per_record_s

        peaks, errors = detect_rpeaks_by_chunk(
            reader=reader,
            lead=lead,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
            method=args.method,
            correct_artifacts=args.correct_artifacts,
        )
        rows = build_rows(reader, lead, peaks)
        summary = build_summary(reader, lead, peaks, errors, args)

        peaks_csv = out_dir / f"{lead}_rpeaks.csv"
        summary_json = out_dir / f"{lead}_summary.json"
        write_csv(peaks_csv, rows)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        if errors:
            error_csv = out_dir / f"{lead}_chunk_errors.csv"
            write_error_csv(error_csv, errors)
            print(f"Chunk errors CSV: {error_csv}")

        print(f"Lead: {lead}")
        print(f"Sampling rate: {fs_hz:.4f} Hz")
        print(f"R peaks: {len(rows)}")
        print(f"Output CSV: {peaks_csv}")
        print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
