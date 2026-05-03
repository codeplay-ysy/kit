from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


WINDOW_SECONDS = 30
STEP_SECONDS = 5
MIN_VALID_RR = 20
RR_MIN = 0.5
RR_MAX = 1.8
BIN_WIDTH = 0.05
AF_OCCUPIED_THRESHOLD = 0.06
AF_MAX_BIN_THRESHOLD = 0.20
AF_POSSIBLE_OCCUPIED_THRESHOLD = 0.045
AF_POSSIBLE_MAX_BIN_THRESHOLD = 0.25
AF_POSSIBLE_ATTACH_GAP_MS = 30_000
AF_BRIDGE_GAP_MS = 30_000
AF_FINAL_MIN_DURATION_MS = 30_000
AF_MIN_STRONG_WINDOWS_PER_EVENT = 1
ECTOPY_OCCUPIED_THRESHOLD = 0.04
ECTOPY_MAX_BIN_THRESHOLD = 0.30


@dataclass(slots=True)
class BeatSeries:
    offsets_ms: np.ndarray
    rr_ms: np.ndarray
    source_csv: Path


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _pick_field(field_map: dict[str, str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        key = _normalize_name(candidate)
        if key in field_map:
            return field_map[key]
    return None


def _parse_float(text: str) -> float | None:
    value = (text or "").strip()
    if not value or value == "-":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int_like(text: str) -> int | None:
    value = _parse_float(text)
    if value is None:
        return None
    return int(round(value))


def load_beat_series(csv_path: Path) -> BeatSeries:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        field_map = {_normalize_name(name): name for name in reader.fieldnames}

        offset_key = _pick_field(
            field_map,
            [
                "merged_milliseconds",
                "time offset(ms)",
                "offset_ms",
                "timestamp_ms",
                "ms",
            ],
        )
        rr_key = _pick_field(
            field_map,
            [
                "rr interval(ms)",
                "rr_ms",
                "rr",
                "rr_raw",
            ],
        )
        if offset_key is None and rr_key is None:
            raise ValueError(f"CSV must contain a time column or RR column: {csv_path}")

        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no rows: {csv_path}")

    offsets: list[int] = []
    rr_values: list[int] = []
    for index, row in enumerate(rows):
        offset_value = _parse_int_like(row[offset_key]) if offset_key else None
        rr_value = _parse_int_like(row[rr_key]) if rr_key else None

        if offset_value is None:
            if index == 0:
                offset_value = 0
            else:
                raise ValueError(
                    f"Missing time offset for row {index + 1} in {csv_path}; "
                    "this script needs a time-aligned beat CSV."
                )
        offsets.append(offset_value)

        if rr_value is None:
            if index > 0:
                rr_value = offsets[index] - offsets[index - 1]
            elif len(rows) > 1:
                next_offset = _parse_int_like(rows[index + 1].get(offset_key, "")) if offset_key else None
                rr_value = next_offset - offset_value if next_offset is not None else 0
            else:
                rr_value = 0
        rr_values.append(rr_value)

    return BeatSeries(
        offsets_ms=np.asarray(offsets, dtype=np.int64),
        rr_ms=np.asarray(rr_values, dtype=np.int64),
        source_csv=csv_path,
    )


def make_windows(duration_ms: int, window_ms: int, step_ms: int) -> list[tuple[int, int]]:
    if duration_ms <= 0:
        return []
    last_start = max(duration_ms - window_ms, 0)
    return [(start_ms, start_ms + window_ms) for start_ms in range(0, last_start + 1, step_ms)]


def rr_2d_bin_count() -> int:
    return int(round((RR_MAX - RR_MIN) / BIN_WIDTH))


def evaluate_rr_2d_filter(rr_ms: np.ndarray | list[int] | list[float]) -> dict[str, Any]:
    values = np.asarray(rr_ms, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]

    bins_per_axis = rr_2d_bin_count()
    total_bins = bins_per_axis**2

    result: dict[str, Any] = {
        "label": "non_af",
        "valid_rr_count": int(values.size),
        "point_count": 0,
        "occupied_bins": 0,
        "total_bins": total_bins,
        "occupied_ratio": 0.0,
        "max_bin_ratio": 0.0,
        "median_rr_ms": None,
    }

    if values.size < MIN_VALID_RR:
        return result

    median_rr = float(np.median(values))
    if not np.isfinite(median_rr) or median_rr <= 0:
        return result

    rr_norm = values / median_rr
    points = np.column_stack((rr_norm[:-1], rr_norm[1:]))
    point_count = int(points.shape[0])
    result["point_count"] = point_count
    result["median_rr_ms"] = round(median_rr, 3)

    if point_count == 0:
        return result

    inside_mask = (
        (points[:, 0] >= RR_MIN)
        & (points[:, 0] <= RR_MAX)
        & (points[:, 1] >= RR_MIN)
        & (points[:, 1] <= RR_MAX)
    )
    inside_points = points[inside_mask]

    counts = np.zeros((bins_per_axis, bins_per_axis), dtype=np.int32)
    if inside_points.size:
        x_idx = np.floor((inside_points[:, 0] - RR_MIN) / BIN_WIDTH).astype(np.int32)
        y_idx = np.floor((inside_points[:, 1] - RR_MIN) / BIN_WIDTH).astype(np.int32)
        x_idx = np.clip(x_idx, 0, bins_per_axis - 1)
        y_idx = np.clip(y_idx, 0, bins_per_axis - 1)
        np.add.at(counts, (x_idx, y_idx), 1)

    occupied_bins = int(np.count_nonzero(counts))
    max_bin_count = int(counts.max()) if occupied_bins else 0
    occupied_ratio = occupied_bins / float(total_bins)
    max_bin_ratio = max_bin_count / float(point_count)

    result.update(
        {
            "occupied_bins": occupied_bins,
            "occupied_ratio": round(occupied_ratio, 6),
            "max_bin_ratio": round(max_bin_ratio, 6),
        }
    )

    if occupied_ratio > AF_OCCUPIED_THRESHOLD and max_bin_ratio < AF_MAX_BIN_THRESHOLD:
        result["label"] = "strong_af"
    elif occupied_ratio > AF_POSSIBLE_OCCUPIED_THRESHOLD and max_bin_ratio < AF_POSSIBLE_MAX_BIN_THRESHOLD:
        result["label"] = "possible_af"
    elif occupied_ratio < ECTOPY_OCCUPIED_THRESHOLD or max_bin_ratio > ECTOPY_MAX_BIN_THRESHOLD:
        result["label"] = "non_af"

    return result


def _window_distance_ms(left: dict[str, Any], right: dict[str, Any]) -> int:
    if int(left["end_ms"]) < int(right["start_ms"]):
        return int(right["start_ms"]) - int(left["end_ms"])
    if int(right["end_ms"]) < int(left["start_ms"]):
        return int(left["start_ms"]) - int(right["end_ms"])
    return 0


def _is_possible_near_strong(window: dict[str, Any], strong_windows: list[dict[str, Any]], attach_gap_ms: int) -> bool:
    return any(_window_distance_ms(window, strong_window) <= attach_gap_ms for strong_window in strong_windows)


def _enhanced_af_segments(
    windows: list[dict[str, Any]],
    bridge_gap_ms: int,
    final_min_duration_ms: int,
    min_strong_windows: int,
) -> list[dict[str, Any]]:
    candidates = [window for window in windows if window.get("enhanced_label") == "enhanced_af"]
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda item: (int(item["start_ms"]), int(item["end_ms"])))
    segments: list[dict[str, Any]] = []
    current = {
        "start_ms": int(ordered[0]["start_ms"]),
        "end_ms": int(ordered[0]["end_ms"]),
        "strong_windows": 1 if ordered[0]["label"] == "strong_af" else 0,
        "possible_windows": 1 if ordered[0]["label"] == "possible_af" else 0,
        "window_count": 1,
    }

    for window in ordered[1:]:
        start_ms = int(window["start_ms"])
        end_ms = int(window["end_ms"])
        if start_ms - int(current["end_ms"]) <= bridge_gap_ms:
            current["end_ms"] = max(int(current["end_ms"]), end_ms)
            current["strong_windows"] = int(current["strong_windows"]) + (1 if window["label"] == "strong_af" else 0)
            current["possible_windows"] = int(current["possible_windows"]) + (1 if window["label"] == "possible_af" else 0)
            current["window_count"] = int(current["window_count"]) + 1
        else:
            segments.append(current)
            current = {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "strong_windows": 1 if window["label"] == "strong_af" else 0,
                "possible_windows": 1 if window["label"] == "possible_af" else 0,
                "window_count": 1,
            }
    segments.append(current)

    return [
        segment
        for segment in segments
        if int(segment["end_ms"]) - int(segment["start_ms"]) >= final_min_duration_ms
        and int(segment["strong_windows"]) >= min_strong_windows
    ]


def apply_enhanced_af_merge(
    rows: list[dict[str, Any]],
    attach_gap_ms: int = AF_POSSIBLE_ATTACH_GAP_MS,
    bridge_gap_ms: int = AF_BRIDGE_GAP_MS,
    final_min_duration_ms: int = AF_FINAL_MIN_DURATION_MS,
    min_strong_windows: int = AF_MIN_STRONG_WINDOWS_PER_EVENT,
) -> list[dict[str, Any]]:
    strong_windows = [row for row in rows if row["label"] == "strong_af"]
    for row in rows:
        if row["label"] == "strong_af":
            row["enhanced_label"] = "enhanced_af"
            row["enhanced_reason"] = "strong_seed"
        elif row["label"] == "possible_af" and _is_possible_near_strong(row, strong_windows, attach_gap_ms):
            row["enhanced_label"] = "enhanced_af"
            row["enhanced_reason"] = f"possible_near_strong_{attach_gap_ms}ms"
        else:
            row["enhanced_label"] = "non_af"
            row["enhanced_reason"] = "possible_without_nearby_strong" if row["label"] == "possible_af" else "non_af"

    segments = _enhanced_af_segments(
        rows,
        bridge_gap_ms=bridge_gap_ms,
        final_min_duration_ms=final_min_duration_ms,
        min_strong_windows=min_strong_windows,
    )
    for row in rows:
        if row["enhanced_label"] != "enhanced_af":
            continue
        in_final_segment = any(
            int(row["start_ms"]) < int(segment["end_ms"]) and int(row["end_ms"]) > int(segment["start_ms"])
            for segment in segments
        )
        if not in_final_segment:
            row["enhanced_label"] = "non_af"
            row["enhanced_reason"] = "filtered_short_or_weak_segment"
    return segments


def slice_window_rr(series: BeatSeries, start_ms: int, end_ms: int) -> np.ndarray:
    mask = (series.offsets_ms >= start_ms) & (series.offsets_ms < end_ms)
    return series.rr_ms[mask]


def classify_series(series: BeatSeries, window_seconds: int, step_seconds: int) -> list[dict[str, Any]]:
    window_ms = window_seconds * 1000
    step_ms = step_seconds * 1000
    duration_ms = int(series.offsets_ms[-1]) + window_ms
    windows = make_windows(duration_ms=duration_ms, window_ms=window_ms, step_ms=step_ms)

    rows: list[dict[str, Any]] = []
    for window_index, (start_ms, end_ms) in enumerate(windows):
        rr_window = slice_window_rr(series, start_ms, end_ms)
        stats = evaluate_rr_2d_filter(rr_window)
        rows.append(
            {
                "window_index": window_index,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_seconds": window_seconds,
                "valid_rr_count": stats["valid_rr_count"],
                "point_count": stats["point_count"],
                "occupied_bins": stats["occupied_bins"],
                "total_bins": stats["total_bins"],
                "occupied_ratio": stats["occupied_ratio"],
                "max_bin_ratio": stats["max_bin_ratio"],
                "median_rr_ms": stats["median_rr_ms"],
                "label": stats["label"],
                "enhanced_label": "non_af",
                "enhanced_reason": "not_evaluated",
            }
        )
    apply_enhanced_af_merge(rows)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "window_index",
        "start_ms",
        "end_ms",
        "duration_seconds",
        "valid_rr_count",
        "point_count",
        "occupied_bins",
        "total_bins",
        "occupied_ratio",
        "max_bin_ratio",
        "median_rr_ms",
        "label",
        "enhanced_label",
        "enhanced_reason",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], series: BeatSeries, window_seconds: int, step_seconds: int) -> dict[str, Any]:
    counts = {}
    enhanced_counts = {}
    for row in rows:
        label = str(row["label"])
        enhanced_label = str(row.get("enhanced_label", "non_af"))
        counts[label] = counts.get(label, 0) + 1
        enhanced_counts[enhanced_label] = enhanced_counts.get(enhanced_label, 0) + 1
    enhanced_segments = apply_enhanced_af_merge([dict(row) for row in rows])
    return {
        "source_csv": str(series.source_csv),
        "window_seconds": window_seconds,
        "step_seconds": step_seconds,
        "window_count": len(rows),
        "label_counts": counts,
        "enhanced_label_counts": enhanced_counts,
        "enhanced_segment_count": len(enhanced_segments),
        "enhanced_segments": enhanced_segments,
        "strong_possible_merge": {
            "possible_attach_gap_ms": AF_POSSIBLE_ATTACH_GAP_MS,
            "bridge_gap_ms": AF_BRIDGE_GAP_MS,
            "final_min_duration_ms": AF_FINAL_MIN_DURATION_MS,
            "min_strong_windows_per_event": AF_MIN_STRONG_WINDOWS_PER_EVENT,
        },
    }


def format_hms(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def enhanced_segments_to_events(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, segment in enumerate(segments, start=1):
        start_ms = int(segment["start_ms"])
        end_ms = int(segment["end_ms"])
        duration_ms = max(0, end_ms - start_ms)
        events.append(
            {
                "type": "af_family",
                "subtype": "fibrillation",
                "layer": "rr_2d_filter",
                "rule": "rr_2d_strong_possible_merge",
                "event_index": index,
                "t0_ms": start_ms,
                "t1_ms": end_ms,
                "time": f"{start_ms} ms ~ {end_ms} ms",
                "duration": format_hms(duration_ms / 1000.0),
                "stats": {
                    "window_count": int(segment["window_count"]),
                    "strong_windows": int(segment["strong_windows"]),
                    "possible_windows": int(segment["possible_windows"]),
                    "possible_attach_gap_ms": AF_POSSIBLE_ATTACH_GAP_MS,
                    "bridge_gap_ms": AF_BRIDGE_GAP_MS,
                    "final_min_duration_ms": AF_FINAL_MIN_DURATION_MS,
                    "min_strong_windows_per_event": AF_MIN_STRONG_WINDOWS_PER_EVENT,
                },
            }
        )
    return events


def write_events_json(path: Path, segments: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    events = enhanced_segments_to_events(segments)
    path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2D RR bin filter for AF candidate screening.")
    parser.add_argument("--csv", required=True, help="Beat CSV with time offsets and RR intervals.")
    parser.add_argument("--out-csv", default="", help="Output CSV path. Default: out/rr_2d_filter/<stem>_rr_2d_windows.csv")
    parser.add_argument("--out-json", default="", help="Output JSON summary path.")
    parser.add_argument("--out-events-json", default="", help="Output enhanced AF events JSON path.")
    parser.add_argument("--window-seconds", type=int, default=WINDOW_SECONDS, help="Sliding window size in seconds.")
    parser.add_argument("--step-seconds", type=int, default=STEP_SECONDS, help="Sliding step in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    series = load_beat_series(csv_path)
    rows = classify_series(series, window_seconds=args.window_seconds, step_seconds=args.step_seconds)

    out_root = Path("out") / "rr_2d_filter"
    out_csv = Path(args.out_csv).resolve() if args.out_csv else out_root / f"{csv_path.stem}_rr_2d_windows.csv"
    out_json = Path(args.out_json).resolve() if args.out_json else out_root / f"{csv_path.stem}_rr_2d_summary.json"
    out_events_json = Path(args.out_events_json).resolve() if args.out_events_json else out_root / f"{csv_path.stem}_rr_2d_events.json"

    write_csv(out_csv, rows)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, series, args.window_seconds, args.step_seconds)
    enhanced_segments = summary["enhanced_segments"]
    out_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_events_json(out_events_json, enhanced_segments)

    print(f"CSV: {csv_path}")
    print(f"Windows: {len(rows)}")
    print(f"Label counts: {summary['label_counts']}")
    print(f"Enhanced label counts: {summary['enhanced_label_counts']}")
    print(f"Enhanced segments: {summary['enhanced_segment_count']}")
    print(f"Output CSV: {out_csv}")
    print(f"Summary JSON: {out_json}")
    print(f"Events JSON: {out_events_json}")


if __name__ == "__main__":
    main()
