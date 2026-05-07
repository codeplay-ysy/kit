from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

AFL_WINDOW_SECONDS = 10 * 60
AFL_STEP_SECONDS = 30
AFL_MIN_VALID_RR = 100
AFL_RR_MIN_MS = 300
AFL_RR_MAX_MS = 2000
AFL_DIFF_MIN_MS = 30
AFL_DIFF_BIN_MS = 30
AFL_TOP_DIFF_PEAKS = 3
AFL_MAX_MULTIPLE = 4
AFL_MULTIPLE_TOLERANCE_MS = 30
AFL_BASE_DIFF_MIN_MS = 40
AFL_BASE_DIFF_MAX_MS = 400
AFL_DIFF_PEAK_COVERAGE_STRONG = 0.50
AFL_DIFF_PEAK_COVERAGE_POSSIBLE = 0.40
AFL_MULTIPLE_RATIO_STRONG = 0.60
AFL_MULTIPLE_RATIO_POSSIBLE = 0.45
AFL_MIN_NONZERO_DIFF_RATIO = 0.15
AFL_MIN_VALID_DIFF = 20
AFL_MEAN_HR_SUPPORT_BPM = 90.0
AFL_STABLE_RR_DOMINANT_RATIO = 0.90
AFL_STABLE_RR_CV = 0.05
AFL_RR_CLUSTER_BIN_MS = 40
AFL_POSSIBLE_ATTACH_GAP_MS = 60_000
AFL_BRIDGE_GAP_MS = 60_000
AFL_FINAL_MIN_DURATION_MS = 60_000
AFL_MIN_STRONG_WINDOWS_PER_EVENT = 1


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
    return None if value is None else int(round(value))


def load_beat_series(csv_path: Path) -> BeatSeries:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        field_map = {_normalize_name(name): name for name in reader.fieldnames}
        offset_key = _pick_field(field_map, ["merged_milliseconds", "time offset(ms)", "offset_ms", "timestamp_ms", "ms"])
        rr_key = _pick_field(field_map, ["rr interval(ms)", "rr_ms", "rr", "rr_raw"])
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
                raise ValueError(f"Missing time offset for row {index + 1} in {csv_path}")
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

    order = np.argsort(np.asarray(offsets, dtype=np.int64))
    return BeatSeries(np.asarray(offsets, dtype=np.int64)[order], np.asarray(rr_values, dtype=np.int64)[order], csv_path)


def make_windows(duration_ms: int, window_ms: int, step_ms: int) -> list[tuple[int, int]]:
    if duration_ms <= 0:
        return []
    last_start = max(duration_ms - window_ms, 0)
    return [(start_ms, start_ms + window_ms) for start_ms in range(0, last_start + 1, step_ms)]


def slice_window_rr(series: BeatSeries, start_ms: int, end_ms: int) -> np.ndarray:
    left = int(np.searchsorted(series.offsets_ms, start_ms, side="left"))
    right = int(np.searchsorted(series.offsets_ms, end_ms, side="left"))
    return series.rr_ms[left:right]


def format_hms(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    return f"{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}:{total_seconds % 60:02d}"


def _valid_rr(rr_ms: np.ndarray | list[int] | list[float]) -> np.ndarray:
    values = np.asarray(rr_ms, dtype=np.float64)
    return values[np.isfinite(values) & (values >= AFL_RR_MIN_MS) & (values <= AFL_RR_MAX_MS)]


def dominant_rr_ratio(rr_ms: np.ndarray) -> float:
    if rr_ms.size == 0:
        return 0.0
    _, counts = np.unique(np.round(rr_ms / AFL_RR_CLUSTER_BIN_MS).astype(np.int32), return_counts=True)
    return float(counts.max() / rr_ms.size) if counts.size else 0.0


def diff_peak_stats(diff_ms: np.ndarray) -> dict[str, Any]:
    if diff_ms.size == 0:
        return {"diff_peak_count": 0, "diff_top_peak_ms": None, "diff_peak_coverage": 0.0, "diff_peak_centers_ms": [], "diff_peak_counts": []}
    bins = np.round(diff_ms / AFL_DIFF_BIN_MS).astype(np.int32)
    unique_bins, counts = np.unique(bins, return_counts=True)
    order = np.argsort(counts)[::-1][:AFL_TOP_DIFF_PEAKS]
    top_bins = unique_bins[order]
    top_counts = counts[order]
    centers = [float(bin_id * AFL_DIFF_BIN_MS) for bin_id in top_bins.tolist()]
    return {
        "diff_peak_count": int(top_counts.size),
        "diff_top_peak_ms": centers[0] if centers else None,
        "diff_peak_coverage": float(np.sum(top_counts) / max(diff_ms.size, 1)),
        "diff_peak_centers_ms": centers,
        "diff_peak_counts": [int(count) for count in top_counts.tolist()],
    }


def multiple_match_count(diff_ms: np.ndarray, base_diff_ms: float) -> int:
    hits = 0
    for value in diff_ms:
        if any(abs(float(value) - multiple * base_diff_ms) <= AFL_MULTIPLE_TOLERANCE_MS for multiple in range(1, AFL_MAX_MULTIPLE + 1)):
            hits += 1
    return hits


def best_multiple_match(diff_ms: np.ndarray, candidates: list[float]) -> tuple[float | None, int, float]:
    best_base: float | None = None
    best_count = 0
    for base_diff_ms in candidates:
        if AFL_BASE_DIFF_MIN_MS <= base_diff_ms <= AFL_BASE_DIFF_MAX_MS:
            count = multiple_match_count(diff_ms, base_diff_ms)
            if count > best_count:
                best_base = float(base_diff_ms)
                best_count = count
    return best_base, best_count, float(best_count / max(diff_ms.size, 1)) if diff_ms.size else 0.0


def evaluate_afl_window(rr_ms: np.ndarray | list[int] | list[float]) -> dict[str, Any]:
    rr = _valid_rr(rr_ms)
    result: dict[str, Any] = {
        "label": "non_afl", "reason": "not_evaluated", "valid_rr_count": int(rr.size), "raw_diff_count": 0,
        "valid_diff_count": 0, "nonzero_diff_ratio": 0.0, "mean_rr_ms": None, "mean_hr": 0.0,
        "rr_cv": 0.0, "dominant_rr_ratio": 0.0, "base_diff_ms": None, "matched_diff_count": 0,
        "multiple_match_ratio": 0.0, "diff_peak_count": 0, "diff_top_peak_ms": None,
        "diff_peak_coverage": 0.0, "diff_peak_centers_ms": [], "diff_peak_counts": [], "afl_score": 0.0,
    }
    if rr.size < AFL_MIN_VALID_RR:
        result["reason"] = "insufficient_rr"
        return result

    mean_rr = float(np.mean(rr))
    rr_cv = float(np.std(rr) / mean_rr) if mean_rr > 0 else 0.0
    raw_diff = np.abs(np.diff(rr))
    valid_diff = raw_diff[raw_diff >= AFL_DIFF_MIN_MS]
    nonzero_ratio = float(valid_diff.size / max(raw_diff.size, 1)) if raw_diff.size else 0.0
    dom_ratio = dominant_rr_ratio(rr)
    result.update({"raw_diff_count": int(raw_diff.size), "valid_diff_count": int(valid_diff.size), "nonzero_diff_ratio": round(nonzero_ratio, 6), "mean_rr_ms": round(mean_rr, 3), "mean_hr": round(60000.0 / mean_rr, 3) if mean_rr > 0 else 0.0, "rr_cv": round(rr_cv, 6), "dominant_rr_ratio": round(dom_ratio, 6)})
    if valid_diff.size < AFL_MIN_VALID_DIFF:
        result["reason"] = "insufficient_nonzero_diff"
        return result

    peaks = diff_peak_stats(valid_diff)
    base_diff, match_count, match_ratio = best_multiple_match(valid_diff, [float(item) for item in peaks["diff_peak_centers_ms"]])
    score = 0.0
    reasons: list[str] = []
    peak_coverage = float(peaks["diff_peak_coverage"])
    if peak_coverage >= AFL_DIFF_PEAK_COVERAGE_STRONG:
        score += 1.0; reasons.append("strong_diff_peak_coverage")
    elif peak_coverage >= AFL_DIFF_PEAK_COVERAGE_POSSIBLE:
        score += 0.5; reasons.append("possible_diff_peak_coverage")
    if match_ratio >= AFL_MULTIPLE_RATIO_STRONG:
        score += 1.0; reasons.append("strong_multiple_match")
    elif match_ratio >= AFL_MULTIPLE_RATIO_POSSIBLE:
        score += 0.5; reasons.append("possible_multiple_match")
    if base_diff is not None:
        score += 1.0; reasons.append("base_diff_in_range")
    if nonzero_ratio >= AFL_MIN_NONZERO_DIFF_RATIO:
        score += 1.0; reasons.append("enough_nonzero_diff")
    if float(result["mean_hr"]) >= AFL_MEAN_HR_SUPPORT_BPM:
        score += 0.5; reasons.append("heart_rate_support")
    stable_rr = dom_ratio > AFL_STABLE_RR_DOMINANT_RATIO and rr_cv < AFL_STABLE_RR_CV
    if stable_rr:
        score -= 1.0; reasons.append("stable_rr_penalty")
    label = "strong_afl" if (not stable_rr and score >= 3.0) else "possible_afl" if (not stable_rr and score >= 2.0) else "non_afl"
    result.update({**peaks, "base_diff_ms": None if base_diff is None else round(base_diff, 3), "matched_diff_count": int(match_count), "multiple_match_ratio": round(match_ratio, 6), "afl_score": round(score, 3), "label": label, "reason": ";".join(reasons) if reasons else "criteria_not_met"})
    return result

def classify_series(series: BeatSeries, window_seconds: int, step_seconds: int) -> list[dict[str, Any]]:
    window_ms = window_seconds * 1000
    step_ms = step_seconds * 1000
    duration_ms = int(series.offsets_ms[-1]) + window_ms if series.offsets_ms.size else 0
    rows: list[dict[str, Any]] = []
    for window_index, (start_ms, end_ms) in enumerate(make_windows(duration_ms, window_ms, step_ms)):
        rows.append({"window_index": window_index, "start_ms": start_ms, "end_ms": end_ms, "duration_seconds": window_seconds, **evaluate_afl_window(slice_window_rr(series, start_ms, end_ms)), "candidate_afl": False, "candidate_reason": "not_evaluated", "final_label": "non_afl", "final_reason": "not_evaluated", "final_event_index": ""})
    apply_afl_merge(rows)
    return rows


def _window_distance_ms(left: dict[str, Any], right: dict[str, Any]) -> int:
    if int(left["end_ms"]) < int(right["start_ms"]):
        return int(right["start_ms"]) - int(left["end_ms"])
    if int(right["end_ms"]) < int(left["start_ms"]):
        return int(left["start_ms"]) - int(right["end_ms"])
    return 0


def apply_afl_merge(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strong = [row for row in rows if row["label"] == "strong_afl"]
    for row in rows:
        if row["label"] == "strong_afl":
            row.update(candidate_afl=True, candidate_reason="strong_seed")
        elif row["label"] == "possible_afl" and any(_window_distance_ms(row, item) <= AFL_POSSIBLE_ATTACH_GAP_MS for item in strong):
            row.update(candidate_afl=True, candidate_reason=f"possible_near_strong_{AFL_POSSIBLE_ATTACH_GAP_MS}ms")
        else:
            row.update(candidate_afl=False, candidate_reason="possible_without_nearby_strong" if row["label"] == "possible_afl" else "non_candidate")
        row.update(final_label="non_afl", final_reason="not_in_final_segment", final_event_index="")

    candidates = sorted([row for row in rows if row.get("candidate_afl")], key=lambda item: (int(item["start_ms"]), int(item["end_ms"])))
    if not candidates:
        return []
    segments = [_new_segment(candidates[0])]
    for row in candidates[1:]:
        if int(row["start_ms"]) - int(segments[-1]["end_ms"]) <= AFL_BRIDGE_GAP_MS:
            _extend_segment(segments[-1], row)
        else:
            segments.append(_new_segment(row))
    final_segments = [segment for segment in segments if int(segment["end_ms"]) - int(segment["start_ms"]) >= AFL_FINAL_MIN_DURATION_MS and int(segment["strong_windows"]) >= AFL_MIN_STRONG_WINDOWS_PER_EVENT]
    for event_index, segment in enumerate(final_segments, start=1):
        segment["event_index"] = event_index
        for row in rows:
            if int(row["start_ms"]) < int(segment["end_ms"]) and int(row["end_ms"]) > int(segment["start_ms"]):
                row.update(final_label="afl", final_reason="candidate_segment" if row.get("candidate_afl") else "covered_by_segment", final_event_index=event_index)
    return final_segments


def _new_segment(row: dict[str, Any]) -> dict[str, Any]:
    return {"start_ms": int(row["start_ms"]), "end_ms": int(row["end_ms"]), "window_count": 1, "strong_windows": 1 if row["label"] == "strong_afl" else 0, "possible_windows": 1 if row["label"] == "possible_afl" else 0, "score_sum": float(row.get("afl_score", 0.0)), "multiple_match_ratio_sum": float(row.get("multiple_match_ratio", 0.0)), "diff_peak_coverage_sum": float(row.get("diff_peak_coverage", 0.0)), "base_diff_values": [float(row["base_diff_ms"])] if row.get("base_diff_ms") not in (None, "") else []}


def _extend_segment(segment: dict[str, Any], row: dict[str, Any]) -> None:
    segment["end_ms"] = max(int(segment["end_ms"]), int(row["end_ms"]))
    segment["window_count"] = int(segment["window_count"]) + 1
    segment["strong_windows"] = int(segment["strong_windows"]) + (1 if row["label"] == "strong_afl" else 0)
    segment["possible_windows"] = int(segment["possible_windows"]) + (1 if row["label"] == "possible_afl" else 0)
    segment["score_sum"] = float(segment["score_sum"]) + float(row.get("afl_score", 0.0))
    segment["multiple_match_ratio_sum"] = float(segment["multiple_match_ratio_sum"]) + float(row.get("multiple_match_ratio", 0.0))
    segment["diff_peak_coverage_sum"] = float(segment["diff_peak_coverage_sum"]) + float(row.get("diff_peak_coverage", 0.0))
    if row.get("base_diff_ms") not in (None, ""):
        segment.setdefault("base_diff_values", []).append(float(row["base_diff_ms"]))


def finalize_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    finalized = []
    for index, segment in enumerate(segments, start=1):
        count = max(1, int(segment.get("window_count", 1)))
        base_values = np.asarray(segment.get("base_diff_values", []), dtype=np.float64)
        finalized.append({**segment, "event_index": int(segment.get("event_index", index)), "mean_afl_score": round(float(segment.get("score_sum", 0.0)) / count, 6), "mean_multiple_match_ratio": round(float(segment.get("multiple_match_ratio_sum", 0.0)) / count, 6), "mean_diff_peak_coverage": round(float(segment.get("diff_peak_coverage_sum", 0.0)) / count, 6), "mean_base_diff_ms": round(float(np.mean(base_values)), 3) if base_values.size else None})
    return finalized


def afl_config(window_seconds: int = AFL_WINDOW_SECONDS, step_seconds: int = AFL_STEP_SECONDS) -> dict[str, Any]:
    return {"window_seconds": window_seconds, "step_seconds": step_seconds, "min_valid_rr": AFL_MIN_VALID_RR, "rr_min_ms": AFL_RR_MIN_MS, "rr_max_ms": AFL_RR_MAX_MS, "diff_min_ms": AFL_DIFF_MIN_MS, "diff_bin_ms": AFL_DIFF_BIN_MS, "top_diff_peaks": AFL_TOP_DIFF_PEAKS, "multiple_tolerance_ms": AFL_MULTIPLE_TOLERANCE_MS, "max_multiple": AFL_MAX_MULTIPLE, "base_diff_min_ms": AFL_BASE_DIFF_MIN_MS, "base_diff_max_ms": AFL_BASE_DIFF_MAX_MS, "possible_attach_gap_ms": AFL_POSSIBLE_ATTACH_GAP_MS, "bridge_gap_ms": AFL_BRIDGE_GAP_MS, "final_min_duration_ms": AFL_FINAL_MIN_DURATION_MS, "min_strong_windows_per_event": AFL_MIN_STRONG_WINDOWS_PER_EVENT}


def segments_to_events(segments: list[dict[str, Any]], window_seconds: int, step_seconds: int) -> list[dict[str, Any]]:
    events = []
    for index, segment in enumerate(finalize_segments(segments), start=1):
        start_ms, end_ms = int(segment["start_ms"]), int(segment["end_ms"])
        events.append({"type": "af_family", "subtype": "flutter", "layer": "rr_afl_filter", "rule": "rr_diff_multiple_10min", "event_index": int(segment.get("event_index", index)), "t0_ms": start_ms, "t1_ms": end_ms, "time": f"{start_ms} ms ~ {end_ms} ms", "duration": format_hms((end_ms - start_ms) / 1000.0), "stats": {"window_count": int(segment["window_count"]), "strong_windows": int(segment["strong_windows"]), "possible_windows": int(segment["possible_windows"]), "mean_afl_score": segment["mean_afl_score"], "mean_multiple_match_ratio": segment["mean_multiple_match_ratio"], "mean_diff_peak_coverage": segment["mean_diff_peak_coverage"], "mean_base_diff_ms": segment["mean_base_diff_ms"], "config": afl_config(window_seconds, step_seconds)}})
    return events


def _csv_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["window_index", "start_ms", "end_ms", "duration_seconds", "valid_rr_count", "raw_diff_count", "valid_diff_count", "nonzero_diff_ratio", "mean_rr_ms", "mean_hr", "rr_cv", "dominant_rr_ratio", "base_diff_ms", "matched_diff_count", "multiple_match_ratio", "diff_peak_count", "diff_top_peak_ms", "diff_peak_coverage", "diff_peak_centers_ms", "diff_peak_counts", "afl_score", "label", "reason", "candidate_afl", "candidate_reason", "final_label", "final_reason", "final_event_index"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: _csv_value(row.get(field, "")) for field in fieldnames} for row in rows)


def summarize(rows: list[dict[str, Any]], series: BeatSeries, segments: list[dict[str, Any]], window_seconds: int, step_seconds: int) -> dict[str, Any]:
    label_counts: dict[str, int] = {}
    final_counts: dict[str, int] = {}
    for row in rows:
        label_counts[str(row.get("label", "non_afl"))] = label_counts.get(str(row.get("label", "non_afl")), 0) + 1
        final_counts[str(row.get("final_label", "non_afl"))] = final_counts.get(str(row.get("final_label", "non_afl")), 0) + 1
    return {"source_csv": str(series.source_csv), "window_count": len(rows), "label_counts": label_counts, "final_label_counts": final_counts, "event_count": len(segments), "events": finalize_segments(segments), "config": afl_config(window_seconds, step_seconds)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RR-diff regularity filter for atrial flutter candidate screening.")
    parser.add_argument("--csv", required=True, help="Beat CSV with time offsets and RR intervals.")
    parser.add_argument("--out-csv", default="", help="Output window CSV path.")
    parser.add_argument("--out-json", default="", help="Output summary JSON path.")
    parser.add_argument("--out-events-json", default="", help="Output AFL events JSON path.")
    parser.add_argument("--window-seconds", type=int, default=AFL_WINDOW_SECONDS, help="Sliding window size in seconds.")
    parser.add_argument("--step-seconds", type=int, default=AFL_STEP_SECONDS, help="Sliding step in seconds.")
    return parser.parse_args()


def default_output_root(csv_path: Path) -> Path:
    parts = csv_path.parts
    for index, part in enumerate(parts):
        if part == "out" and index + 1 < len(parts):
            return Path(*parts[: index + 2]) / "rr_afl_filter"
    return Path("out") / csv_path.stem / "rr_afl_filter"


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    series = load_beat_series(csv_path)
    rows = classify_series(series, args.window_seconds, args.step_seconds)
    segments = apply_afl_merge(rows)
    out_root = default_output_root(csv_path)
    out_csv = Path(args.out_csv).resolve() if args.out_csv else out_root / f"{csv_path.stem}_rr_afl_windows.csv"
    out_json = Path(args.out_json).resolve() if args.out_json else out_root / f"{csv_path.stem}_rr_afl_summary.json"
    out_events_json = Path(args.out_events_json).resolve() if args.out_events_json else out_root / f"{csv_path.stem}_rr_afl_events.json"
    write_csv(out_csv, rows)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summarize(rows, series, segments, args.window_seconds, args.step_seconds), ensure_ascii=False, indent=2), encoding="utf-8")
    out_events_json.parent.mkdir(parents=True, exist_ok=True)
    out_events_json.write_text(json.dumps(segments_to_events(segments, args.window_seconds, args.step_seconds), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"windows={len(rows)} -> {out_csv}")
    print(f"events={len(segments)} -> {out_events_json}")
    print(f"summary -> {out_json}")


if __name__ == "__main__":
    main()
