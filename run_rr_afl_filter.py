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
AFL_BIN_MS = 40
AFL_TOL_MS = 50
AFL_MAX_MULTIPLE = 4
AFL_MIN_CLUSTER_COUNT = 8
AFL_MIN_CLUSTER_RATIO = 0.03
AFL_MAX_CANDIDATE_CLUSTERS = 8
AFL_MIN_MATCHED_CLUSTERS = 2
AFL_MIN_MATCHED_RR_RATIO = 0.60
AFL_MIN_MINOR_COUNT = 5
AFL_MIN_MINOR_RATIO = 0.03
AFL_STABLE_TOP1_RATIO = 0.90
AFL_STABLE_TOP2_RATIO = 0.05
AFL_MIN_TRANSITION_COVERAGE = 0.45
AFL_TRANSITION_TOP_K = 6
AFL_EVENT_GAP_MS = 10_000
AFL_FINAL_MIN_DURATION_MS = 60_000


@dataclass(slots=True)
class BeatSeries:
    offsets_ms: np.ndarray
    rr_ms: np.ndarray
    source_csv: Path


def _norm(name: str) -> str:
    return name.strip().lower()


def _pick(fields: dict[str, str], names: list[str]) -> str | None:
    for name in names:
        if _norm(name) in fields:
            return fields[_norm(name)]
    return None


def _num(text: str) -> float | None:
    value = (text or "").strip()
    if not value or value == "-":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _int_like(text: str) -> int | None:
    value = _num(text)
    return None if value is None else int(round(value))


def load_beat_series(csv_path: Path) -> BeatSeries:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        fields = {_norm(name): name for name in reader.fieldnames}
        offset_key = _pick(fields, ["merged_milliseconds", "time offset(ms)", "offset_ms", "timestamp_ms", "ms"])
        rr_key = _pick(fields, ["rr interval(ms)", "rr_ms", "rr", "rr_raw"])
        if offset_key is None and rr_key is None:
            raise ValueError(f"CSV must contain a time column or RR column: {csv_path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no rows: {csv_path}")

    offsets: list[int] = []
    rr_values: list[int] = []
    for index, row in enumerate(rows):
        offset = _int_like(row[offset_key]) if offset_key else None
        rr = _int_like(row[rr_key]) if rr_key else None
        if offset is None:
            if index != 0:
                raise ValueError(f"Missing time offset for row {index + 1} in {csv_path}")
            offset = 0
        offsets.append(offset)
        if rr is None:
            if index > 0:
                rr = offsets[index] - offsets[index - 1]
            elif len(rows) > 1 and offset_key:
                next_offset = _int_like(rows[index + 1].get(offset_key, ""))
                rr = next_offset - offset if next_offset is not None else 0
            else:
                rr = 0
        rr_values.append(rr)
    order = np.argsort(np.asarray(offsets, dtype=np.int64))
    return BeatSeries(np.asarray(offsets, dtype=np.int64)[order], np.asarray(rr_values, dtype=np.int64)[order], csv_path)


def make_windows(duration_ms: int, window_ms: int, step_ms: int) -> list[tuple[int, int]]:
    if duration_ms <= 0:
        return []
    return [(start, start + window_ms) for start in range(0, max(duration_ms - window_ms, 0) + 1, step_ms)]


def format_hms(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def valid_mask(rr_ms: np.ndarray) -> np.ndarray:
    values = np.asarray(rr_ms, dtype=np.float64)
    return np.isfinite(values) & (values >= AFL_RR_MIN_MS) & (values <= AFL_RR_MAX_MS)


def rr_clusters(rr: np.ndarray) -> list[dict[str, float | int]]:
    if rr.size == 0:
        return []
    bins = np.round(rr / AFL_BIN_MS).astype(np.int32)
    clusters: list[dict[str, float | int]] = []
    for bin_id in sorted(set(bins.tolist())):
        values = rr[bins == bin_id]
        clusters.append({"center_ms": float(np.mean(values)), "count": int(values.size), "ratio": float(values.size / rr.size)})
    return sorted(clusters, key=lambda item: int(item["count"]), reverse=True)


def candidate_clusters(clusters: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    kept = [item for item in clusters if int(item["count"]) >= AFL_MIN_CLUSTER_COUNT or float(item["ratio"]) >= AFL_MIN_CLUSTER_RATIO]
    return kept[:AFL_MAX_CANDIDATE_CLUSTERS]


def match_allowed(rr: np.ndarray, allowed: list[float]) -> np.ndarray:
    if rr.size == 0 or not allowed:
        return np.zeros(rr.size, dtype=bool)
    values = np.asarray(allowed, dtype=np.float64)
    return np.min(np.abs(rr[:, None] - values[None, :]), axis=1) <= AFL_TOL_MS

def transition_coverage(rr: np.ndarray) -> float:
    if rr.size < 2:
        return 0.0
    left = np.round(rr[:-1] / AFL_BIN_MS).astype(np.int32)
    right = np.round(rr[1:] / AFL_BIN_MS).astype(np.int32)
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(left.tolist(), right.tolist(), strict=False):
        counts[pair] = counts.get(pair, 0) + 1
    return float(sum(sorted(counts.values(), reverse=True)[:AFL_TRANSITION_TOP_K]) / max(left.size, 1))


def find_pattern(rr: np.ndarray) -> tuple[dict[str, Any], list[dict[str, float | int]]]:
    clusters = rr_clusters(rr)
    no = {"found": False, "base_rr_ms": None, "allowed_rr_ms": [], "matched_rr_ratio": 0.0, "matched_cluster_count": 0, "minor_cluster_count": 0, "minor_cluster_ratio": 0.0, "rr_cluster_coverage": 0.0, "transition_coverage": 0.0, "reason": "criteria_not_met"}
    if rr.size < AFL_MIN_VALID_RR:
        no["reason"] = "insufficient_rr"
        return no, clusters
    if len(clusters) < 2:
        no["reason"] = "single_cluster"
        return no, clusters
    if float(clusters[0]["ratio"]) >= AFL_STABLE_TOP1_RATIO and float(clusters[1]["ratio"]) < AFL_STABLE_TOP2_RATIO:
        no["reason"] = "stable_sinus_like"
        return no, clusters

    candidates = candidate_clusters(clusters)
    coverage = float(sum(float(item["ratio"]) for item in candidates))
    trans_cov = transition_coverage(rr)
    best: dict[str, Any] | None = None
    for base_item in sorted(candidates, key=lambda item: float(item["center_ms"])):
        base = float(base_item["center_ms"])
        matched_clusters: list[dict[str, float | int]] = []
        allowed: list[float] = []
        for cluster in candidates:
            center = float(cluster["center_ms"])
            for multiple in range(1, AFL_MAX_MULTIPLE + 1):
                if abs(center - multiple * base) <= AFL_TOL_MS:
                    matched_clusters.append(cluster)
                    allowed.append(float(multiple * base))
                    break
        if len(matched_clusters) < AFL_MIN_MATCHED_CLUSTERS:
            continue
        allowed = sorted(set(round(item, 3) for item in allowed))
        matched_ratio = float(np.mean(match_allowed(rr, allowed)))
        minor_count = min(int(item["count"]) for item in matched_clusters)
        minor_ratio = min(float(item["ratio"]) for item in matched_clusters)
        if matched_ratio < AFL_MIN_MATCHED_RR_RATIO:
            continue
        if minor_count < AFL_MIN_MINOR_COUNT and minor_ratio < AFL_MIN_MINOR_RATIO:
            continue
        if trans_cov < AFL_MIN_TRANSITION_COVERAGE:
            continue
        pattern = {"found": True, "base_rr_ms": round(base, 3), "allowed_rr_ms": allowed, "matched_rr_ratio": matched_ratio, "matched_cluster_count": len(matched_clusters), "minor_cluster_count": minor_count, "minor_cluster_ratio": minor_ratio, "rr_cluster_coverage": coverage, "transition_coverage": trans_cov, "reason": "rr_cluster_integer_template"}
        if best is None or float(pattern["matched_rr_ratio"]) > float(best["matched_rr_ratio"]):
            best = pattern
    return (best or no), clusters


def evaluate_window(rr_ms: np.ndarray) -> dict[str, Any]:
    rr = np.asarray(rr_ms, dtype=np.float64)
    rr = rr[valid_mask(rr)]
    pattern, clusters = find_pattern(rr)
    return {"label": "afl" if pattern["found"] else "non_afl", "reason": pattern["reason"], "valid_rr_count": int(rr.size), "cluster_count": len(clusters), "cluster_centers_ms": [round(float(item["center_ms"]), 3) for item in clusters], "cluster_counts": [int(item["count"]) for item in clusters], "cluster_ratios": [round(float(item["ratio"]), 6) for item in clusters], "base_rr_ms": pattern["base_rr_ms"], "allowed_rr_ms": pattern["allowed_rr_ms"], "matched_rr_ratio": round(float(pattern["matched_rr_ratio"]), 6), "matched_cluster_count": pattern["matched_cluster_count"], "minor_cluster_count": pattern["minor_cluster_count"], "minor_cluster_ratio": round(float(pattern["minor_cluster_ratio"]), 6), "rr_cluster_coverage": round(float(pattern["rr_cluster_coverage"]), 6), "transition_coverage": round(float(pattern["transition_coverage"]), 6)}


def mask_segments(series: BeatSeries, mask: np.ndarray) -> list[dict[str, Any]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    segments: list[dict[str, Any]] = []
    start = prev = int(indices[0])
    for raw in indices[1:]:
        index = int(raw)
        if int(series.offsets_ms[index]) - int(series.offsets_ms[prev]) <= AFL_EVENT_GAP_MS:
            prev = index
        else:
            segments.append({"start_index": start, "end_index": prev, "start_ms": int(series.offsets_ms[start]), "end_ms": int(series.offsets_ms[prev])})
            start = prev = index
    segments.append({"start_index": start, "end_index": prev, "start_ms": int(series.offsets_ms[start]), "end_ms": int(series.offsets_ms[prev])})
    return [item for item in segments if int(item["end_ms"]) - int(item["start_ms"]) >= AFL_FINAL_MIN_DURATION_MS]

def classify_series(series: BeatSeries, window_seconds: int, step_seconds: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    votes = np.zeros(series.rr_ms.size, dtype=np.int32)
    window_ms, step_ms = window_seconds * 1000, step_seconds * 1000
    duration_ms = int(series.offsets_ms[-1]) + window_ms if series.offsets_ms.size else 0
    for window_index, (start_ms, end_ms) in enumerate(make_windows(duration_ms, window_ms, step_ms)):
        left = int(np.searchsorted(series.offsets_ms, start_ms, side="left"))
        right = int(np.searchsorted(series.offsets_ms, end_ms, side="left"))
        rr_window = series.rr_ms[left:right]
        stats = evaluate_window(rr_window)
        if stats["label"] == "afl" and stats["allowed_rr_ms"]:
            local_valid = valid_mask(rr_window)
            local_match = np.zeros(rr_window.size, dtype=bool)
            local_match[local_valid] = match_allowed(np.asarray(rr_window, dtype=np.float64)[local_valid], [float(item) for item in stats["allowed_rr_ms"]])
            votes[left:right][local_match] += 1
        rows.append({"window_index": window_index, "start_ms": start_ms, "end_ms": end_ms, "duration_seconds": window_seconds, **stats, "final_label": "non_afl", "final_event_index": ""})
    segments = mask_segments(series, votes > 0)
    for event_index, segment in enumerate(segments, start=1):
        segment["event_index"] = event_index
        overlap = [row for row in rows if int(row["start_ms"]) < int(segment["end_ms"]) and int(row["end_ms"]) > int(segment["start_ms"])]
        for row in overlap:
            row["final_label"] = "afl"
            row["final_event_index"] = event_index
        segment["window_count"] = len(overlap)
        segment["pattern_windows"] = sum(row["label"] == "afl" for row in overlap)
        segment["mean_matched_rr_ratio"] = round(float(np.mean([float(row["matched_rr_ratio"]) for row in overlap])) if overlap else 0.0, 6)
        segment["allowed_rr_ms"] = sorted({value for row in overlap for value in row.get("allowed_rr_ms", [])})
    return rows, segments


def afl_config(window_seconds: int = AFL_WINDOW_SECONDS, step_seconds: int = AFL_STEP_SECONDS) -> dict[str, Any]:
    return {"window_seconds": window_seconds, "step_seconds": step_seconds, "min_valid_rr": AFL_MIN_VALID_RR, "rr_min_ms": AFL_RR_MIN_MS, "rr_max_ms": AFL_RR_MAX_MS, "cluster_bin_ms": AFL_BIN_MS, "rr_tolerance_ms": AFL_TOL_MS, "max_multiple": AFL_MAX_MULTIPLE, "min_cluster_count": AFL_MIN_CLUSTER_COUNT, "min_cluster_ratio": AFL_MIN_CLUSTER_RATIO, "min_matched_rr_ratio": AFL_MIN_MATCHED_RR_RATIO, "min_transition_coverage": AFL_MIN_TRANSITION_COVERAGE, "event_gap_ms": AFL_EVENT_GAP_MS, "final_min_duration_ms": AFL_FINAL_MIN_DURATION_MS}


def segments_to_events(segments: list[dict[str, Any]], window_seconds: int, step_seconds: int) -> list[dict[str, Any]]:
    events = []
    for index, segment in enumerate(segments, start=1):
        start_ms, end_ms = int(segment["start_ms"]), int(segment["end_ms"])
        events.append({"type": "af_family", "subtype": "flutter", "layer": "rr_afl_filter", "rule": "rr_cluster_integer_template", "event_index": int(segment.get("event_index", index)), "t0_ms": start_ms, "t1_ms": end_ms, "time": f"{start_ms} ms ~ {end_ms} ms", "duration": format_hms((end_ms - start_ms) / 1000.0), "stats": {"window_count": int(segment.get("window_count", 0)), "pattern_windows": int(segment.get("pattern_windows", 0)), "mean_matched_rr_ratio": segment.get("mean_matched_rr_ratio", 0.0), "allowed_rr_ms": segment.get("allowed_rr_ms", []), "config": afl_config(window_seconds, step_seconds)}})
    return events


def _csv_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (list, tuple)) else "" if value is None else str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["window_index", "start_ms", "end_ms", "duration_seconds", "label", "final_label", "final_event_index", "reason", "valid_rr_count", "cluster_count", "cluster_centers_ms", "cluster_counts", "cluster_ratios", "base_rr_ms", "allowed_rr_ms", "matched_rr_ratio", "matched_cluster_count", "minor_cluster_count", "minor_cluster_ratio", "rr_cluster_coverage", "transition_coverage"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: _csv_value(row.get(field, "")) for field in fieldnames} for row in rows)


def summarize(rows: list[dict[str, Any]], series: BeatSeries, segments: list[dict[str, Any]], window_seconds: int, step_seconds: int) -> dict[str, Any]:
    labels: dict[str, int] = {}
    finals: dict[str, int] = {}
    for row in rows:
        labels[str(row.get("label", "non_afl"))] = labels.get(str(row.get("label", "non_afl")), 0) + 1
        finals[str(row.get("final_label", "non_afl"))] = finals.get(str(row.get("final_label", "non_afl")), 0) + 1
    return {"source_csv": str(series.source_csv), "window_count": len(rows), "label_counts": labels, "final_label_counts": finals, "event_count": len(segments), "events": segments, "config": afl_config(window_seconds, step_seconds)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RR-cluster integer-template filter for atrial flutter screening.")
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
    rows, segments = classify_series(series, args.window_seconds, args.step_seconds)
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


