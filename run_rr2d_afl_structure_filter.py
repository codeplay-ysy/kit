from __future__ import annotations

import argparse, csv, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

WINDOW_SECONDS = 10 * 60
STEP_SECONDS = 30
RR_MIN_MS = 300
RR_MAX_MS = 2000
MIN_VALID_RR = 40
LOCAL_RADIUS = 8
SPECIAL_RATIOS = [4/3, 1.5, 2.0, 3.0, 4.0]
RATIO_LOG_TOL = 0.07
SELF_LOOP_LOG_TOL = 0.08
VALUE_BIN_MS = 50
VALUE_MERGE_MS = 80
MAX_VALUE_CLUSTER_COUNT = 6
MAX_BRIDGE_GAP = 3
MIN_SPECIAL_COUNT_AFL = 3
MIN_SPECIAL_COUNT_SUSPICIOUS = 2
MIN_BRIDGED_RUN_AFL = 3
MIN_BRIDGED_RUN_SUSPICIOUS = 2
MIN_VALUE_CENTER_COUNT = 2
ECTOPY_RECIPROCAL_TOL = 0.10
ECTOPY_STRONG_RATIO = 0.72
EVENT_GAP_MS = 10_000
FINAL_MIN_DURATION_MS = 30_000
SEGMENT_CANDIDATE_GAP_MS = 15_000
SEGMENT_MIN_CANDIDATES = 3
SEGMENT_MIN_AFL_CANDIDATES = 4
SEGMENT_MAX_CLEAR_AFL_SLOPE_TYPES = 4
SEGMENT_ECTOPY_FRACTION = 0.25

@dataclass(slots=True)
class BeatSeries:
    offsets_ms: np.ndarray
    rr_ms: np.ndarray
    source_csv: Path


def _norm(name: str) -> str: return name.strip().lower()

def _pick(fields: dict[str, str], names: list[str]) -> str | None:
    for name in names:
        if _norm(name) in fields: return fields[_norm(name)]
    return None

def _num(text: str) -> float | None:
    value = (text or "").strip()
    if not value or value == "-": return None
    try: return float(value)
    except ValueError: return None

def _int_like(text: str) -> int | None:
    value = _num(text)
    return None if value is None else int(round(value))


def load_beat_series(csv_path: Path) -> BeatSeries:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames: raise ValueError(f"CSV has no header: {csv_path}")
        fields = {_norm(name): name for name in reader.fieldnames}
        offset_key = _pick(fields, ["merged_milliseconds", "time offset(ms)", "offset_ms", "timestamp_ms", "ms"])
        rr_key = _pick(fields, ["rr interval(ms)", "rr_ms", "rr", "rr_raw"])
        rows = list(reader)
    if not rows or (offset_key is None and rr_key is None): raise ValueError(f"Bad CSV: {csv_path}")
    offsets, rr_values = [], []
    for i, row in enumerate(rows):
        offset = _int_like(row[offset_key]) if offset_key else None
        rr = _int_like(row[rr_key]) if rr_key else None
        if offset is None: offset = 0 if i == 0 else (_ for _ in ()).throw(ValueError(f"Missing offset row {i+1}"))
        offsets.append(offset)
        if rr is None: rr = offsets[i] - offsets[i-1] if i > 0 else 0
        rr_values.append(rr)
    order = np.argsort(np.asarray(offsets, dtype=np.int64))
    return BeatSeries(np.asarray(offsets, dtype=np.int64)[order], np.asarray(rr_values, dtype=np.int64)[order], csv_path)


def make_windows(duration_ms: int, window_ms: int, step_ms: int) -> list[tuple[int, int]]:
    return [] if duration_ms <= 0 else [(s, s + window_ms) for s in range(0, max(duration_ms - window_ms, 0) + 1, step_ms)]

def format_hms(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{total // 3600:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"

def valid_rr(rr_ms: np.ndarray) -> np.ndarray:
    values = np.asarray(rr_ms, dtype=np.float64)
    return values[np.isfinite(values) & (values >= RR_MIN_MS) & (values <= RR_MAX_MS)]

def match_special_ratio(ratio: float) -> float | None:
    if ratio <= 0: return None
    normalized = ratio if ratio >= 1.0 else 1.0 / ratio
    diffs = [abs(np.log(normalized / t)) for t in SPECIAL_RATIOS]
    best = int(np.argmin(diffs))
    if diffs[best] > RATIO_LOG_TOL:
        return None
    matched = float(SPECIAL_RATIOS[best])
    return matched if ratio >= 1.0 else round(1.0 / matched, 6)

def is_self_loop(ratio: float) -> bool:
    return ratio > 0 and abs(np.log(ratio)) <= SELF_LOOP_LOG_TOL

def reciprocal_ratio(slopes: list[float]) -> float:
    if len(slopes) < 2: return 0.0
    hits = sum(abs(a * b - 1.0) <= ECTOPY_RECIPROCAL_TOL for a, b in zip(slopes[:-1], slopes[1:], strict=False))
    return float(hits / max(len(slopes) - 1, 1))


def value_centers(values: np.ndarray) -> list[float]:
    if values.size == 0: return []
    bins = np.round(values / VALUE_BIN_MS).astype(np.int32)
    raw = [(float(np.mean(values[bins == b])), int(np.sum(bins == b))) for b in sorted(set(bins.tolist()))]
    merged, current = [], [raw[0]]
    for center, count in raw[1:]:
        cur = sum(v * c for v, c in current) / max(sum(c for _, c in current), 1)
        if abs(center - cur) <= VALUE_MERGE_MS: current.append((center, count))
        else:
            total = sum(c for _, c in current)
            merged.append((sum(v * c for v, c in current) / max(total, 1), total))
            current = [(center, count)]
    total = sum(c for _, c in current)
    merged.append((sum(v * c for v, c in current) / max(total, 1), total))
    return [round(c, 3) for c, _ in sorted(merged, key=lambda x: x[1], reverse=True)[:MAX_VALUE_CLUSTER_COUNT]]


def bridged_run(indices: list[int], ratios: np.ndarray) -> int:
    if not indices: return 0
    best = current = 1
    for left, right in zip(indices[:-1], indices[1:], strict=False):
        gap = right - left - 1
        if gap <= MAX_BRIDGE_GAP and all(is_self_loop(float(v)) for v in ratios[left + 1:right]): current += 1
        else:
            best = max(best, current)
            current = 1
    return max(best, current)


def evaluate_local(rr: np.ndarray, ratios: np.ndarray, matched: list[float | None], center: int) -> dict[str, Any]:
    left = max(0, center - LOCAL_RADIUS)
    right = min(ratios.size, center + LOCAL_RADIUS + 1)
    local_ratios = ratios[left:right]
    local_matches = matched[left:right]
    special_indices = [i for i, value in enumerate(local_matches) if value is not None]
    if not special_indices:
        return {"label": "non_afl", "reason": "no_special_points", "special_count": 0, "bridged_run": 0, "centers": [], "reciprocal_ratio": 0.0, "range_start": left, "range_end": right}
    slopes = [float(local_matches[i]) for i in special_indices if local_matches[i] is not None]
    centers = value_centers(rr[left:right + 1])
    count = len(special_indices)
    run = bridged_run(special_indices, local_ratios)
    recip = reciprocal_ratio(slopes)
    if count < MIN_SPECIAL_COUNT_SUSPICIOUS or len(centers) < MIN_VALUE_CENTER_COUNT:
        return {"label": "non_afl", "reason": "isolated_or_unstable_local_structure", "special_count": count, "bridged_run": run, "centers": centers, "reciprocal_ratio": recip, "range_start": left, "range_end": right}
    if recip >= ECTOPY_STRONG_RATIO:
        return {"label": "suspicious", "reason": "ectopy_like_reciprocal_pattern", "special_count": count, "bridged_run": run, "centers": centers, "reciprocal_ratio": recip, "range_start": left, "range_end": right}
    if count >= MIN_SPECIAL_COUNT_AFL and run >= MIN_BRIDGED_RUN_AFL:
        return {"label": "afl", "reason": "repeated_special_slopes_with_bridge_and_states", "special_count": count, "bridged_run": run, "centers": centers, "reciprocal_ratio": recip, "range_start": left, "range_end": right}
    if run >= MIN_BRIDGED_RUN_SUSPICIOUS:
        return {"label": "suspicious", "reason": "structured_local_pattern", "special_count": count, "bridged_run": run, "centers": centers, "reciprocal_ratio": recip, "range_start": left, "range_end": right}
    return {"label": "non_afl", "reason": "weak_local_structure", "special_count": count, "bridged_run": run, "centers": centers, "reciprocal_ratio": recip, "range_start": left, "range_end": right}


def build_local_candidates(series: BeatSeries) -> list[dict[str, Any]]:
    rr = np.asarray(series.rr_ms, dtype=np.float64)
    valid = np.isfinite(rr) & (rr >= RR_MIN_MS) & (rr <= RR_MAX_MS)
    ratios = np.full(max(rr.size - 1, 0), np.nan, dtype=np.float64)
    matched: list[float | None] = []
    for i in range(ratios.size):
        if valid[i] and valid[i + 1] and rr[i] > 0:
            ratios[i] = rr[i + 1] / rr[i]
            matched.append(match_special_ratio(float(ratios[i])))
        else:
            matched.append(None)
    special_indices = [i for i, value in enumerate(matched) if value is not None]
    candidates: list[dict[str, Any]] = []
    for index in special_indices:
        local = evaluate_local(rr, ratios, matched, index)
        if local["label"] == "non_afl":
            continue
        start_i = max(0, int(local["range_start"]))
        end_i = min(series.offsets_ms.size - 1, int(local["range_end"]) + 1)
        candidates.append({
            "start_index": start_i,
            "end_index": end_i,
            "start_ms": int(series.offsets_ms[start_i]),
            "end_ms": int(series.offsets_ms[end_i]),
            "label": str(local["label"]),
            "reason": str(local["reason"]),
            "special_count": int(local["special_count"]),
            "bridged_run": int(local["bridged_run"]),
            "centers": local["centers"],
            "reciprocal_ratio": float(local["reciprocal_ratio"]),
            "slope": float(matched[index] or 0.0),
        })
    return candidates


def merge_candidate_segments(candidates: list[dict[str, Any]], gap_ms: int = SEGMENT_CANDIDATE_GAP_MS) -> list[dict[str, Any]]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: (int(item["start_ms"]), int(item["end_ms"])))
    groups: list[list[dict[str, Any]]] = [[ordered[0]]]
    for candidate in ordered[1:]:
        current_end = max(int(item["end_ms"]) for item in groups[-1])
        if int(candidate["start_ms"]) - current_end <= gap_ms:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])
    segments: list[dict[str, Any]] = []
    for group in groups:
        start_index = min(int(item["start_index"]) for item in group)
        end_index = max(int(item["end_index"]) for item in group)
        start_ms = min(int(item["start_ms"]) for item in group)
        end_ms = max(int(item["end_ms"]) for item in group)
        duration_ms = end_ms - start_ms
        labels = [str(item["label"]) for item in group]
        reasons = [str(item["reason"]) for item in group]
        slopes = [float(item["slope"]) for item in group if float(item["slope"]) > 0]
        centers = sorted({float(center) for item in group for center in item.get("centers", [])})
        ectopy_count = sum(reason == "ectopy_like_reciprocal_pattern" for reason in reasons)
        afl_local_count = sum(label == "afl" for label in labels)
        candidate_count = len(group)
        slope_types = len({round(slope if slope >= 1 else 1 / slope, 3) for slope in slopes})
        mean_recip = float(np.mean([float(item["reciprocal_ratio"]) for item in group])) if group else 0.0
        if candidate_count < SEGMENT_MIN_CANDIDATES or duration_ms < FINAL_MIN_DURATION_MS:
            label, reason = "non_afl", "short_or_sparse_candidate_segment"
        elif ectopy_count / max(candidate_count, 1) >= SEGMENT_ECTOPY_FRACTION:
            label, reason = "suspicious", "segment_ectopy_like_overlap"
        elif afl_local_count >= SEGMENT_MIN_AFL_CANDIDATES and slope_types <= SEGMENT_MAX_CLEAR_AFL_SLOPE_TYPES and mean_recip < ECTOPY_STRONG_RATIO:
            label, reason = "afl", "segment_repeated_stable_local_structure"
        else:
            label, reason = "suspicious", "segment_structured_but_not_clear_afl"
        segments.append({
            "start_index": start_index,
            "end_index": end_index,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": duration_ms,
            "label": label,
            "reason": reason,
            "candidate_count": candidate_count,
            "afl_local_count": afl_local_count,
            "ectopy_local_count": ectopy_count,
            "slope_type_count": slope_types,
            "mean_reciprocal_ratio": round(mean_recip, 6),
            "candidate_rr_centers_ms": centers[:MAX_VALUE_CLUSTER_COUNT],
        })
    return segments


def evaluate_window(rr_ms: np.ndarray) -> dict[str, Any]:
    rr = valid_rr(np.asarray(rr_ms, dtype=np.float64))
    base = {"label": "non_afl", "reason": "insufficient_rr", "suspicious_reason": "", "valid_rr_count": int(rr.size), "ray_match_count": 0, "ray_match_ratio": 0.0, "max_run_length": 0, "mean_run_length": 0.0, "run_count": 0, "isolated_ratio": 0.0, "matched_slopes": [], "candidate_rr_centers_ms": [], "candidate_rr_center_count": 0, "self_loop_ratio": 0.0, "reciprocal_alternation_ratio": 0.0, "afl_feature_point_count": 0, "ectopy_feature_point_count": 0}
    if rr.size < MIN_VALID_RR or rr.size < 3: return base
    left_rr, right_rr = rr[:-1], rr[1:]
    ratios = right_rr / np.maximum(left_rr, 1.0)
    matched = [match_special_ratio(float(r)) for r in ratios.tolist()]
    special_indices = [i for i, v in enumerate(matched) if v is not None]
    if not special_indices:
        base["reason"] = "no_special_points"
        return base
    local_results = [evaluate_local(rr, ratios, matched, i) for i in special_indices]
    afl_hits = [x for x in local_results if x["label"] == "afl"]
    suspicious_hits = [x for x in local_results if x["label"] == "suspicious"]
    ectopy_hits = [x for x in suspicious_hits if x["reason"] == "ectopy_like_reciprocal_pattern"]
    best = max(local_results, key=lambda x: (x["reason"] == "ectopy_like_reciprocal_pattern", x["label"] == "afl", x["bridged_run"], x["special_count"]))
    if ectopy_hits and afl_hits:
        label = "suspicious"
    elif ectopy_hits and len(ectopy_hits) >= max(2, len(afl_hits)):
        label = "suspicious"
    elif afl_hits:
        label = "afl"
    elif suspicious_hits:
        label = "suspicious"
    else:
        label = "non_afl"
    slopes = [float(v) for v in matched if v is not None]
    slope_counts: dict[float, int] = {}
    for value in slopes: slope_counts[value] = slope_counts.get(value, 0) + 1
    self_loops = sum(is_self_loop(float(r)) for r in ratios.tolist())
    base.update({
        "label": label,
        "reason": best["reason"],
        "suspicious_reason": best["reason"] if label == "suspicious" else "",
        "valid_rr_count": int(rr.size),
        "ray_match_count": int(len(special_indices)),
        "ray_match_ratio": round(float(len(special_indices) / max(ratios.size, 1)), 6),
        "max_run_length": int(max((x["bridged_run"] for x in local_results), default=0)),
        "mean_run_length": round(float(np.mean([x["bridged_run"] for x in local_results])) if local_results else 0.0, 6),
        "run_count": int(len(local_results)),
        "isolated_ratio": round(float(sum(x["special_count"] <= 1 for x in local_results) / max(len(local_results), 1)), 6),
        "matched_slopes": [float(k) for k in sorted(slope_counts.keys())],
        "candidate_rr_centers_ms": best["centers"],
        "candidate_rr_center_count": int(len(best["centers"])),
        "self_loop_ratio": round(float(self_loops / max(ratios.size, 1)), 6),
        "reciprocal_alternation_ratio": round(float(max((x["reciprocal_ratio"] for x in local_results), default=0.0)), 6),
        "afl_feature_point_count": int(sum(x["special_count"] for x in afl_hits)),
        "ectopy_feature_point_count": int(sum(x["special_count"] for x in ectopy_hits)),
        "local_hit_ranges": [(int(x["range_start"]), int(x["range_end"])) for x in local_results if x["label"] in {"afl", "suspicious"}],
    })
    return base


def mask_segments(series: BeatSeries, mask: np.ndarray) -> list[dict[str, Any]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0: return []
    segments, start, prev = [], int(indices[0]), int(indices[0])
    for raw in indices[1:]:
        index = int(raw)
        if int(series.offsets_ms[index]) - int(series.offsets_ms[prev]) <= EVENT_GAP_MS: prev = index
        else:
            segments.append({"start_index": start, "end_index": prev, "start_ms": int(series.offsets_ms[start]), "end_ms": int(series.offsets_ms[prev])})
            start = prev = index
    segments.append({"start_index": start, "end_index": prev, "start_ms": int(series.offsets_ms[start]), "end_ms": int(series.offsets_ms[prev])})
    return [s for s in segments if int(s["end_ms"]) - int(s["start_ms"]) >= FINAL_MIN_DURATION_MS]


def classify_series(series: BeatSeries, window_seconds: int, step_seconds: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = build_local_candidates(series)
    candidate_segments = merge_candidate_segments(candidates)
    rows: list[dict[str, Any]] = []
    votes = np.zeros(series.rr_ms.size, dtype=np.int32)
    window_ms, step_ms = window_seconds * 1000, step_seconds * 1000
    duration_ms = int(series.offsets_ms[-1]) + window_ms if series.offsets_ms.size else 0
    for window_index, (start_ms, end_ms) in enumerate(make_windows(duration_ms, window_ms, step_ms)):
        overlapping = [seg for seg in candidate_segments if int(seg["start_ms"]) < end_ms and int(seg["end_ms"]) > start_ms and seg["label"] != "non_afl"]
        afl_segments = [seg for seg in overlapping if seg["label"] == "afl"]
        suspicious_segments = [seg for seg in overlapping if seg["label"] == "suspicious"]
        if afl_segments and not suspicious_segments:
            label = "afl"
            best = max(afl_segments, key=lambda item: int(item["candidate_count"]))
        elif afl_segments or suspicious_segments:
            label = "suspicious"
            best = max(overlapping, key=lambda item: int(item["candidate_count"]))
        else:
            label = "non_afl"
            best = {"reason": "no_candidate_segment", "candidate_count": 0, "slope_type_count": 0, "candidate_rr_centers_ms": [], "mean_reciprocal_ratio": 0.0, "afl_local_count": 0, "ectopy_local_count": 0}
        rows.append({
            "window_index": window_index,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_seconds": window_seconds,
            "label": label,
            "final_label": label,
            "final_event_index": "",
            "reason": best["reason"],
            "suspicious_reason": best["reason"] if label == "suspicious" else "",
            "valid_rr_count": int(np.sum((series.offsets_ms >= start_ms) & (series.offsets_ms < end_ms))),
            "ray_match_count": int(best.get("candidate_count", 0)),
            "ray_match_ratio": 0.0,
            "max_run_length": int(best.get("candidate_count", 0)),
            "mean_run_length": 0.0,
            "run_count": int(len(overlapping)),
            "isolated_ratio": 0.0,
            "matched_slopes": [],
            "candidate_rr_centers_ms": best.get("candidate_rr_centers_ms", []),
            "candidate_rr_center_count": len(best.get("candidate_rr_centers_ms", [])),
            "self_loop_ratio": 0.0,
            "reciprocal_alternation_ratio": float(best.get("mean_reciprocal_ratio", 0.0)),
            "afl_feature_point_count": int(best.get("afl_local_count", 0)),
            "ectopy_feature_point_count": int(best.get("ectopy_local_count", 0)),
        })
    final_segments = [seg for seg in candidate_segments if seg["label"] in {"afl", "suspicious"}]
    for event_index, segment in enumerate(final_segments, start=1):
        segment["event_index"] = event_index
        segment["window_count"] = sum(int(row["start_ms"]) < int(segment["end_ms"]) and int(row["end_ms"]) > int(segment["start_ms"]) for row in rows)
        segment["label_counts"] = {"afl": int(segment["label"] == "afl"), "suspicious": int(segment["label"] == "suspicious"), "non_afl": 0}
        for row in rows:
            if int(row["start_ms"]) < int(segment["end_ms"]) and int(row["end_ms"]) > int(segment["start_ms"]):
                row["final_event_index"] = event_index
    return rows, final_segments


def config(window_seconds: int = WINDOW_SECONDS, step_seconds: int = STEP_SECONDS) -> dict[str, Any]:
    return {"window_seconds": window_seconds, "step_seconds": step_seconds, "rr_min_ms": RR_MIN_MS, "rr_max_ms": RR_MAX_MS, "min_valid_rr": MIN_VALID_RR, "local_radius": LOCAL_RADIUS, "special_ratios": SPECIAL_RATIOS, "ratio_log_tol": RATIO_LOG_TOL, "max_bridge_gap": MAX_BRIDGE_GAP, "event_gap_ms": EVENT_GAP_MS, "final_min_duration_ms": FINAL_MIN_DURATION_MS}


def segments_to_events(segments: list[dict[str, Any]], window_seconds: int, step_seconds: int) -> list[dict[str, Any]]:
    events = []
    for index, segment in enumerate(segments, start=1):
        label = str(segment.get("label", "suspicious"))
        subtype = "flutter" if label == "afl" else "suspicious_flutter_like"
        start_ms, end_ms = int(segment["start_ms"]), int(segment["end_ms"])
        events.append({"type": "af_family", "subtype": subtype, "layer": "rr2d_afl_structure_filter", "rule": "rr2d_local_special_point_structure", "event_index": int(segment.get("event_index", index)), "t0_ms": start_ms, "t1_ms": end_ms, "time": f"{start_ms} ms ~ {end_ms} ms", "duration": format_hms((end_ms - start_ms) / 1000.0), "stats": {"window_count": int(segment.get("window_count", 0)), "label_counts": segment.get("label_counts", {}), "config": config(window_seconds, step_seconds)}})
    return events


def _csv_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False) if isinstance(value, (list, tuple, dict)) else "" if value is None else str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["window_index", "start_ms", "end_ms", "duration_seconds", "label", "final_label", "final_event_index", "reason", "suspicious_reason", "valid_rr_count", "ray_match_count", "ray_match_ratio", "max_run_length", "mean_run_length", "run_count", "isolated_ratio", "matched_slopes", "candidate_rr_centers_ms", "candidate_rr_center_count", "self_loop_ratio", "reciprocal_alternation_ratio", "afl_feature_point_count", "ectopy_feature_point_count"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: _csv_value(row.get(field, "")) for field in fields} for row in rows)


def summarize(rows: list[dict[str, Any]], series: BeatSeries, segments: list[dict[str, Any]], window_seconds: int, step_seconds: int) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("label", "non_afl"))
        counts[label] = counts.get(label, 0) + 1
    return {"source_csv": str(series.source_csv), "window_count": len(rows), "label_counts": counts, "event_count": len(segments), "events": segments, "config": config(window_seconds, step_seconds)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RR2D local-structure AFL filter.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-events-json", default="")
    parser.add_argument("--window-seconds", type=int, default=WINDOW_SECONDS)
    parser.add_argument("--step-seconds", type=int, default=STEP_SECONDS)
    return parser.parse_args()


def default_output_root(csv_path: Path) -> Path:
    parts = csv_path.parts
    for index, part in enumerate(parts):
        if part == "out" and index + 1 < len(parts): return Path(*parts[: index + 2]) / "rr_afl_filter_rr2d"
    return Path("out") / csv_path.stem / "rr_afl_filter_rr2d"


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    if not csv_path.exists(): raise SystemExit(f"CSV not found: {csv_path}")
    series = load_beat_series(csv_path)
    rows, segments = classify_series(series, args.window_seconds, args.step_seconds)
    out_root = default_output_root(csv_path)
    out_csv = Path(args.out_csv).resolve() if args.out_csv else out_root / f"{csv_path.stem}_rr2d_afl_windows.csv"
    out_json = Path(args.out_json).resolve() if args.out_json else out_root / f"{csv_path.stem}_rr2d_afl_summary.json"
    out_events_json = Path(args.out_events_json).resolve() if args.out_events_json else out_root / f"{csv_path.stem}_rr2d_afl_events.json"
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
