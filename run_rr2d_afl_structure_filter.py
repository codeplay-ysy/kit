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
SPECIAL_RATIOS = [4 / 3, 1.5, 2.0, 3.0, 4.0]
RATIO_LOG_TOL = 0.07
SELF_LOOP_LOG_TOL = 0.08
VALUE_BIN_MS = 50
VALUE_MERGE_MS = 80
MAX_VALUE_CLUSTER_COUNT = 6
EVENT_GAP_MS = 10_000
FINAL_MIN_DURATION_MS = 30_000
COARSE_WINDOW_SECONDS = 90
COARSE_STEP_SECONDS = 15
COARSE_MIN_VALID_TRANSITIONS = 12
COARSE_MIN_ALL_SPECIAL_DENSITY = 0.50
COARSE_MIN_JUMP_SPECIAL_DENSITY = 0.08
COARSE_MIN_JUMP_SHARE_WITHIN_SPECIAL = 0.10
COARSE_MIN_JUMP_DOMINANT_RATIO_FRACTION = 0.45
COARSE_MAX_NOISE_FRACTION = 0.22
COARSE_MAX_RR_CENTER_COUNT = 6
SEGMENT_BRIDGE_GAP_MS = 20_000
SEGMENT_MIN_TRANSITIONS = 10
SEGMENT_MIN_ALL_SPECIAL_DENSITY = 0.65
SEGMENT_MIN_JUMP_SPECIAL_DENSITY = 0.12
SEGMENT_MIN_JUMP_SHARE_WITHIN_SPECIAL = 0.18
SEGMENT_MIN_JUMP_DOMINANT_RATIO_FRACTION = 0.55
SEGMENT_MAX_NOISE_FRACTION = 0.25
SEGMENT_MAX_AFL_SLOPE_TYPES = 4
ECTOPY_RECIPROCAL_TOL = 0.10
ECTOPY_STRONG_RATIO = 0.72
SEGMENT_SUSPICIOUS_RECIPROCAL_RATIO = 0.55


@dataclass(slots=True)
class BeatSeries:
    offsets_ms: np.ndarray
    rr_ms: np.ndarray
    source_csv: Path


@dataclass(slots=True)
class TransitionSeries:
    start_ms: np.ndarray
    end_ms: np.ndarray
    rr0_ms: np.ndarray
    rr1_ms: np.ndarray
    ratio: np.ndarray
    matched_ratio: np.ndarray
    is_all_special: np.ndarray
    is_jump_special: np.ndarray
    is_self_loop: np.ndarray
    is_noise: np.ndarray
    valid: np.ndarray


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
        rows = list(reader)
    if not rows or (offset_key is None and rr_key is None):
        raise ValueError(f"Bad CSV: {csv_path}")
    offsets, rr_values = [], []
    for i, row in enumerate(rows):
        offset = _int_like(row[offset_key]) if offset_key else None
        rr = _int_like(row[rr_key]) if rr_key else None
        if offset is None:
            offset = 0 if i == 0 else (_ for _ in ()).throw(ValueError(f"Missing offset row {i + 1}"))
        offsets.append(offset)
        if rr is None:
            rr = offsets[i] - offsets[i - 1] if i > 0 else 0
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
    if ratio <= 0:
        return None
    normalized = ratio if ratio >= 1.0 else 1.0 / ratio
    diffs = [abs(np.log(normalized / target)) for target in SPECIAL_RATIOS]
    best = int(np.argmin(diffs))
    if diffs[best] > RATIO_LOG_TOL:
        return None
    matched = float(SPECIAL_RATIOS[best])
    return matched if ratio >= 1.0 else round(1.0 / matched, 6)


def is_self_loop(ratio: float) -> bool:
    return ratio > 0 and abs(np.log(ratio)) <= SELF_LOOP_LOG_TOL


def reciprocal_ratio(slopes: list[float]) -> float:
    if len(slopes) < 2:
        return 0.0
    hits = sum(abs(a * b - 1.0) <= ECTOPY_RECIPROCAL_TOL for a, b in zip(slopes[:-1], slopes[1:], strict=False))
    return float(hits / max(len(slopes) - 1, 1))


def value_centers(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return []
    bins = np.round(values / VALUE_BIN_MS).astype(np.int32)
    raw = [(float(np.mean(values[bins == b])), int(np.sum(bins == b))) for b in sorted(set(bins.tolist()))]
    merged, current = [], [raw[0]]
    for center, count in raw[1:]:
        cur = sum(v * c for v, c in current) / max(sum(c for _, c in current), 1)
        if abs(center - cur) <= VALUE_MERGE_MS:
            current.append((center, count))
        else:
            total = sum(c for _, c in current)
            merged.append((sum(v * c for v, c in current) / max(total, 1), total))
            current = [(center, count)]
    total = sum(c for _, c in current)
    merged.append((sum(v * c for v, c in current) / max(total, 1), total))
    return [round(c, 3) for c, _ in sorted(merged, key=lambda x: x[1], reverse=True)[:MAX_VALUE_CLUSTER_COUNT]]


def build_transitions(series: BeatSeries) -> TransitionSeries:
    rr = np.asarray(series.rr_ms, dtype=np.float64)
    size = max(rr.size - 1, 0)
    start_ms = np.asarray(series.offsets_ms[:-1], dtype=np.int64) if size else np.asarray([], dtype=np.int64)
    end_ms = np.asarray(series.offsets_ms[1:], dtype=np.int64) if size else np.asarray([], dtype=np.int64)
    rr0_ms = rr[:-1] if size else np.asarray([], dtype=np.float64)
    rr1_ms = rr[1:] if size else np.asarray([], dtype=np.float64)
    ratio = np.full(size, np.nan, dtype=np.float64)
    matched_ratio = np.full(size, np.nan, dtype=np.float64)
    is_all_special = np.zeros(size, dtype=bool)
    is_jump_special = np.zeros(size, dtype=bool)
    is_self = np.zeros(size, dtype=bool)
    valid = np.isfinite(rr0_ms) & np.isfinite(rr1_ms) & (rr0_ms >= RR_MIN_MS) & (rr0_ms <= RR_MAX_MS) & (rr1_ms >= RR_MIN_MS) & (rr1_ms <= RR_MAX_MS) & (rr0_ms > 0)
    for i in range(size):
        if not valid[i]:
            continue
        current_ratio = float(rr1_ms[i] / rr0_ms[i])
        ratio[i] = current_ratio
        is_self[i] = is_self_loop(current_ratio)
        if is_self[i]:
            matched_ratio[i] = 1.0
            is_all_special[i] = True
            continue
        matched = match_special_ratio(current_ratio)
        if matched is not None:
            matched_ratio[i] = matched
            is_all_special[i] = True
            is_jump_special[i] = True
    is_noise = valid & (~is_all_special)
    return TransitionSeries(start_ms, end_ms, rr0_ms, rr1_ms, ratio, matched_ratio, is_all_special, is_jump_special, is_self, is_noise, valid)


def _transition_mask(transitions: TransitionSeries, start_ms: int, end_ms: int) -> np.ndarray:
    return transitions.valid & (transitions.start_ms < end_ms) & (transitions.end_ms > start_ms)


def _dominant_ratio_fraction(matched_values: np.ndarray) -> tuple[float, list[float]]:
    if matched_values.size == 0:
        return 0.0, []
    normalized = np.array([round(value if value >= 1.0 else 1.0 / value, 3) for value in matched_values.tolist()], dtype=np.float64)
    unique, counts = np.unique(normalized, return_counts=True)
    order = np.argsort(counts)[::-1]
    top_counts = counts[order][: min(2, counts.size)]
    top_values = unique[order][: min(2, counts.size)]
    return float(np.sum(top_counts) / max(np.sum(counts), 1)), [float(v) for v in top_values.tolist()]


def extract_window_features(series: BeatSeries, transitions: TransitionSeries, start_ms: int, end_ms: int) -> dict[str, Any]:
    mask = _transition_mask(transitions, start_ms, end_ms)
    indices = np.flatnonzero(mask)
    valid_count = int(indices.size)
    rr_mask = (series.offsets_ms >= start_ms) & (series.offsets_ms < end_ms)
    rr_values = valid_rr(np.asarray(series.rr_ms[rr_mask], dtype=np.float64))
    centers = value_centers(rr_values)
    if valid_count == 0:
        return {"valid_transition_count": 0, "all_special_count": 0, "jump_special_count": 0, "all_special_density": 0.0, "jump_special_density": 0.0, "jump_share_within_special": 0.0, "all_special_dominant_fraction": 0.0, "all_special_dominant_ratios": [], "jump_special_dominant_fraction": 0.0, "jump_special_dominant_ratios": [], "noise_fraction": 0.0, "self_loop_ratio": 0.0, "reciprocal_ratio": 0.0, "rr_centers": centers, "rr_center_count": len(centers), "matched_slopes": []}
    all_special_mask = transitions.is_all_special[indices]
    jump_special_mask = transitions.is_jump_special[indices]
    self_mask = transitions.is_self_loop[indices]
    noise_mask = transitions.is_noise[indices]
    matched_values = transitions.matched_ratio[indices][all_special_mask]
    matched_values = matched_values[np.isfinite(matched_values)]
    jump_values = transitions.matched_ratio[indices][jump_special_mask]
    jump_values = jump_values[np.isfinite(jump_values)]
    all_slopes = [float(v) for v in matched_values.tolist()]
    jump_slopes = [float(v) for v in jump_values.tolist()]
    all_dominant_fraction, all_dominant_ratios = _dominant_ratio_fraction(matched_values)
    jump_dominant_fraction, jump_dominant_ratios = _dominant_ratio_fraction(jump_values)
    all_special_count = int(np.sum(all_special_mask))
    jump_special_count = int(np.sum(jump_special_mask))
    return {
        "valid_transition_count": valid_count,
        "all_special_count": all_special_count,
        "jump_special_count": jump_special_count,
        "all_special_density": float(all_special_count / max(valid_count, 1)),
        "jump_special_density": float(jump_special_count / max(valid_count, 1)),
        "jump_share_within_special": float(jump_special_count / max(all_special_count, 1)),
        "all_special_dominant_fraction": all_dominant_fraction,
        "all_special_dominant_ratios": all_dominant_ratios,
        "jump_special_dominant_fraction": jump_dominant_fraction,
        "jump_special_dominant_ratios": jump_dominant_ratios,
        "noise_fraction": float(np.sum(noise_mask) / max(valid_count, 1)),
        "self_loop_ratio": float(np.sum(self_mask) / max(valid_count, 1)),
        "reciprocal_ratio": reciprocal_ratio(jump_slopes),
        "rr_centers": centers,
        "rr_center_count": len(centers),
        "matched_slopes": sorted({round(v, 6) for v in all_slopes}),
    }


def classify_coarse_window(features: dict[str, Any]) -> tuple[str, str]:
    if int(features["valid_transition_count"]) < COARSE_MIN_VALID_TRANSITIONS:
        return "background", "insufficient_valid_transitions"
    if float(features["jump_special_density"]) < COARSE_MIN_JUMP_SPECIAL_DENSITY:
        return "background", "low_jump_special_density"
    if float(features["noise_fraction"]) > COARSE_MAX_NOISE_FRACTION:
        return "background", "high_noise_fraction"
    if int(features["rr_center_count"]) > COARSE_MAX_RR_CENTER_COUNT:
        return "background", "too_many_rr_centers"
    if float(features["all_special_density"]) >= COARSE_MIN_ALL_SPECIAL_DENSITY:
        return "structured", "high_all_special_density"
    if float(features["jump_share_within_special"]) >= COARSE_MIN_JUMP_SHARE_WITHIN_SPECIAL and float(features["jump_special_dominant_fraction"]) >= COARSE_MIN_JUMP_DOMINANT_RATIO_FRACTION:
        return "structured", "jump_special_supported_structure"
    return "structured", "jump_special_sparse_but_admitted"


def build_coarse_windows(series: BeatSeries, transitions: TransitionSeries, coarse_window_seconds: int, coarse_step_seconds: int) -> list[dict[str, Any]]:
    coarse_window_ms = coarse_window_seconds * 1000
    coarse_step_ms = coarse_step_seconds * 1000
    duration_ms = int(series.offsets_ms[-1]) + coarse_window_ms if series.offsets_ms.size else 0
    rows: list[dict[str, Any]] = []
    for index, (start_ms, end_ms) in enumerate(make_windows(duration_ms, coarse_window_ms, coarse_step_ms)):
        features = extract_window_features(series, transitions, start_ms, end_ms)
        label, reason = classify_coarse_window(features)
        rows.append({
            "window_index": index,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "label": label,
            "reason": reason,
            **features,
        })
    return rows


def merge_structured_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    structured = [row for row in windows if str(row.get("label")) == "structured"]
    if not structured:
        return []
    structured.sort(key=lambda item: (int(item["start_ms"]), int(item["end_ms"])))
    groups: list[list[dict[str, Any]]] = [[structured[0]]]
    for row in structured[1:]:
        if int(row["start_ms"]) <= int(groups[-1][-1]["end_ms"]):
            groups[-1].append(row)
        else:
            groups.append([row])
    return [
        {
            "start_ms": min(int(item["start_ms"]) for item in group),
            "end_ms": max(int(item["end_ms"]) for item in group),
            "window_count": len(group),
            "mean_all_special_density": round(float(np.mean([float(item["all_special_density"]) for item in group])), 6),
            "mean_jump_special_density": round(float(np.mean([float(item["jump_special_density"]) for item in group])), 6),
            "mean_jump_share_within_special": round(float(np.mean([float(item["jump_share_within_special"]) for item in group])), 6),
            "mean_jump_dominant_ratio_fraction": round(float(np.mean([float(item["jump_special_dominant_fraction"]) for item in group])), 6),
            "mean_noise_fraction": round(float(np.mean([float(item["noise_fraction"]) for item in group])), 6),
            "mean_reciprocal_ratio": round(float(np.mean([float(item["reciprocal_ratio"]) for item in group])), 6),
        }
        for group in groups
    ]


def _segment_indices_from_transition_mask(series: BeatSeries, indices: np.ndarray) -> tuple[int, int] | None:
    if indices.size == 0:
        return None
    return int(indices[0]), min(int(indices[-1]) + 1, series.offsets_ms.size - 1)


def refine_region(series: BeatSeries, transitions: TransitionSeries, region: dict[str, Any]) -> list[dict[str, Any]]:
    region_mask = _transition_mask(transitions, int(region["start_ms"]), int(region["end_ms"]))
    special_indices = np.flatnonzero(region_mask & transitions.is_jump_special)
    if special_indices.size == 0:
        return []
    groups: list[list[int]] = [[int(special_indices[0])]]
    for raw_index in special_indices[1:]:
        index = int(raw_index)
        prev = groups[-1][-1]
        gap_ms = int(transitions.start_ms[index]) - int(transitions.end_ms[prev])
        if gap_ms <= SEGMENT_BRIDGE_GAP_MS:
            groups[-1].append(index)
        else:
            groups.append([index])
    out: list[dict[str, Any]] = []
    for group in groups:
        group_indices = np.asarray(group, dtype=np.int64)
        valid_indices = np.flatnonzero(region_mask & transitions.valid)
        segment_start_ms = int(transitions.start_ms[group_indices[0]])
        segment_end_ms = int(transitions.end_ms[group_indices[-1]])
        local_valid = valid_indices[(transitions.start_ms[valid_indices] >= segment_start_ms) & (transitions.end_ms[valid_indices] <= segment_end_ms)]
        if local_valid.size == 0:
            local_valid = group_indices
        bounds = _segment_indices_from_transition_mask(series, local_valid)
        if bounds is None:
            continue
        start_index, end_index = bounds
        rr_slice = np.asarray(series.rr_ms[start_index:end_index + 1], dtype=np.float64)
        centers = value_centers(valid_rr(rr_slice))
        matched_values = transitions.matched_ratio[group_indices]
        matched_values = matched_values[np.isfinite(matched_values)]
        local_all_special = transitions.is_all_special[local_valid]
        local_jump_special = transitions.is_jump_special[local_valid]
        all_values = transitions.matched_ratio[local_valid][local_all_special]
        all_values = all_values[np.isfinite(all_values)]
        jump_values = transitions.matched_ratio[local_valid][local_jump_special]
        jump_values = jump_values[np.isfinite(jump_values)]
        all_dominant_fraction, all_dominant_ratios = _dominant_ratio_fraction(all_values)
        jump_dominant_fraction, jump_dominant_ratios = _dominant_ratio_fraction(jump_values)
        valid_count = int(local_valid.size)
        jump_special_count = int(group_indices.size)
        all_special_count = int(np.sum(local_all_special))
        slopes = [float(v) for v in jump_values.tolist()]
        out.append({
            "start_index": start_index,
            "end_index": end_index,
            "start_ms": int(series.offsets_ms[start_index]),
            "end_ms": int(series.offsets_ms[end_index]),
            "duration_ms": int(series.offsets_ms[end_index]) - int(series.offsets_ms[start_index]),
            "window_count": int(region.get("window_count", 0)),
            "valid_transition_count": valid_count,
            "all_special_count": all_special_count,
            "jump_special_count": jump_special_count,
            "all_special_density": float(all_special_count / max(valid_count, 1)),
            "jump_special_density": float(jump_special_count / max(valid_count, 1)),
            "jump_share_within_special": float(jump_special_count / max(all_special_count, 1)),
            "all_special_dominant_fraction": all_dominant_fraction,
            "all_special_dominant_ratios": all_dominant_ratios,
            "jump_special_dominant_fraction": jump_dominant_fraction,
            "jump_special_dominant_ratios": jump_dominant_ratios,
            "noise_fraction": float(np.sum(transitions.is_noise[local_valid]) / max(valid_count, 1)),
            "self_loop_ratio": float(np.sum(transitions.is_self_loop[local_valid]) / max(valid_count, 1)),
            "reciprocal_ratio": reciprocal_ratio(slopes),
            "candidate_rr_centers_ms": centers,
            "candidate_rr_center_count": len(centers),
            "slope_type_count": len({round(v if v >= 1.0 else 1.0 / v, 3) for v in slopes}),
            "matched_slopes": sorted({round(v, 6) for v in all_values.tolist()}),
        })
    return out


def classify_segment(segment: dict[str, Any]) -> tuple[str, str]:
    if int(segment["duration_ms"]) < FINAL_MIN_DURATION_MS:
        return "non_afl", "short_segment"
    if int(segment["valid_transition_count"]) < SEGMENT_MIN_TRANSITIONS:
        return "non_afl", "sparse_segment"
    if float(segment["all_special_density"]) < SEGMENT_MIN_ALL_SPECIAL_DENSITY:
        return "non_afl", "low_all_special_density"
    if float(segment["jump_special_density"]) < SEGMENT_MIN_JUMP_SPECIAL_DENSITY:
        return "non_afl", "low_jump_special_density"
    if float(segment["noise_fraction"]) > SEGMENT_MAX_NOISE_FRACTION:
        return "suspicious", "structured_but_noisy"
    if float(segment["jump_share_within_special"]) < SEGMENT_MIN_JUMP_SHARE_WITHIN_SPECIAL:
        return "suspicious", "jump_special_share_too_low"
    if float(segment["jump_special_dominant_fraction"]) < SEGMENT_MIN_JUMP_DOMINANT_RATIO_FRACTION:
        return "suspicious", "jump_ratio_not_dominant"
    if float(segment["reciprocal_ratio"]) >= ECTOPY_STRONG_RATIO:
        return "suspicious", "ectopy_like_reciprocal_pattern"
    if int(segment["candidate_rr_center_count"]) < 2:
        return "suspicious", "too_few_rr_centers_for_afl_structure"
    if int(segment["candidate_rr_center_count"]) > MAX_VALUE_CLUSTER_COUNT:
        return "suspicious", "too_many_rr_centers_for_clear_afl"
    if int(segment["slope_type_count"]) > SEGMENT_MAX_AFL_SLOPE_TYPES:
        return "suspicious", "too_many_special_ratio_types"
    if float(segment["reciprocal_ratio"]) >= SEGMENT_SUSPICIOUS_RECIPROCAL_RATIO:
        return "suspicious", "structured_reciprocal_heavy_pattern"
    return "afl", "segment_repeated_stable_special_ratio_structure"


def build_segments(series: BeatSeries, transitions: TransitionSeries, regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refined: list[dict[str, Any]] = []
    for region in regions:
        refined.extend(refine_region(series, transitions, region))
    if not refined:
        return []
    ordered = sorted(refined, key=lambda item: (int(item["start_ms"]), int(item["end_ms"])))
    merged: list[dict[str, Any]] = [ordered[0].copy()]
    for current in ordered[1:]:
        previous = merged[-1]
        if int(current["start_ms"]) - int(previous["end_ms"]) <= EVENT_GAP_MS:
            previous["start_index"] = min(int(previous["start_index"]), int(current["start_index"]))
            previous["end_index"] = max(int(previous["end_index"]), int(current["end_index"]))
            previous["start_ms"] = min(int(previous["start_ms"]), int(current["start_ms"]))
            previous["end_ms"] = max(int(previous["end_ms"]), int(current["end_ms"]))
            previous["duration_ms"] = int(previous["end_ms"]) - int(previous["start_ms"])
            previous["window_count"] = int(previous["window_count"]) + int(current["window_count"])
            previous["valid_transition_count"] = int(previous["valid_transition_count"]) + int(current["valid_transition_count"])
            previous["all_special_count"] = int(previous["all_special_count"]) + int(current["all_special_count"])
            previous["jump_special_count"] = int(previous["jump_special_count"]) + int(current["jump_special_count"])
            previous["candidate_rr_centers_ms"] = sorted({*previous["candidate_rr_centers_ms"], *current["candidate_rr_centers_ms"]})[:MAX_VALUE_CLUSTER_COUNT]
            previous["candidate_rr_center_count"] = len(previous["candidate_rr_centers_ms"])
            previous["matched_slopes"] = sorted({*previous["matched_slopes"], *current["matched_slopes"]})
            previous["slope_type_count"] = len({round(v if v >= 1.0 else 1.0 / v, 3) for v in previous["matched_slopes"] if abs(v - 1.0) > 1e-6})
            previous["all_special_density"] = float(int(previous["all_special_count"]) / max(int(previous["valid_transition_count"]), 1))
            previous["jump_special_density"] = float(int(previous["jump_special_count"]) / max(int(previous["valid_transition_count"]), 1))
            previous["jump_share_within_special"] = float(int(previous["jump_special_count"]) / max(int(previous["all_special_count"]), 1))
            all_values = np.asarray(previous["matched_slopes"], dtype=np.float64)
            jump_values = all_values[np.abs(all_values - 1.0) > 1e-6]
            previous["all_special_dominant_fraction"], previous["all_special_dominant_ratios"] = _dominant_ratio_fraction(all_values)
            previous["jump_special_dominant_fraction"], previous["jump_special_dominant_ratios"] = _dominant_ratio_fraction(jump_values)
            previous["noise_fraction"] = float(np.mean([float(previous["noise_fraction"]), float(current["noise_fraction"])]))
            previous["self_loop_ratio"] = float(np.mean([float(previous["self_loop_ratio"]), float(current["self_loop_ratio"])]))
            previous["reciprocal_ratio"] = float(np.mean([float(previous["reciprocal_ratio"]), float(current["reciprocal_ratio"])]))
        else:
            merged.append(current.copy())
    final_segments: list[dict[str, Any]] = []
    for segment in merged:
        label, reason = classify_segment(segment)
        segment["label"] = label
        segment["reason"] = reason
        final_segments.append(segment)
    return final_segments


def classify_series(series: BeatSeries, window_seconds: int, step_seconds: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    transitions = build_transitions(series)
    coarse_windows = build_coarse_windows(series, transitions, COARSE_WINDOW_SECONDS, COARSE_STEP_SECONDS)
    regions = merge_structured_windows(coarse_windows)
    all_segments = build_segments(series, transitions, regions)
    final_segments = [segment for segment in all_segments if str(segment.get("label")) in {"afl", "suspicious"}]
    rows: list[dict[str, Any]] = []
    window_ms, step_ms = window_seconds * 1000, step_seconds * 1000
    duration_ms = int(series.offsets_ms[-1]) + window_ms if series.offsets_ms.size else 0
    for window_index, (start_ms, end_ms) in enumerate(make_windows(duration_ms, window_ms, step_ms)):
        overlapping = [seg for seg in final_segments if int(seg["start_ms"]) < end_ms and int(seg["end_ms"]) > start_ms]
        afl_segments = [seg for seg in overlapping if seg["label"] == "afl"]
        suspicious_segments = [seg for seg in overlapping if seg["label"] == "suspicious"]
        if afl_segments:
            label = "afl"
            best = max(afl_segments, key=lambda item: int(item["duration_ms"]))
        elif suspicious_segments:
            label = "suspicious"
            best = max(suspicious_segments, key=lambda item: int(item["duration_ms"]))
        else:
            label = "non_afl"
            best = {"reason": "no_candidate_segment", "all_special_count": 0, "jump_special_count": 0, "all_special_density": 0.0, "jump_special_density": 0.0, "jump_share_within_special": 0.0, "noise_fraction": 0.0, "candidate_rr_centers_ms": [], "candidate_rr_center_count": 0, "self_loop_ratio": 0.0, "reciprocal_ratio": 0.0, "matched_slopes": []}
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
            "ray_match_count": int(best.get("jump_special_count", 0)),
            "ray_match_ratio": round(float(best.get("jump_special_density", 0.0)), 6),
            "max_run_length": int(best.get("jump_special_count", 0)),
            "mean_run_length": 0.0,
            "run_count": int(len(overlapping)),
            "isolated_ratio": round(float(best.get("noise_fraction", 0.0)), 6),
            "matched_slopes": best.get("matched_slopes", []),
            "candidate_rr_centers_ms": best.get("candidate_rr_centers_ms", []),
            "candidate_rr_center_count": int(best.get("candidate_rr_center_count", 0)),
            "self_loop_ratio": round(float(best.get("self_loop_ratio", 0.0)), 6),
            "reciprocal_alternation_ratio": round(float(best.get("reciprocal_ratio", 0.0)), 6),
            "afl_feature_point_count": int(best.get("jump_special_count", 0)) if label == "afl" else 0,
            "ectopy_feature_point_count": int(best.get("jump_special_count", 0)) if label == "suspicious" and "reciprocal" in str(best.get("reason", "")) else 0,
        })
    for event_index, segment in enumerate(final_segments, start=1):
        segment["event_index"] = event_index
        segment["window_count"] = sum(int(row["start_ms"]) < int(segment["end_ms"]) and int(row["end_ms"]) > int(segment["start_ms"]) for row in rows)
        segment["label_counts"] = {"afl": int(segment["label"] == "afl"), "suspicious": int(segment["label"] == "suspicious"), "non_afl": 0}
        for row in rows:
            if int(row["start_ms"]) < int(segment["end_ms"]) and int(row["end_ms"]) > int(segment["start_ms"]):
                row["final_event_index"] = event_index
    return rows, final_segments


def config(window_seconds: int = WINDOW_SECONDS, step_seconds: int = STEP_SECONDS) -> dict[str, Any]:
    return {
        "window_seconds": window_seconds,
        "step_seconds": step_seconds,
        "rr_min_ms": RR_MIN_MS,
        "rr_max_ms": RR_MAX_MS,
        "min_valid_rr": MIN_VALID_RR,
        "special_ratios": SPECIAL_RATIOS,
        "ratio_log_tol": RATIO_LOG_TOL,
        "self_loop_log_tol": SELF_LOOP_LOG_TOL,
        "coarse_window_seconds": COARSE_WINDOW_SECONDS,
        "coarse_step_seconds": COARSE_STEP_SECONDS,
        "coarse_min_valid_transitions": COARSE_MIN_VALID_TRANSITIONS,
        "coarse_min_all_special_density": COARSE_MIN_ALL_SPECIAL_DENSITY,
        "coarse_min_jump_special_density": COARSE_MIN_JUMP_SPECIAL_DENSITY,
        "coarse_min_jump_share_within_special": COARSE_MIN_JUMP_SHARE_WITHIN_SPECIAL,
        "coarse_min_jump_dominant_ratio_fraction": COARSE_MIN_JUMP_DOMINANT_RATIO_FRACTION,
        "coarse_max_noise_fraction": COARSE_MAX_NOISE_FRACTION,
        "segment_min_transitions": SEGMENT_MIN_TRANSITIONS,
        "segment_min_all_special_density": SEGMENT_MIN_ALL_SPECIAL_DENSITY,
        "segment_min_jump_special_density": SEGMENT_MIN_JUMP_SPECIAL_DENSITY,
        "segment_min_jump_share_within_special": SEGMENT_MIN_JUMP_SHARE_WITHIN_SPECIAL,
        "segment_min_jump_dominant_ratio_fraction": SEGMENT_MIN_JUMP_DOMINANT_RATIO_FRACTION,
        "segment_max_noise_fraction": SEGMENT_MAX_NOISE_FRACTION,
        "event_gap_ms": EVENT_GAP_MS,
        "final_min_duration_ms": FINAL_MIN_DURATION_MS,
        "ectopy_strong_ratio": ECTOPY_STRONG_RATIO,
    }


def segments_to_events(segments: list[dict[str, Any]], window_seconds: int, step_seconds: int) -> list[dict[str, Any]]:
    events = []
    for index, segment in enumerate(segments, start=1):
        label = str(segment.get("label", "suspicious"))
        subtype = "flutter" if label == "afl" else "suspicious_flutter_like"
        start_ms, end_ms = int(segment["start_ms"]), int(segment["end_ms"])
        events.append({
            "type": "af_family",
            "subtype": subtype,
            "layer": "rr2d_afl_structure_filter",
            "rule": "rr2d_segmented_special_ratio_structure",
            "event_index": int(segment.get("event_index", index)),
            "t0_ms": start_ms,
            "t1_ms": end_ms,
            "time": f"{start_ms} ms ~ {end_ms} ms",
            "duration": format_hms((end_ms - start_ms) / 1000.0),
            "stats": {"window_count": int(segment.get("window_count", 0)), "label_counts": segment.get("label_counts", {}), "reason": segment.get("reason", ""), "config": config(window_seconds, step_seconds)},
        })
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
    parser = argparse.ArgumentParser(description="RR2D segmented-structure AFL filter.")
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
        if part == "out" and index + 1 < len(parts):
            return Path(*parts[: index + 2]) / "rr_afl_filter_rr2d"
    return Path("out") / csv_path.stem / "rr_afl_filter_rr2d"


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
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
