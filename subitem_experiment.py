from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from edf_reader import EdfReader


LEAD_CHANNELS = {
    "V5": "ECG_CH1",
    "V1": "ECG_CH3",
}

PRE_MS = 40
POST_MS = 80
RR_LOCAL_SPAN = 10
CONTEXT_SIDE = 10
PEAK_SEARCH_MS = 10
EARLY_RATIO = 0.85
ESCAPE_RATIO = 1.15
SUPRA_CORR_THRESHOLD = 0.90
VENT_CORR_THRESHOLD = 0.80
VENT_ENERGY_RATIO_THRESHOLD = 1.20


def _time_only(timestamp: str) -> str:
    return timestamp.split(" ", 1)[1] if " " in timestamp else timestamp


@dataclass(slots=True)
class RhythmInterval:
    start_ms: float
    end_ms: float
    subtype: str
    layer: str
    rule: str


def _load_af_intervals(path: str | Path | None) -> list[RhythmInterval]:
    if not path:
        return []
    events_path = Path(path)
    if not events_path.exists():
        raise SystemExit(f"AF events JSON not found: {events_path}")
    events = json.loads(events_path.read_text(encoding="utf-8"))
    if not isinstance(events, list):
        raise SystemExit(f"AF events JSON must contain a list: {events_path}")
    intervals: list[RhythmInterval] = []
    for event in events:
        if not isinstance(event, dict) or "t0_ms" not in event or "t1_ms" not in event:
            continue
        start_ms = float(event["t0_ms"])
        end_ms = float(event["t1_ms"])
        if end_ms > start_ms:
            intervals.append(
                RhythmInterval(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    subtype=str(event.get("subtype") or "fibrillation"),
                    layer=str(event.get("layer") or "unknown"),
                    rule=str(event.get("rule") or "unknown"),
                )
            )
    return sorted(intervals, key=lambda item: (item.start_ms, item.end_ms))


def _mark_af_beats(offsets_ms: list[float], intervals: list[RhythmInterval]) -> list[RhythmInterval | None]:
    rhythm_marks: list[RhythmInterval | None] = [None] * len(offsets_ms)
    interval_index = 0
    for beat_index, offset_ms in enumerate(offsets_ms):
        while interval_index < len(intervals) and intervals[interval_index].end_ms < offset_ms:
            interval_index += 1
        if interval_index < len(intervals):
            interval = intervals[interval_index]
            if interval.start_ms <= offset_ms <= interval.end_ms:
                rhythm_marks[beat_index] = interval
    return rhythm_marks


@dataclass(slots=True)
class BeatRow:
    beat_order: int
    merged_milliseconds: float
    merged_timestamp: str
    source: str
    primary_lead: str
    rr_prev_ms: int
    rr_local_ms: int
    rr_ratio: float | None
    rr_raw_state: str
    rr_state: str
    suppressed_by_prev_early: bool
    corr_v1: float | None = None
    corr_v5: float | None = None
    corr_avg: float | None = None
    corr_min: float | None = None
    energy_v1: float | None = None
    energy_v5: float | None = None
    energy_ratio_v1: float | None = None
    energy_ratio_v5: float | None = None
    energy_ratio_max: float | None = None
    symbol: str = "U"
    note: str = ""

    def to_row(self) -> dict[str, str]:
        return {
            "beat_order": str(self.beat_order),
            "merged_milliseconds": f"{self.merged_milliseconds:.1f}",
            "merged_timestamp": _time_only(self.merged_timestamp),
            "symbol": self.symbol,
            "source": self.source,
            "primary_lead": self.primary_lead,
            "rr_prev_ms": str(self.rr_prev_ms),
            "rr_local_ms": str(self.rr_local_ms),
            "rr_ratio": "" if self.rr_ratio is None else f"{self.rr_ratio:.4f}",
            "rr_raw_state": self.rr_raw_state,
            "rr_state": self.rr_state,
            "suppressed_by_prev_early": "yes" if self.suppressed_by_prev_early else "no",
            "corr_v1": "" if self.corr_v1 is None else f"{self.corr_v1:.4f}",
            "corr_v5": "" if self.corr_v5 is None else f"{self.corr_v5:.4f}",
            "corr_avg": "" if self.corr_avg is None else f"{self.corr_avg:.4f}",
            "corr_min": "" if self.corr_min is None else f"{self.corr_min:.4f}",
            "energy_v1": "" if self.energy_v1 is None else f"{self.energy_v1:.4f}",
            "energy_v5": "" if self.energy_v5 is None else f"{self.energy_v5:.4f}",
            "energy_ratio_v1": "" if self.energy_ratio_v1 is None else f"{self.energy_ratio_v1:.4f}",
            "energy_ratio_v5": "" if self.energy_ratio_v5 is None else f"{self.energy_ratio_v5:.4f}",
            "energy_ratio_max": "" if self.energy_ratio_max is None else f"{self.energy_ratio_max:.4f}",
            "note": self.note,
        }


@dataclass(slots=True)
class ExperimentResult:
    csv_path: Path
    summary_path: Path
    beats: list[BeatRow]
    summary: dict[str, Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _load_merged_rows(path: str | Path, start_time: Any) -> list[dict[str, Any]]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"beat_order", "merged_milliseconds"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Merged CSV is missing required columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Merged CSV has no rows: {csv_path}")

    def _sort_key(row: dict[str, Any]) -> tuple[int, float]:
        beat_order = int(row.get("beat_order", 0) or 0)
        merged_ms = float(row["merged_milliseconds"])
        return beat_order, merged_ms

    rows.sort(key=_sort_key)
    for index, row in enumerate(rows, start=1):
        row.setdefault("beat_order", str(index))
        row.setdefault("merged_timestamp", "")
        row.setdefault("source", "")
        row.setdefault("primary_lead", "")
        row["beat_order"] = int(row["beat_order"])
        row["merged_milliseconds"] = float(row["merged_milliseconds"])
        if not row["merged_timestamp"]:
            delta_ms = int(round(float(row["merged_milliseconds"])))
            row["merged_timestamp"] = (start_time + timedelta(milliseconds=delta_ms)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return rows


def _load_signals(reader: EdfReader) -> tuple[dict[str, np.ndarray], float]:
    duration_ms = int(round(reader.header.duration_s * 1000))
    windows = reader.read_window(0, duration_ms, labels=[LEAD_CHANNELS["V5"], LEAD_CHANNELS["V1"]])
    signals = {name: np.asarray(window.values, dtype=np.float32) for name, window in zip(("V5", "V1"), windows, strict=True)}
    return signals, float(windows[0].fs_hz)


def _normalize(values: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr - float(np.mean(arr))
    scale = float(np.std(arr))
    if scale < 1e-8:
        return None
    return arr / scale


def _pearson_corr(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None or left.size != right.size or left.size < 2:
        return None
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std < 1e-8 or right_std < 1e-8:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    if np.isnan(value):
        return None
    return value


def _rr_local_from_history(normal_rr_history: list[int], span: int = RR_LOCAL_SPAN) -> int:
    # Only use previous beats that are already confirmed as normal (`N`).
    rr_values = normal_rr_history[-span:]
    if not rr_values:
        return 0
    return int(round(float(np.median(np.asarray(rr_values, dtype=np.float32)))))


def _rr_state(rr_prev_ms: int, rr_local_ms: int) -> str:
    if rr_prev_ms <= 0 or rr_local_ms <= 0:
        return "normal"
    if rr_prev_ms < EARLY_RATIO * rr_local_ms:
        return "early"
    if rr_prev_ms > ESCAPE_RATIO * rr_local_ms:
        return "escape"
    return "normal"


def _template_eligible(raw_states: list[str], index: int) -> bool:
    if raw_states[index] != "normal":
        return False
    if index > 0 and raw_states[index - 1] != "normal":
        return False
    if index + 1 < len(raw_states) and raw_states[index + 1] != "normal":
        return False
    return True


def _segment_from_signal(signal: np.ndarray, fs_hz: float, center_ms: float) -> np.ndarray | None:
    start_idx = int(round((center_ms - PRE_MS) * fs_hz / 1000.0))
    end_idx = int(round((center_ms + POST_MS) * fs_hz / 1000.0))
    if start_idx < 0 or end_idx > signal.size or end_idx <= start_idx:
        return None
    segment = signal[start_idx:end_idx]
    expected = int(round((PRE_MS + POST_MS) * fs_hz / 1000.0))
    if segment.size != expected:
        return None
    return _normalize(segment)


def _window_energy_from_signal(signal: np.ndarray, fs_hz: float, center_ms: float) -> float | None:
    """Return squared energy inside the fixed window around the refined R peak."""
    start_idx = int(round((center_ms - PRE_MS) * fs_hz / 1000.0))
    end_idx = int(round((center_ms + POST_MS) * fs_hz / 1000.0))
    if start_idx < 0 or end_idx > signal.size or end_idx <= start_idx:
        return None
    segment = np.asarray(signal[start_idx:end_idx], dtype=np.float32)
    expected = int(round((PRE_MS + POST_MS) * fs_hz / 1000.0))
    if segment.size != expected:
        return None
    return float(np.sum(np.square(segment)))


def _refine_peak_center_ms(signal: np.ndarray, fs_hz: float, center_ms: float, search_ms: int = PEAK_SEARCH_MS) -> float:
    """Search left/right around the expected R peak and anchor the window on the local extremum.

    We use the largest absolute deflection so the same logic works for upright and inverted leads.
    """
    search_start = int(round((center_ms - search_ms) * fs_hz / 1000.0))
    search_end = int(round((center_ms + search_ms) * fs_hz / 1000.0))
    search_start = max(0, search_start)
    search_end = min(signal.size, search_end)
    if search_end <= search_start:
        return center_ms

    local = np.asarray(signal[search_start:search_end], dtype=np.float32)
    if local.size == 0:
        return center_ms
    peak_offset = int(np.argmax(np.abs(local)))
    peak_sample = search_start + peak_offset
    return peak_sample * 1000.0 / fs_hz


def _build_template(
    segment_getter: Any,
    indices: list[int],
    lead: str,
) -> np.ndarray | None:
    segments = [segment_getter(index, lead) for index in indices]
    segments = [segment for segment in segments if segment is not None]
    if len(segments) < 2:
        return None
    return np.median(np.stack(segments, axis=0), axis=0)


def _build_energy_reference(
    energy_getter: Any,
    indices: list[int],
    lead: str,
) -> float | None:
    energies = [energy_getter(index, lead) for index in indices]
    energies = [energy for energy in energies if energy is not None]
    if len(energies) < 2:
        return None
    return float(np.median(np.asarray(energies, dtype=np.float32)))


def _morph_symbol(rr_state: str, corr_avg: float | None, energy_ratio_max: float | None = None) -> tuple[str, str]:
    if rr_state not in {"early", "escape"} or corr_avg is None:
        return "N", "clear"
    if corr_avg >= SUPRA_CORR_THRESHOLD:
        return ("S", "supraventricular") if rr_state == "early" else ("Se", "supraventricular")
    if corr_avg < VENT_CORR_THRESHOLD:
        if energy_ratio_max is None:
            return "U", "uncertain"
        if energy_ratio_max > VENT_ENERGY_RATIO_THRESHOLD:
            return ("V", "ventricular") if rr_state == "early" else ("Ve", "ventricular")
        return ("S", "supraventricular") if rr_state == "early" else ("Se", "supraventricular")
    return "U", "uncertain"


def _compute_rr_series(offsets_ms: list[float]) -> tuple[list[int], list[int], list[str], list[str], list[bool]]:
    rr_prev_ms: list[int] = []
    rr_local_ms: list[int] = []
    raw_states: list[str] = []
    final_states: list[str] = []
    suppressed_flags: list[bool] = []
    normal_rr_history: list[int] = []

    for index, offset_ms in enumerate(offsets_ms):
        raw_prev_rr = 0 if index == 0 else int(round(offsets_ms[index] - offsets_ms[index - 1]))
        if index > 0 and final_states[index - 1] == "early" and rr_prev_ms:
            prev_rr = int(round((raw_prev_rr + rr_prev_ms[-1]) / 2.0))
            suppressed = True
        else:
            prev_rr = raw_prev_rr
            suppressed = False

        local_rr = _rr_local_from_history(normal_rr_history)
        raw_state = _rr_state(prev_rr, local_rr)
        final_state = "normal" if suppressed else raw_state

        rr_prev_ms.append(prev_rr)
        rr_local_ms.append(local_rr)
        raw_states.append(raw_state)
        final_states.append(final_state)
        suppressed_flags.append(suppressed)
        if final_state == "normal" and prev_rr > 0:
            normal_rr_history.append(prev_rr)

    return rr_prev_ms, rr_local_ms, raw_states, final_states, suppressed_flags


def run_experiment(
    edf_path: str | Path,
    merged_csv_path: str | Path,
    out_csv_path: str | Path,
    out_summary_path: str | Path,
    af_events_json_path: str | Path | None = None,
) -> ExperimentResult:
    reader = EdfReader(edf_path)
    merged_rows = _load_merged_rows(merged_csv_path, reader.header.start_time)
    offsets_ms = [float(row["merged_milliseconds"]) for row in merged_rows]
    af_intervals = _load_af_intervals(af_events_json_path)
    rhythm_marks = _mark_af_beats(offsets_ms, af_intervals)
    rr_prev_ms, rr_local_ms, raw_states, final_states, suppressed_flags = _compute_rr_series(offsets_ms)

    signals, fs_hz = _load_signals(reader)
    eligible_indices = [index for index in range(len(offsets_ms)) if _template_eligible(raw_states, index)]

    @lru_cache(maxsize=None)
    def _refined_center(index: int, lead: str) -> float:
        return _refine_peak_center_ms(signals[lead], fs_hz, offsets_ms[index])

    @lru_cache(maxsize=None)
    def _segment(index: int, lead: str) -> np.ndarray | None:
        return _segment_from_signal(signals[lead], fs_hz, _refined_center(index, lead))

    @lru_cache(maxsize=None)
    def _energy(index: int, lead: str) -> float | None:
        return _window_energy_from_signal(signals[lead], fs_hz, _refined_center(index, lead))

    rows: list[BeatRow] = []
    for index, row in enumerate(merged_rows):
        final_state = final_states[index]
        symbol = "U"
        note = ""
        corr_v1 = corr_v5 = corr_avg = corr_min = None
        energy_v1 = energy_v5 = energy_ratio_v1 = energy_ratio_v5 = energy_ratio_max = None

        rhythm_mark = rhythm_marks[index]
        if rhythm_mark is not None and rhythm_mark.subtype in {"flutter", "fibrillation", "mixed_or_uncertain"}:
            symbol = "AF"
            note = f"af_family_{rhythm_mark.subtype}_event"
        elif rhythm_mark is not None and rhythm_mark.subtype == "suspicious_flutter_like":
            symbol = "N"
            note = "af_family_suspicious_flutter_like_event"
        elif final_state in {"early", "escape"}:
            context_indices = [candidate for candidate in eligible_indices if candidate < index][-CONTEXT_SIDE:]
            if len(context_indices) >= 2:
                templates = {
                    lead: _build_template(_segment, context_indices, lead)
                    for lead in ("V1", "V5")
                }
                energy_refs = {
                    lead: _build_energy_reference(_energy, context_indices, lead)
                    for lead in ("V1", "V5")
                }
                candidate_segments = {
                    lead: _segment(index, lead)
                    for lead in ("V1", "V5")
                }
                if all(template is not None for template in templates.values()) and all(segment is not None for segment in candidate_segments.values()):
                    corr_v1 = _pearson_corr(templates["V1"], candidate_segments["V1"])
                    corr_v5 = _pearson_corr(templates["V5"], candidate_segments["V5"])
                    if corr_v1 is not None and corr_v5 is not None:
                        corr_avg = max(corr_v1, corr_v5)
                        corr_min = min(corr_v1, corr_v5)
                        energy_v1 = _energy(index, "V1")
                        energy_v5 = _energy(index, "V5")
                        if energy_v1 is not None and energy_refs["V1"] not in (None, 0) and energy_v5 is not None and energy_refs["V5"] not in (None, 0):
                            energy_ratio_v1 = energy_v1 / float(energy_refs["V1"])
                            energy_ratio_v5 = energy_v5 / float(energy_refs["V5"])
                            energy_ratio_max = max(energy_ratio_v1, energy_ratio_v5)
                        symbol, note = _morph_symbol(final_state, corr_avg, energy_ratio_max)
                    else:
                        note = "pearson_failed"
                else:
                    note = "template_unavailable"
            else:
                note = "context_too_small"
        else:
            symbol = "N"

        rows.append(
            BeatRow(
                beat_order=int(row["beat_order"]),
                merged_milliseconds=float(row["merged_milliseconds"]),
                merged_timestamp=str(row["merged_timestamp"]),
                source=str(row["source"]),
                primary_lead=str(row["primary_lead"]),
                rr_prev_ms=rr_prev_ms[index],
                rr_local_ms=rr_local_ms[index],
                rr_ratio=(None if rr_local_ms[index] <= 0 else rr_prev_ms[index] / float(rr_local_ms[index])),
                rr_raw_state=raw_states[index],
                rr_state=final_states[index],
                suppressed_by_prev_early=suppressed_flags[index],
                corr_v1=corr_v1,
                corr_v5=corr_v5,
                corr_avg=corr_avg,
                corr_min=corr_min,
                energy_v1=energy_v1,
                energy_v5=energy_v5,
                energy_ratio_v1=energy_ratio_v1,
                energy_ratio_v5=energy_ratio_v5,
                energy_ratio_max=energy_ratio_max,
                symbol=symbol,
                note=note,
            )
        )

    csv_path = Path(out_csv_path).resolve()
    summary_path = Path(out_summary_path).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].to_row().keys()))
        writer.writeheader()
        writer.writerows(row.to_row() for row in rows)

    symbol_counts: dict[str, int] = {}
    rr_state_counts: dict[str, int] = {}
    raw_rr_state_counts: dict[str, int] = {}
    for row in rows:
        symbol_counts[row.symbol] = symbol_counts.get(row.symbol, 0) + 1
        rr_state_counts[row.rr_state] = rr_state_counts.get(row.rr_state, 0) + 1
        raw_rr_state_counts[row.rr_raw_state] = raw_rr_state_counts.get(row.rr_raw_state, 0) + 1

    summary = {
        "input": {
            "edf_path": str(Path(edf_path).resolve()),
            "merged_csv_path": str(Path(merged_csv_path).resolve()),
            "af_events_json_path": str(Path(af_events_json_path).resolve()) if af_events_json_path else "",
        },
        "beat_count": len(rows),
        "af_event_count": len(af_intervals),
        "af_labeled_beat_count": int(sum(mark is not None for mark in rhythm_marks)),
        "af_event_subtype_counts": {
            subtype: sum(interval.subtype == subtype for interval in af_intervals)
            for subtype in sorted({interval.subtype for interval in af_intervals})
        },
        "symbol_counts": symbol_counts,
        "rr_state_counts": rr_state_counts,
        "raw_rr_state_counts": raw_rr_state_counts,
        "suppressed_count": int(sum(suppressed_flags)),
        "eligible_template_beats": len(eligible_indices),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return ExperimentResult(csv_path=csv_path, summary_path=summary_path, beats=rows, summary=summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Beat subitem experiment for PAC/PVC labeling.")
    parser.add_argument("--edf", required=True, help="EDF path.")
    parser.add_argument("--merged-csv", required=True, help="Merged R-peak CSV path.")
    parser.add_argument("--af-events-json", default="", help="Optional AF global events JSON. Beats inside events are labeled AF before subitem classification.")
    parser.add_argument("--out-csv", required=True, help="Output beat-level CSV path.")
    parser.add_argument("--out-summary", required=True, help="Output summary JSON path.")
    args = parser.parse_args(argv)

    result = run_experiment(
        edf_path=args.edf,
        merged_csv_path=args.merged_csv,
        out_csv_path=args.out_csv,
        out_summary_path=args.out_summary,
        af_events_json_path=args.af_events_json,
    )
    print(f"Output CSV: {result.csv_path}")
    print(f"Summary JSON: {result.summary_path}")
    print(f"beats={len(result.beats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
