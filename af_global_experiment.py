from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np


AF_WINDOW_MS = 30_000
AF_WINDOW_STEP_MS = 5_000
AF_FINAL_MIN_DURATION_MS = 30_000
AF_EVENT_GAP_MS = 5_000
AF_MIN_BEATS = 10
AF_MIN_NN_COUNT = 15
AF_GRAY_MIN_NN_COUNT = 50
AF_CV_FAST_NEGATIVE = 0.15
AF_CV_POSITIVE = 0.20
AF_SD1_THRESHOLD_MS = 60.0
AF_SD2_THRESHOLD_MS = 80.0
AF_SD1_SD2_FIB_THRESHOLD = 0.70
AF_SDNN_THRESHOLD_MS = 100.0
AF_RMSSD_THRESHOLD_MS = 80.0
AF_PNN50_THRESHOLD = 0.60
AF_GRAY_SD1_SD2_THRESHOLD = 0.85
AF_GRAY_PNN50_THRESHOLD = 0.50
AF_GRAY_TEMPORAL_HITS_MIN = 1
SQI_SUBWINDOW_MS = 10_000
SQI_KURTOSIS_THRESHOLD = 5.0
SQI_MIN_LEAD_PASSES = 2
SQI_MIN_SUBWINDOW_PASSES = 2
RR_MIN_MS = 300
RR_MAX_MS = 2000
RR_CLUSTER_BIN_MS = 40.0
AFL_CLUSTER_RATIO_THRESHOLD = 0.70
AFL_MULTIPLE_RATIO_THRESHOLD = 0.40
AFL_FORCE_FIB_CLUSTER_MAX = 0.50
AFL_FORCE_FIB_MULTIPLE_MAX = 0.10
AFL_FORCE_FIB_RATIO_MIN = 0.80
AFL_RATIO_MIN = 0.50
NN_LIKE_SYMBOLS = {"N", "AF", "Af"}


@dataclass(slots=True)
class SignalSpec:
    label: str
    phys_min: float
    phys_max: float
    dig_min: int
    dig_max: int
    samples_per_record: int
    record_offset_samples: int

    @property
    def record_offset_bytes(self) -> int:
        return self.record_offset_samples * 2


@dataclass(slots=True)
class EdfHeader:
    path: Path
    start_time: datetime
    duration_s: float
    duration_per_record_s: float
    header_bytes: int
    num_data_records: int
    signals: list[SignalSpec]

    @property
    def record_size_samples(self) -> int:
        return sum(signal.samples_per_record for signal in self.signals)

    @property
    def record_size_bytes(self) -> int:
        return self.record_size_samples * 2

    def signal(self, label: str) -> SignalSpec:
        for signal in self.signals:
            if signal.label == label:
                return signal
        raise KeyError(f"Unknown EDF signal: {label}")


@dataclass(slots=True)
class SignalWindow:
    label: str
    fs_hz: float
    values: np.ndarray


@dataclass(slots=True)
class BeatRecord:
    id: int
    t_ms: int
    symbol: str
    rr_ms: int
    local_hr_bpm: float


@dataclass(slots=True)
class CaseData:
    start_time: datetime
    duration_s: float
    beats: list[BeatRecord]
    beat_times_ms: np.ndarray
    rr_ms: np.ndarray
    local_hr_bpm: np.ndarray
    symbols: np.ndarray
    edf_path: Path

    def time_slice(self, start_ms: int, end_ms: int) -> slice:
        left = int(np.searchsorted(self.beat_times_ms, start_ms, side="left"))
        right = int(np.searchsorted(self.beat_times_ms, end_ms, side="right"))
        return slice(left, right)


@dataclass(slots=True)
class WindowFeature:
    start_ms: int
    end_ms: int
    beat_start: int
    beat_end: int
    subtype: str
    stats: dict[str, float | int | bool | str | None]


@dataclass(slots=True)
class RhythmSpan:
    start_ms: int
    end_ms: int
    windows: list[WindowFeature]


def _parse_ascii_int(blob: bytes) -> int:
    return int(blob.decode("latin-1").strip() or 0)


def _parse_ascii_float(blob: bytes) -> float:
    return float(blob.decode("latin-1").strip() or 0.0)


def _decode_fields(blob: bytes, width: int, count: int) -> list[str]:
    return [blob[index * width : (index + 1) * width].decode("latin-1").strip() for index in range(count)]


def _parse_edf_datetime(date_text: str, time_text: str) -> datetime:
    day, month, year = [int(part) for part in date_text.split(".")]
    hour, minute, second = [int(part) for part in time_text.split(".")]
    year += 2000 if year < 85 else 1900
    return datetime(year, month, day, hour, minute, second)


class EdfReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.header = self._read_header()

    def _read_header(self) -> EdfHeader:
        with self.path.open("rb") as handle:
            fixed_header = handle.read(256)
            signal_count = _parse_ascii_int(fixed_header[252:256])
            signal_blob = handle.read(signal_count * 256)

        start_time = _parse_edf_datetime(
            fixed_header[168:176].decode("latin-1").strip(),
            fixed_header[176:184].decode("latin-1").strip(),
        )
        header_bytes = _parse_ascii_int(fixed_header[184:192])
        num_data_records = _parse_ascii_int(fixed_header[236:244])
        duration_per_record_s = _parse_ascii_float(fixed_header[244:252])

        cursor = 0
        labels = _decode_fields(signal_blob[cursor : cursor + signal_count * 16], 16, signal_count)
        cursor += signal_count * 16
        cursor += signal_count * 80
        cursor += signal_count * 8
        phys_mins = _decode_fields(signal_blob[cursor : cursor + signal_count * 8], 8, signal_count)
        cursor += signal_count * 8
        phys_maxs = _decode_fields(signal_blob[cursor : cursor + signal_count * 8], 8, signal_count)
        cursor += signal_count * 8
        dig_mins = _decode_fields(signal_blob[cursor : cursor + signal_count * 8], 8, signal_count)
        cursor += signal_count * 8
        dig_maxs = _decode_fields(signal_blob[cursor : cursor + signal_count * 8], 8, signal_count)
        cursor += signal_count * 8
        cursor += signal_count * 80
        samples_per_record = _decode_fields(signal_blob[cursor : cursor + signal_count * 8], 8, signal_count)

        signals: list[SignalSpec] = []
        sample_offset = 0
        for index in range(signal_count):
            signals.append(
                SignalSpec(
                    label=labels[index],
                    phys_min=float(phys_mins[index] or 0.0),
                    phys_max=float(phys_maxs[index] or 0.0),
                    dig_min=int(dig_mins[index] or 0),
                    dig_max=int(dig_maxs[index] or 0),
                    samples_per_record=int(samples_per_record[index] or 0),
                    record_offset_samples=sample_offset,
                )
            )
            sample_offset += signals[-1].samples_per_record

        return EdfHeader(
            path=self.path,
            start_time=start_time,
            duration_s=num_data_records * duration_per_record_s,
            duration_per_record_s=duration_per_record_s,
            header_bytes=header_bytes,
            num_data_records=num_data_records,
            signals=signals,
        )

    def ecg_signals(self) -> list[SignalSpec]:
        return [signal for signal in self.header.signals if signal.label.startswith("ECG_")]

    def read_window(self, start_ms: int, end_ms: int, labels: list[str] | None = None) -> list[SignalWindow]:
        return [
            self._read_single_signal(label=label, start_ms=start_ms, end_ms=end_ms)
            for label in (labels or [signal.label for signal in self.ecg_signals()])
        ]

    def _read_single_signal(self, label: str, start_ms: int, end_ms: int) -> SignalWindow:
        signal = self.header.signal(label)
        fs_hz = signal.samples_per_record / self.header.duration_per_record_s
        if end_ms <= start_ms:
            return SignalWindow(label=label, fs_hz=fs_hz, values=np.array([], dtype=np.float32))

        sample_start = max(0, int(np.floor(start_ms * fs_hz / 1000.0)))
        sample_end = max(sample_start, int(np.ceil(end_ms * fs_hz / 1000.0)))
        record_start = sample_start // signal.samples_per_record
        record_end = max(record_start, (sample_end - 1) // signal.samples_per_record)

        chunks: list[np.ndarray] = []
        with self.path.open("rb") as handle:
            for record_index in range(record_start, min(record_end + 1, self.header.num_data_records)):
                byte_offset = (
                    self.header.header_bytes
                    + record_index * self.header.record_size_bytes
                    + signal.record_offset_bytes
                )
                handle.seek(byte_offset)
                raw = handle.read(signal.samples_per_record * 2)
                chunks.append(np.frombuffer(raw, dtype="<i2").copy())

        if not chunks:
            values = np.array([], dtype=np.float32)
        else:
            digital = np.concatenate(chunks)
            local_start = sample_start - record_start * signal.samples_per_record
            local_end = local_start + (sample_end - sample_start)
            digital = digital[local_start:local_end]
            scale = (signal.phys_max - signal.phys_min) / max(signal.dig_max - signal.dig_min, 1)
            values = ((digital - signal.dig_min) * scale + signal.phys_min).astype(np.float32, copy=False)
        return SignalWindow(label=label, fs_hz=fs_hz, values=values)


def load_case(edf_path: Path, csv_path: Path) -> CaseData:
    edf_reader = EdfReader(edf_path)
    rows = read_csv_rows(csv_path)
    offsets = np.array([int(row["offset_ms"]) for row in rows], dtype=np.int64)
    rr_values = parse_rr_values(offsets, [str(row["rr_raw"]) for row in rows])
    local_hr = compute_local_hr(rr_values)
    symbols = np.array([str(row["symbol"]) for row in rows], dtype="<U8")
    beats = [
        BeatRecord(
            id=index + 1,
            t_ms=int(offset),
            symbol=str(symbols[index]),
            rr_ms=int(rr_values[index]),
            local_hr_bpm=float(local_hr[index]),
        )
        for index, offset in enumerate(offsets)
    ]
    return CaseData(
        start_time=edf_reader.header.start_time,
        duration_s=edf_reader.header.duration_s,
        beats=beats,
        beat_times_ms=offsets,
        rr_ms=rr_values,
        local_hr_bpm=local_hr,
        symbols=symbols,
        edf_path=edf_path,
    )


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        fields = {name.strip().lower(): name for name in reader.fieldnames}

        if "time offset(ms)" in fields:
            offset_key = fields["time offset(ms)"]
        elif "merged_milliseconds" in fields:
            offset_key = fields["merged_milliseconds"]
        elif len(reader.fieldnames) >= 3:
            offset_key = reader.fieldnames[2]
        else:
            raise ValueError(f"CSV has no timestamp column: {csv_path}")

        symbol_key = fields.get("beat symbol") or fields.get("symbol")
        rr_key = fields.get("rr interval(ms)")
        rows = [
            {
                "offset_ms": str(int(round(float(row[offset_key].strip())))),
                "symbol": row[symbol_key].strip() if symbol_key else "N",
                "rr_raw": row[rr_key].strip() if rr_key else "-",
            }
            for row in reader
        ]
    if not rows:
        raise ValueError(f"CSV has no beat rows: {csv_path}")
    return rows


def parse_rr_values(offsets: np.ndarray, rr_raw: list[str]) -> np.ndarray:
    rr_values = np.zeros(len(rr_raw), dtype=np.int32)
    for index, raw in enumerate(rr_raw):
        if raw not in {"", "-"}:
            rr_values[index] = int(raw)
        elif index > 0:
            rr_values[index] = int(offsets[index] - offsets[index - 1])
        elif len(offsets) > 1:
            rr_values[index] = int(offsets[index + 1] - offsets[index])
    return rr_values


def compute_local_hr(rr_ms: np.ndarray) -> np.ndarray:
    local_hr = np.zeros_like(rr_ms, dtype=np.float32)
    for index in range(len(rr_ms)):
        left = max(0, index - 2)
        right = min(len(rr_ms), index + 3)
        valid = rr_ms[left:right][rr_ms[left:right] > 0]
        if valid.size:
            local_hr[index] = 60000.0 / float(np.median(valid))
    return local_hr


def sample_std(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.std(values.astype(np.float64), ddof=1))


def sample_var(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.var(values.astype(np.float64), ddof=1))


def build_nn_like_rr(rr_ms: np.ndarray, symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rr_values = rr_ms.astype(np.float64, copy=False)
    range_mask = (rr_values >= RR_MIN_MS) & (rr_values <= RR_MAX_MS)
    if rr_values.size == 0 or symbols.size == 0:
        return rr_values[range_mask], rr_values[range_mask]
    current_allowed = np.isin(symbols, list(NN_LIKE_SYMBOLS))
    previous_allowed = np.zeros_like(current_allowed, dtype=bool)
    if current_allowed.size > 1:
        previous_allowed[1:] = current_allowed[:-1]
    return rr_values[range_mask], rr_values[range_mask & current_allowed & previous_allowed]


def dominant_cluster_ratio(rr_ms: np.ndarray, bin_width_ms: float = RR_CLUSTER_BIN_MS) -> tuple[float, np.ndarray]:
    if rr_ms.size == 0:
        return 0.0, np.array([], dtype=np.int32)
    binned = np.round(rr_ms / bin_width_ms).astype(np.int32)
    bin_counts = Counter(int(item) for item in binned.tolist())
    used_bins: set[int] = set()
    top_coverage = 0
    cluster_labels: dict[int, int] = {}
    next_cluster_id = 0
    for center_bin, _ in bin_counts.most_common():
        if center_bin in used_bins:
            continue
        cluster_bins = {center_bin - 1, center_bin, center_bin + 1}
        available_bins = {bin_id for bin_id in cluster_bins if bin_id not in used_bins}
        top_coverage += sum(bin_counts.get(bin_id, 0) for bin_id in available_bins)
        used_bins.update(cluster_bins)
        for bin_id in available_bins:
            if bin_id in bin_counts:
                cluster_labels[bin_id] = next_cluster_id
        next_cluster_id += 1
        if next_cluster_id >= 3:
            break
    fallback_labels = {bin_id: index for index, bin_id in enumerate(sorted(bin_counts))}
    labels = np.array(
        [cluster_labels.get(int(item), fallback_labels[int(item)] + next_cluster_id) for item in binned.tolist()],
        dtype=np.int32,
    )
    return float(top_coverage / max(labels.size, 1)), labels


def multiple_ratio(rr_ms: np.ndarray) -> float:
    if rr_ms.size < 2:
        return 0.0
    hits = 0
    pair_count = 0
    for left, right in zip(rr_ms[:-1], rr_ms[1:], strict=False):
        smaller = min(left, right)
        larger = max(left, right)
        if smaller <= 0:
            continue
        pair_count += 1
        ratio = larger / smaller
        if 1.85 <= ratio <= 2.15 or 2.80 <= ratio <= 3.20:
            hits += 1
    return float(hits / max(pair_count, 1))


def periodic_coverage(labels: np.ndarray, template_length: int) -> float:
    if labels.size < template_length * 2:
        return 0.0
    best = 0
    for start in range(labels.size - template_length + 1):
        template = labels[start : start + template_length]
        cursor = start
        matched = 0
        while cursor + template_length <= labels.size and np.array_equal(labels[cursor : cursor + template_length], template):
            matched += template_length
            cursor += template_length
        best = max(best, matched)
    return float(best / max(labels.size, 1))


def compute_rr_features(rr_ms: np.ndarray, symbols: np.ndarray) -> dict[str, Any]:
    raw_rr, nn_like_rr = build_nn_like_rr(rr_ms, symbols)
    if raw_rr.size == 0:
        mean_rr = sdnn_ms = rmssd_ms = pnn50_ratio = prr31 = sd1_ms = sd2_ms = 0.0
        rr_cluster_ratio = rr_multiple_ratio = 0.0
        rr_periodic = False
    else:
        diffs = np.diff(nn_like_rr)
        mean_rr = float(np.mean(nn_like_rr.astype(np.float64))) if nn_like_rr.size else 0.0
        sdnn_ms = sample_std(nn_like_rr)
        rmssd_ms = float(np.sqrt(np.mean(np.square(diffs)))) if diffs.size else 0.0
        pnn50_ratio = float(np.mean(np.abs(diffs) > 50.0)) if diffs.size else 0.0
        prr31 = float(np.mean(np.abs(diffs) >= 31.0)) if diffs.size else 0.0
        sd1_ms = rmssd_ms / np.sqrt(2.0) if diffs.size else 0.0
        sd2_ms = float(np.sqrt(max(0.0, 2.0 * sample_var(nn_like_rr) - 0.5 * sample_var(diffs)))) if nn_like_rr.size >= 2 else 0.0
        rr_cluster_ratio, cluster_labels = dominant_cluster_ratio(nn_like_rr)
        rr_multiple_ratio = multiple_ratio(nn_like_rr)
        rr_periodic = periodic_coverage(cluster_labels, 2) >= 0.40 or periodic_coverage(cluster_labels, 3) >= 0.40

    return {
        "mean_rr": mean_rr,
        "mean_hr": 60000.0 / mean_rr if mean_rr else 0.0,
        "rr_std": sdnn_ms,
        "cv": sdnn_ms / mean_rr if mean_rr else 0.0,
        "rmssd_ratio": rmssd_ms / mean_rr if mean_rr else 0.0,
        "pnn50": pnn50_ratio,
        "prr31": prr31,
        "sd1": sd1_ms,
        "sd2": sd2_ms,
        "sd1_meanrr": sd1_ms / mean_rr if mean_rr else 0.0,
        "sd1_sd2": sd1_ms / sd2_ms if sd2_ms else 0.0,
        "sdnn_ms": sdnn_ms,
        "rmssd_ms": rmssd_ms,
        "pnn50_ratio": pnn50_ratio,
        "sd1_ms": sd1_ms,
        "sd2_ms": sd2_ms,
        "nn_count": int(nn_like_rr.size),
        "raw_rr_count": int(raw_rr.size),
        "kurtosis_sqi": None,
        "sqi_pass": None,
        "rr_cluster_ratio": rr_cluster_ratio,
        "rr_multiple_ratio": rr_multiple_ratio,
        "rr_periodic": rr_periodic,
        "af_stage": "rr_only" if raw_rr.size else "insufficient_rr",
        "pac_density": float(np.mean(symbols == "S")) if symbols.size else 0.0,
        "pvc_density": float(np.mean(symbols == "V")) if symbols.size else 0.0,
        "noise_ratio": float(np.mean(symbols == "X")) if symbols.size else 0.0,
        "beat_count": float(symbols.size),
    }


def signal_kurtosis(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    centered = values.astype(np.float64) - float(np.mean(values))
    std = float(np.std(centered))
    if std == 0.0:
        return 0.0
    return float(np.mean(np.power(centered / std, 4)))


class AFGlobalExperiment:
    def __init__(self, ignore_sqi: bool = False) -> None:
        self.ignore_sqi = ignore_sqi
        self.sqi_cache: dict[tuple[int, int], tuple[float | None, bool | None]] = {}
        self.edf_reader: EdfReader | None = None

    def run(self, case: CaseData) -> tuple[list[dict[str, Any]], list[WindowFeature]]:
        self.sqi_cache = {}
        self.edf_reader = None
        positive_windows = self.scan_af_windows(case)
        events = self.build_events(case, positive_windows)
        return events, positive_windows

    def scan_af_windows(self, case: CaseData) -> list[WindowFeature]:
        if not case.beat_times_ms.size:
            return []
        last_start = max(int(case.beat_times_ms[-1] - AF_WINDOW_MS), 0)
        windows: list[WindowFeature] = []
        for start_ms in range(0, last_start + AF_WINDOW_STEP_MS, AF_WINDOW_STEP_MS):
            end_ms = start_ms + AF_WINDOW_MS
            beat_slice = case.time_slice(start_ms, end_ms)
            if beat_slice.stop - beat_slice.start < AF_MIN_BEATS:
                continue
            stats = compute_rr_features(case.rr_ms[beat_slice], case.symbols[beat_slice])
            stats.update(self.window_sqi(case, start_ms, end_ms, stats))
            stats.update(classify_af_window(stats))
            if stats["af_stage"] != "confirmed_positive":
                continue
            windows.append(
                WindowFeature(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    beat_start=beat_slice.start,
                    beat_end=max(beat_slice.start, beat_slice.stop - 1),
                    subtype=str(stats["subtype"]),
                    stats=stats,
                )
            )
        return windows

    def window_sqi(self, case: CaseData, start_ms: int, end_ms: int, stats: dict[str, Any]) -> dict[str, Any]:
        if float(stats["cv"]) < AF_CV_FAST_NEGATIVE or int(stats["nn_count"]) < AF_MIN_NN_COUNT:
            return {"kurtosis_sqi": None, "sqi_pass": None}
        if self.ignore_sqi:
            return {"kurtosis_sqi": 1.0, "sqi_pass": True}
        return dict(zip(["kurtosis_sqi", "sqi_pass"], self.evaluate_window_sqi(case, start_ms, end_ms), strict=True))

    def evaluate_window_sqi(self, case: CaseData, start_ms: int, end_ms: int) -> tuple[float | None, bool | None]:
        cache_key = (start_ms, end_ms)
        if cache_key in self.sqi_cache:
            return self.sqi_cache[cache_key]
        if self.edf_reader is None:
            self.edf_reader = EdfReader(case.edf_path)
        passed_subwindows = 0
        total_subwindows = 0
        for subwindow_start in range(start_ms, end_ms, SQI_SUBWINDOW_MS):
            subwindow_end = min(end_ms, subwindow_start + SQI_SUBWINDOW_MS)
            if subwindow_end <= subwindow_start:
                continue
            total_subwindows += 1
            lead_passes = sum(
                signal_kurtosis(window.values) > SQI_KURTOSIS_THRESHOLD
                for window in self.edf_reader.read_window(subwindow_start, subwindow_end)
            )
            if lead_passes >= SQI_MIN_LEAD_PASSES:
                passed_subwindows += 1
        result = (None, None) if total_subwindows == 0 else (
            float(passed_subwindows / total_subwindows),
            passed_subwindows >= SQI_MIN_SUBWINDOW_PASSES,
        )
        self.sqi_cache[cache_key] = result
        return result

    def build_events(self, case: CaseData, windows: list[WindowFeature]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for span in merge_positive_windows(windows):
            if span.end_ms - span.start_ms < AF_FINAL_MIN_DURATION_MS:
                continue
            beat_slice = case.time_slice(span.start_ms, span.end_ms)
            if beat_slice.stop - beat_slice.start < AF_MIN_BEATS:
                continue
            stats = compute_rr_features(case.rr_ms[beat_slice], case.symbols[beat_slice])
            subtype = event_subtype(span.windows, stats)
            stats.update(event_stats(span, stats))
            beat0 = case.beats[max(0, beat_slice.start)].id
            beat1 = case.beats[max(0, beat_slice.stop - 1)].id
            events.append(
                {
                    "type": "af_family",
                    "subtype": subtype,
                    "layer": "global",
                    "rule": "af_hrv_lorenz_afl_rules",
                    "t0_ms": span.start_ms,
                    "t1_ms": span.end_ms,
                    "beat0": beat0,
                    "beat1": beat1,
                    "time": format_time_range(case, span.start_ms, span.end_ms),
                    "duration": format_hms((span.end_ms - span.start_ms) / 1000.0),
                    "stats": compact_stats(stats),
                }
            )
        return events


def classify_af_window(stats: dict[str, Any]) -> dict[str, Any]:
    cv = float(stats["cv"])
    nn_count = int(stats["nn_count"])
    sd1_sd2 = float(stats["sd1_sd2"])
    temporal_hits = sum(
        [
            float(stats["sdnn_ms"]) > AF_SDNN_THRESHOLD_MS,
            float(stats["rmssd_ms"]) > AF_RMSSD_THRESHOLD_MS,
            float(stats["pnn50_ratio"]) > AF_PNN50_THRESHOLD,
        ]
    )
    flutter_hits = sum(
        [
            float(stats["rr_cluster_ratio"]) >= AFL_CLUSTER_RATIO_THRESHOLD,
            float(stats["rr_multiple_ratio"]) > AFL_MULTIPLE_RATIO_THRESHOLD,
            bool(stats["rr_periodic"]),
        ]
    )
    lorenz_support = float(stats["sd1_ms"]) > AF_SD1_THRESHOLD_MS and float(stats["sd2_ms"]) > AF_SD2_THRESHOLD_MS
    forced_fibrillation = (
        float(stats["rr_cluster_ratio"]) < AFL_FORCE_FIB_CLUSTER_MAX
        and float(stats["rr_multiple_ratio"]) <= AFL_FORCE_FIB_MULTIPLE_MAX
        and sd1_sd2 > AFL_FORCE_FIB_RATIO_MIN
    )
    result = {
        "temporal_positive_count": temporal_hits,
        "flutter_rule_hits": flutter_hits,
        "lorenz_support": lorenz_support,
        "flutter_ratio_band": AFL_RATIO_MIN <= sd1_sd2 <= AF_SD1_SD2_FIB_THRESHOLD,
        "forced_fibrillation": forced_fibrillation,
        "gray_promoted": False,
        "subtype": "fibrillation",
        "af_stage": "rejected",
    }
    if nn_count < AF_MIN_NN_COUNT:
        result["af_stage"] = "insufficient_nn"
        return result
    if cv < AF_CV_FAST_NEGATIVE:
        result["af_stage"] = "fast_negative"
        return result
    if not bool(stats["sqi_pass"]):
        result["af_stage"] = "sqi_rejected"
        return result
    if AF_CV_FAST_NEGATIVE <= cv <= AF_CV_POSITIVE:
        gray_promoted = (
            sd1_sd2 >= AF_GRAY_SD1_SD2_THRESHOLD
            and float(stats["sd1_ms"]) >= AF_SD1_THRESHOLD_MS
            and float(stats["sd2_ms"]) >= AF_SD2_THRESHOLD_MS
            and nn_count >= AF_GRAY_MIN_NN_COUNT
            and float(stats["pnn50_ratio"]) >= AF_GRAY_PNN50_THRESHOLD
            and temporal_hits >= AF_GRAY_TEMPORAL_HITS_MIN
        )
        result["gray_promoted"] = gray_promoted
        result["af_stage"] = "confirmed_positive" if gray_promoted else "gray_negative"
        return result
    core_ratio_support = sd1_sd2 > AF_SD1_SD2_FIB_THRESHOLD or (sd1_sd2 >= AFL_RATIO_MIN and flutter_hits >= 2)
    if not lorenz_support or not core_ratio_support:
        result["af_stage"] = "core_rejected"
        return result
    if temporal_hits < 2:
        result["af_stage"] = "temporal_suspicious"
        return result
    result["subtype"] = "flutter" if flutter_hits >= 2 and not forced_fibrillation and sd1_sd2 >= AFL_RATIO_MIN else "fibrillation"
    result["af_stage"] = "confirmed_positive"
    return result


def merge_positive_windows(windows: list[WindowFeature]) -> list[RhythmSpan]:
    if not windows:
        return []
    ordered = sorted(windows, key=lambda item: (item.start_ms, item.end_ms))
    spans = [RhythmSpan(start_ms=ordered[0].start_ms, end_ms=ordered[0].end_ms, windows=[ordered[0]])]
    for window in ordered[1:]:
        current = spans[-1]
        if window.start_ms - current.end_ms <= AF_EVENT_GAP_MS:
            current.end_ms = max(current.end_ms, window.end_ms)
            current.windows.append(window)
        else:
            spans.append(RhythmSpan(start_ms=window.start_ms, end_ms=window.end_ms, windows=[window]))
    return spans


def event_subtype(windows: list[WindowFeature], stats: dict[str, Any]) -> str:
    forced_fibrillation = (
        float(stats["rr_cluster_ratio"]) < AFL_FORCE_FIB_CLUSTER_MAX
        and float(stats["rr_multiple_ratio"]) <= AFL_FORCE_FIB_MULTIPLE_MAX
        and float(stats["sd1_sd2"]) > AFL_FORCE_FIB_RATIO_MIN
    )
    flutter_hits = sum(
        [
            float(stats["rr_cluster_ratio"]) >= AFL_CLUSTER_RATIO_THRESHOLD,
            float(stats["rr_multiple_ratio"]) > AFL_MULTIPLE_RATIO_THRESHOLD,
            bool(stats["rr_periodic"]),
        ]
    )
    if flutter_hits >= 2 and not forced_fibrillation and float(stats["sd1_sd2"]) >= AFL_RATIO_MIN:
        return "flutter"
    votes = Counter(window.subtype for window in windows)
    return votes.most_common(1)[0][0] if votes else "fibrillation"


def event_stats(span: RhythmSpan, stats: dict[str, Any]) -> dict[str, Any]:
    temporal_hits = sum(
        [
            float(stats["sdnn_ms"]) > AF_SDNN_THRESHOLD_MS,
            float(stats["rmssd_ms"]) > AF_RMSSD_THRESHOLD_MS,
            float(stats["pnn50_ratio"]) > AF_PNN50_THRESHOLD,
        ]
    )
    flutter_hits = sum(
        [
            float(stats["rr_cluster_ratio"]) >= AFL_CLUSTER_RATIO_THRESHOLD,
            float(stats["rr_multiple_ratio"]) > AFL_MULTIPLE_RATIO_THRESHOLD,
            bool(stats["rr_periodic"]),
        ]
    )
    sqi_scores = [window.stats.get("kurtosis_sqi") for window in span.windows if window.stats.get("kurtosis_sqi") is not None]
    return {
        "kurtosis_sqi": float(np.mean(np.array(sqi_scores, dtype=np.float64))) if sqi_scores else None,
        "sqi_pass": True if sqi_scores else None,
        "af_stage": "confirmed_event",
        "gray_promoted": any(bool(window.stats.get("gray_promoted")) for window in span.windows),
        "gray_promoted_window_count": sum(bool(window.stats.get("gray_promoted")) for window in span.windows),
        "subtype_window_votes": dict(Counter(window.subtype for window in span.windows)),
        "positive_window_count": len(span.windows),
        "temporal_positive_count": temporal_hits,
        "flutter_rule_hits": flutter_hits,
        "flutter_ratio_band": AFL_RATIO_MIN <= float(stats["sd1_sd2"]) <= AF_SD1_SD2_FIB_THRESHOLD,
        "forced_fibrillation": (
            float(stats["rr_cluster_ratio"]) < AFL_FORCE_FIB_CLUSTER_MAX
            and float(stats["rr_multiple_ratio"]) <= AFL_FORCE_FIB_MULTIPLE_MAX
            and float(stats["sd1_sd2"]) > AFL_FORCE_FIB_RATIO_MIN
        ),
        "window_start_ms": span.start_ms,
        "window_end_ms": span.end_ms,
    }


def compact_stats(stats: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in stats.items():
        if value is None:
            continue
        if isinstance(value, (np.floating, float)):
            compact[key] = round(float(value), 4)
        elif isinstance(value, (np.integer, int)):
            compact[key] = int(value)
        elif isinstance(value, dict):
            compact[key] = compact_stats(value)
        else:
            compact[key] = value
    return compact


def format_hms(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_time_range(case: CaseData, start_ms: int, end_ms: int) -> str:
    start_time = case.start_time + timedelta(milliseconds=start_ms)
    end_time = case.start_time + timedelta(milliseconds=end_ms)
    start_text = start_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    end_text = end_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return start_text if start_ms == end_ms else f"{start_text} ~ {end_text}"


def write_windows_csv(path: Path, windows: list[WindowFeature]) -> None:
    fieldnames = [
        "start_ms",
        "end_ms",
        "beat_start",
        "beat_end",
        "subtype",
        "af_stage",
        "cv",
        "rmssd_ratio",
        "pnn50_ratio",
        "sd1_ms",
        "sd2_ms",
        "sd1_sd2",
        "rr_cluster_ratio",
        "rr_multiple_ratio",
        "rr_periodic",
        "temporal_positive_count",
        "flutter_rule_hits",
        "kurtosis_sqi",
        "sqi_pass",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for window in windows:
            stats = compact_stats(window.stats)
            writer.writerow({field: getattr(window, field, stats.get(field, "")) for field in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone machine-side AF global experiment.")
    parser.add_argument("--edf", required=True, help="Path to EDF file.")
    parser.add_argument("--csv", required=True, help="Path to beat CSV file.")
    parser.add_argument("--out-dir", default="", help="Output directory. Default: out/af_global_<csv_stem>/")
    parser.add_argument("--ignore-sqi", action="store_true", help="Force SQI pass for every AF window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    edf_path = Path(args.edf)
    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir) if args.out_dir else Path("out") / f"af_global_{csv_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    case = load_case(edf_path=edf_path, csv_path=csv_path)
    events, windows = AFGlobalExperiment(ignore_sqi=args.ignore_sqi).run(case)

    events_path = out_dir / "af_global_events.json"
    windows_path = out_dir / "af_global_positive_windows.csv"
    summary_path = out_dir / "af_global_summary.json"
    events_path.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    write_windows_csv(windows_path, windows)
    summary_path.write_text(
        json.dumps(
            {
                "edf": str(edf_path),
                "csv": str(csv_path),
                "ignore_sqi": bool(args.ignore_sqi),
                "event_count": len(events),
                "positive_window_count": len(windows),
                "events_json": str(events_path),
                "positive_windows_csv": str(windows_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"events={len(events)} -> {events_path}")
    print(f"positive_windows={len(windows)} -> {windows_path}")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
