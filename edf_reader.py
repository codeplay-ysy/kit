from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


def _parse_ascii_int(blob: bytes) -> int:
    return int(blob.decode("latin-1").strip() or 0)


def _parse_ascii_float(blob: bytes) -> float:
    return float(blob.decode("latin-1").strip() or 0.0)


def _parse_edf_datetime(date_text: str, time_text: str) -> datetime:
    day, month, year = [int(part) for part in date_text.split(".")]
    hour, minute, second = [int(part) for part in time_text.split(".")]
    year += 2000 if year < 85 else 1900
    return datetime(year, month, day, hour, minute, second)


def _decode_fields(blob: bytes, width: int, count: int) -> list[str]:
    return [
        blob[index * width : (index + 1) * width].decode("latin-1").strip()
        for index in range(count)
    ]


@dataclass(slots=True)
class SignalSpec:
    label: str
    phys_dimension: str
    phys_min: float
    phys_max: float
    dig_min: int
    dig_max: int
    samples_per_record: int
    prefilter: str
    transducer: str
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

    def signal(self, label: str) -> SignalSpec:
        for signal in self.signals:
            if signal.label == label:
                return signal
        raise KeyError(label)

    @property
    def record_size_bytes(self) -> int:
        return sum(signal.samples_per_record * 2 for signal in self.signals)


@dataclass(slots=True)
class SignalWindow:
    label: str
    fs_hz: float
    start_ms: int
    end_ms: int
    values: np.ndarray

    def to_summary(self) -> dict[str, float | int]:
        if self.values.size == 0:
            return {"samples": 0, "fs_hz": round(self.fs_hz, 4)}
        return {
            "samples": int(self.values.size),
            "fs_hz": round(self.fs_hz, 4),
            "min": round(float(self.values.min()), 4),
            "max": round(float(self.values.max()), 4),
            "mean": round(float(self.values.mean()), 4),
            "std": round(float(self.values.std()), 4),
        }


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
        transducers = _decode_fields(signal_blob[cursor : cursor + signal_count * 80], 80, signal_count)
        cursor += signal_count * 80
        phys_dimensions = _decode_fields(signal_blob[cursor : cursor + signal_count * 8], 8, signal_count)
        cursor += signal_count * 8
        phys_mins = _decode_fields(signal_blob[cursor : cursor + signal_count * 8], 8, signal_count)
        cursor += signal_count * 8
        phys_maxs = _decode_fields(signal_blob[cursor : cursor + signal_count * 8], 8, signal_count)
        cursor += signal_count * 8
        dig_mins = _decode_fields(signal_blob[cursor : cursor + signal_count * 8], 8, signal_count)
        cursor += signal_count * 8
        dig_maxs = _decode_fields(signal_blob[cursor : cursor + signal_count * 8], 8, signal_count)
        cursor += signal_count * 8
        prefilters = _decode_fields(signal_blob[cursor : cursor + signal_count * 80], 80, signal_count)
        cursor += signal_count * 80
        samples_per_record = _decode_fields(signal_blob[cursor : cursor + signal_count * 8], 8, signal_count)

        signals: list[SignalSpec] = []
        sample_offset = 0
        for index in range(signal_count):
            signals.append(
                SignalSpec(
                    label=labels[index],
                    phys_dimension=phys_dimensions[index],
                    phys_min=float(phys_mins[index] or 0.0),
                    phys_max=float(phys_maxs[index] or 0.0),
                    dig_min=int(dig_mins[index] or 0),
                    dig_max=int(dig_maxs[index] or 0),
                    samples_per_record=int(samples_per_record[index] or 0),
                    prefilter=prefilters[index],
                    transducer=transducers[index],
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

    def read_window(
        self,
        start_ms: int,
        end_ms: int,
        labels: list[str] | None = None,
    ) -> list[SignalWindow]:
        labels = labels or [signal.label for signal in self.ecg_signals()]
        windows: list[SignalWindow] = []
        for label in labels:
            windows.append(self._read_single_signal(label=label, start_ms=start_ms, end_ms=end_ms))
        return windows

    def _read_single_signal(self, label: str, start_ms: int, end_ms: int) -> SignalWindow:
        signal = self.header.signal(label)
        fs_hz = signal.samples_per_record / self.header.duration_per_record_s
        if end_ms <= start_ms:
            return SignalWindow(label=label, fs_hz=fs_hz, start_ms=start_ms, end_ms=end_ms, values=np.array([]))

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

        return SignalWindow(
            label=label,
            fs_hz=fs_hz,
            start_ms=start_ms,
            end_ms=end_ms,
            values=values,
        )
