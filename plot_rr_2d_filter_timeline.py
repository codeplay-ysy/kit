from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


LABEL_COLORS = {
    "af": "#d73027",
    "strong_af": "#ef5350",
    "possible_af": "#fb8c00",
    "enhanced_af": "#8e24aa",
    "mid_af": "#00897b",
    "long_af": "#1e88e5",
    "non_af": "#9e9e9e",
    "afl": "#6a3d9a",
    "strong_afl": "#7b3294",
    "possible_afl": "#ab47bc",
    "suspicious": "#e6ab02",
    "non_afl": "#bdbdbd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RR 2D filter window labels on a timeline.")
    parser.add_argument("--windows-csv", required=True, help="RR 2D filter window CSV path.")
    parser.add_argument("--beat-csv", default="", help="Optional beat CSV for full timeline duration.")
    parser.add_argument("--out-figure", required=True, help="Output figure path, e.g. .pdf or .png.")
    parser.add_argument(
        "--label-column",
        choices=["label", "enhanced_label", "mid_label", "long_label", "final_label"],
        default="enhanced_label",
        help="Window label column to plot. Use label/final_label for AFL window outputs.",
    )
    return parser.parse_args()


def load_windows(path: str | Path, label_column: str = "enhanced_label") -> list[dict[str, Any]]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"start_ms", "end_ms", "label"}
        missing = required - fieldnames
        if missing:
            raise SystemExit(f"Missing required columns in {csv_path}: {sorted(missing)}")
        actual_label_column = label_column if label_column in fieldnames else "label"
        return [
            {
                "start_ms": int(float(row["start_ms"])),
                "end_ms": int(float(row["end_ms"])),
                "label": row.get(actual_label_column) or row["label"],
                "raw_label": row["label"],
                "occupied_ratio": float(row.get("occupied_ratio") or 0.0),
                "max_bin_ratio": float(row.get("max_bin_ratio") or 0.0),
            }
            for row in reader
        ]


def load_duration_ms(beat_csv: str | Path, windows: list[dict[str, Any]]) -> int:
    max_ms = max((int(window["end_ms"]) for window in windows), default=0)
    if not beat_csv:
        return max_ms
    csv_path = Path(beat_csv)
    if not csv_path.exists():
        return max_ms
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        time_key = None
        for candidate in ("merged_milliseconds", "time offset(ms)", "offset_ms"):
            if candidate in fields:
                time_key = candidate
                break
        if time_key is None and len(fields) >= 2:
            time_key = fields[1]
        if time_key:
            for row in reader:
                max_ms = max(max_ms, int(float(row[time_key])))
    return max_ms


def format_hour_tick(hour: float) -> str:
    total_minutes = int(round(hour * 60.0))
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def merge_label_segments(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not windows:
        return []
    ordered = sorted(windows, key=lambda item: (int(item["start_ms"]), int(item["end_ms"])))
    segments = [
        {
            "start_ms": int(ordered[0]["start_ms"]),
            "end_ms": int(ordered[0]["end_ms"]),
            "label": str(ordered[0]["label"]),
        }
    ]
    for window in ordered[1:]:
        label = str(window["label"])
        start_ms = int(window["start_ms"])
        end_ms = int(window["end_ms"])
        current = segments[-1]
        if label == current["label"] and start_ms <= int(current["end_ms"]):
            current["end_ms"] = max(int(current["end_ms"]), end_ms)
        else:
            segments.append({"start_ms": start_ms, "end_ms": end_ms, "label": label})
    return segments


def plot_timeline(windows: list[dict[str, Any]], duration_ms: int, out_path: str | Path) -> None:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    duration_h = duration_ms / 3_600_000.0 if duration_ms else 0.0
    fig, ax = plt.subplots(figsize=(18, 5), constrained_layout=True)
    ax.hlines(y=1.0, xmin=0, xmax=duration_h, color="#333333", linewidth=1.3)

    for segment in merge_label_segments(windows):
        start_h = segment["start_ms"] / 3_600_000.0
        width_h = max((segment["end_ms"] - segment["start_ms"]) / 3_600_000.0, 0.001)
        color = LABEL_COLORS.get(str(segment["label"]), "#7f7f7f")
        ax.broken_barh([(start_h, width_h)], (0.78, 0.44), facecolors=color, edgecolors="none", alpha=0.85)

    label_counts = {label: sum(window["label"] == label for window in windows) for label in LABEL_COLORS if any(window["label"] == label for window in windows)}
    ax.set_title("RR 2D Filter Timeline", fontsize=20, fontweight="bold", pad=14)
    ax.set_xlabel("Elapsed time (HH:MM)", fontsize=13, fontweight="bold")
    ax.set_yticks([1.0], ["RR 2D windows"])
    ax.tick_params(axis="both", labelsize=11)

    tick_step = 1.0 if duration_h <= 24 else 2.0
    ticks = [tick * tick_step for tick in range(int(duration_h / tick_step) + 1)]
    if duration_h and (not ticks or ticks[-1] < duration_h):
        ticks.append(duration_h)
    ax.set_xticks(ticks, [format_hour_tick(tick) for tick in ticks], rotation=45, ha="right")
    ax.set_xlim(0, max(duration_h, 0.1))
    ax.set_ylim(0.4, 1.45)
    ax.grid(axis="x", color="#cccccc", linewidth=0.8, alpha=0.8)

    legend_handles = [Patch(facecolor=color, edgecolor="none", label=label) for label, color in LABEL_COLORS.items()]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=11)
    summary = "    ".join(f"{label}={count}" for label, count in label_counts.items())
    ax.text(0.01, 0.05, summary, transform=ax.transAxes, fontsize=12, fontweight="bold", ha="left", va="bottom")

    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    windows = load_windows(args.windows_csv, label_column=args.label_column)
    duration_ms = load_duration_ms(args.beat_csv, windows)
    plot_timeline(windows=windows, duration_ms=duration_ms, out_path=args.out_figure)
    print(f"windows={len(windows)}")
    print(f"figure -> {Path(args.out_figure).resolve()}")


if __name__ == "__main__":
    main()
