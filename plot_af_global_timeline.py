from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


SUBTYPE_COLORS = {
    "fibrillation": "#d73027",
    "flutter": "#4575b4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot AF global events on a time axis.")
    parser.add_argument("--events-json", required=True, help="AF global events JSON path.")
    parser.add_argument("--windows-csv", default="", help="Optional AF positive windows CSV path.")
    parser.add_argument("--beat-csv", default="", help="Optional beat CSV for full timeline duration.")
    parser.add_argument("--out-figure", required=True, help="Output figure path, e.g. .pdf or .png.")
    return parser.parse_args()


def load_events(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        events = json.load(handle)
    if not isinstance(events, list):
        raise SystemExit(f"Events JSON must contain a list: {path}")
    return sorted(events, key=lambda event: (int(event["t0_ms"]), int(event["t1_ms"])))


def load_windows(path: str | Path) -> list[dict[str, Any]]:
    if not path:
        return []
    csv_path = Path(path)
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            {
                "start_ms": int(float(row["start_ms"])),
                "end_ms": int(float(row["end_ms"])),
                "subtype": row.get("subtype", ""),
            }
            for row in csv.DictReader(handle)
        ]


def load_duration_ms(beat_csv: str | Path, events: list[dict[str, Any]], windows: list[dict[str, Any]]) -> int:
    max_ms = 0
    if beat_csv:
        csv_path = Path(beat_csv)
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames or []
                if len(fields) >= 2:
                    time_key = fields[1]
                    for row in reader:
                        max_ms = max(max_ms, int(float(row[time_key])))
    for event in events:
        max_ms = max(max_ms, int(event["t1_ms"]))
    for window in windows:
        max_ms = max(max_ms, int(window["end_ms"]))
    return max_ms


def format_hour_tick(hour: float) -> str:
    total_minutes = int(round(hour * 60.0))
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def plot_timeline(events: list[dict[str, Any]], windows: list[dict[str, Any]], duration_ms: int, out_path: str | Path) -> None:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    duration_h = duration_ms / 3_600_000.0 if duration_ms else 0.0
    fig, ax = plt.subplots(figsize=(18, 5.5), constrained_layout=True)

    ax.hlines(y=1.0, xmin=0, xmax=duration_h, color="#333333", linewidth=1.5)
    ax.hlines(y=0.35, xmin=0, xmax=duration_h, color="#999999", linewidth=1.0, alpha=0.7)

    for window in windows:
        start_h = window["start_ms"] / 3_600_000.0
        width_h = max((window["end_ms"] - window["start_ms"]) / 3_600_000.0, 0.001)
        color = SUBTYPE_COLORS.get(str(window["subtype"]), "#7f7f7f")
        ax.broken_barh([(start_h, width_h)], (0.25, 0.2), facecolors=color, alpha=0.18, edgecolors="none")

    for event in events:
        start_ms = int(event["t0_ms"])
        end_ms = int(event["t1_ms"])
        start_h = start_ms / 3_600_000.0
        width_h = max((end_ms - start_ms) / 3_600_000.0, 0.002)
        subtype = str(event.get("subtype", ""))
        color = SUBTYPE_COLORS.get(subtype, "#7f7f7f")
        ax.broken_barh([(start_h, width_h)], (0.78, 0.44), facecolors=color, edgecolors="#111111", linewidth=0.8)

    event_count = len(events)
    total_af_ms = sum(max(0, int(event["t1_ms"]) - int(event["t0_ms"])) for event in events)
    fibrillation_count = sum(str(event.get("subtype")) == "fibrillation" for event in events)
    flutter_count = sum(str(event.get("subtype")) == "flutter" for event in events)

    ax.set_title("AF Global Timeline", fontsize=20, fontweight="bold", pad=16)
    ax.set_xlabel("Elapsed time (HH:MM)", fontsize=14, fontweight="bold")
    ax.set_yticks([1.0, 0.35], ["AF events", "positive windows"])
    ax.tick_params(axis="both", labelsize=12)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontweight("bold")

    tick_step = 1.0 if duration_h <= 24 else 2.0
    ticks = [tick * tick_step for tick in range(int(duration_h / tick_step) + 1)]
    if duration_h and (not ticks or ticks[-1] < duration_h):
        ticks.append(duration_h)
    ax.set_xticks(ticks, [format_hour_tick(tick) for tick in ticks], rotation=45, ha="right")
    ax.set_xlim(0, max(duration_h, 0.1))
    ax.set_ylim(0.05, 1.45)
    ax.grid(axis="x", color="#cccccc", linewidth=0.8, alpha=0.8)

    legend_handles = [
        Patch(facecolor=SUBTYPE_COLORS["fibrillation"], edgecolor="#111111", label="fibrillation"),
        Patch(facecolor=SUBTYPE_COLORS["flutter"], edgecolor="#111111", label="flutter"),
        Patch(facecolor="#7f7f7f", edgecolor="none", alpha=0.18, label="positive windows"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=11)

    summary = (
        f"events={event_count}    fibrillation={fibrillation_count}    flutter={flutter_count}    "
        f"total AF duration={total_af_ms / 60_000.0:.1f} min"
    )
    ax.text(0.01, 0.04, summary, transform=ax.transAxes, fontsize=12, fontweight="bold", ha="left", va="bottom")

    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    events = load_events(args.events_json)
    windows = load_windows(args.windows_csv)
    duration_ms = load_duration_ms(args.beat_csv, events, windows)
    plot_timeline(events=events, windows=windows, duration_ms=duration_ms, out_path=args.out_figure)
    print(f"events={len(events)}")
    print(f"positive_windows={len(windows)}")
    print(f"figure -> {Path(args.out_figure).resolve()}")


if __name__ == "__main__":
    main()
