from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


OVERLAP_GAP_MS = 0


def format_hms(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    return f"{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}:{total_seconds % 60:02d}"


def load_events(path: str | Path, source: str) -> list[dict[str, Any]]:
    event_path = Path(path)
    if not event_path.exists():
        return []
    data = json.loads(event_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Events JSON must contain a list: {event_path}")
    events: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict) or "t0_ms" not in item or "t1_ms" not in item:
            continue
        start_ms = int(float(item["t0_ms"]))
        end_ms = int(float(item["t1_ms"]))
        if end_ms <= start_ms:
            continue
        events.append({**item, "t0_ms": start_ms, "t1_ms": end_ms, "_source": source})
    return sorted(events, key=lambda event: (int(event["t0_ms"]), int(event["t1_ms"])))


def _subtype_priority(subtypes: set[str]) -> str:
    if "flutter" in subtypes and "fibrillation" in subtypes:
        return "mixed_or_uncertain"
    if "flutter" in subtypes:
        return "flutter"
    if "fibrillation" in subtypes:
        return "fibrillation"
    return sorted(subtypes)[0] if subtypes else "unknown"


def merge_events(events: list[dict[str, Any]], gap_ms: int = OVERLAP_GAP_MS) -> list[dict[str, Any]]:
    if not events:
        return []
    ordered = sorted(events, key=lambda event: (int(event["t0_ms"]), int(event["t1_ms"])))
    groups: list[list[dict[str, Any]]] = [[ordered[0]]]
    for event in ordered[1:]:
        current = groups[-1]
        current_end = max(int(item["t1_ms"]) for item in current)
        if int(event["t0_ms"]) - current_end <= gap_ms:
            current.append(event)
        else:
            groups.append([event])

    merged: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        start_ms = min(int(item["t0_ms"]) for item in group)
        end_ms = max(int(item["t1_ms"]) for item in group)
        subtypes = {str(item.get("subtype", "unknown")) for item in group}
        sources = sorted({str(item.get("_source", item.get("layer", "unknown"))) for item in group})
        rules = sorted({str(item.get("rule", "")) for item in group if item.get("rule")})
        merged.append(
            {
                "type": "af_family",
                "subtype": _subtype_priority(subtypes),
                "layer": "af_family_event_fusion",
                "rule": "merge_overlapping_af_afl_events",
                "event_index": index,
                "t0_ms": start_ms,
                "t1_ms": end_ms,
                "time": f"{start_ms} ms ~ {end_ms} ms",
                "duration": format_hms((end_ms - start_ms) / 1000.0),
                "stats": {
                    "source_count": len(group),
                    "sources": sources,
                    "subtypes": sorted(subtypes),
                    "rules": rules,
                    "source_events": [
                        {
                            "source": item.get("_source", item.get("layer", "unknown")),
                            "subtype": item.get("subtype", "unknown"),
                            "t0_ms": int(item["t0_ms"]),
                            "t1_ms": int(item["t1_ms"]),
                            "rule": item.get("rule", ""),
                        }
                        for item in group
                    ],
                },
            }
        )
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge AF fibrillation and AFL flutter events into one af_family JSON.")
    parser.add_argument("--af-events-json", default="", help="RR2D AF events JSON.")
    parser.add_argument("--afl-events-json", default="", help="RR-diff AFL events JSON.")
    parser.add_argument("--out-events-json", required=True, help="Merged af_family events JSON output path.")
    parser.add_argument("--gap-ms", type=int, default=OVERLAP_GAP_MS, help="Maximum gap for merging events.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events: list[dict[str, Any]] = []
    if args.af_events_json:
        events.extend(load_events(args.af_events_json, "rr_2d_filter"))
    if args.afl_events_json:
        events.extend(load_events(args.afl_events_json, "rr_afl_filter"))
    merged = merge_events(events, gap_ms=args.gap_ms)
    out_path = Path(args.out_events_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"input_events={len(events)}")
    print(f"merged_events={len(merged)} -> {out_path}")


if __name__ == "__main__":
    main()
