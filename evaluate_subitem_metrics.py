from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NO_PRED = "__NO_PRED__"
NO_TRUE = "__NO_TRUE__"
DEFAULT_LABEL_ORDER = ["N", "AF", "Af", "S", "Se", "V", "Ve"]


@dataclass(slots=True)
class Beat:
    order: int
    time_ms: float
    label: str
    row: dict[str, str]


@dataclass(slots=True)
class Match:
    truth: Beat
    pred: Beat
    diff_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate subitem beat labels against ground truth with time tolerance.")
    parser.add_argument("--truth-csv", required=True, help="Ground-truth CSV. Column 2 is time in ms; column 3 is label.")
    parser.add_argument("--pred-csv", required=True, help="Subitem prediction CSV.")
    parser.add_argument("--tolerance-ms", type=float, default=80.0, help="Maximum absolute timestamp difference for a match.")
    parser.add_argument("--pred-time-column", default="merged_milliseconds", help="Prediction timestamp column in ms.")
    parser.add_argument("--pred-label-column", default="symbol", help="Prediction label column.")
    parser.add_argument("--out-figure", required=True, help="Output figure path, e.g. .pdf or .png.")
    return parser.parse_args()


def _clean_label(value: str) -> str:
    return value.strip() or "<EMPTY>"


def load_truth(path: str | Path) -> list[Beat]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if len(fieldnames) < 3:
            raise SystemExit(f"Truth CSV must have at least 3 columns: {csv_path}")

        time_column = fieldnames[1]
        label_column = fieldnames[2]
        beats: list[Beat] = []
        for index, row in enumerate(reader, start=1):
            beats.append(
                Beat(
                    order=index,
                    time_ms=float(row[time_column]),
                    label=_clean_label(row[label_column]),
                    row=row,
                )
            )
    return sorted(beats, key=lambda beat: beat.time_ms)


def load_predictions(path: str | Path, time_column: str, label_column: str) -> list[Beat]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = {time_column, label_column} - fieldnames
        if missing:
            raise SystemExit(f"Prediction CSV is missing required columns: {sorted(missing)}")

        beats: list[Beat] = []
        for index, row in enumerate(reader, start=1):
            beats.append(
                Beat(
                    order=int(row.get("beat_order") or index),
                    time_ms=float(row[time_column]),
                    label=_clean_label(row[label_column]),
                    row=row,
                )
            )
    return sorted(beats, key=lambda beat: beat.time_ms)


def match_beats(truths: list[Beat], preds: list[Beat], tolerance_ms: float) -> tuple[list[Match], list[Beat], list[Beat]]:
    matches: list[Match] = []
    unmatched_truths: list[Beat] = []
    unmatched_preds: list[Beat] = []

    pred_index = 0
    for truth in truths:
        lower_bound = truth.time_ms - tolerance_ms
        upper_bound = truth.time_ms + tolerance_ms

        while pred_index < len(preds) and preds[pred_index].time_ms < lower_bound:
            unmatched_preds.append(preds[pred_index])
            pred_index += 1

        candidate_index: int | None = None
        candidate_diff: float | None = None
        scan_index = pred_index
        while scan_index < len(preds) and preds[scan_index].time_ms <= upper_bound:
            diff = abs(preds[scan_index].time_ms - truth.time_ms)
            if candidate_diff is None or diff < candidate_diff:
                candidate_index = scan_index
                candidate_diff = diff
            scan_index += 1

        if candidate_index is None:
            unmatched_truths.append(truth)
            continue

        unmatched_preds.extend(preds[pred_index:candidate_index])
        pred = preds[candidate_index]
        matches.append(Match(truth=truth, pred=pred, diff_ms=float(candidate_diff or 0.0)))
        pred_index = candidate_index + 1

    unmatched_preds.extend(preds[pred_index:])
    return matches, unmatched_truths, unmatched_preds


def build_confusion(pairs: list[tuple[str, str]]) -> tuple[list[str], dict[str, Counter[str]]]:
    observed_labels = {label for pair in pairs for label in pair}
    labels = [label for label in DEFAULT_LABEL_ORDER if label in observed_labels]
    labels.extend(sorted(observed_labels - set(labels)))
    matrix: dict[str, Counter[str]] = {label: Counter() for label in labels}
    for true_label, pred_label in pairs:
        matrix.setdefault(true_label, Counter())[pred_label] += 1
    return labels, matrix


def compute_metrics(labels: list[str], matrix: dict[str, Counter[str]], total_truth: int, total_pred: int, matched_count: int) -> dict[str, Any]:
    correct = sum(matrix.get(label, Counter()).get(label, 0) for label in labels)
    total_matched = sum(sum(row.values()) for row in matrix.values())
    total_fp = total_matched - correct
    total_fn = total_matched - correct

    per_label: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = matrix.get(label, Counter()).get(label, 0)
        fp = sum(matrix.get(true_label, Counter()).get(label, 0) for true_label in labels if true_label != label)
        fn = sum(count for pred_label, count in matrix.get(label, Counter()).items() if pred_label != label)
        support = sum(matrix.get(label, Counter()).values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }

    macro_precision = sum(float(item["precision"]) for item in per_label.values()) / len(labels) if labels else 0.0
    macro_recall = sum(float(item["recall"]) for item in per_label.values()) / len(labels) if labels else 0.0
    macro_f1 = sum(float(item["f1"]) for item in per_label.values()) / len(labels) if labels else 0.0

    return {
        "total_truth": total_truth,
        "total_predictions": total_pred,
        "matched_count": matched_count,
        "unmatched_truth_count": total_truth - matched_count,
        "unmatched_prediction_count": total_pred - matched_count,
        "matched_accuracy": round(correct / total_matched, 6) if total_matched else 0.0,
        "overall_precision": round(correct / (correct + total_fp), 6) if correct + total_fp else 0.0,
        "overall_recall": round(correct / (correct + total_fn), 6) if correct + total_fn else 0.0,
        "match_recall_vs_truth": round(matched_count / total_truth, 6) if total_truth else 0.0,
        "match_precision_vs_predictions": round(matched_count / total_pred, 6) if total_pred else 0.0,
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "macro_f1": round(macro_f1, 6),
        "per_label": per_label,
    }


def _matrix_values(labels: list[str], matrix: dict[str, Counter[str]]) -> list[list[int]]:
    return [[matrix.get(true_label, Counter()).get(pred_label, 0) for pred_label in labels] for true_label in labels]


def _plot_confusion(ax: Any, labels: list[str], matrix: dict[str, Counter[str]], title: str) -> None:
    values = _matrix_values(labels, matrix)
    display_values = [[count if count > 0 else 0 for count in row] for row in values]
    max_count = max((count for row in values for count in row), default=0)

    image = ax.imshow(display_values, cmap="Blues", origin="upper")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted label", fontsize=16, fontweight="bold", labelpad=12)
    ax.set_ylabel("True label", fontsize=16, fontweight="bold", labelpad=10)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labeltop=True, labelbottom=False, top=True, bottom=False)
    ax.set_xticks([index - 0.5 for index in range(len(labels) + 1)], minor=True)
    ax.set_yticks([index - 0.5 for index in range(len(labels) + 1)], minor=True)
    ax.grid(which="minor", color="#222222", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", labelsize=14, width=1.5, length=6)
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_fontweight("bold")

    threshold = max_count * 0.55
    for row_index, row in enumerate(values):
        for col_index, count in enumerate(row):
            if count == 0:
                continue
            color = "white" if count >= threshold else "black"
            ax.text(col_index, row_index, str(count), ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def write_figure(
    path: str | Path,
    labels: list[str],
    matrix: dict[str, Counter[str]],
    unmatched_labels: list[str],
    unmatched_matrix: dict[str, Counter[str]],
    metrics: dict[str, Any],
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.2, 0.8], height_ratios=[1.0, 1.0])
    matched_ax = fig.add_subplot(grid[:, 0])
    unmatched_ax = fig.add_subplot(grid[0, 1])
    metrics_ax = fig.add_subplot(grid[1, 1])

    _plot_confusion(matched_ax, labels, matrix, "Matched Confusion Matrix")
    _plot_confusion(unmatched_ax, unmatched_labels, unmatched_matrix, "Unmatched Matrix")

    metrics_ax.axis("off")
    per_label = metrics["per_label"]
    label_lines = ["Per-label metrics", "label     precision    recall"]
    for label in DEFAULT_LABEL_ORDER:
        if label not in per_label:
            continue
        values = per_label[label]
        label_lines.append(f"{label:<6}    {float(values['precision']):>8.4f}  {float(values['recall']):>8.4f}")

    summary_lines = [
        "Subitem Evaluation",
        "",
        f"Tolerance: {metrics['tolerance_ms']} ms",
        f"Truth beats: {metrics['total_truth']}",
        f"Predicted beats: {metrics['total_predictions']}",
        f"Matched: {metrics['matched_count']}",
        f"Unmatched truth: {metrics['unmatched_truth_count']}",
        f"Unmatched predictions: {metrics['unmatched_prediction_count']}",
        "",
        f"Overall precision: {metrics['overall_precision']:.6f}",
        f"Overall recall: {metrics['overall_recall']:.6f}",
        "",
        *label_lines,
    ]
    metrics_ax.text(0.02, 0.98, "\n".join(summary_lines), va="top", ha="left", fontsize=11, family="monospace")

    fig.suptitle("Subitem Label Evaluation", fontsize=18)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    truths = load_truth(args.truth_csv)
    preds = load_predictions(args.pred_csv, time_column=args.pred_time_column, label_column=args.pred_label_column)
    matches, unmatched_truths, unmatched_preds = match_beats(truths, preds, tolerance_ms=args.tolerance_ms)

    matched_pairs = [(match.truth.label, match.pred.label) for match in matches]
    labels, matrix = build_confusion(matched_pairs)

    unmatched_pairs = [(beat.label, NO_PRED) for beat in unmatched_truths]
    unmatched_pairs.extend((NO_TRUE, beat.label) for beat in unmatched_preds)
    unmatched_labels, unmatched_matrix = build_confusion(unmatched_pairs)

    metrics = compute_metrics(
        labels=labels,
        matrix=matrix,
        total_truth=len(truths),
        total_pred=len(preds),
        matched_count=len(matches),
    )
    metrics["tolerance_ms"] = args.tolerance_ms
    write_figure(
        path=args.out_figure,
        labels=labels,
        matrix=matrix,
        unmatched_labels=unmatched_labels,
        unmatched_matrix=unmatched_matrix,
        metrics=metrics,
    )

    print(f"Matched: {len(matches)}")
    print(f"Unmatched truth: {len(unmatched_truths)}")
    print(f"Unmatched predictions: {len(unmatched_preds)}")
    print(f"Figure: {Path(args.out_figure).resolve()}")


if __name__ == "__main__":
    main()
