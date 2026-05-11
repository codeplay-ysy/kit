from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LEADS = ("ECG_CH1", "ECG_CH3")
Stage = Literal["preprocess", "analysis"]


@dataclass(frozen=True)
class CaseFiles:
    edf: Path
    truth_csv: Path
    tag: str


@dataclass(frozen=True)
class CommandSpec:
    description: str
    args: list[str]
    stage: Stage
    outputs: tuple[Path, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ECG processing pipeline for all EDF/CSV case pairs.")
    parser.add_argument("--root", default=str(SCRIPT_DIR), help="Folder containing EDF and ground-truth CSV files.")
    parser.add_argument("--window-seconds", type=int, default=15, help="Quality-window length in seconds.")
    parser.add_argument("--leads", nargs="+", default=list(DEFAULT_LEADS), help="R-peak ECG leads to process.")
    parser.add_argument("--case", action="append", default=[], help="Run only cases whose EDF stem or case tag matches this value. Can be used multiple times.")
    parser.add_argument(
        "--stage",
        choices=["all", "preprocess", "analysis"],
        default="all",
        help="all: run both stages; preprocess: only steps 1-4; analysis: only steps 5-12.",
    )
    parser.add_argument("--rerun-preprocess", action="store_true", help="Rerun preprocessing steps even when their expected outputs already exist.")
    parser.add_argument("--skip-existing-analysis", action="store_true", help="Also skip analysis steps when their expected outputs already exist.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip a whole case if its final evaluation PDF already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue with the next case after a command fails.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run each pipeline step.")
    return parser.parse_args()


def case_tag_from_stem(stem: str) -> str:
    return stem.split("_", 1)[1] if "_" in stem else stem


def discover_cases(root: Path, selected: list[str]) -> list[CaseFiles]:
    selected_set = {item.strip() for item in selected if item.strip()}
    cases: list[CaseFiles] = []
    for edf_path in sorted(root.glob("*.edf"), key=lambda path: path.name):
        truth_csv = edf_path.with_suffix(".csv")
        if not truth_csv.exists():
            print(f"[SKIP] {edf_path.name}: missing matching truth CSV {truth_csv.name}")
            continue

        tag = case_tag_from_stem(edf_path.stem)
        if selected_set and edf_path.stem not in selected_set and tag not in selected_set:
            continue

        cases.append(CaseFiles(edf=edf_path, truth_csv=truth_csv, tag=tag))
    return cases


def output_path(path: Path) -> Path:
    return path if path.is_absolute() else SCRIPT_DIR / path


def existing_outputs(outputs: tuple[Path, ...]) -> bool:
    return bool(outputs) and all(output_path(path).exists() for path in outputs)


def build_commands(case: CaseFiles, leads: list[str], window_seconds: int) -> list[CommandSpec]:
    out_root = Path("out") / case.tag
    quality_csv = out_root / "quality" / f"{case.tag}_ecg_quality_{window_seconds}s.csv"
    ch1_csv = out_root / "rpeaks" / "ECG_CH1_rpeaks.csv"
    ch3_csv = out_root / "rpeaks" / "ECG_CH3_rpeaks.csv"
    merged_csv = out_root / "merged" / f"merged_rpeaks_by_quality_window_v5_priority_{case.tag}.csv"
    merged_audit_csv = out_root / "merged" / f"merged_rpeaks_by_quality_window_v5_priority_{case.tag}_window_audit.csv"
    merged_summary_json = out_root / "merged" / f"merged_rpeaks_by_quality_window_v5_priority_{case.tag}_summary.json"
    rr2d_windows_csv = out_root / "rr_2d_filter" / f"{case.tag}_rr_2d_windows.csv"
    rr2d_summary_json = out_root / "rr_2d_filter" / f"{case.tag}_rr_2d_summary.json"
    rr2d_events_json = out_root / "rr_2d_filter" / f"{case.tag}_rr_2d_events.json"
    afl_windows_csv = out_root / "rr_afl_filter" / f"merged_rpeaks_by_quality_window_v5_priority_{case.tag}_rr_afl_windows.csv"
    afl_summary_json = out_root / "rr_afl_filter" / f"merged_rpeaks_by_quality_window_v5_priority_{case.tag}_rr_afl_summary.json"
    afl_events_json = out_root / "rr_afl_filter" / f"merged_rpeaks_by_quality_window_v5_priority_{case.tag}_rr_afl_events.json"
    af_family_events_json = out_root / "af_family_events.json"
    subitem_csv = out_root / "subitem" / "subitem_experiment_beats_with_af_family.csv"
    subitem_summary_json = out_root / "subitem" / "subitem_experiment_summary_with_af_family.json"
    subitem_evaluation_pdf = out_root / "subitem" / "subitem_evaluation_with_af_family.pdf"
    rr2d_timeline_pdf = out_root / "rr_2d_filter" / f"{case.tag}_rr_2d_timeline_final_label.pdf"
    afl_label_timeline_pdf = out_root / "rr_afl_filter" / f"{case.tag}_rr_afl_label_timeline.pdf"
    afl_final_timeline_pdf = out_root / "rr_afl_filter" / f"{case.tag}_rr_afl_final_timeline.pdf"

    commands: list[CommandSpec] = []
    lead_outputs = {
        "ECG_CH1": ch1_csv,
        "ECG_CH3": ch3_csv,
    }
    for lead in leads:
        commands.append(
            CommandSpec(
                description=f"R-peaks {lead}",
                args=["run_nk_ecg_peaks.py", "--edf", str(case.edf), "--lead", lead],
                stage="preprocess",
                outputs=(lead_outputs.get(lead, out_root / "rpeaks" / f"{lead}_rpeaks.csv"),),
            )
        )

    commands.extend(
        [
            CommandSpec(
                description="ECG quality windows",
                args=["run_ecg_quality_windows.py", "--edf", str(case.edf), "--window-seconds", str(window_seconds)],
                stage="preprocess",
                outputs=(quality_csv,),
            ),
            CommandSpec(
                description="Merge R-peaks by quality window",
                args=[
                    "merge_rpeaks_by_quality_window.py",
                    "--quality-csv",
                    str(quality_csv),
                    "--ch1-csv",
                    str(ch1_csv),
                    "--ch3-csv",
                    str(ch3_csv),
                    "--window-seconds",
                    str(window_seconds),
                ],
                stage="preprocess",
                outputs=(merged_csv, merged_audit_csv, merged_summary_json),
            ),
            CommandSpec(
                description="RR 2D AF filter",
                args=[
                    "run_rr_2d_af_filter.py",
                    "--csv",
                    str(merged_csv),
                    "--out-csv",
                    str(rr2d_windows_csv),
                    "--out-json",
                    str(rr2d_summary_json),
                    "--out-events-json",
                    str(rr2d_events_json),
                ],
                stage="analysis",
                outputs=(rr2d_windows_csv, rr2d_summary_json, rr2d_events_json),
            ),
            CommandSpec(
                description="RR AFL filter",
                args=["run_rr_afl_filter.py", "--csv", str(merged_csv)],
                stage="analysis",
                outputs=(afl_windows_csv, afl_summary_json, afl_events_json),
            ),
            CommandSpec(
                description="Merge AF family events",
                args=[
                    "merge_af_family_events.py",
                    "--af-events-json",
                    str(rr2d_events_json),
                    "--afl-events-json",
                    str(afl_events_json),
                    "--out-events-json",
                    str(af_family_events_json),
                ],
                stage="analysis",
                outputs=(af_family_events_json,),
            ),
            CommandSpec(
                description="Subitem experiment",
                args=[
                    "subitem_experiment.py",
                    "--edf",
                    str(case.edf),
                    "--merged-csv",
                    str(merged_csv),
                    "--af-events-json",
                    str(af_family_events_json),
                    "--out-csv",
                    str(subitem_csv),
                    "--out-summary",
                    str(subitem_summary_json),
                ],
                stage="analysis",
                outputs=(subitem_csv, subitem_summary_json),
            ),
            CommandSpec(
                description="Evaluate subitem metrics",
                args=[
                    "evaluate_subitem_metrics.py",
                    "--truth-csv",
                    str(case.truth_csv),
                    "--pred-csv",
                    str(subitem_csv),
                    "--tolerance-ms",
                    "80",
                    "--out-figure",
                    str(subitem_evaluation_pdf),
                ],
                stage="analysis",
                outputs=(subitem_evaluation_pdf,),
            ),
            CommandSpec(
                description="Plot RR 2D final timeline",
                args=[
                    "plot_rr_2d_filter_timeline.py",
                    "--windows-csv",
                    str(rr2d_windows_csv),
                    "--beat-csv",
                    str(merged_csv),
                    "--label-column",
                    "final_label",
                    "--out-figure",
                    str(rr2d_timeline_pdf),
                ],
                stage="analysis",
                outputs=(rr2d_timeline_pdf,),
            ),
            CommandSpec(
                description="Plot AFL label timeline",
                args=[
                    "plot_rr_2d_filter_timeline.py",
                    "--windows-csv",
                    str(afl_windows_csv),
                    "--beat-csv",
                    str(merged_csv),
                    "--label-column",
                    "label",
                    "--out-figure",
                    str(afl_label_timeline_pdf),
                ],
                stage="analysis",
                outputs=(afl_label_timeline_pdf,),
            ),
            CommandSpec(
                description="Plot AFL final timeline",
                args=[
                    "plot_rr_2d_filter_timeline.py",
                    "--windows-csv",
                    str(afl_windows_csv),
                    "--beat-csv",
                    str(merged_csv),
                    "--label-column",
                    "final_label",
                    "--out-figure",
                    str(afl_final_timeline_pdf),
                ],
                stage="analysis",
                outputs=(afl_final_timeline_pdf,),
            ),
        ]
    )
    return commands


def selected_by_stage(spec: CommandSpec, stage: str) -> bool:
    return stage == "all" or spec.stage == stage


def should_skip_step(spec: CommandSpec, args: argparse.Namespace) -> bool:
    if spec.stage == "preprocess":
        return not args.rerun_preprocess and existing_outputs(spec.outputs)
    if spec.stage == "analysis":
        return args.skip_existing_analysis and existing_outputs(spec.outputs)
    return False


def quote_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def run_command(command: list[str], cwd: Path, dry_run: bool) -> int:
    print(f"$ {quote_command(command)}", flush=True)
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=cwd, check=False)
    return int(completed.returncode)


def run_case(case: CaseFiles, args: argparse.Namespace) -> bool:
    final_pdf = SCRIPT_DIR / "out" / case.tag / "subitem" / "subitem_evaluation_with_af_family.pdf"
    if args.skip_existing and final_pdf.exists():
        print(f"\n===== SKIP {case.tag}: final evaluation exists =====", flush=True)
        return True

    print(f"\n===== CASE {case.tag} =====", flush=True)
    print(f"EDF: {case.edf.name}", flush=True)
    print(f"Truth CSV: {case.truth_csv.name}", flush=True)

    selected_commands = [spec for spec in build_commands(case, args.leads, args.window_seconds) if selected_by_stage(spec, args.stage)]
    for index, spec in enumerate(selected_commands, start=1):
        print(f"\n[{index}] {spec.description} ({spec.stage})", flush=True)
        if should_skip_step(spec, args):
            outputs = ", ".join(str(path) for path in spec.outputs)
            print(f"[SKIP] existing output(s): {outputs}", flush=True)
            continue

        command = [args.python, *spec.args]
        return_code = run_command(command, cwd=SCRIPT_DIR, dry_run=args.dry_run)
        if return_code != 0:
            print(f"[FAIL] {case.tag}: step {index} returned {return_code}", flush=True)
            return False
    return True


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root folder not found: {root}")

    cases = discover_cases(root, args.case)
    if not cases:
        raise SystemExit("No EDF/CSV case pairs found.")

    print(f"Found {len(cases)} case(s): {', '.join(case.tag for case in cases)}", flush=True)
    print("Preprocess steps are skipped by default when expected outputs already exist.", flush=True)
    if args.stage == "analysis":
        print("Running analysis stage only: steps 5-12.", flush=True)
    failed: list[str] = []
    for case in cases:
        ok = run_case(case, args)
        if not ok:
            failed.append(case.tag)
            if not args.continue_on_error:
                break

    if failed:
        raise SystemExit(f"Failed case(s): {', '.join(failed)}")
    print("\nAll requested cases finished successfully.", flush=True)


if __name__ == "__main__":
    main()
