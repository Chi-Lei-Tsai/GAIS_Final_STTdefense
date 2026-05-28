"""Build a target-model manifest from rows that passed a guard run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

def _find_eval_root(script_path: Path) -> Path:
    for parent in [script_path.parent, *script_path.parents]:
        if parent.name == "eval" and (parent / "src").is_dir():
            return parent
    raise RuntimeError(f"Could not locate eval root from {script_path}")


EVAL_ROOT = _find_eval_root(Path(__file__).resolve())
sys.path.insert(0, str(EVAL_ROOT))

from src.manifest import ManifestRow, read_manifest, write_manifest  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_guard_pass(row: dict[str, Any], pass_decisions: set[str]) -> bool:
    if row.get("error"):
        return False
    decision = (row.get("guard_decision") or {}).get("decision")
    if decision not in pass_decisions:
        return False
    if row.get("abstained") is True:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-manifest", type=Path, required=True)
    ap.add_argument("--guard-results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--pass-decisions",
        nargs="+",
        default=["answer_reconstructed"],
        help="Guard decision values that count as passing the input onward.",
    )
    ap.add_argument(
        "--task-type",
        default="compliance",
        choices=["compliance", "safety_judgment"],
        help="Task type to write for the target-model manifest.",
    )
    args = ap.parse_args()

    source_by_id = {row["sample_id"]: row for row in read_manifest(args.source_manifest)}
    pass_decisions = set(args.pass_decisions)
    rows: list[ManifestRow] = []

    total = 0
    passed = 0
    missing_source = 0
    for result in read_jsonl(args.guard_results):
        total += 1
        if not is_guard_pass(result, pass_decisions):
            continue
        passed += 1
        source = source_by_id.get(result["sample_id"])
        if source is None:
            missing_source += 1
            continue
        extra = dict(source.get("extra") or {})
        extra.update({
            "source_guard_results": str(args.guard_results),
            "guard_decision": result.get("guard_decision"),
            "guard_response": result.get("response"),
            "guard_judgment_verdict": result.get("judgment_verdict"),
            "guard_is_attack_success": result.get("is_attack_success"),
            "guard_structured_input": result.get("structured_input"),
        })
        rows.append(
            ManifestRow(
                sample_id=source["sample_id"],
                audio_path=source["audio_path"],
                category=source["category"],
                benchmark=source["benchmark"],
                duration_sec=source.get("duration_sec"),
                description=source.get("description"),
                source_file=source.get("source_file"),
                expected_behavior=source.get("expected_behavior", "refuse"),
                task_type=args.task_type,
                injection_start_sec=source.get("injection_start_sec"),
                injection_end_sec=source.get("injection_end_sec"),
                primary_speaker_id=source.get("primary_speaker_id"),
                extra=extra,
            )
        )

    write_manifest(rows, args.out)
    print(f"Read {total} guard rows from {args.guard_results}")
    print(f"Guard-passed rows: {passed}")
    print(f"Rows written: {len(rows)}")
    if missing_source:
        print(f"Missing source rows: {missing_source}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

