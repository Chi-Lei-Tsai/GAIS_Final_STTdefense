"""Build a transcript manifest from structured-provenance result JSONL."""

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


def normalize_text(text: str | None) -> str:
    return " ".join((text or "").split())


def transcript_from_structured(row: dict[str, Any], speaker_prefix_mode: str) -> str:
    structured = row.get("structured_input") or {}
    pieces: list[str] = []
    for segment in structured.get("segments") or []:
        if segment.get("kind") != "speech":
            continue
        transcript = normalize_text(segment.get("transcript"))
        if transcript:
            if speaker_prefix_mode == "speaker":
                speaker = segment.get("speaker") or "speaker"
                pieces.append(f"{speaker}: {transcript}")
            else:
                pieces.append(transcript)
    return " ".join(pieces)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structured-results", type=Path, required=True)
    ap.add_argument("--source-manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--task-source",
        choices=["primary_audio", "text_input", "uncertain"],
        default="primary_audio",
    )
    ap.add_argument(
        "--speaker-prefix-mode",
        choices=["speaker", "none"],
        default="speaker",
        help="Whether to prefix each transcript span with the diarized speaker label.",
    )
    args = ap.parse_args()

    source_by_id = {row["sample_id"]: row for row in read_manifest(args.source_manifest)}
    rows: list[ManifestRow] = []
    for result in read_jsonl(args.structured_results):
        sample_id = result["sample_id"]
        source = source_by_id.get(sample_id, {})
        transcript = transcript_from_structured(result, args.speaker_prefix_mode)
        if not transcript:
            continue
        rows.append(
            ManifestRow(
                sample_id=sample_id,
                audio_path=source.get("audio_path", ""),
                category=result.get("category") or source.get("category") or "unknown",
                benchmark=result.get("benchmark") or source.get("benchmark") or "unknown",
                duration_sec=result.get("duration_sec") or source.get("duration_sec"),
                description=source.get("description") or transcript[:120],
                source_file=result.get("source_file") or source.get("source_file"),
                expected_behavior=source.get("expected_behavior", result.get("expected_behavior", "answer")),
                task_type="transcript_guard",
                extra={
                    **(source.get("extra") or {}),
                    "dialogue_transcript": transcript,
                    "task_source": args.task_source,
                    "source_structured_result": str(args.structured_results),
                    "source_task_type": result.get("task_type"),
                },
            )
        )

    write_manifest(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

