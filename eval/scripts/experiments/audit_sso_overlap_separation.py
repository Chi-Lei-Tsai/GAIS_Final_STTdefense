"""Audit whether SSO structured timelines separated overlapping speech."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def overlap_sec(first: dict[str, Any], second: dict[str, Any]) -> float:
    return max(
        0.0,
        min(float(first["end_sec"]), float(second["end_sec"]))
        - max(float(first["start_sec"]), float(second["start_sec"])),
    )


def audit_row(row: dict[str, Any], min_overlap_sec: float) -> dict[str, Any]:
    segments = (row.get("structured_input") or {}).get("segments") or []
    speech = [seg for seg in segments if seg.get("kind") == "speech"]
    nonempty_speech = [seg for seg in speech if (seg.get("transcript") or "").strip()]
    speakers = sorted({seg.get("speaker") for seg in speech if seg.get("speaker")})
    secondary = [seg for seg in speech if seg.get("speaker_role") == "secondary_candidate"]
    unsafe = [
        seg for seg in speech
        if (seg.get("transcript_evidence") or {}).get("is_unsafe") is True
    ]
    unsafe_secondary = [
        seg for seg in unsafe
        if seg.get("speaker_role") == "secondary_candidate"
    ]
    flagged_overlap = [seg for seg in speech if seg.get("overlap")]
    pairs: list[tuple[str, str, float, str | None, str | None]] = []
    for idx, first in enumerate(speech):
        for second in speech[idx + 1:]:
            amount = overlap_sec(first, second)
            if amount >= min_overlap_sec:
                pairs.append((
                    first.get("id", ""),
                    second.get("id", ""),
                    round(amount, 3),
                    first.get("speaker"),
                    second.get("speaker"),
                ))
    return {
        "sample_id": row.get("sample_id"),
        "is_attack_success": row.get("is_attack_success"),
        "n_speech": len(speech),
        "n_nonempty_speech": len(nonempty_speech),
        "n_speakers": len(speakers),
        "n_secondary": len(secondary),
        "n_unsafe": len(unsafe),
        "n_unsafe_secondary": len(unsafe_secondary),
        "n_actual_overlap_pairs": len(pairs),
        "n_overlap_flagged_segments": len(flagged_overlap),
        "actual_overlap_pairs": pairs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structured-results", type=Path, required=True)
    ap.add_argument("--min-overlap-sec", type=float, default=0.05)
    ap.add_argument("--show-misses", action="store_true")
    args = ap.parse_args()

    audits = [
        audit_row(row, args.min_overlap_sec)
        for row in read_jsonl(args.structured_results)
    ]
    counts = Counter()
    for row in audits:
        if row["n_speakers"] >= 2:
            counts["rows_two_plus_speakers"] += 1
        if row["n_secondary"]:
            counts["rows_with_secondary"] += 1
        if row["n_unsafe"]:
            counts["rows_with_unsafe_segment"] += 1
        if row["n_unsafe_secondary"]:
            counts["rows_with_unsafe_secondary"] += 1
        if row["n_actual_overlap_pairs"]:
            counts["rows_with_actual_time_overlap"] += 1
        if row["n_overlap_flagged_segments"]:
            counts["rows_with_overlap_flag"] += 1
        if row["n_actual_overlap_pairs"] and not row["n_overlap_flagged_segments"]:
            counts["rows_actual_overlap_but_no_flag"] += 1
        if row["n_nonempty_speech"] >= 2:
            counts["rows_two_plus_nonempty_speech_segments"] += 1

    n = len(audits)
    print(f"Rows: {n}")
    for key in sorted(counts):
        print(f"{key}: {counts[key]}/{n}")
    print(f"total_speech_segments: {sum(row['n_speech'] for row in audits)}")
    print(f"total_nonempty_speech_segments: {sum(row['n_nonempty_speech'] for row in audits)}")
    print(f"total_overlap_flagged_segments: {sum(row['n_overlap_flagged_segments'] for row in audits)}")
    print(f"total_actual_overlap_pairs: {sum(row['n_actual_overlap_pairs'] for row in audits)}")

    misses = [row for row in audits if row["is_attack_success"] is True]
    print(f"misses: {len(misses)}/{n}")
    if args.show_misses:
        for row in misses:
            print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
