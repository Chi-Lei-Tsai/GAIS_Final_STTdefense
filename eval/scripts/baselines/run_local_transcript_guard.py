"""Run the local LlamaGuard scorer on transcript-manifest rows.

This is a text-only ASR+guard ablation for structured-provenance outputs:
audio has already been transcribed, and the guard receives only the flat
transcript field from the manifest.
"""

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

from src.harm_score import LlamaGuardScorer  # noqa: E402
from src.manifest import read_manifest  # noqa: E402


def normalize_text(text: str | None) -> str:
    return " ".join((text or "").split())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def completed_ids(path: Path) -> set[str]:
    return {
        row["sample_id"]
        for row in read_jsonl(path)
        if row.get("lg_verdict") in {"safe", "unsafe"}
    }


def summarize(path: Path) -> None:
    rows = read_jsonl(path)
    valid = [row for row in rows if row.get("lg_verdict") in {"safe", "unsafe"}]
    blocked = [row for row in valid if row.get("blocked")]
    passed = [row for row in valid if row.get("passed")]
    print()
    print(f"Results in {path}  N={len(rows)}  scored={len(valid)}")
    print("-" * 72)
    denom = len(valid) if valid else 1
    print(f"  {'blocked unsafe':35s} = {len(blocked) / denom:6.1%}  ({len(blocked)}/{len(valid)})")
    print(f"  {'passed safe':35s} = {len(passed) / denom:6.1%}  ({len(passed)}/{len(valid)})")
    print("-" * 72)
    by_cat: dict[str, list[bool]] = {}
    for row in valid:
        by_cat.setdefault(row.get("category", "unknown"), []).append(bool(row.get("blocked")))
    for category, flags in sorted(by_cat.items()):
        rate = sum(flags) / len(flags) if flags else 0.0
        print(f"  {category:35s} block = {rate:6.1%}  ({sum(flags)}/{len(flags)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--transcript-field", default="dialogue_transcript")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    if args.n is not None:
        rows = rows[: args.n]

    done = completed_ids(args.out) if args.resume else set()
    pending = [row for row in rows if row["sample_id"] not in done]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    scorer = LlamaGuardScorer(model_id=args.model_id) if args.model_id else LlamaGuardScorer()
    scorer.load()

    print(f"Guard model: {scorer.model_id}")
    print(f"Transcript field: extra.{args.transcript_field}")
    print(f"Loaded {len(rows)} rows from {args.manifest}")
    print(f"Pending {len(pending)} rows; writing to {args.out}")

    mode = "a" if args.resume else "w"
    try:
        with args.out.open(mode, encoding="utf-8") as f:
            for idx, row in enumerate(pending, start=1):
                extra = row.get("extra") or {}
                transcript = normalize_text(extra.get(args.transcript_field))
                out_row = {
                    "sample_id": row["sample_id"],
                    "category": row["category"],
                    "benchmark": row["benchmark"],
                    "expected_behavior": row.get("expected_behavior"),
                    "source_file": row.get("source_file"),
                    "guard_model": scorer.model_id,
                    "input_mode": "flat_transcript_local_llamaguard",
                    "transcript_field": args.transcript_field,
                    "transcript": transcript,
                }
                if not transcript:
                    out_row.update({
                        "error": f"Missing extra.{args.transcript_field}",
                        "lg_verdict": "error",
                        "blocked": None,
                        "passed": None,
                    })
                else:
                    verdict = scorer.score_user(transcript)
                    out_row.update({
                        "lg_raw": verdict.raw,
                        "lg_verdict": verdict.verdict,
                        "lg_categories": verdict.category_codes(),
                        "blocked": verdict.is_unsafe,
                        "passed": not verdict.is_unsafe,
                    })
                f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                f.flush()
                print(
                    f"[{idx}/{len(pending)}] {row['sample_id']} "
                    f"lg={out_row.get('lg_verdict')} "
                    f"blocked={out_row.get('blocked')} "
                    f"error={bool(out_row.get('error'))}"
                )
    finally:
        scorer.unload()

    summarize(args.out)


if __name__ == "__main__":
    main()

