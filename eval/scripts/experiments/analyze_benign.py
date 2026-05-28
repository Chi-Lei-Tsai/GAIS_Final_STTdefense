"""Summarize over-refusal rates from defense runs on benign samples.

Reads results/benign/<defense>.jsonl files. For each defense, counts:
  - explicit abstain: defense.abstained == True
  - refusal-shaped response: response looks like a refusal but defense
    didn't set abstained
  - any-refusal: union of the two

A high any-refusal rate on benign samples = the defense is over-refusing
and destroying utility on legitimate content.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

def _find_eval_root(script_path: Path) -> Path:
    for parent in [script_path.parent, *script_path.parents]:
        if parent.name == "eval" and (parent / "src").is_dir():
            return parent
    raise RuntimeError(f"Could not locate eval root from {script_path}")


EVAL_ROOT = _find_eval_root(Path(__file__).resolve())
sys.path.insert(0, str(EVAL_ROOT))
from src.defenses import looks_like_refusal  # noqa: E402

DEFAULT_DIR = EVAL_ROOT / "results" / "benign"


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=Path, default=DEFAULT_DIR)
    args = ap.parse_args()

    files = sorted(args.results_dir.glob("*.jsonl"))
    if not files:
        sys.exit(f"no result files in {args.results_dir}")

    name_w = max(20, max(len(f.stem) for f in files) + 2)
    cat_w = 14
    print(f"{'defense':<{name_w}} | {'overall':<{cat_w}} | "
          f"{'late_content':<{cat_w}} | {'dialogue':<{cat_w}} | abstain | shape_refused")
    print("-" * (name_w + 3 + (cat_w + 3) * 3 + 28))

    for f in files:
        rows = read_jsonl(f)
        if not rows:
            continue

        late = [r for r in rows if r["category"] == "benign_late_content_required"]
        dial = [r for r in rows if r["category"] == "benign_multi_speaker_dialogue"]

        def refuse_rate(rs: list[dict]) -> tuple[int, int, int, int]:
            n = len(rs)
            if n == 0:
                return 0, 0, 0, 0
            abst = sum(1 for r in rs if r.get("abstained"))
            shape = sum(
                1 for r in rs if not r.get("abstained") and looks_like_refusal(r["response"])
            )
            any_ref = abst + shape
            return any_ref, abst, shape, n

        ovr_a, ovr_e, ovr_s, ovr_n = refuse_rate(rows)
        late_a, _, _, late_n = refuse_rate(late)
        dial_a, _, _, dial_n = refuse_rate(dial)

        def fmt(a: int, n: int) -> str:
            if n == 0:
                return "    -    "
            return f"{a / n * 100:5.1f}% ({a}/{n})"

        print(
            f"{f.stem:<{name_w}} | "
            f"{fmt(ovr_a, ovr_n):<{cat_w}} | "
            f"{fmt(late_a, late_n):<{cat_w}} | "
            f"{fmt(dial_a, dial_n):<{cat_w}} | "
            f"{ovr_e:5d}   | {ovr_s:5d}"
        )


if __name__ == "__main__":
    main()

