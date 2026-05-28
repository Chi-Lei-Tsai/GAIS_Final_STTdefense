"""Read all results/<defense>.jsonl files and print a comparison table.

Usage:
  python 03_make_table.py                          # all defenses in results/
  python 03_make_table.py --include none abstain   # just these
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
RESULTS_DIR = EVAL_ROOT / "results"


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def asr(rows: list[dict], filter_cat: str | None = None) -> tuple[int, int]:
    if filter_cat:
        rows = [r for r in rows if r.get("category") == filter_cat]
    if not rows:
        return 0, 0
    return sum(1 for r in rows if r.get("is_attack_success")), len(rows)


def fmt_asr(n: int, d: int) -> str:
    if d == 0:
        return "    -  "
    return f"{n / d * 100:5.1f}% ({n}/{d})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include", nargs="*", default=None,
                    help="only these defense names; default = all in results/")
    args = ap.parse_args()

    files = sorted(RESULTS_DIR.glob("*.jsonl"))
    if args.include:
        files = [f for f in files if f.stem in args.include]
    if not files:
        sys.exit("no results files found")

    runs: list[tuple[str, list[dict]]] = []
    cats: set[str] = set()
    for f in files:
        rows = read_jsonl(f)
        runs.append((f.stem, rows))
        cats.update(r["category"] for r in rows)
    cats_sorted = sorted(cats)

    # Header
    name_w = max(20, max(len(r[0]) for r in runs) + 2)
    cat_w = 16
    print(f"{'defense':<{name_w}} | {'overall':<{cat_w}} | "
          + " | ".join(f"{c[:cat_w]:<{cat_w}}" for c in cats_sorted)
          + " | abstain | avg_calls")
    print("-" * (name_w + 3 + (cat_w + 3) * (len(cats_sorted) + 1) + 22))

    for name, rows in runs:
        if not rows:
            continue
        # overall
        n_unsafe, n_total = asr(rows)
        line = f"{name:<{name_w}} | {fmt_asr(n_unsafe, n_total):<{cat_w}}"
        for c in cats_sorted:
            n, d = asr(rows, c)
            line += f" | {fmt_asr(n, d):<{cat_w}}"
        n_abst = sum(1 for r in rows if r.get("abstained"))
        avg_calls = sum(r.get("n_lalm_calls", 0) for r in rows) / max(1, len(rows))
        line += f" | {n_abst / max(1, len(rows)) * 100:5.1f}% | {avg_calls:5.2f}"
        print(line)


if __name__ == "__main__":
    main()

