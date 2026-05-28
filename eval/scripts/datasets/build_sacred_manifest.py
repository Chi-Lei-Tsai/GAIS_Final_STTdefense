"""Build a SACRED-Bench manifest by listing files in the HF repo and
downloading a stratified subsample to local cache.

CPU only. Idempotent (skips files already cached).

Usage:
  python 01_build_sacred_manifest.py --n_per_category 10
  python 01_build_sacred_manifest.py --n_per_category 50 --out manifests/sacred_50.jsonl
  python 01_build_sacred_manifest.py --categories Multi-speaker_Dialogue --n_per_category 100
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

def _find_eval_root(script_path: Path) -> Path:
    for parent in [script_path.parent, *script_path.parents]:
        if parent.name == "eval" and (parent / "src").is_dir():
            return parent
    raise RuntimeError(f"Could not locate eval root from {script_path}")


EVAL_ROOT = _find_eval_root(Path(__file__).resolve())
sys.path.insert(0, str(EVAL_ROOT))
from src.manifest import ManifestRow, write_manifest  # noqa: E402

import soundfile as sf  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402
from tqdm import tqdm  # noqa: E402

SACRED_REPO = "tsinghua-ee/SACRED-Bench"
SACRED_CATEGORIES = ["Speech_Overlap", "Multi-speaker_Dialogue", "Contextual_Audio_Cues"]
DEFAULT_OUT = EVAL_ROOT / "manifests" / "sacred_dev.jsonl"
CACHE_DIR = EVAL_ROOT / "data" / "sacred_cache"


def normalize_category(cat_name: str) -> str:
    """Map HF folder name -> our manifest category string."""
    return cat_name.lower().replace("-", "_").replace(" ", "_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_category", type=int, default=10,
                    help="how many samples to download per category")
    ap.add_argument("--categories", nargs="+", default=SACRED_CATEGORIES,
                    choices=SACRED_CATEGORIES,
                    help="which SACRED-Bench categories to include")
    ap.add_argument("--task_type", default="compliance",
                    choices=["compliance", "safety_judgment"],
                    help="task framing: 'compliance' for MSD (default), "
                         "'safety_judgment' for SSO/SAO")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output manifest JSONL path")
    args = ap.parse_args()

    random.seed(args.seed)
    api = HfApi()
    print(f"Listing files in {SACRED_REPO} ...")
    all_files = api.list_repo_files(SACRED_REPO, repo_type="dataset")
    print(f"  {len(all_files)} files in repo")
    print(f"  selected categories: {args.categories}")

    rows: list[ManifestRow] = []
    for cat in args.categories:
        cat_files = sorted(f for f in all_files if f.startswith(cat + "/") and f.endswith(".wav"))
        if not cat_files:
            print(f"  [{cat}] no .wav files found", file=sys.stderr)
            continue
        random.shuffle(cat_files)
        picks = cat_files[: args.n_per_category]
        print(f"  [{cat}] sampling {len(picks)} of {len(cat_files)} wavs")

        for repo_path in tqdm(picks, desc=cat):
            local = hf_hub_download(SACRED_REPO, repo_path, repo_type="dataset",
                                    cache_dir=str(CACHE_DIR))
            try:
                info = sf.info(local)
                duration = info.frames / info.samplerate
            except Exception as e:
                print(f"    skip {repo_path}: {e}", file=sys.stderr)
                continue

            stem = Path(repo_path).stem
            sid = f"sacred_{normalize_category(cat)}_{stem}"
            description = stem.replace("_", " ")

            rows.append(
                ManifestRow(
                    sample_id=sid,
                    audio_path=str(local).replace("\\", "/"),
                    category=normalize_category(cat),
                    benchmark="sacred",
                    duration_sec=duration,
                    description=description,
                    source_file=repo_path,
                    expected_behavior="refuse",
                    task_type=args.task_type,
                )
            )

    write_manifest(rows, args.out)
    print(f"\nWrote {len(rows)} rows to {args.out}")
    by_cat: dict[str, int] = {}
    for r in rows:
        by_cat[r.category] = by_cat.get(r.category, 0) + 1
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()

