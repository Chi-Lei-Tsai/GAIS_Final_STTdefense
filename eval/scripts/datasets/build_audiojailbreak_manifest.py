"""Build a small AudioJailbreak manifest for the structured guard.

The HF dataset has metadata splits plus audio files. For the first local
probe we use the Origin/origin split and resolve each row's speech file
against the repo's `audio/total_wav` directory.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

def _find_eval_root(script_path: Path) -> Path:
    for parent in [script_path.parent, *script_path.parents]:
        if parent.name == "eval" and (parent / "src").is_dir():
            return parent
    raise RuntimeError(f"Could not locate eval root from {script_path}")


EVAL_ROOT = _find_eval_root(Path(__file__).resolve())
sys.path.insert(0, str(EVAL_ROOT))

from src.manifest import ManifestRow, write_manifest  # noqa: E402

REPO_ID = "MBZUAI/AudioJailbreak"
CACHE_DIR = EVAL_ROOT / "data" / "audiojailbreak_cache"
DEFAULT_OUT = EVAL_ROOT / "manifests" / "audiojailbreak_jailbreakbench_n20.jsonl"
DEFAULT_CATEGORIES = [
    "Disinformation",
    "Fraud/Deception",
    "Harassment/Discrimination",
    "Physical harm",
    "Privacy",
    "Sexual/Adult content",
]


def slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower())
    return text.strip("_") or "unknown"


APT_AUDIO_FOLDERS = {
    "Diva": "Diva_response_jsonl",
    "Gemini2.0_flash": "Gemini2.0_flash_response_jsonl",
    "gpt_4o": "gpt4o_response_jsonl",
    "qwen2": "qwen2_response_jsonl",
    # The SALMONN metadata points at gpt4o_response_jsonl/new_BO paths and
    # this HF repo does not include SALMONN BO audio files, only JSON outputs.
    # Try gpt4o APT audio before falling back to base audio.
    "SALMONN": "gpt4o_response_jsonl",
}


def candidate_repo_paths(
    speech_path: str,
    *,
    config: str,
    split: str,
    prefer_apt_audio: bool,
    prefer_source_audio: bool,
) -> list[str]:
    raw = speech_path.strip().replace("\\", "/").lstrip("./")
    name = Path(raw).stem
    suffix = Path(raw).suffix
    paths: list[str] = []
    if prefer_apt_audio and config == "APT":
        folder = APT_AUDIO_FOLDERS.get(split)
        if folder:
            paths.append(f"inference/response/{folder}/BO/{name}.mp3")
            paths.append(f"inference/response/{folder}/BO/{name}.wav")
    source_paths: list[str] = [raw]
    if suffix.lower() == ".mp3":
        source_paths.append(raw[:-4] + ".wav")
    if raw.startswith("audio/"):
        source_paths.append(raw)
    else:
        source_paths.append("audio/" + raw)
    total_wav_path = f"audio/total_wav/{name}.wav"
    if prefer_source_audio:
        paths.extend(source_paths)
        paths.append(total_wav_path)
    else:
        paths.append(total_wav_path)
        paths.extend(source_paths)
    return list(dict.fromkeys(paths))


def resolve_repo_path(
    speech_path: str,
    all_files: set[str],
    *,
    config: str,
    split: str,
    prefer_apt_audio: bool,
    prefer_source_audio: bool,
) -> str | None:
    for path in candidate_repo_paths(
        speech_path,
        config=config,
        split=split,
        prefer_apt_audio=prefer_apt_audio,
        prefer_source_audio=prefer_source_audio,
    ):
        if path in all_files:
            return path
    return None


def speech_source(speech_path: str) -> str:
    path = speech_path.replace("\\", "/")
    if "jailbreakbench" in path:
        return "jailbreakbench"
    if "Do_Not_Answer" in path:
        return "Do_Not_Answer"
    if "jailbreak_llms" in path:
        return "jailbreak_llms"
    return "other"


def read_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                completed.add(json.loads(line)["sample_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def list_local_repo_files(root: Path) -> set[str]:
    root = root.resolve()
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="Origin")
    ap.add_argument("--split", default="origin")
    ap.add_argument(
        "--n",
        type=int,
        default=20,
        help=(
            "Number of rows to sample after filters. Use --all to keep every "
            "matched row."
        ),
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Keep every row that matches the filters instead of sampling --n rows.",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--local-repo-root",
        type=Path,
        default=None,
        help=(
            "Path to a local MBZUAI/AudioJailbreak dataset repo. When set, "
            "metadata and audio are read from that directory instead of "
            "fetching/listing files through Hugging Face."
        ),
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Append missing rows to an existing manifest instead of rewriting it.",
    )
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument(
        "--all-categories",
        action="store_true",
        help="Do not apply the default category subset.",
    )
    ap.add_argument(
        "--speech-sources",
        nargs="+",
        default=["jailbreakbench"],
        choices=["jailbreakbench", "Do_Not_Answer", "jailbreak_llms", "other"],
        help=(
            "Audio source folders to include. Default avoids very long "
            "jailbreak_llms roleplay clips for quick local probes."
        ),
    )
    ap.add_argument(
        "--max-duration-sec",
        type=float,
        default=75.0,
        help="Skip audio longer than this. Use <= 0 for no duration cap.",
    )
    ap.add_argument(
        "--prefer-apt-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For config=APT, prefer inference/response/*/BO perturbed MP3s when present.",
    )
    ap.add_argument(
        "--prefer-source-audio",
        action="store_true",
        help=(
            "Prefer the dataset row's speech_path before audio/total_wav. "
            "Useful when very large total_wav files are slow to fetch."
        ),
    )
    args = ap.parse_args()

    random.seed(args.seed)
    categories = None if args.all_categories else args.categories
    max_duration_sec = None if args.max_duration_sec <= 0 else args.max_duration_sec
    local_repo_root = args.local_repo_root.resolve() if args.local_repo_root else None
    dataset_source = str(local_repo_root) if local_repo_root else REPO_ID
    print(f"Loading {dataset_source} config={args.config} split={args.split}")
    ds = load_dataset(dataset_source, args.config, split=args.split)

    if local_repo_root:
        all_files = list_local_repo_files(local_repo_root)
    else:
        api = HfApi()
        all_files = set(api.list_repo_files(REPO_ID, repo_type="dataset"))

    rows_by_category: dict[str, list[dict]] = {}
    for row in ds:
        cat = str(row.get("category") or "unknown")
        if categories and cat not in categories:
            continue
        source = speech_source(str(row.get("speech_path") or ""))
        if args.speech_sources and source not in args.speech_sources:
            continue
        repo_path = resolve_repo_path(
            str(row.get("speech_path") or ""),
            all_files,
            config=args.config,
            split=args.split,
            prefer_apt_audio=args.prefer_apt_audio,
            prefer_source_audio=args.prefer_source_audio,
        )
        if repo_path is None:
            continue
        row = dict(row)
        row["_repo_path"] = repo_path
        rows_by_category.setdefault(cat, []).append(row)

    if not rows_by_category:
        sys.exit("No rows matched the requested categories and audio files.")

    if args.all:
        picks = []
        for cat in sorted(rows_by_category):
            picks.extend(rows_by_category[cat])
    else:
        picks: list[dict] = []
        cats = sorted(rows_by_category)
        cursor = 0
        while len(picks) < args.n and cats:
            cat = cats[cursor % len(cats)]
            bucket = rows_by_category[cat]
            if bucket:
                if len(bucket) == len(rows_by_category[cat]):
                    random.shuffle(bucket)
                picks.append(bucket.pop())
            cats = [c for c in cats if rows_by_category[c]]
            cursor += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed_ids(args.out) if args.resume else set()
    mode = "a" if args.resume else "w"
    by_cat: dict[str, int] = {}
    written = 0
    skipped_completed = 0

    print(f"Matched {len(picks)} rows before duration/audio validation")
    print(f"Already completed {len(completed)} rows; writing to {args.out}")

    out_f = args.out.open(mode, encoding="utf-8")
    try:
        for i, row in enumerate(picks, start=1):
            repo_path = row["_repo_path"]
            category = str(row.get("category") or "unknown")
            idx = row.get("index", i)
            dataset_tag = f"{slug(args.config)}_{slug(args.split)}"
            sample_id = f"audiojailbreak_{dataset_tag}_{slug(category)}_{idx}"
            if sample_id in completed:
                skipped_completed += 1
                continue

            if local_repo_root:
                local = local_repo_root / repo_path
            else:
                local = Path(hf_hub_download(
                    REPO_ID,
                    repo_path,
                    repo_type="dataset",
                    cache_dir=str(CACHE_DIR),
                ))
            try:
                info = sf.info(local)
                duration = float(info.frames) / max(1, int(info.samplerate))
            except Exception as exc:
                print(f"Skipping {repo_path}: {exc}", file=sys.stderr)
                continue
            if max_duration_sec is not None and duration > max_duration_sec:
                print(
                    f"Skipping {repo_path}: duration {duration:.1f}s exceeds "
                    f"{max_duration_sec:.1f}s",
                    file=sys.stderr,
                )
                continue

            manifest_row = ManifestRow(
                sample_id=sample_id,
                audio_path=str(local).replace("\\", "/"),
                category=f"audiojailbreak_{slug(category)}",
                benchmark="audiojailbreak",
                duration_sec=duration,
                description=str(row.get("goal") or row.get("prompt") or "")[:500],
                source_file=repo_path,
                expected_behavior="refuse",
                task_type="safety_judgment",
                extra={
                    "dataset_config": args.config,
                    "dataset_split": args.split,
                    "category": row.get("category"),
                    "goal": row.get("goal"),
                    "prompt": row.get("prompt"),
                    "attack_type": row.get("attack_type"),
                    "target_model": row.get("target_model"),
                    "platform": row.get("platform"),
                    "source": row.get("source"),
                    "speech_source": speech_source(str(row.get("speech_path") or "")),
                    "behavior": row.get("behavior"),
                    "speech_path": row.get("speech_path"),
                    "resolved_repo_path": repo_path,
                    "used_apt_audio": repo_path.startswith("inference/response/"),
                },
            )
            out_f.write(json.dumps(manifest_row.as_dict(), ensure_ascii=False) + "\n")
            out_f.flush()
            written += 1
            by_cat[manifest_row.category] = by_cat.get(manifest_row.category, 0) + 1
            print(f"[{i}/{len(picks)}] wrote {sample_id}")
    finally:
        out_f.close()

    print(f"Wrote {written} new rows to {args.out}")
    if skipped_completed:
        print(f"Skipped {skipped_completed} completed rows")
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()

