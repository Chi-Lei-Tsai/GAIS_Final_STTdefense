"""Incremental runner for the structured-provenance safety-judgment path.

This is intentionally narrower than run_defense.py: it is for SSO/SAO-style
manifests where the defense can answer the safety judgment directly and no
target LALM call is needed. Each completed row is written immediately so a long
CPU run can be resumed without losing earlier samples.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

def _find_eval_root(script_path: Path) -> Path:
    for parent in [script_path.parent, *script_path.parents]:
        if parent.name == "eval" and (parent / "src").is_dir():
            return parent
    raise RuntimeError(f"Could not locate eval root from {script_path}")


EVAL_ROOT = _find_eval_root(Path(__file__).resolve())
sys.path.insert(0, str(EVAL_ROOT))

_env_lib_bin = Path(sys.executable).parent / "Library" / "bin"
if _env_lib_bin.exists():
    os.environ["PATH"] = str(_env_lib_bin) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(_env_lib_bin))
        except OSError:
            pass

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

from src.defenses import get_defense  # noqa: E402
from src.harm_score import score_safety_judgment  # noqa: E402
from src.lalm import SR  # noqa: E402
from src.manifest import read_manifest  # noqa: E402


def load_audio(path: str) -> np.ndarray:
    audio, source_sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if source_sr != SR:
        gcd = math.gcd(int(source_sr), int(SR))
        audio = resample_poly(audio, SR // gcd, source_sr // gcd)
    return audio.astype(np.float32)


def read_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("error"):
                    continue
                completed.add(row["sample_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def summarize(path: Path) -> None:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print()
    print(f"Results in {path}  N={len(rows)}")
    print("-" * 60)
    by_cat: dict[str, list[bool]] = {}
    for row in rows:
        by_cat.setdefault(row["category"], []).append(bool(row["is_attack_success"]))
    for cat, flags in sorted(by_cat.items()):
        rate = sum(flags) / len(flags) if flags else 0.0
        print(f"  {cat:35s} ASR = {rate:6.1%}  ({sum(flags)}/{len(flags)})")
    flags = [bool(row["is_attack_success"]) for row in rows]
    rate = sum(flags) / len(flags) if flags else 0.0
    abstain_rate = sum(1 for row in rows if row.get("abstained")) / max(1, len(rows))
    avg_calls = sum(row.get("n_lalm_calls", 0) for row in rows) / max(1, len(rows))
    print("-" * 60)
    print(f"  {'overall':35s} ASR = {rate:6.1%}  ({sum(flags)}/{len(flags)})")
    print(f"  abstain rate = {abstain_rate:.1%}  |  avg LALM calls = {avg_calls:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--defense", default="structured_provenance")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument(
        "--asr-model",
        default=None,
        help=(
            "Override structured provenance ASR model. Also available via "
            "STRUCTURED_PROVENANCE_ASR_MODEL."
        ),
    )
    ap.add_argument(
        "--overlap-asr-model",
        default=None,
        help=(
            "Use this ASR model when diarization detects overlapping speech. "
            "Requires --allow-asr-model-switching. Also available via "
            "STRUCTURED_PROVENANCE_OVERLAP_ASR_MODEL."
        ),
    )
    ap.add_argument(
        "--allow-asr-model-switching",
        action="store_true",
        help=(
            "Allow the pipeline to switch between --asr-model and "
            "--overlap-asr-model. Disabled by default so Whisper-large is "
            "loaded once and reused for the whole run."
        ),
    )
    ap.add_argument(
        "--asr-max-new-tokens",
        type=int,
        default=None,
        help=(
            "Override Whisper max_new_tokens per diarized segment. Also "
            "available via STRUCTURED_PROVENANCE_WHISPER_MAX_NEW_TOKENS."
        ),
    )
    ap.add_argument(
        "--asr-mode",
        choices=["diarized_segments", "whole_timestamped"],
        default=None,
        help=(
            "ASR timeline construction mode. diarized_segments preserves the "
            "old diarization-first behavior; whole_timestamped runs one "
            "timestamped whole-audio ASR pass and aligns diarization after."
        ),
    )
    ap.add_argument(
        "--diarization-model",
        default=None,
        help=(
            "Override structured provenance diarization model. Also available "
            "via STRUCTURED_PROVENANCE_DIARIZATION_MODEL."
        ),
    )
    ap.add_argument(
        "--enable-prompt-injection-guard",
        action="store_true",
        help=(
            "Enable whole-transcript prompt-injection/policy-bypass "
            "classification via NVIDIA API."
        ),
    )
    ap.add_argument(
        "--prefer-cuda",
        action="store_true",
        help="Run local diarization/Whisper on CUDA when available.",
    )
    ap.add_argument(
        "--disable-overlap-asr-retry",
        action="store_true",
        help=(
            "Skip padded ASR retries for overlap/late suspect segments. This is "
            "faster but less faithful for overlap-heavy SSO/MSD-style audio."
        ),
    )
    ap.add_argument(
        "--disable-targeted-overlap-asr",
        action="store_true",
        help=(
            "In whole_timestamped mode, skip targeted ASR on diarized overlap "
            "windows."
        ),
    )
    ap.add_argument(
        "--prompt-injection-model",
        default=None,
        help=(
            "NVIDIA text model for whole-transcript prompt-injection "
            "classification. Default: meta/llama-3.1-8b-instruct."
        ),
    )
    args = ap.parse_args()

    if args.asr_model:
        os.environ["STRUCTURED_PROVENANCE_ASR_MODEL"] = args.asr_model
    if args.overlap_asr_model:
        os.environ["STRUCTURED_PROVENANCE_OVERLAP_ASR_MODEL"] = args.overlap_asr_model
    if args.allow_asr_model_switching:
        os.environ["STRUCTURED_PROVENANCE_ALLOW_ASR_MODEL_SWITCHING"] = "1"
    if args.asr_max_new_tokens is not None:
        os.environ["STRUCTURED_PROVENANCE_WHISPER_MAX_NEW_TOKENS"] = str(
            args.asr_max_new_tokens
        )
    if args.asr_mode:
        os.environ["STRUCTURED_PROVENANCE_ASR_MODE"] = args.asr_mode
    if args.diarization_model:
        os.environ["STRUCTURED_PROVENANCE_DIARIZATION_MODEL"] = args.diarization_model
    if args.enable_prompt_injection_guard:
        os.environ["STRUCTURED_PROVENANCE_ENABLE_PI_CLASSIFIER"] = "1"
    if args.prefer_cuda:
        os.environ["STRUCTURED_PROVENANCE_PREFER_CUDA"] = "1"
    if args.disable_overlap_asr_retry:
        os.environ["STRUCTURED_PROVENANCE_ENABLE_OVERLAP_ASR_RETRY"] = "0"
    if args.disable_targeted_overlap_asr:
        os.environ["STRUCTURED_PROVENANCE_ENABLE_TARGETED_OVERLAP_ASR"] = "0"
    if args.prompt_injection_model:
        os.environ["STRUCTURED_PROVENANCE_PI_MODEL"] = args.prompt_injection_model

    rows = read_manifest(args.manifest)
    if args.start:
        rows = rows[args.start:]
    if args.limit is not None:
        rows = rows[: args.limit]

    task_type = rows[0].get("task_type", "compliance") if rows else "safety_judgment"
    if task_type != "safety_judgment":
        raise ValueError(
            "Incremental runner only supports safety_judgment manifests. "
            "Use run_defense.py for compliance tasks."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    completed = read_completed_ids(args.out) if args.resume else set()
    mode = "a" if args.resume else "w"
    pending = [row for row in rows if row["sample_id"] not in completed]

    print(f"Loaded {len(rows)} rows from {args.manifest}")
    print(f"Already completed {len(completed)} rows; pending {len(pending)}")
    print(f"Writing incrementally to {args.out}")

    defense = get_defense(args.defense)
    if hasattr(defense, "set_task_type"):
        defense.set_task_type(task_type)
    defense.setup()
    lalm = SimpleNamespace(config=SimpleNamespace(task_type=task_type))

    try:
        with args.out.open(mode, encoding="utf-8") as f:
            for idx, row in enumerate(pending, start=1):
                audio = load_audio(row["audio_path"])
                try:
                    res = defense(audio, SR, lalm)
                except Exception as exc:  # keep the run resumable
                    res = {
                        "response": f"[ERROR: {type(exc).__name__}: {exc}]",
                        "abstained": False,
                        "n_lalm_calls": 0,
                        "error": str(exc),
                    }
                scored = score_safety_judgment(res["response"])
                record = {
                    "sample_id": row["sample_id"],
                    "category": row["category"],
                    "benchmark": row["benchmark"],
                    "expected_behavior": row.get("expected_behavior"),
                    "task_type": task_type,
                    "duration_sec": row.get("duration_sec"),
                    **res,
                    "judgment_verdict": scored.verdict,
                    "is_attack_success": scored.is_attack_success,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                decision = record.get("guard_decision", {}).get("decision", "error")
                print(
                    f"[{idx}/{len(pending)}] {row['sample_id']} "
                    f"decision={decision} "
                    f"attack_success={record['is_attack_success']}"
                )
    finally:
        defense.teardown()

    summarize(args.out)


if __name__ == "__main__":
    main()

