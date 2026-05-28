"""Run one defense across a manifest, score outputs with Llama-Guard.

Usage:
  python run_defense.py --defense none --manifest eval/manifests/sacred_dev.jsonl --n 30
  python run_defense.py --defense fixed_trim_2s --out eval/results/fixed_trim_2s.jsonl

Two-phase to fit on 8 GB VRAM:
  Phase 1: load LALM, defense produces response per sample.
  Phase 2: free LALM, load Llama-Guard, score each response.
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

# Make conda-env ffmpeg DLLs findable on Windows.
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
from tqdm import tqdm  # noqa: E402

from src.defenses import get_defense  # noqa: E402
from src.harm_score import LlamaGuardScorer, score_safety_judgment  # noqa: E402
from src.lalm import (  # noqa: E402
    DEFAULT_INSTRUCTION,
    SAFETY_JUDGMENT_INSTRUCTION,
    SR,
    Qwen2AudioLALM,
)
from src.manifest import read_manifest  # noqa: E402

DEFAULT_MANIFEST = EVAL_ROOT / "manifests" / "sacred_dev.jsonl"
RESULTS_DIR = EVAL_ROOT / "results"


def load_audio(path: str) -> np.ndarray:
    audio, source_sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if source_sr != SR:
        gcd = math.gcd(int(source_sr), int(SR))
        audio = resample_poly(audio, SR // gcd, source_sr // gcd)
    return audio.astype(np.float32)


def run_lalm_phase(rows: list[dict], defense, task_type: str) -> list[dict]:
    print(f"[Phase 1] Defense: {defense.name}  (task_type={task_type})")
    if hasattr(defense, "set_task_type"):
        defense.set_task_type(task_type)

    needs_lalm = (
        defense.needs_lalm(task_type)
        if hasattr(defense, "needs_lalm")
        else True
    )
    if needs_lalm:
        lalm = Qwen2AudioLALM()
        # Switch the LALM's default instruction based on the manifest's task framing
        if task_type == "safety_judgment":
            lalm.config.instruction = SAFETY_JUDGMENT_INSTRUCTION
        else:
            lalm.config.instruction = DEFAULT_INSTRUCTION
        lalm.config.task_type = task_type
        lalm.load()
    else:
        lalm = SimpleNamespace(config=SimpleNamespace(task_type=task_type))
    defense.setup()

    out = []
    for r in tqdm(rows, desc=f"defense:{defense.name}"):
        audio = load_audio(r["audio_path"])
        try:
            res = defense(audio, SR, lalm)
        except Exception as e:
            res = {
                "response": f"[ERROR: {type(e).__name__}: {e}]",
                "abstained": False,
                "n_lalm_calls": 0,
                "error": str(e),
            }
        out.append({
            "sample_id": r["sample_id"],
            "category": r["category"],
            "benchmark": r["benchmark"],
            "expected_behavior": r.get("expected_behavior"),
            "task_type": task_type,
            "duration_sec": r.get("duration_sec"),
            **res,
        })

    defense.teardown()
    if needs_lalm:
        lalm.unload()
    return out


def run_score_phase_compliance(rows: list[dict], defense) -> list[dict]:
    print("[Phase 2] Llama-Guard scoring + defense finalize")
    scorer = LlamaGuardScorer()
    scorer.load()

    out = []
    for r in tqdm(rows, desc="harm_score"):
        v = scorer.score_agent(r["response"])
        scored = {
            **r,
            "harm_verdict": v.verdict,
            "harm_raw": v.raw,
            "harm_categories": v.category_codes(),
            "is_attack_success": v.is_unsafe,
        }
        finalized = defense.finalize(scored, scorer)
        if finalized is not None and finalized.get("response") != scored["response"]:
            v_final = scorer.score_agent(finalized["response"])
            scored = {
                **finalized,
                "harm_verdict": v_final.verdict,
                "harm_raw": v_final.raw,
                "harm_categories": v_final.category_codes(),
                "is_attack_success": v_final.is_unsafe,
            }
        elif finalized is not None:
            scored = finalized
        out.append(scored)

    scorer.unload()
    return out


def run_score_phase_safety_judgment(rows: list[dict]) -> list[dict]:
    """For SSO/SAO: parse the LALM response for harm identification.
    No external model needed; we just keyword-match the response.
    Defense.finalize is NOT called here (it relies on LG-1B logic
    which doesn't apply to the safety_judgment task)."""
    print("[Phase 2] safety_judgment keyword scoring")
    out = []
    for r in tqdm(rows, desc="safety_judge"):
        v = score_safety_judgment(r["response"])
        out.append({
            **r,
            "judgment_verdict": v.verdict,
            "is_attack_success": v.is_attack_success,
        })
    return out


def summarize(rows: list[dict], defense_name: str) -> None:
    print()
    print(f"Results for defense={defense_name}, N={len(rows)}")
    print("-" * 60)
    by_cat: dict[str, list[bool]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(bool(r["is_attack_success"]))
    for cat, flags in sorted(by_cat.items()):
        rate = sum(flags) / len(flags) if flags else 0.0
        print(f"  {cat:35s} ASR = {rate:6.1%}  ({sum(flags)}/{len(flags)})")
    overall = [r["is_attack_success"] for r in rows]
    rate = sum(overall) / len(overall) if overall else 0.0
    print("-" * 60)
    print(f"  {'overall':35s} ASR = {rate:6.1%}  ({sum(overall)}/{len(overall)})")

    abstain_rate = sum(1 for r in rows if r.get("abstained")) / max(1, len(rows))
    n_calls = sum(r.get("n_lalm_calls", 0) for r in rows)
    avg_calls = n_calls / max(1, len(rows))
    print(f"  abstain rate = {abstain_rate:.1%}  |  avg LALM calls = {avg_calls:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defense", required=True, help="defense name (see src/defenses.py registry)")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--n", type=int, default=None,
                    help="cap to first N rows of the manifest (after stratification)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output JSONL (default: results/<defense>.jsonl)")
    ap.add_argument("--task-type", choices=["compliance", "safety_judgment", "dialogue_safety", "msd_dialogue_safety"], default=None,
                    help="override manifest task_type for experiments")
    ap.add_argument("--asr-model", default=None,
                    help="Override structured provenance ASR model.")
    ap.add_argument("--overlap-asr-model", default=None,
                    help="Use this ASR model when diarization detects overlapping speech. Requires --allow-asr-model-switching.")
    ap.add_argument("--allow-asr-model-switching", action="store_true",
                    help="Allow switching between --asr-model and --overlap-asr-model. Disabled by default to keep Whisper loaded once.")
    ap.add_argument("--asr-max-new-tokens", type=int, default=None,
                    help="Override Whisper max_new_tokens.")
    ap.add_argument("--asr-mode", choices=["diarized_segments", "whole_timestamped"], default=None,
                    help="Structured provenance ASR mode.")
    ap.add_argument("--diarization-model", default=None,
                    help="Override structured provenance diarization model.")
    ap.add_argument("--prefer-cuda", action="store_true",
                    help="Run local diarization/Whisper on CUDA when available.")
    ap.add_argument("--enable-prompt-injection-guard", action="store_true",
                    help="Enable whole-transcript NVIDIA prompt-injection classifier.")
    args = ap.parse_args()

    if args.asr_model:
        os.environ["STRUCTURED_PROVENANCE_ASR_MODEL"] = args.asr_model
    if args.overlap_asr_model:
        os.environ["STRUCTURED_PROVENANCE_OVERLAP_ASR_MODEL"] = args.overlap_asr_model
    if args.allow_asr_model_switching:
        os.environ["STRUCTURED_PROVENANCE_ALLOW_ASR_MODEL_SWITCHING"] = "1"
    if args.asr_max_new_tokens is not None:
        os.environ["STRUCTURED_PROVENANCE_WHISPER_MAX_NEW_TOKENS"] = str(args.asr_max_new_tokens)
    if args.asr_mode:
        os.environ["STRUCTURED_PROVENANCE_ASR_MODE"] = args.asr_mode
    if args.diarization_model:
        os.environ["STRUCTURED_PROVENANCE_DIARIZATION_MODEL"] = args.diarization_model
    if args.prefer_cuda:
        os.environ["STRUCTURED_PROVENANCE_PREFER_CUDA"] = "1"
    if args.enable_prompt_injection_guard:
        os.environ["STRUCTURED_PROVENANCE_ENABLE_PI_CLASSIFIER"] = "1"

    rows = read_manifest(args.manifest)
    if args.n is not None and args.n < len(rows):
        # Stratified pick: round-robin by category.
        by_cat: dict[str, list[dict]] = {}
        for r in rows:
            by_cat.setdefault(r["category"], []).append(r)
        cats = list(by_cat)
        picked: list[dict] = []
        i = 0
        while len(picked) < args.n and any(by_cat.values()):
            cat = cats[i % len(cats)]
            if by_cat[cat]:
                picked.append(by_cat[cat].pop(0))
            i += 1
        rows = picked
    print(f"Loaded {len(rows)} samples from {args.manifest}")

    out_path = args.out or (RESULTS_DIR / f"{args.defense}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine task type from the manifest (assume homogeneous)
    task_type = args.task_type or (rows[0].get("task_type", "compliance") if rows else "compliance")

    defense = get_defense(args.defense)
    rows = run_lalm_phase(rows, defense, task_type)
    if task_type == "safety_judgment":
        rows = run_score_phase_safety_judgment(rows)
    else:
        rows = run_score_phase_compliance(rows, defense)

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summarize(rows, args.defense)
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

