"""Run NVIDIA chat models on SACRED MSD transcript + revised_query only.

This is an oracle-transcript baseline: SACRED's `GPT_description` is the
script used to synthesize the audio, not an ASR transcript produced by
our pipeline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

def _find_eval_root(script_path: Path) -> Path:
    for parent in [script_path.parent, *script_path.parents]:
        if parent.name == "eval" and (parent / "src").is_dir():
            return parent
    raise RuntimeError(f"Could not locate eval root from {script_path}")


EVAL_ROOT = _find_eval_root(Path(__file__).resolve())
REPO_ROOT = EVAL_ROOT.parent
sys.path.insert(0, str(EVAL_ROOT))

from src.manifest import read_manifest  # noqa: E402


def _load_baseline_module():
    path = EVAL_ROOT / "scripts" / "baselines" / "run_nvidia_api_baseline.py"
    spec = importlib.util.spec_from_file_location("nvidia_api_baseline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = _load_baseline_module()
NvidiaAudioChatClient = baseline.NvidiaAudioChatClient
default_system_prompt = baseline.default_system_prompt
resolve_model = baseline.resolve_model
safe_error = baseline.safe_error


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def record_key(row_index: int | None, row: dict[str, Any]) -> str:
    return f"{row_index}:{row.get('sample_id')}:{row.get('source_file')}"


def completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    for row in read_jsonl(path):
        if row.get("error"):
            continue
        keys.add(record_key(row.get("row_index"), row))
    return keys


def normalize_text(text: str | None) -> str:
    return " ".join((text or "").split())


def build_prompt(row: dict[str, Any], transcript_field: str, prompt_mode: str) -> str:
    extra = row.get("extra") or {}
    transcript = normalize_text(extra.get(transcript_field))
    revised_query = normalize_text(extra.get("revised_query"))
    if not transcript:
        raise RuntimeError(f"Missing extra.{transcript_field}")
    if not revised_query:
        raise RuntimeError("Missing extra.revised_query")
    if prompt_mode == "bare_query_transcript":
        return f"{revised_query}\n\n{transcript}"
    return (
        "You are given the transcript of an audio conversation. "
        "Answer the user's question based on the transcript.\n\n"
        "Conversation transcript:\n"
        f"{transcript}\n\n"
        "User question:\n"
        f"{revised_query}"
    )


def summarize(path: Path) -> None:
    rows = read_jsonl(path)
    valid = [row for row in rows if not row.get("error")]
    blanks = [
        row for row in valid
        if not (row.get("response") and row.get("response").strip())
    ]
    print()
    print(f"Results in {path}  N={len(rows)}  valid={len(valid)}  errors={len(rows) - len(valid)}")
    print(f"  blank responses = {len(blanks)}/{len(valid) if valid else 0}")
    by_finish: dict[str, int] = {}
    for row in valid:
        key = str(row.get("finish_reason"))
        by_finish[key] = by_finish.get(key, 0) + 1
    for key, count in sorted(by_finish.items()):
        print(f"  finish_reason={key}: {count}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--blank-retry-max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout-sec", type=int, default=180)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument(
        "--transcript-field",
        default="dialogue_transcript",
        choices=["dialogue_transcript", "tts_transcript"],
    )
    ap.add_argument(
        "--prompt-mode",
        default="labeled",
        choices=["labeled", "bare_query_transcript"],
        help=(
            "labeled uses explanatory transcript/query labels; "
            "bare_query_transcript sends only revised_query, blank line, transcript."
        ),
    )
    ap.add_argument("--system-prompt", default=None)
    ap.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Do not send the model-specific default system prompt.",
    )
    ap.add_argument("--thinking-token-budget", type=int, default=None)
    args = ap.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing NVIDIA API key. Set ${args.api_key_env}.")

    model = resolve_model(args.model)
    rows = list(enumerate(read_manifest(args.manifest)))
    if args.n is not None:
        rows = rows[: args.n]
    completed = completed_keys(args.out) if args.resume else set()
    pending = [
        item for item in rows
        if record_key(item[0], item[1]) not in completed
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    client = NvidiaAudioChatClient(
        api_key=api_key,
        timeout_sec=args.timeout_sec,
        max_retries=args.max_retries,
    )
    if args.no_system_prompt:
        system_prompt = None
    else:
        system_prompt = (
            args.system_prompt
            if args.system_prompt is not None
            else default_system_prompt(model)
        )

    print(f"Model: {model}")
    print(f"Transcript field: extra.{args.transcript_field}")
    print(f"Prompt mode: {args.prompt_mode}")
    print(f"Loaded {len(rows)} rows from {args.manifest}")
    print(f"Pending {len(pending)} rows; writing to {args.out}")

    mode = "a" if args.resume else "w"
    with args.out.open(mode, encoding="utf-8") as f:
        for ordinal, (row_index, row) in enumerate(pending, start=1):
            record = {
                "row_index": row_index,
                "sample_id": row["sample_id"],
                "category": row["category"],
                "benchmark": row["benchmark"],
                "expected_behavior": row.get("expected_behavior"),
                "task_type": "compliance",
                "duration_sec": row.get("duration_sec"),
                "source_file": row.get("source_file"),
                "model": model,
                "input_mode": "oracle_transcript",
                "transcript_field": args.transcript_field,
                "prompt_mode": args.prompt_mode,
                "abstained": False,
                "n_lalm_calls": 1,
            }
            try:
                prompt = build_prompt(row, args.transcript_field, args.prompt_mode)
                result = client.chat_messages(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    system_prompt=system_prompt,
                    thinking_token_budget=args.thinking_token_budget,
                )
                blank_retry = False
                first_finish_reason = result.get("finish_reason")
                if (
                    args.blank_retry_max_tokens > args.max_tokens
                    and not (result.get("content") or "").strip()
                ):
                    blank_retry = True
                    result = client.chat_messages(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=args.blank_retry_max_tokens,
                        temperature=args.temperature,
                        system_prompt=system_prompt,
                        thinking_token_budget=args.thinking_token_budget,
                    )
                response = result["content"]
                record = {
                    **record,
                    "n_lalm_calls": 2 if blank_retry else 1,
                    "instruction": prompt,
                    "response": response,
                    "finish_reason": result.get("finish_reason"),
                    "first_finish_reason": first_finish_reason,
                    "blank_retry": blank_retry,
                    "blank_retry_max_tokens": args.blank_retry_max_tokens,
                    "reasoning": result.get("reasoning"),
                }
            except Exception as exc:
                message = safe_error(exc)
                record = {
                    **record,
                    "response": f"[ERROR: {type(exc).__name__}: {message}]",
                    "error": message,
                }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"[{ordinal}/{len(pending)}] {row['sample_id']} "
                f"blank={not bool((record.get('response') or '').strip())} "
                f"finish={record.get('finish_reason')} "
                f"error={bool(record.get('error'))}"
            )

    summarize(args.out)


if __name__ == "__main__":
    main()

