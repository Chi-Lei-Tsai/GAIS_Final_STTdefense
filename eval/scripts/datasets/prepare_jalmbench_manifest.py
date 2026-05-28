"""Convert a JALMBench parquet slice into WAV files plus repo manifest."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

def _find_eval_root(script_path: Path) -> Path:
    for parent in [script_path.parent, *script_path.parents]:
        if parent.name == "eval" and (parent / "src").is_dir():
            return parent
    raise RuntimeError(f"Could not locate eval root from {script_path}")


EVAL_ROOT = _find_eval_root(Path(__file__).resolve())
REPO_ROOT = EVAL_ROOT.parent
sys.path.insert(0, str(EVAL_ROOT))

from src.manifest import ManifestRow, write_manifest  # noqa: E402


def safe_slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "row"


def audio_payload(row: dict[str, Any]) -> tuple[np.ndarray, int]:
    audio = row.get("audio") or {}
    samples = audio.get("array")
    sampling_rate = audio.get("sampling_rate")
    if samples is None or sampling_rate is None:
        raise RuntimeError("Missing audio.array or audio.sampling_rate")
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim != 1:
        array = array.reshape(-1)
    return array, int(sampling_rate)


def text_preview(text: str | None, limit: int = 120) -> str:
    text = " ".join((text or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--out-manifest", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--subset-name", default=None)
    ap.add_argument("--overwrite-audio", action="store_true")
    args = ap.parse_args()

    subset = args.subset_name or args.parquet.stem
    parquet = pq.ParquetFile(args.parquet)
    args.audio_dir.mkdir(parents=True, exist_ok=True)

    rows: list[ManifestRow] = []
    seen = 0
    for batch in parquet.iter_batches():
        for item in batch.to_pylist():
            if args.n is not None and seen >= args.n:
                break
            seen += 1

            source_id = str(item.get("id") or seen)
            sample_id = (
                f"jalmbench_{safe_slug(subset).lower()}_"
                f"{seen:04d}_{safe_slug(source_id)}"
            )
            samples, sampling_rate = audio_payload(item)
            audio_path = args.audio_dir / f"{sample_id}.wav"
            if args.overwrite_audio or not audio_path.exists():
                sf.write(audio_path, samples, sampling_rate)

            duration_sec = len(samples) / sampling_rate if sampling_rate else None
            original_text = item.get("original_text")
            text = item.get("text")
            rows.append(
                ManifestRow(
                    sample_id=sample_id,
                    audio_path=str(audio_path.resolve()),
                    category=f"jalmbench_{safe_slug(subset).lower()}",
                    benchmark="jalmbench",
                    duration_sec=duration_sec,
                    description=text_preview(text or original_text),
                    source_file=f"{args.parquet.as_posix()}#{source_id}",
                    expected_behavior="refuse",
                    task_type="compliance",
                    extra={
                        "jalmbench_subset": subset,
                        "jalmbench_id": source_id,
                        "source": item.get("source"),
                        "text": text,
                        "original_text": original_text,
                    },
                )
            )
        if args.n is not None and seen >= args.n:
            break

    write_manifest(rows, args.out_manifest)
    print(f"Wrote {len(rows)} rows to {args.out_manifest}")
    print(f"Wrote audio under {args.audio_dir}")


if __name__ == "__main__":
    main()

