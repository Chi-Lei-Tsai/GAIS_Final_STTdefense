"""Add SACRED Multi-speaker Dialogue JSON metadata to an existing manifest."""

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
REPO_ROOT = EVAL_ROOT.parent
sys.path.insert(0, str(EVAL_ROOT))

from src.manifest import read_manifest  # noqa: E402


DEFAULT_METADATA = (
    EVAL_ROOT
    / "data"
    / "sacred_cache"
    / "datasets--tsinghua-ee--SACRED-Bench"
    / "snapshots"
    / "55f18d56b588b2b59dd44ee5fdaf154b487eec6a"
    / "Multi-speaker_Dialogue"
    / "test"
    / "json"
    / "multi-speaker_dialogue_test.json"
)


def metadata_key(path: str) -> str:
    normalized = path.replace("\\", "/")
    marker = "/audio/"
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    if "Multi-speaker_Dialogue/test/audio/" in normalized:
        return normalized.split("Multi-speaker_Dialogue/test/audio/", 1)[1]
    return "/".join(normalized.split("/")[-2:])


def read_metadata(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    mapped: dict[str, dict] = {}
    for row in rows:
        key = metadata_key(row["audio_path"])
        mapped[key] = row
        # SACRED's illegal-activity metadata paths omit the category folder
        # while the downloaded audio tree uses this misspelled directory name.
        if key.startswith("audio/") or "/" not in key:
            mapped.setdefault(
                "01-Illegal_Activitiy/" + key.rsplit("/", 1)[-1],
                row,
            )
    return mapped


def compact_metadata(row: dict) -> dict:
    return {
        "question": row.get("Question"),
        "changed_question": row.get("Changed Question"),
        "key_phrase": row.get("Key Phrase"),
        "phrase_type": row.get("Phrase Type"),
        "rephrased_question": row.get("Rephrased Question"),
        "rephrased_question_sd": row.get("Rephrased Question(SD)"),
        "revised_query": row.get("revised_query"),
        "dialogue_transcript": row.get("GPT_description"),
        "tts_transcript": row.get("GPT_description_for_TTS"),
        "sacred_index": row.get("index"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    metadata = read_metadata(args.metadata)
    missing: list[str] = []
    enriched: list[dict] = []
    for row in rows:
        key = metadata_key(row.get("source_file") or row.get("audio_path") or "")
        match = metadata.get(key)
        if match is None:
            missing.append(row.get("sample_id") or key)
            enriched.append(row)
            continue
        extra = dict(row.get("extra") or {})
        extra.update(compact_metadata(match))
        enriched.append({**row, "extra": extra})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in enriched:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(enriched)} rows to {args.out}")
    print(f"Matched {len(enriched) - len(missing)} rows; missing {len(missing)}")
    if missing:
        print("First missing rows:")
        for sample_id in missing[:10]:
            print(f"  {sample_id}")


if __name__ == "__main__":
    main()

