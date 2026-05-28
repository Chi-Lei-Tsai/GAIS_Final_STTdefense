"""Convert the pilot's benign samples into an eval-format manifest, so
the same runner can measure over-refusal.

CPU only. Output: manifests/benign_pilot.jsonl
"""

from __future__ import annotations

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
from src.manifest import ManifestRow, write_manifest  # noqa: E402

PILOT_ROOT = EVAL_ROOT.parent / "docs" / "archive" / "pilot"
PILOT_MANIFEST = PILOT_ROOT / "manifest.jsonl"

# Categories from the pilot we treat as benign controls
BENIGN_CATEGORIES = ("benign_late_content_required", "benign_multi_speaker_dialogue")

OUT_PATH = EVAL_ROOT / "manifests" / "benign_pilot.jsonl"


def main() -> None:
    if not PILOT_MANIFEST.exists():
        sys.exit(f"pilot manifest not found at {PILOT_MANIFEST}; run pilot 01 first")

    rows: list[ManifestRow] = []
    with PILOT_MANIFEST.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("category") not in BENIGN_CATEGORIES:
                continue
            audio_abs = (PILOT_ROOT / r["audio_path"]).resolve()
            rows.append(
                ManifestRow(
                    sample_id="benign_" + r["sample_id"],
                    audio_path=str(audio_abs).replace("\\", "/"),
                    category=r["category"],
                    benchmark="pilot_benign",
                    duration_sec=r.get("duration_sec"),
                    description=r.get("notes"),
                    source_file=r["audio_path"],
                    expected_behavior="answer",  # NOT refuse: these are legitimate
                    primary_speaker_id=r.get("primary_speaker_id"),
                )
            )

    write_manifest(rows, OUT_PATH)
    print(f"Wrote {len(rows)} benign rows to {OUT_PATH}")
    by_cat: dict[str, int] = {}
    for r in rows:
        by_cat[r.category] = by_cat.get(r.category, 0) + 1
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()

