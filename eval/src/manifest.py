"""Manifest schema and JSONL I/O.

Each row describes one audio sample with enough metadata to run any
defense + harm-scoring pipeline against it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ManifestRow:
    sample_id: str                       # globally unique
    audio_path: str                      # absolute or repo-relative
    category: str                        # e.g. "speech_overlap", "benign_clean"
    benchmark: str                       # e.g. "sacred", "synthetic", "jalmbench"
    duration_sec: float | None = None
    description: str | None = None       # human-readable note (e.g. attack topic from filename)
    source_file: str | None = None       # original file path inside source dataset
    expected_behavior: str = "answer"    # "answer" | "refuse" | "use_late_content" | ...
    # SACRED-Bench task framing:
    #   "compliance"        — open-ended; attack success = model produces harmful output (MSD)
    #   "safety_judgment"   — ask model if audio is safe; attack success = model says yes (SSO, SAO)
    task_type: str = "compliance"
    # Optional ground-truth annotations (for synthetic / annotated data only)
    injection_start_sec: float | None = None
    injection_end_sec: float | None = None
    primary_speaker_id: str | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        if d["duration_sec"] is not None:
            d["duration_sec"] = round(d["duration_sec"], 3)
        for k in ("injection_start_sec", "injection_end_sec"):
            if d.get(k) is not None:
                d[k] = round(d[k], 3)
        return d


def write_manifest(rows: list[ManifestRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.as_dict(), ensure_ascii=False) + "\n")


def read_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
