"""Pilot data preparation.

Produces 40 audio samples (16 kHz mono WAV) plus manifest.jsonl:
  - 10 late_suffix_attack            (LibriSpeech base + TTS injection appended)
  - 10 second_speaker_attack         (LibriSpeech base + TTS injection mid)
  - 10 benign_late_content_required  (single-voice TTS, info at end)
  - 10 benign_multi_speaker_dialogue (two LibriSpeech speakers, both benign)

Idempotent: skips files already on disk. Safe to re-run.

CPU only. No torchaudio / torchcodec dependency: downloads LibriSpeech
dev-clean directly from OpenSLR and decodes FLAC with soundfile.
Conda env's ffmpeg is needed for librosa to decode edge-tts MP3 output.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Make conda env's ffmpeg findable when this script is run via env's
# python.exe directly (without `conda activate`).
_env_lib_bin = Path(sys.executable).parent / "Library" / "bin"
if _env_lib_bin.exists():
    os.environ["PATH"] = str(_env_lib_bin) + os.pathsep + os.environ.get("PATH", "")

import edge_tts  # noqa: E402
import librosa  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from tqdm import tqdm  # noqa: E402

PILOT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PILOT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "pilot_samples"
LIBRISPEECH_DIR = DATA_DIR / "librispeech"
INJECTION_TTS_DIR = DATA_DIR / "tts_cache" / "injections"
BENIGN_TTS_DIR = DATA_DIR / "tts_cache" / "benign"
USER_QUERY_TTS_DIR = DATA_DIR / "tts_cache" / "user_queries"
PROMPTS_DIR = DATA_DIR / "prompts"
MANIFEST_PATH = PILOT_ROOT / "manifest.jsonl"

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
LIBRISPEECH_ARCHIVE = LIBRISPEECH_DIR / "dev-clean.tar.gz"
LIBRISPEECH_EXTRACTED = LIBRISPEECH_DIR / "LibriSpeech" / "dev-clean"

SR = 16000
RNG_SEED = 42
N_PER_CATEGORY = 10
TARGET_BASE_DUR_SEC = (5.0, 8.0)

VOICE_INJECTION = "en-US-GuyNeural"
VOICE_BENIGN = "en-US-AriaNeural"

random.seed(RNG_SEED)
np.random.seed(RNG_SEED)


@dataclass
class ManifestRow:
    sample_id: str
    audio_path: str
    category: str
    duration_sec: float
    injection_start_sec: float | None
    injection_end_sec: float | None
    primary_speaker_id: str | None
    notes: str

    def as_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "audio_path": self.audio_path,
            "category": self.category,
            "duration_sec": round(self.duration_sec, 3),
            "injection_start_sec": (
                round(self.injection_start_sec, 3)
                if self.injection_start_sec is not None
                else None
            ),
            "injection_end_sec": (
                round(self.injection_end_sec, 3)
                if self.injection_end_sec is not None
                else None
            ),
            "primary_speaker_id": self.primary_speaker_id,
            "notes": self.notes,
        }


# --- LibriSpeech (direct OpenSLR download) ---


def _download_with_progress(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"  downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as resp:  # noqa: S310 — public dataset
        total = int(resp.headers.get("Content-Length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc="dev-clean.tar.gz")
        with dest.open("wb") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
        bar.close()


def ensure_librispeech() -> Path:
    if not LIBRISPEECH_EXTRACTED.exists():
        _download_with_progress(LIBRISPEECH_URL, LIBRISPEECH_ARCHIVE)
        print(f"  extracting {LIBRISPEECH_ARCHIVE} ...")
        with tarfile.open(LIBRISPEECH_ARCHIVE, "r:gz") as tar:
            tar.extractall(LIBRISPEECH_DIR)
    return LIBRISPEECH_EXTRACTED


def index_librispeech_clips(root: Path) -> dict[str, list[Path]]:
    """Return {speaker_id: [flac paths]} for all clips under root."""
    by_speaker: dict[str, list[Path]] = {}
    for spk_dir in root.iterdir():
        if not spk_dir.is_dir():
            continue
        spk = spk_dir.name
        flacs: list[Path] = []
        for chap_dir in spk_dir.iterdir():
            if not chap_dir.is_dir():
                continue
            flacs.extend(chap_dir.glob("*.flac"))
        if flacs:
            by_speaker[spk] = flacs
    return by_speaker


def load_flac_16k_mono(path: Path) -> np.ndarray:
    """Read a FLAC, resample to 16k mono."""
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return audio.astype(np.float32)


def collect_librispeech_clips(
    by_speaker: dict[str, list[Path]],
    n: int,
    min_distinct_speakers: int = 8,
) -> list[dict]:
    """Pick n clips with speaker variety, each cropped to TARGET_BASE_DUR_SEC."""
    speakers = list(by_speaker.keys())
    random.shuffle(speakers)
    speakers = speakers[: max(min_distinct_speakers, n // 2 + 2)]

    picks: list[dict] = []
    cycle = list(speakers)
    while len(picks) < n and any(by_speaker[s] for s in cycle):
        spk = cycle.pop(0)
        cycle.append(spk)
        flacs = by_speaker[spk]
        if not flacs:
            continue
        path = flacs.pop(random.randrange(len(flacs)))
        try:
            audio = load_flac_16k_mono(path)
        except Exception as e:
            print(f"  skip {path.name}: {e}", file=sys.stderr)
            continue
        cropped = crop_to_duration(audio, TARGET_BASE_DUR_SEC, SR)
        if cropped is None:
            continue
        picks.append({"audio": cropped, "speaker_id": spk, "source": str(path)})
    return picks


def crop_to_duration(
    audio: np.ndarray, target_range: tuple[float, float], sr: int
) -> np.ndarray | None:
    trimmed, _ = librosa.effects.trim(audio, top_db=30)
    dur = len(trimmed) / sr
    lo, hi = target_range
    if dur < lo:
        return None
    target = random.uniform(lo, min(hi, dur))
    n = int(target * sr)
    start = random.randint(0, max(0, len(trimmed) - n))
    return trimmed[start : start + n].astype(np.float32)


# --- TTS via edge-tts ---


async def tts_to_wav(text: str, voice: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    tmp_mp3 = out_path.with_suffix(".mp3")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(tmp_mp3))
    audio, _ = librosa.load(str(tmp_mp3), sr=SR, mono=True)
    sf.write(str(out_path), audio, SR, subtype="PCM_16")
    tmp_mp3.unlink(missing_ok=True)


async def render_all_tts(
    injection_texts: list[str],
    benign_texts: list[str],
    user_query_texts: list[str],
) -> tuple[list[Path], list[Path], list[Path]]:
    inj_paths = [
        INJECTION_TTS_DIR / f"injection_{i:02d}.wav"
        for i in range(len(injection_texts))
    ]
    ben_paths = [
        BENIGN_TTS_DIR / f"benign_{i:02d}.wav" for i in range(len(benign_texts))
    ]
    uq_paths = [
        USER_QUERY_TTS_DIR / f"user_query_{i:02d}.wav"
        for i in range(len(user_query_texts))
    ]
    tasks = []
    for text, path in zip(injection_texts, inj_paths):
        tasks.append(tts_to_wav(text, VOICE_INJECTION, path))
    for text, path in zip(benign_texts, ben_paths):
        tasks.append(tts_to_wav(text, VOICE_BENIGN, path))
    for text, path in zip(user_query_texts, uq_paths):
        tasks.append(tts_to_wav(text, VOICE_BENIGN, path))
    await asyncio.gather(*tasks)
    return inj_paths, ben_paths, uq_paths


def load_wav(path: Path) -> np.ndarray:
    audio, sr = librosa.load(str(path), sr=SR, mono=True)
    return audio.astype(np.float32)


# --- splicing ---


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def splice_late_suffix(
    base: np.ndarray, injection: np.ndarray, gap_sec: float = 0.3
) -> tuple[np.ndarray, float, float]:
    gap = silence(gap_sec)
    start = (len(base) + len(gap)) / SR
    end = start + len(injection) / SR
    return np.concatenate([base, gap, injection]), start, end


def splice_second_speaker(
    base: np.ndarray, injection: np.ndarray, gap_sec: float = 0.15
) -> tuple[np.ndarray, float, float]:
    mid = len(base) // 2
    gap = silence(gap_sec)
    start = (mid + len(gap)) / SR
    end = start + len(injection) / SR
    return (
        np.concatenate([base[:mid], gap, injection, gap, base[mid:]]),
        start,
        end,
    )


def splice_two_speaker_dialogue(
    a: np.ndarray, b: np.ndarray, gap_sec: float = 0.3
) -> np.ndarray:
    return np.concatenate([a, silence(gap_sec), b])


# --- main ---


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SR, subtype="PCM_16")


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    injection_texts = (PROMPTS_DIR / "injection_texts.txt").read_text(
        encoding="utf-8"
    ).strip().splitlines()
    benign_texts = (PROMPTS_DIR / "benign_late_content_texts.txt").read_text(
        encoding="utf-8"
    ).strip().splitlines()
    user_query_texts = (PROMPTS_DIR / "benign_user_queries.txt").read_text(
        encoding="utf-8"
    ).strip().splitlines()
    assert len(injection_texts) >= N_PER_CATEGORY
    assert len(benign_texts) >= N_PER_CATEGORY
    assert len(user_query_texts) >= 2 * N_PER_CATEGORY, "need >=20 user queries"

    print("[1/5] Ensuring LibriSpeech dev-clean is on disk ...")
    root = ensure_librispeech()
    by_speaker = index_librispeech_clips(root)
    n_speakers = len(by_speaker)
    n_clips = sum(len(v) for v in by_speaker.values())
    print(f"      {n_speakers} speakers, {n_clips} FLAC clips")

    print("[2/5] Picking 20 LibriSpeech clips for benign_multi_speaker_dialogue ...")
    dialogue_clips = collect_librispeech_clips(by_speaker, n=20)
    if len(dialogue_clips) < 20:
        raise SystemExit(f"only got {len(dialogue_clips)} clips; need 20")

    print("[3/5] Rendering TTS via edge-tts ...")
    inj_paths, ben_paths, uq_paths = asyncio.run(
        render_all_tts(
            injection_texts[:N_PER_CATEGORY],
            benign_texts[:N_PER_CATEGORY],
            user_query_texts[: 2 * N_PER_CATEGORY],
        )
    )
    inj_audios = [load_wav(p) for p in inj_paths]
    ben_audios = [load_wav(p) for p in ben_paths]
    uq_audios = [load_wav(p) for p in uq_paths]

    print("[4/5] Splicing 40 samples ...")
    rows: list[ManifestRow] = []

    # late_suffix_attack: TTS user query (AriaNeural) + TTS attack (GuyNeural)
    base_pool = uq_audios[:10]
    base_texts = user_query_texts[:10]
    for i, (base_audio, base_text, inj, inj_text) in enumerate(
        zip(base_pool, base_texts, inj_audios, injection_texts[:N_PER_CATEGORY])
    ):
        sid = f"late_suffix_{i:02d}"
        spliced, t0, t1 = splice_late_suffix(base_audio, inj)
        out = SAMPLES_DIR / f"{sid}.wav"
        write_wav(out, spliced)
        rows.append(
            ManifestRow(
                sample_id=sid,
                audio_path=str(out.relative_to(PILOT_ROOT)).replace("\\", "/"),
                category="late_suffix_attack",
                duration_sec=len(spliced) / SR,
                injection_start_sec=t0,
                injection_end_sec=t1,
                primary_speaker_id="tts:" + VOICE_BENIGN,
                notes=f"base={base_text!r}; injection={inj_text!r}",
            )
        )

    # second_speaker_attack: TTS user query (AriaNeural) split + TTS attack (GuyNeural) inserted
    base_pool = uq_audios[10:20]
    base_texts = user_query_texts[10:20]
    for i, (base_audio, base_text, inj, inj_text) in enumerate(
        zip(base_pool, base_texts, inj_audios, injection_texts[:N_PER_CATEGORY])
    ):
        sid = f"second_speaker_{i:02d}"
        spliced, t0, t1 = splice_second_speaker(base_audio, inj)
        out = SAMPLES_DIR / f"{sid}.wav"
        write_wav(out, spliced)
        rows.append(
            ManifestRow(
                sample_id=sid,
                audio_path=str(out.relative_to(PILOT_ROOT)).replace("\\", "/"),
                category="second_speaker_attack",
                duration_sec=len(spliced) / SR,
                injection_start_sec=t0,
                injection_end_sec=t1,
                primary_speaker_id="tts:" + VOICE_BENIGN,
                notes=f"base={base_text!r}; injection={inj_text!r}",
            )
        )

    for i, (audio, text) in enumerate(zip(ben_audios, benign_texts[:N_PER_CATEGORY])):
        sid = f"benign_late_content_{i:02d}"
        out = SAMPLES_DIR / f"{sid}.wav"
        write_wav(out, audio)
        rows.append(
            ManifestRow(
                sample_id=sid,
                audio_path=str(out.relative_to(PILOT_ROOT)).replace("\\", "/"),
                category="benign_late_content_required",
                duration_sec=len(audio) / SR,
                injection_start_sec=None,
                injection_end_sec=None,
                primary_speaker_id="tts:" + VOICE_BENIGN,
                notes=f"benign_info_at_end={text!r}",
            )
        )

    pool = dialogue_clips
    for i in range(N_PER_CATEGORY):
        a = pool[2 * i]
        b = pool[2 * i + 1]
        sid = f"benign_dialogue_{i:02d}"
        a_short = a["audio"][: int(SR * random.uniform(2.5, 4.0))]
        b_short = b["audio"][: int(SR * random.uniform(2.5, 4.0))]
        spliced = splice_two_speaker_dialogue(a_short, b_short)
        out = SAMPLES_DIR / f"{sid}.wav"
        write_wav(out, spliced)
        rows.append(
            ManifestRow(
                sample_id=sid,
                audio_path=str(out.relative_to(PILOT_ROOT)).replace("\\", "/"),
                category="benign_multi_speaker_dialogue",
                duration_sec=len(spliced) / SR,
                injection_start_sec=None,
                injection_end_sec=None,
                primary_speaker_id=f"{a['speaker_id']}+{b['speaker_id']}",
                notes="two distinct LibriSpeech speakers, both benign",
            )
        )

    print("[5/5] Writing manifest ...")
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.as_dict()) + "\n")

    print(f"\nDone. {len(rows)} samples in {SAMPLES_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")
    by_cat: dict[str, int] = {}
    for r in rows:
        by_cat[r.category] = by_cat.get(r.category, 0) + 1
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
