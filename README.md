# Structured Provenance Audio Guard

Prototype defense for audio-language models against unsafe audio
instructions and audio prompt-injection. The current pipeline converts an
audio file into a provenance-aware timeline, attaches ASR/diarization
evidence, and runs a deterministic guard over that structure before any
target audio-language model is allowed to answer.

## Current Pipeline

The supported path is:

1. Diarize the audio with pyannote.
2. Transcribe the full audio with `openai/whisper-large-v3` and timestamps.
3. Align ASR chunks to diarization turns and mark overlaps/non-speech gaps.
4. Classify whole-transcript prompt-injection/unsafe-task evidence with the
   NVIDIA text API when enabled.
5. Return a pass/refuse safety judgment from the structured guard.

Whisper-large is the default ASR model for the structured-provenance defense.
It is loaded once during defense setup and reused across rows. ASR model
switching is disabled by default because alternating between Whisper-small and
Whisper-large repeatedly unloads/reloads model weights and is slower on GPU.
Older Whisper-small and routed-ASR experiments are preserved under
`eval/scripts/archive/` and `docs/archive/`.

## Repository Layout

- `eval/src/` - defense implementation, provenance timeline builder, guard,
  scoring, and model wrappers.
- `eval/scripts/current/` - supported runners for the active pipeline.
- `eval/scripts/datasets/` - manifest builders and dataset preparation tools.
- `eval/scripts/baselines/` - NVIDIA API baselines and judge/rescoring tools.
- `eval/scripts/experiments/` - ablations and exploratory scripts.
- `eval/scripts/archive/` - superseded launch scripts kept for provenance.
- `docs/current_results_report.md` - current experimental report.
- `docs/archive/` - older experiment summaries and previous ProvGuard notes.

Generated data, manifests, results, caches, and API keys are intentionally
ignored by git.

## Setup

Create a Python environment with the repo dependencies, then copy
`.env.example` to `.env` and fill in keys as needed.

Required for the active AudioJailbreak pipeline:

- Hugging Face access for pyannote and dataset downloads.
- `NVIDIA_API_KEY` if using the prompt-injection classifier or NVIDIA
  target-model/judge baselines.
- A CUDA GPU is strongly recommended for full benchmark runs.

## Run AudioJailbreak Origin

On a cloud GPU:

```powershell
& "<env>\python.exe" -m pip install -r requirements.txt

& ".\eval\scripts\current\run_audiojailbreak_origin_whisper_large.ps1" `
  -Python "<env>\python.exe"
```

If AudioJailbreak is already downloaded locally, pass the dataset repository
root. The directory should contain paths such as `audio/total_wav/...` and the
dataset metadata files from `MBZUAI/AudioJailbreak`.

```powershell
& ".\eval\scripts\current\run_audiojailbreak_origin_whisper_large.ps1" `
  -Python "<env>\python.exe" `
  -AudioJailbreakRoot "D:\datasets\AudioJailbreak"
```

The runner:

- builds/resumes the full AudioJailbreak Origin manifest;
- runs the structured provenance guard with Whisper-large;
- builds a manifest of guard-passed rows;
- sends guard-passed audio to Nemotron;
- judges Nemotron outputs with Llama-Guard-4.

Main output:

`eval/results/structured_provenance_audiojailbreak_origin_whisper_large.jsonl`

## Useful Direct Commands

Run the structured guard on any safety-judgment manifest:

```powershell
& "<env>\python.exe" -u `
  eval\scripts\current\run_structured_provenance_incremental.py `
  --manifest eval\manifests\audiojailbreak_origin_full_n1495.jsonl `
  --out eval\results\structured_provenance_audiojailbreak_origin_whisper_large.jsonl `
  --asr-model openai/whisper-large-v3 `
  --asr-mode whole_timestamped `
  --enable-prompt-injection-guard `
  --prefer-cuda `
  --resume
```

Run the defense runner for compliance or MSD-style manifests:

```powershell
& "<env>\python.exe" -u `
  eval\scripts\current\run_defense.py `
  --defense structured_provenance `
  --manifest eval\manifests\sacred_msd_n100.jsonl `
  --task-type msd_dialogue_safety `
  --asr-model openai/whisper-large-v3 `
  --asr-mode whole_timestamped `
  --prefer-cuda
```

## Key Docs

- `structured_provenance_pipeline.md` - design and guard contract.
- `docs/current_results_report.md` - latest results and caveats.
- `docs/benchmark_availability.md` - benchmark availability notes.
