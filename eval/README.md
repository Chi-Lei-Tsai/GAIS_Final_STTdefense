# Evaluation Code

This directory contains the active implementation and evaluation tooling for
the structured-provenance audio guard.

## Layout

```
eval/
  src/               core defense, provenance builder, guard, scoring
  scripts/current/   supported entrypoints for the active pipeline
  scripts/datasets/  manifest builders and dataset preparation
  scripts/baselines/ NVIDIA API baselines and judge/rescore utilities
  scripts/experiments/
                     ablations and exploratory scripts
  scripts/archive/   superseded launch scripts
  manifests/         generated JSONL manifests, ignored by git
  results/           generated JSONL results/logs, ignored by git
  data/              downloaded datasets/caches, ignored by git
```

The current structured-provenance pipeline defaults to
`openai/whisper-large-v3`. For full benchmark runs, use a CUDA machine and the
runner in `scripts/current/run_audiojailbreak_origin_whisper_large.ps1`.

## Current Entrypoints

```powershell
& "<env>\python.exe" -u eval\scripts\current\run_structured_provenance_incremental.py --help
& "<env>\python.exe" -u eval\scripts\current\run_defense.py --help
```

AudioJailbreak Origin full pipeline:

```powershell
& ".\eval\scripts\current\run_audiojailbreak_origin_whisper_large.ps1" `
  -Python "<env>\python.exe"
```

Use `-AudioJailbreakRoot "<path-to-local-MBZUAI-AudioJailbreak>"` when the
dataset repo is already downloaded on the machine.
