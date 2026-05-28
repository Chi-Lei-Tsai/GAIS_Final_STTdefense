# ProvGuard Pilot Experiment

Archived go/no-go experiment for the earlier ProvGuard ablation project.
See `docs/archive/previous_provguard/` for the old proposal docs and
`docs/archive/experiment_summary_2026-05.md` for the structured-provenance
experiment history.

## What this tests

Three premises that need to hold for the full project to be worth pursuing:

1. **LALM is sensitive to injection presence** — muting the injected segment
   on an attack sample meaningfully changes Qwen2-Audio's output (lower
   Llama-Guard "unsafe" logit, semantically different response).
2. **Guard models over-refuse on benign controls** — SALMONN-Guard
   incorrectly flags `benign_late_content_required` and
   `benign_multi_speaker_dialogue` samples often enough that ProvGuard's
   utility-preservation wedge is real.
3. **Cheap attribution is feasible** — naive lateness + speaker-change
   rules localize the injected segment well above chance.

## Decision rule

| Premise 1 | Premise 2 | Premise 3 | Decision |
|---|---|---|---|
| Pass | Pass | Pass | Full project, headline = stacking |
| Pass | Pass | Fail | Full project, deemphasize localization |
| Pass | Fail | * | Pivot to interpretability/measurement-only |
| Fail | * | * | **Rethink** |

Specific thresholds in each script's docstring.

## Environment

- Conda env: `GAIS`
- GPU: 8 GB VRAM (RTX 5070). Models loaded sequentially with 4-bit
  quantization where needed.
- Models used (HF):
  - `Qwen/Qwen2-Audio-7B-Instruct` — 4-bit NF4
  - `meta-llama/Llama-Guard-3-1B` — fp16
  - `tsinghua-ee/SACRED-Bench` SALMONN-Guard checkpoint — 4-bit
  - `pyannote/speaker-diarization-3.1` — for sub-experiment 3

## Setup

```powershell
conda activate GAIS
pip install -r requirements.txt
huggingface-cli login    # token with access to Llama-Guard, pyannote
```

## Run order

```powershell
python scripts/01_prepare_data.py        # CPU only, ~5 min
python scripts/02_run_qwen.py            # GPU, ~15-30 min
python scripts/03_run_salmonn_guard.py   # GPU, ~10 min
python scripts/04_cheap_attribution.py   # GPU (light), ~5 min
python scripts/05_report.py              # CPU, instant
```

Each script writes to `results/<name>.jsonl`. `05_report.py` reads all of
them and prints the go/no-go verdict.

## Layout

```
pilot/
├── README.md
├── requirements.txt
├── manifest.jsonl                 # produced by 01
├── data/
│   ├── librispeech/               # downloaded LibriSpeech dev-clean
│   ├── tts_cache/                 # cached edge-tts outputs
│   ├── prompts/                   # injection / benign info text lists
│   └── pilot_samples/             # 40 final spliced WAVs
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_run_qwen.py
│   ├── 03_run_salmonn_guard.py
│   ├── 04_cheap_attribution.py
│   └── 05_report.py
└── results/
    ├── 02_qwen_outputs.jsonl
    ├── 03_salmonn_guard_verdicts.jsonl
    ├── 04_attribution_hits.jsonl
    └── 05_summary.txt
```
