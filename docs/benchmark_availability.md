# External Benchmark Availability

Last checked: 2026-05-20.

## Multi-AudioJail

- Paper: https://arxiv.org/pdf/2504.01094
- OpenReview: https://openreview.net/forum?id=yGa8CYT8kS
- Public Hugging Face dataset: https://huggingface.co/datasets/jroh/Multi-AudioJail
- Local snapshot: `eval/data/multi_audiojail_cache/jroh--Multi-AudioJail/`

Status: not currently usable from the public dataset snapshot. The
Hugging Face repository exists, but it currently contains only repository
metadata (`.gitattributes`) and no benchmark audio, manifest, or dataset
card. The OpenReview page still describes the dataset as planned for
release.

Paper notes:

- The benchmark is named `MULTI-AUDIOJAIL`.
- It is built from 520 AdvBench harmful instructions translated into
  five additional languages.
- The paper reports 102,720 generated audio files across multilingual,
  natural-accent, synthetic-accent, and perturbation settings.
- Perturbations include reverberation, echo, and whisper effects.
- Their evaluation uses Llama Guard 3 for jailbreak success rate and
  Whisper-large-v3 WER as a transcription-understanding metric.

Next action once released: replace the metadata-only local snapshot with
the actual dataset files, then create manifests for a small first slice
covering clean multilingual audio, reverberation, echo, and whisper.

## Duplicate URL Note

The user request listed `https://arxiv.org/pdf/2504.01094` twice. I
treated this as one benchmark request because both links resolve to the
same Multi-AudioJail paper.

## JALMBench

- Paper: https://arxiv.org/abs/2505.17568
- Public Hugging Face dataset: https://huggingface.co/datasets/AnonymousUser000/JALMBench
- Local cache: `eval/data/jalmbench_cache/AnonymousUser000--JALMBench/`
- Prepared manifest: `eval/manifests/jalmbench_aharm_n246.jsonl`
- Extracted audio: `eval/data/jalmbench_cache/audio/AHarm/`

Status: available. The public dataset contains 1,460 files, about
138 GB total. I downloaded the dataset card and the small
`HarmfulQuery/AHarm.parquet` slice.

Repository layout:

- `HarmfulQuery`: 2 parquet files, about 0.21 GB total.
- `Text_Transferred_Jailbreak`: 177 parquet files, about 17.96 GB total.
- `Audio_Originated_Jailbreak`: 1,279 parquet files, about 119.76 GB
  total.

Dataset card notes:

- JALMBench contains 245,355 audio samples and 11,316 text prompts.
- Main categories are harmful query audio, text-transferred jailbreaks,
  and audio-originated jailbreaks.
- The harmful-query category includes `AHarm` and `ADiv`.
- Text-transferred methods include ICA, DAN, DI, and PAP.
- Audio-originated methods include SSJ, AMSE, BoN, and AdvWave.

Prepared local slice:

- `AHarm` rows: 246.
- WAV files extracted: 246.
- Sampling rate: 16 kHz.
- Duration range: 1.559 to 9.705 seconds.
- Median duration: 5.089 seconds.
- Sources: AdvBench, HarmBench, JailbreakBench, MM-SafetyBench.

The converter script is
`eval/scripts/datasets/prepare_jalmbench_manifest.py`. It converts a
JALMBench parquet slice into WAV files plus the repo's normal JSONL
manifest format.
