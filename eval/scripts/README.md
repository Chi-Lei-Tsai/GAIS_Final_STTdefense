# Scripts

The script tree is split by intent so the active path is easy to find.

- `current/` - supported entrypoints for the active structured-provenance
  pipeline.
- `datasets/` - manifest builders and dataset preparation scripts.
- `baselines/` - hosted NVIDIA baselines and judge/rescore utilities.
- `experiments/` - exploratory ablations that are useful but not the default
  pipeline.
- `archive/` - superseded launch scripts from earlier local runs.

Current full AudioJailbreak Origin run:

```powershell
& ".\eval\scripts\current\run_audiojailbreak_origin_whisper_large.ps1" `
  -Python "<env>\python.exe"
```
