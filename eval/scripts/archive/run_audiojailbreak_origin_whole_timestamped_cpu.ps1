param(
  [string]$Python = "C:\Users\ChiLeiTsai\miniconda3\envs\GAIS\python.exe",
  [string]$Manifest = "eval\manifests\audiojailbreak_origin_full_n1495.jsonl",
  [string]$GuardResults = "eval\results\structured_provenance_audiojailbreak_origin_full_pi_guard_whole_timestamped.jsonl",
  [string]$PassedManifest = "eval\manifests\audiojailbreak_origin_full_guard_passed_for_nemotron_whole_timestamped.jsonl",
  [string]$NemotronResults = "eval\results\nvidia_api_baselines\nemotron_audiojailbreak_origin_guard_passed_full_audio_whole_timestamped.jsonl",
  [string]$JudgeResults = "eval\results\nvidia_api_baselines\nemotron_audiojailbreak_origin_guard_passed_lg4_context_whole_timestamped.jsonl",
  [string]$WholeAsrMaxNewTokens = "192"
)

$ErrorActionPreference = "Stop"

function Assert-NativeStepSucceeded {
  param([string]$StepName)
  if ($LASTEXITCODE -ne 0) {
    throw "$StepName failed with exit code $LASTEXITCODE"
  }
}

$env:STRUCTURED_PROVENANCE_WHOLE_ASR_MAX_NEW_TOKENS = $WholeAsrMaxNewTokens
$env:STRUCTURED_PROVENANCE_PREFER_CUDA = "0"
$env:CUDA_VISIBLE_DEVICES = "-1"

Write-Host "[1/5] Build/resume full AudioJailbreak Origin manifest"
& $Python `
  eval\scripts\datasets\build_audiojailbreak_manifest.py `
  --config Origin `
  --split origin `
  --all `
  --all-categories `
  --speech-sources jailbreakbench Do_Not_Answer jailbreak_llms other `
  --max-duration-sec 0 `
  --prefer-source-audio `
  --out $Manifest `
  --resume
Assert-NativeStepSucceeded "Build/resume full AudioJailbreak Origin manifest"

Write-Host "[2/5] Run structured provenance guard with whole-audio timestamped ASR on CPU"
& $Python -u `
  eval\scripts\current\run_structured_provenance_incremental.py `
  --manifest $Manifest `
  --out $GuardResults `
  --asr-model openai/whisper-large-v3 `
  --asr-max-new-tokens 440 `
  --asr-mode whole_timestamped `
  --enable-prompt-injection-guard `
  --resume
Assert-NativeStepSucceeded "Run structured provenance guard"

Write-Host "[3/5] Build manifest for guard-passed rows"
& $Python `
  eval\scripts\datasets\build_guard_passed_manifest.py `
  --source-manifest $Manifest `
  --guard-results $GuardResults `
  --out $PassedManifest `
  --task-type compliance
Assert-NativeStepSucceeded "Build manifest for guard-passed rows"

$passedCount = 0
if (Test-Path $PassedManifest) {
  $passedCount = (Get-Content $PassedManifest | Measure-Object -Line).Lines
}
Write-Host "Guard-passed rows: $passedCount"

if ($passedCount -gt 0) {
  Write-Host "[4/5] Send guard-passed full audio to Nemotron"
  & $Python -u `
    eval\scripts\baselines\run_nvidia_api_baseline.py `
    --model nemotron `
    --manifest $PassedManifest `
    --out $NemotronResults `
    --task-type compliance `
    --max-tokens 1024 `
    --blank-retry-max-tokens 4096 `
    --no-score-compliance `
    --resume
  Assert-NativeStepSucceeded "Send guard-passed full audio to Nemotron"

  Write-Host "[5/5] Judge Nemotron outputs with LG4 context scorer"
  & $Python -u `
    eval\scripts\baselines\rescore_audiojailbreak_lg4_context.py `
    --results $NemotronResults `
    --manifest $PassedManifest `
    --out $JudgeResults `
    --model meta/llama-guard-4-12b `
    --resume
  Assert-NativeStepSucceeded "Judge Nemotron outputs with LG4 context scorer"
} else {
  Write-Host "[4/5] No guard-passed rows; skipping Nemotron"
  Write-Host "[5/5] No guard-passed rows; skipping LG4 judge"
}

Write-Host "Pipeline complete."

