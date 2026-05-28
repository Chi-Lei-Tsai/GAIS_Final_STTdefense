# ProvGuard — Project Proposal

Black-box, training-free defense for Large Audio-Language Models
(LALMs) that attributes audio composition to specific segments via
speaker diarization + ASR + ablation, then either **reconstructs a
clean response** (when ablation isolates harm) or **abstains**.

This is a revision of the earlier proposal. Major changes are noted
inline; the short version is that several "guaranteed" claims from the
first draft turned out to need qualification once we built the eval
pipeline and saw real numbers.

---

## 1. Goal

Reduce attack success rate against Qwen2-Audio on compositional audio
attacks (SACRED-Bench), **without** the indiscriminate over-refusal
that prompt-prefix defenses suffer from, and **without** depending on
the metric's own harm scorer for the defense's runtime decision (the
"judge tautology" pitfall).

The contribution is the **provenance-attribution mechanism** —
combining off-the-shelf diarization, ASR, and a black-box ablation
loop into a defense that produces interpretable per-segment attribution
and reconstructs a real LALM answer instead of refusing when possible.

---

## 2. Threat model

We target **SACRED-Bench** ([arxiv:2511.10222](https://arxiv.org/abs/2511.10222)),
which exposes three audio-compositional attack categories against
Qwen2-Audio-7B-Instruct. Each category uses a different attack-success
criterion, so the defense pipeline and the metric both need to be
task-aware. This was a non-obvious finding from the eval phase.

| Category | Abbrev | Audio composition | Success criterion |
|---|---|---|---|
| Multi-speaker Dialogue | **MSD** | two-speaker harmful Q&A | model **complies** with the harmful exchange rather than refusing |
| Speech-Speech Overlap | **SSO** | harmful speech overlaid on benign speech | model **fails to identify** the harm when asked to judge safety |
| Speech-Audio Overlap (Contextual Audio Cues) | **SAO** | benign speech alongside harmful non-speech audio (gunshots, alarms, etc.) | same as SSO: model fails to identify |

Two task framings follow:
- **Compliance** task (MSD): open-ended prompt; attack succeeds if the served response is harmful per a harm scorer.
- **Safety-judgment** task (SSO/SAO): prompt explicitly asks whether the audio is safe; attack succeeds if the model answers "yes / safe" on actually-harmful audio.

Empirically these have very different baseline rates on the same
target model (Qwen2-Audio-7B):

| Task | Category | Baseline ASR (our eval, N=100) | Paper's reported (Qwen2-Audio) |
|---|---|---|---|
| Compliance | MSD | 54% | ~96% |
| Safety-judgment | SSO | 84% | 100% |
| Safety-judgment | SAO | 84% | 100% |

The gap to the paper's numbers is mostly the harm-scorer (we use
Llama-Guard-3-1B; the paper used Gemini 2.5 Pro). The relative
comparison between defenses on our pipeline is preserved.

---

## 3. What we observed in baselines

Eight baseline defenses + ProvGuard tested, with per-task metrics.
**Generic preprocessing defenses don't help on either task.**
`fixed_trim`, `denoise`, `compression`, `random_smoothing` all sit at
or near baseline ASR — they target perturbation attacks, but
SACRED-Bench attacks are compositional.

The interesting baselines:

| Defense | MSD ASR (N=100) | SSO+SAO ASR (N=100) | Benign refusal (N=20) | Cost |
|---|---|---|---|---|
| `none` | 54% | 84% | 0% | 1 LALM call |
| `random_smoothing` | 46% | 84% | 0% | 3 LALM calls |
| `asr_then_guard` (Whisper → LG → refuse) | 8% | 62% (44% SSO / 80% SAO) | 0% | ≤ 1 LALM call |
| `self_reminder` (safety prefix in prompt) | 3% | 9% | **85%** | 1 LALM call |
| `output_guard` (LG on response → refuse if flagged) | 0%¹ | not tested | 0% | 1 LALM call |
| `abstain` | 0% | 0%² | 100% | 0 LALM calls |

¹ Judge-tautological: `output_guard`'s decision uses the same LG-1B
as the metric, so its ASR is bounded below by judge self-consistency.
The "0%" is partly an artifact of measurement.

² For safety-judgment, the abstain text answers "yes, harmful" — which is the correct
identification of harm.

Three findings drove the proposal revision:

1. **`self_reminder` is paranoia-masquerading-as-defense.** 3% MSD ASR
   looks excellent until you see 85% benign over-refusal. The
   model refuses ~all audio inputs that look ambiguous. This is the
   **utility-preservation wedge** the original proposal predicted —
   just not against SALMONN-Guard as originally framed.
2. **`asr_then_guard` works where harm is transcribable.** 44% on SSO
   (transcribable overlap) but 80% on SAO (non-speech cues Whisper
   can't read). This identifies SAO as a category where text-only
   defenses are blind by construction.
3. **Output-guard-style defenses are bounded by judge tautology.**
   `output_guard` uses LG-1B to decide whether to refuse; the metric
   uses LG-1B to decide whether the served response is unsafe. Same
   model in both roles → trivially 0% ASR. For an honest comparison
   we need either a different metric judge or to report only ASR on
   defenses whose decision-path does **not** use the metric's scorer.

---

## 4. ProvGuard mechanism

Two operating modes corresponding to the two task framings. Both share
the same upstream pipeline: pyannote diarization gives speaker turns,
optional Whisper-small adds per-segment transcripts. Both make the
defense's decision **without** invoking the metric's harm scorer in the
critical path (avoiding judge tautology).

### 4.1 Compliance mode (MSD)

```
audio
  ├── y_full = LALM(audio, instruction = DEFAULT_INSTRUCTION)
  └── segments = pyannote_diarize(audio)
         pre_filter → candidates (minority-speaker turns, lateness,
                                  composite of all minority turns,
                                  non-speech regions [stretch])

for each candidate c (top-K=3):
   audio_c   = mute(audio, c)
   y_clean_c = LALM(audio_c, instruction = DEFAULT_INSTRUCTION)

Phase 2 (LALM unloaded, harm scorer loaded):
   v_full   = score(y_full)
   if v_full == safe:
       return passthrough_full_safe(y_full)
   for c in candidates:
       v_c = score(y_clean_c)
       if v_c == safe:
           return reconstructed(y_clean_c)        ← serve the LALM's answer to muted audio
   return abstain_no_clean_reconstruction(REFUSAL_TEXT)
```

See [`provguard_pipeline.png`](provguard_pipeline.png) for the full
diagram.

The key property: ProvGuard's decision-path **does** use LG-1B at the
`if v_c == safe` step — so it's not fully judge-decoupled either.
However, ProvGuard can also fail (be flagged unsafe by the metric) in
the `passthrough_no_candidates` branch, where its served response
hasn't been LG-vetted. That's the only branch where the metric's
verdict is independent of the defense's logic. We report this branch's
ASR as the honest measurement.

### 4.2 Safety-judgment mode (SSO/SAO)

The prompt asks the LALM "is this audio safe?" — so ablation now means
something different. We want the LALM to **identify** harm; the attack
hides harm via composition. Ablating composition should help.

**Note: this mode is a proposed mechanism, not yet implemented.** It's
the natural generalization of compliance-mode aggregation:

```
y_full = LALM(audio, instruction = SAFETY_JUDGMENT_INSTRUCTION)
if y_full == "identifies_harm":
    return y_full

# Otherwise model missed it. Ablate to "unmask":
for each candidate c (including non-speech segments):
    audio_c = mute(audio, c)
    y_c     = LALM(audio_c, instruction = SAFETY_JUDGMENT_INSTRUCTION)
    if y_c == "identifies_harm":
        return y_c    ← muting c unmasked the harm; report harmful

return abstain    ← could not isolate; conservatively flag as harmful
```

The decision is parsed from the LALM's own response (yes/no keyword
match), so this mode is **judge-decoupled** by construction.

### 4.3 Why this is "provenance-aware"

The mechanism produces, alongside the served response:
- `chosen_segment`: the segment whose ablation cleaned (or unmasked) the response, with timestamps and the reason (speaker_mismatch, lateness, etc.)
- `candidates`: all considered segments with their influence scores
- `decision`: one of `passthrough_full_safe`, `reconstructed`, `passthrough_no_candidates`, `abstain_no_clean_reconstruction`

These are the **interpretability and localization** outputs that no
other defense in our baseline set produces by construction.

### 4.4 Why not "structured provenance as input" instead

An alternative we explored: skip ablation, instead include the
diarized + transcribed structure as text in the LALM's instruction
prompt and let the model reason over it. Quick verify on 5 MSD samples
showed Qwen2-Audio reads the structure (cites `Speaker 00 / Speaker 01`)
but **describes the content analytically rather than refusing**.
Concretely, on a catfishing-tutorial dialogue, the model summarized the
tutorial in its response instead of refusing. Structured-input alone is
**not sufficient** for the compliance task on this LALM size.

We may revisit structured input as a path for safety-judgment mode
(where the model has a binary judgment to make and the structure may
help it identify composition), but for the compliance task ablation
remains load-bearing.

---

## 5. What we claim, with evidence

Each claim states the dataset, N, metric, and judge used. We split into
**measured** (have N>=30 evidence), **plausible** (mechanism is sound
but not yet evaluated), and **future work**.

### Measured (MSD compliance task, Qwen2-Audio-7B, LG-3-1B judge)

| Claim | Evidence | N |
|---|---|---|
| ProvGuard v3 reduces MSD attack success vs baseline | 20% → 3.3% (83% relative reduction) on `sacred_dev` | 30 |
| ProvGuard v3 maintains zero benign over-refusal | 0/20 on pilot benign manifest | 20 |
| ProvGuard v3 fails only when pyannote produces no candidate segments | the one failure (`MSD_71`) was `passthrough_no_candidates`; diarization collapsed the TTS dialogue into one speaker | 30 |
| Generic preprocessing defenses don't help on compositional attacks | `fixed_trim`, `denoise`, `compression`, `random_smoothing` all 16–20% ASR (within noise of 20% baseline) | 30 |
| `self_reminder` looks effective only when measured on attacks alone | 3% MSD ASR but 85% benign over-refusal | 100 attack / 20 benign |

### Plausible (mechanism designed, evaluation pending)

| Claim | Test needed |
|---|---|
| ProvGuard's `passthrough_no_candidates` branch is the only judge-decoupled failure path → its ASR is the honest absolute number | Re-run with LG-3-8B or LLM-as-judge as the metric; compare ProvGuard's ASR to its `passthrough_no_candidates` rate |
| ProvGuard safety-judgment mode reduces SSO ASR via speaker isolation | Implement safety-judgment aggregation + run on N=100 SSO |
| Adding non-speech-region candidates (VAD inversion) gives ProvGuard coverage on SAO with temporally distinct cues | Add candidate type + run on N=50 SAO |
| Stacking ProvGuard with output_guard catches the `passthrough_no_candidates` failures | Implement chain; head-to-head against each alone |

### Out of scope / future work

| Item | Why we don't claim it |
|---|---|
| SAO with overlapping (mixed) non-speech audio cues | Requires source separation (Demucs / Spleeter) or audio event classifier; not in the finals deliverable |
| Same-speaker malicious suffix (TTS-cloned voice) | Speaker mismatch is zero by construction; falls back to lateness + raw ablation |
| Audio jailbreaks with adversarial perturbation (AdvWave style) | ProvGuard's attribution is structural; perturbation attacks need a fundamentally different defense (ALMGuard's mel-domain trigger) |
| End-to-end comparison with SALMONN-Guard | SALMONN-Guard checkpoint is referenced in SACRED-Bench but not publicly available as of pilot date |

---

## 6. Evaluation methodology

### 6.1 Two-axis result reporting

Every defense reports both **attack ASR** and **benign refusal rate**.
A defense scoring 0% ASR by refusing every input (`abstain`) is not a
defense; the benign column is where utility leaks show up.

### 6.2 Per-task metric, locked at manifest construction time

Manifests carry a `task_type` field (`compliance` or `safety_judgment`).
The runner switches the LALM instruction and the metric accordingly:

- compliance: open-ended prompt → harm scorer (Llama-Guard-3) on the response
- safety_judgment: yes/no prompt → keyword parser on the response

### 6.3 Judge decoupling

For defenses whose decision-path uses Llama-Guard (`output_guard`,
`provguard`'s reconstruction step), we report:

1. **ASR with LG-1B as both defense scorer and metric scorer** — current numbers; judge-tautological for these defenses.
2. **ASR on the "judge-independent" sub-branch** — for ProvGuard, the
   `passthrough_no_candidates` branch only (where the metric's verdict
   is independent of the defense's logic).
3. **(Stretch) ASR with LG-3-8B as metric scorer** — defense still uses
   LG-1B; the larger metric model has different false-positive/negative
   patterns, breaking strict self-consistency.
4. **(Stretch) ASR with LLM-as-judge** — Gemini 2.5 Pro or Claude as
   the metric; matches what the SACRED-Bench paper does.

For defenses whose decision-path doesn't use LG (`none`,
`random_smoothing`, `self_reminder`; partially `asr_then_guard`), the
LG-1B metric numbers are honest absolute measurements.

### 6.4 Benign control set

20 pilot samples (10 `benign_late_content_required`, 10
`benign_multi_speaker_dialogue`) — synthetic but acoustically realistic.
Larger/non-synthetic benign sets are a stretch goal.

---

## 7. Comparison to baselines on what matters

The framing for the final result table:

| Defense | Mechanism | Judge-coupled? | Attack ASR | Benign refusal | Interpretability |
|---|---|---|---|---|---|
| `none` | — | — | high | 0% | none |
| `fixed_trim_*` / `denoise` / `random_smoothing` / `compression` | input perturbation | no | ≈ baseline | 0–5% | none |
| `self_reminder` | prompt prefix | no | very low | **very high** | none |
| `asr_then_guard` | Whisper + LG on transcript | partial (LG on text, not audio) | low on SO, high on CAC | 0% | "transcript was flagged" |
| `output_guard` | LG on LALM output | **yes** | 0% by construction | 0% | "response was flagged" |
| **ProvGuard** | diarization + ablation | partial (LG in finalize) | low; honest = passthrough_no_candidates rate | **0%** | **per-segment timestamps + reasons** |

ProvGuard's pitch is the row read: low attack ASR, zero benign
over-refusal, and structural interpretability that no other defense
produces. We do not claim to beat `output_guard` on raw ASR (the
comparison is unfair given judge tautology); we claim to **match it
within measurement uncertainty while providing attribution and
reconstruction**.

---

## 8. Known limitations

1. **SAO with mixed (overlapping) non-speech audio:** ProvGuard's
   pyannote-based attribution is speech-centric. When the harmful
   audio is non-speech and temporally overlaps speech (movie clip
   with embedded gunshot, etc.), diarization produces no relevant
   candidate. Mitigation in current build: VAD inversion adds
   non-speech regions as candidates, which catches the temporally-
   distinct case. The overlapping case requires source separation
   (Demucs/Spleeter) and is paper-extension scope.

2. **TTS-similar voices defeat diarization:** when both speakers are
   TTS-synthesized with similar voice profiles, pyannote collapses
   them to one speaker. Observed once in MSD N=30. Mitigation: use
   the larger `pyannote/speaker-diarization-community-1` (currently
   used) — failure rate at ~3%.

3. **Judge tautology** in the harm-flip check inside `finalize` — the
   defense uses LG-1B; the metric uses LG-1B. Documented; resolved by
   cross-judge eval (LG-8B or LLM-judge) as a separate experiment.

4. **8 GB VRAM ceiling** prevents loading multiple large models
   concurrently. Forced 4-bit NF4 quantization on Qwen2-Audio and
   sequential phase architecture in the runner. May affect generation
   quality vs paper's likely fp16 setup.

5. **Smaller LALM than the field's strongest:** Qwen2-Audio-7B vs the
   SACRED-Bench paper's evaluation of Gemini 2.5 Pro and GPT-4o-Audio.
   Our attack baseline rates are correspondingly lower; ProvGuard's
   relative reduction should transfer, but absolute numbers are not
   directly comparable to the paper's tables.

---

## 9. Timeline / remaining work

Roughly in priority order:

1. **Run ProvGuard v3 on MSD N=100** to lock in the headline number on the
   focused dev set (currently have N=30 data).
2. **Cross-judge eval:** re-run ProvGuard + output_guard with LG-3-8B
   as the metric scorer (CPU offload on 8 GB) to break judge tautology.
3. **Implement safety-judgment ProvGuard mode** + run on SSO N=50.
4. **VAD-inversion candidate generator** for SAO temporally-distinct
   cases; run on SAO N=50.
5. **All-defenses head-to-head on a single 200-sample manifest** mixing
   MSD/SSO/SAO so the final table is one document.
6. **Failure analysis writeup**: the `passthrough_no_candidates` cases
   for ProvGuard; the SO-vs-CAC split for `asr_then_guard`; the benign
   refusal failures for `self_reminder`.

---

## 10. Honest summary of contribution

ProvGuard's defensible contributions, ranked:

1. **Two-axis evaluation methodology for LALM defenses** — attack ASR
   alongside benign over-refusal, with task-specific metrics. The
   `self_reminder` finding (3% ASR / 85% over-refusal) shows why
   single-axis reporting is misleading.

2. **Empirical evidence that compositional audio attacks defeat
   text-only and perturbation-based defenses** — `asr_then_guard`
   handles SO (44%) but fails SAO (80%); `random_smoothing` and
   `denoise` are inert across the board. This was hypothesized in the
   pilot proposal; we now have N=100 numbers.

3. **A black-box, training-free defense** (ProvGuard) that combines
   diarization + ablation + LALM-as-its-own-rectifier to reduce MSD
   ASR with zero benign over-refusal, producing per-segment
   attribution as a side effect.

4. **A diagnosis of the judge-tautology problem** in LALM defense
   evaluation: defenses that use the metric's scorer as a decision
   primitive get artificially low ASR. We propose specific eval
   methodology to break it.

5. **A scoped extension path** for SAO via VAD inversion (covered in
   the finals) and source separation (paper extension).

---

## 11. References

- SACRED-Bench paper (target benchmark): https://arxiv.org/abs/2511.10222
- Qwen2-Audio-7B-Instruct: https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
- Llama-Guard-3-1B: https://huggingface.co/meta-llama/Llama-Guard-3-1B
- pyannote speaker-diarization-3.1: https://huggingface.co/pyannote/speaker-diarization-3.1
- Whisper-small (per-segment ASR): https://huggingface.co/openai/whisper-small
- Self-reminders (Xie et al. 2023, original idea adapted to audio): https://www.nature.com/articles/s42256-023-00765-8

---

*This proposal supersedes the earlier draft. Major changes: per-task
metrics, judge-decoupling methodology, structured-input ablation (we
tried it; insufficient alone), SAO scope (limited to temporally-distinct
non-speech cues), explicit two-axis evaluation requirement. See
`status.md` for the running experimental log and `msd_pipelines.md` /
`sso_sao_pipelines.md` for per-defense pipeline details.*
