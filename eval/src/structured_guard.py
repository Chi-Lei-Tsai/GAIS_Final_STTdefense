"""Deterministic guard over a provenance-labeled segment timeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.provenance import ProvenanceTimeline, Segment


TRUSTED_LABELS = {"trusted_primary_user", "trusted_context"}
UNTRUSTED_LABELS = {
    "untrusted_secondary_speaker",
    "untrusted_late_suffix",
    "untrusted_background_or_overlap",
}
UNCERTAIN_LABELS = {"unknown_low_confidence"}
DIALOGUE_SAFETY_TASKS = {"dialogue_safety", "msd_dialogue_safety"}


@dataclass
class GuardDecision:
    decision: str
    primary_intent_segment_ids: list[str] = field(default_factory=list)
    included_segment_ids: list[str] = field(default_factory=list)
    excluded_segment_ids: list[str] = field(default_factory=list)
    safety_relevant_segment_ids: list[str] = field(default_factory=list)
    reconstructed_user_request: str = ""
    confidence: str = "medium"
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "decision": self.decision,
            "primary_intent_segment_ids": self.primary_intent_segment_ids,
            "included_segment_ids": self.included_segment_ids,
            "excluded_segment_ids": self.excluded_segment_ids,
            "safety_relevant_segment_ids": self.safety_relevant_segment_ids,
            "reconstructed_user_request": self.reconstructed_user_request,
            "confidence": self.confidence,
            "reason": self.reason,
        }


def _label(seg: Segment) -> str:
    return seg.provenance.label


def _join_transcripts(segments: list[Segment]) -> str:
    parts = [s.transcript.strip() for s in segments if s.transcript.strip()]
    return " ".join(parts).strip()


def _is_safety_relevant(seg: Segment) -> bool:
    """Return whether a segment contains direct harmful-content evidence.

    Provenance suspicion is not harm evidence by itself. A late segment,
    secondary speaker, overlap, or non-speech region may tell the guard what
    to inspect, but it should not make a safety-judgment task answer "yes,
    harmful" unless a semantic/audio evidence model actually claims harm.
    """
    flags = set(seg.risk_flags)
    if "unsafe_claimed_by_model" in flags:
        return True
    return seg.transcript_evidence.is_unsafe is True


def _contains_primary_unsafe_request(seg: Segment) -> bool:
    return _label(seg) == "trusted_primary_user" and (
        "unsafe_claimed_by_model" in set(seg.risk_flags)
        or seg.transcript_evidence.is_unsafe is True
    )


def _timeline_transcript_is_unsafe(timeline: ProvenanceTimeline) -> bool:
    evidence = timeline.metadata.get("timeline_transcript_evidence") or {}
    return evidence.get("is_unsafe") is True


def _timeline_prompt_injection_evidence(timeline: ProvenanceTimeline) -> dict:
    evidence = timeline.metadata.get("prompt_injection_evidence") or {}
    if evidence.get("was_classified") is not True:
        return {}
    confidence = str(evidence.get("confidence") or "low").lower()
    if confidence == "low":
        return {}
    if any(
        evidence.get(key) is True
        for key in (
            "prompt_injection_present",
            "policy_bypass_present",
            "roleplay_jailbreak_present",
            "unsafe_task",
        )
    ):
        return evidence
    return {}


def _is_meaningful_uncertainty(seg: Segment) -> bool:
    """Tiny empty diarization/ASR fragments should not drive refusal."""

    if _label(seg) not in UNCERTAIN_LABELS:
        return False
    duration = max(0.0, seg.end_sec - seg.start_sec)
    if seg.kind == "speech" and not seg.transcript.strip() and duration < 0.5:
        return False
    return True


class StructuredGuard:
    """Rule guard for the first structured-provenance prototype.

    The point of this first guard is not to be a perfect classifier. It
    provides a constrained reasoning contract over the provenance object
    so the final LALM call only sees a reconstructed trusted request.
    """

    def decide(self, timeline: ProvenanceTimeline) -> GuardDecision:
        if timeline.task_type == "safety_judgment":
            return self._decide_safety_judgment(timeline)
        if timeline.task_type in DIALOGUE_SAFETY_TASKS:
            return self._decide_dialogue_safety(timeline)
        return self._decide_compliance(timeline)

    def _decide_dialogue_safety(self, timeline: ProvenanceTimeline) -> GuardDecision:
        """Compliance-style refusal policy for harmful multi-speaker dialogue.

        MSD is structurally different from prompt-injection benchmarks: the
        dialogue itself is the object being judged, so unsafe evidence from a
        secondary speaker is still safety evidence rather than something to
        silently ignore.
        """
        segments = timeline.segments
        safety_relevant = [s for s in segments if _is_safety_relevant(s)]
        timeline_text_unsafe = _timeline_transcript_is_unsafe(timeline)
        prompt_injection_evidence = _timeline_prompt_injection_evidence(timeline)
        if timeline_text_unsafe:
            for seg in segments:
                if (
                    seg.kind == "speech"
                    and seg.transcript.strip()
                    and seg not in safety_relevant
                ):
                    safety_relevant.append(seg)

        if prompt_injection_evidence:
            return GuardDecision(
                decision="refuse_prompt_injection",
                primary_intent_segment_ids=[
                    s.id for s in segments if _label(s) == "trusted_primary_user"
                ],
                included_segment_ids=[],
                excluded_segment_ids=[
                    s.id
                    for s in segments
                    if _label(s) in UNTRUSTED_LABELS or _label(s) in UNCERTAIN_LABELS
                ],
                safety_relevant_segment_ids=[
                    s.id for s in segments if s.kind == "speech" and s.transcript.strip()
                ],
                reconstructed_user_request="",
                confidence=str(prompt_injection_evidence.get("confidence") or "medium"),
                reason=(
                    "Whole-transcript classifier identified prompt-injection "
                    "policy-bypass, or unsafe-task evidence: "
                    f"{prompt_injection_evidence.get('reason')}"
                ),
            )

        if safety_relevant:
            return GuardDecision(
                decision="refuse_unsafe_dialogue",
                primary_intent_segment_ids=[
                    s.id for s in segments if _label(s) == "trusted_primary_user"
                ],
                included_segment_ids=[],
                excluded_segment_ids=[
                    s.id
                    for s in segments
                    if _label(s) in UNTRUSTED_LABELS or _label(s) in UNCERTAIN_LABELS
                ],
                safety_relevant_segment_ids=[s.id for s in safety_relevant],
                reconstructed_user_request="",
                confidence="high",
                reason=(
                    "Timeline-level transcript evidence indicates harmful dialogue."
                    if timeline_text_unsafe
                    else "A dialogue segment contains direct harmful-content evidence."
                ),
            )

        return self._decide_compliance(timeline)

    def _decide_compliance(self, timeline: ProvenanceTimeline) -> GuardDecision:
        segments = timeline.segments
        primary = [s for s in segments if _label(s) == "trusted_primary_user"]
        trusted_context = [s for s in segments if _label(s) == "trusted_context"]
        trusted = primary + trusted_context
        excluded = [
            s
            for s in segments
            if _label(s) in UNTRUSTED_LABELS or _label(s) in UNCERTAIN_LABELS
        ]
        safety_relevant = [s for s in segments if _is_safety_relevant(s)]
        prompt_injection_evidence = _timeline_prompt_injection_evidence(timeline)

        if prompt_injection_evidence:
            return GuardDecision(
                decision="refuse_prompt_injection",
                primary_intent_segment_ids=[s.id for s in primary],
                included_segment_ids=[],
                excluded_segment_ids=[s.id for s in excluded],
                safety_relevant_segment_ids=[
                    s.id for s in segments if s.kind == "speech" and s.transcript.strip()
                ],
                reconstructed_user_request="",
                confidence=str(prompt_injection_evidence.get("confidence") or "medium"),
                reason=(
                    "Whole-transcript classifier identified prompt-injection "
                    "policy-bypass, or unsafe-task evidence: "
                    f"{prompt_injection_evidence.get('reason')}"
                ),
            )

        if any(_contains_primary_unsafe_request(s) for s in primary):
            return GuardDecision(
                decision="refuse_unsafe_primary_request",
                primary_intent_segment_ids=[s.id for s in primary],
                included_segment_ids=[],
                excluded_segment_ids=[s.id for s in segments if s not in primary],
                safety_relevant_segment_ids=[s.id for s in safety_relevant],
                confidence="high",
                reason="Trusted primary-user segment appears to contain unsafe content.",
            )

        reconstructed = _join_transcripts(trusted)
        if not reconstructed:
            decision = (
                "abstain_uncertain_provenance"
                if safety_relevant or excluded
                else "ask_clarification"
            )
            return GuardDecision(
                decision=decision,
                primary_intent_segment_ids=[],
                included_segment_ids=[],
                excluded_segment_ids=[s.id for s in excluded],
                safety_relevant_segment_ids=[s.id for s in safety_relevant],
                reconstructed_user_request="",
                confidence="low",
                reason="No trusted transcript could be reconstructed from the provenance timeline.",
            )

        untrusted_instruction = [
            s
            for s in excluded
            if s.transcript_evidence.is_instruction is True
            or "instruction_claimed_by_model" in set(s.risk_flags)
            or "unsafe_claimed_by_model" in set(s.risk_flags)
        ]
        if untrusted_instruction:
            return GuardDecision(
                decision="ignore_untrusted_injection",
                primary_intent_segment_ids=[s.id for s in primary],
                included_segment_ids=[s.id for s in trusted],
                excluded_segment_ids=[s.id for s in excluded],
                safety_relevant_segment_ids=[s.id for s in safety_relevant],
                reconstructed_user_request=reconstructed,
                confidence="high",
                reason="Untrusted segment has model-supplied instruction or safety evidence; reconstructing from trusted segments only.",
            )

        if excluded:
            return GuardDecision(
                decision="answer_reconstructed",
                primary_intent_segment_ids=[s.id for s in primary],
                included_segment_ids=[s.id for s in trusted],
                excluded_segment_ids=[s.id for s in excluded],
                safety_relevant_segment_ids=[s.id for s in safety_relevant],
                reconstructed_user_request=reconstructed,
                confidence="medium",
                reason="Some segments were excluded as uncertain or untrusted, but no unsafe instruction was detected.",
            )

        return GuardDecision(
            decision="answer_reconstructed",
            primary_intent_segment_ids=[s.id for s in primary],
            included_segment_ids=[s.id for s in trusted],
            excluded_segment_ids=[],
            safety_relevant_segment_ids=[s.id for s in safety_relevant],
            reconstructed_user_request=reconstructed,
            confidence="high",
            reason="All instruction-bearing content appears to come from trusted segments.",
        )

    def _decide_safety_judgment(self, timeline: ProvenanceTimeline) -> GuardDecision:
        segments = timeline.segments
        safety_relevant = [s for s in segments if _is_safety_relevant(s)]
        timeline_text_unsafe = _timeline_transcript_is_unsafe(timeline)
        prompt_injection_evidence = _timeline_prompt_injection_evidence(timeline)
        if timeline_text_unsafe:
            for seg in segments:
                if (
                    seg.kind == "speech"
                    and seg.transcript.strip()
                    and seg not in safety_relevant
                ):
                    safety_relevant.append(seg)
        uncertain = [s for s in segments if _is_meaningful_uncertainty(s)]
        included = [
            s
            for s in segments
            if _label(s) in TRUSTED_LABELS or _label(s) == "non_speech_context"
        ]
        excluded = [s for s in segments if _label(s) in UNTRUSTED_LABELS]

        if prompt_injection_evidence and not safety_relevant:
            return GuardDecision(
                decision="safety_judgment_harmful",
                primary_intent_segment_ids=[
                    s.id for s in segments if _label(s) == "trusted_primary_user"
                ],
                included_segment_ids=[s.id for s in included],
                excluded_segment_ids=[s.id for s in excluded],
                safety_relevant_segment_ids=[
                    s.id for s in segments if s.kind == "speech" and s.transcript.strip()
                ],
                reconstructed_user_request="",
                confidence=str(prompt_injection_evidence.get("confidence") or "medium"),
                reason=(
                    "Whole-transcript classifier identified prompt-injection "
                    "policy-bypass, or unsafe-task evidence: "
                    f"{prompt_injection_evidence.get('reason')}"
                ),
            )

        if safety_relevant:
            return GuardDecision(
                decision="safety_judgment_harmful",
                primary_intent_segment_ids=[
                    s.id for s in segments if _label(s) == "trusted_primary_user"
                ],
                included_segment_ids=[s.id for s in included],
                excluded_segment_ids=[s.id for s in excluded],
                safety_relevant_segment_ids=[s.id for s in safety_relevant],
                reconstructed_user_request="",
                confidence="high",
                reason=(
                    "Timeline-level transcript evidence indicates harmful content."
                    if timeline_text_unsafe
                    else "Structured timeline contains direct harmful-content evidence."
                ),
            )

        if uncertain:
            return GuardDecision(
                decision="abstain_uncertain_provenance",
                primary_intent_segment_ids=[
                    s.id for s in segments if _label(s) == "trusted_primary_user"
                ],
                included_segment_ids=[s.id for s in included],
                excluded_segment_ids=[s.id for s in excluded + uncertain],
                safety_relevant_segment_ids=[],
                reconstructed_user_request="",
                confidence="low",
                reason="No harmful segment was identified, but provenance is too uncertain for a clean safety judgment.",
            )

        return GuardDecision(
            decision="answer_reconstructed",
            primary_intent_segment_ids=[
                s.id for s in segments if _label(s) == "trusted_primary_user"
            ],
            included_segment_ids=[s.id for s in included],
            excluded_segment_ids=[s.id for s in excluded],
            safety_relevant_segment_ids=[],
            reconstructed_user_request="No harmful content was identified in the structured audio timeline.",
            confidence="medium",
            reason="No safety-relevant segment was identified by the structured provenance guard.",
        )
