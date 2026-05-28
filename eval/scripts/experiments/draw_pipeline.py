"""Render the ProvGuard pipeline diagram to PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[3] / "provguard_pipeline.png"

# --- palette ---
COLOR_INPUT = "#E3F2FD"     # light blue   — data
COLOR_PROC = "#E8F5E9"      # light green  — processing
COLOR_LALM = "#E1F5FE"      # cyan-ish     — LALM calls
COLOR_GUARD = "#F3E5F5"     # light purple — Llama-Guard
COLOR_DECIDE = "#FFF3E0"    # light orange — decision
COLOR_OUT_OK = "#C8E6C9"    # green        — serve LALM answer
COLOR_OUT_ABSTAIN = "#FFCDD2"  # red       — abstain
EDGE = "#37474F"


def box(ax, x, y, w, h, text, color="white"):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.12",
        linewidth=1.2, edgecolor=EDGE, facecolor=color, zorder=2,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=9.5, family="monospace", zorder=3)


def diamond(ax, x, y, w, h, text):
    pts = [[x + w / 2, y + h], [x + w, y + h / 2],
           [x + w / 2, y], [x, y + h / 2]]
    poly = plt.Polygon(pts, edgecolor=EDGE, facecolor=COLOR_DECIDE,
                       linewidth=1.2, zorder=2)
    ax.add_patch(poly)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=9.5, family="monospace", zorder=3)


def arrow(ax, x1, y1, x2, y2, label=None, label_xy=None,
          connectionstyle=None, linestyle="-"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->,head_length=0.35,head_width=0.22",
        mutation_scale=12, linewidth=1.3, color=EDGE, zorder=1,
        connectionstyle=connectionstyle, linestyle=linestyle,
    )
    ax.add_patch(a)
    if label:
        if label_xy is None:
            label_xy = ((x1 + x2) / 2, (y1 + y2) / 2)
        ax.text(label_xy[0], label_xy[1], label, ha="center", va="center",
                fontsize=8.5, style="italic", color="#455A64",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.95),
                zorder=4)


def phase_divider(ax, y, text):
    ax.axhline(y=y, xmin=0.04, xmax=0.96, color="#90A4AE",
               linestyle=(0, (4, 3)), linewidth=1.0, zorder=0)
    ax.text(0.5, y + 0.1, text, ha="left", va="bottom",
            fontsize=9, color="#546E7A", style="italic")


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 14))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # -- input
    box(ax, 5.25, 13.0, 2.5, 0.6, "input audio (16 kHz mono)", color=COLOR_INPUT)

    # =========== Phase 1: LALM + diarization-based candidate generation ===========

    # left branch: y_full from Qwen2-Audio
    box(ax, 0.6, 11.5, 3.6, 0.8, "Qwen2-Audio(audio)\n=> y_full",
        color=COLOR_LALM)
    arrow(ax, 5.7, 13.0, 2.4, 12.3)

    # right branch: pyannote diarization -> pre-filter -> per-candidate ablation
    box(ax, 8.0, 11.5, 4.5, 0.8,
        "pyannote diarization\n(speaker turns + boundaries)",
        color=COLOR_PROC)
    arrow(ax, 7.3, 13.0, 10.25, 12.3)

    box(ax, 8.0, 9.7, 4.5, 1.1,
        "cheap pre-filter\nminority-speaker turns + lateness >=75%\n"
        "+ composite candidate (all minority)",
        color=COLOR_PROC)
    arrow(ax, 10.25, 11.5, 10.25, 10.8)

    box(ax, 8.0, 7.7, 4.5, 1.4,
        "for each candidate c (top-K=3):\n"
        "  audio_c = mute(audio, c.start, c.end)\n"
        "  y_clean[c] = Qwen2-Audio(audio_c)",
        color=COLOR_LALM)
    arrow(ax, 10.25, 9.7, 10.25, 9.1)

    # -- Phase divider
    phase_divider(ax, 7.2, "Phase 2  —  free Qwen2-Audio, load Llama-Guard")

    # =========== Phase 2: harm scoring + decision ===========

    box(ax, 3.0, 5.6, 7.0, 1.0,
        "Llama-Guard-3-1B  (manual chat template)\n"
        "v_full = score(y_full);  v_clean[c] = score(y_clean[c]) for each c",
        color=COLOR_GUARD)
    # y_full from left branch -> harm scorer
    arrow(ax, 2.4, 11.5, 4.0, 6.6,
          label="y_full", label_xy=(2.3, 8.6))
    # y_clean[c] from right branch -> harm scorer
    arrow(ax, 10.25, 7.7, 9.0, 6.6,
          label="y_clean[c]", label_xy=(10.5, 7.05))

    # decision 1: was y_full safe?
    diamond(ax, 4.5, 3.5, 4.0, 1.5, "v_full == safe ?")
    arrow(ax, 6.5, 5.6, 6.5, 5.0)

    # outputs
    # passthrough (full safe)
    box(ax, 0.2, 1.7, 3.5, 1.0,
        "passthrough_full_safe\nserve y_full",
        color=COLOR_OUT_OK)
    arrow(ax, 5.0, 3.85, 2.0, 2.7, label="yes", label_xy=(3.3, 3.55))

    # decision 2: any y_clean safe?
    diamond(ax, 4.5, 1.2, 4.0, 1.5, "any v_clean[c] == safe ?")
    arrow(ax, 6.5, 3.5, 6.5, 2.7, label="no", label_xy=(7.05, 3.1))

    # reconstructed
    box(ax, 9.3, 1.7, 3.5, 1.0,
        "reconstructed\nserve y_clean[c*]\n(c* = first to flip)",
        color=COLOR_OUT_OK)
    arrow(ax, 8.0, 2.0, 9.3, 2.2, label="yes", label_xy=(8.6, 2.45))

    # abstain
    box(ax, 4.5, -0.1, 4.0, 0.9,
        "abstain_no_clean_reconstruction\nserve refusal text",
        color=COLOR_OUT_ABSTAIN)
    arrow(ax, 6.5, 1.2, 6.5, 0.8, label="no", label_xy=(7.0, 1.0))

    # title
    ax.text(6.5, 13.85, "ProvGuard pipeline",
            ha="center", va="bottom", fontsize=17, fontweight="bold")
    ax.text(6.5, 13.65,
            "black-box, training-free, plug-and-use defense for LALMs (v3 with abstain fallback)",
            ha="center", va="bottom", fontsize=10, style="italic", color="#546E7A")

    # legend
    legend_handles = [
        mpatches.Patch(color=COLOR_INPUT, label="input / data"),
        mpatches.Patch(color=COLOR_PROC, label="processing"),
        mpatches.Patch(color=COLOR_LALM, label="LALM call (Qwen2-Audio)"),
        mpatches.Patch(color=COLOR_GUARD, label="harm score (Llama-Guard)"),
        mpatches.Patch(color=COLOR_DECIDE, label="decision"),
        mpatches.Patch(color=COLOR_OUT_OK, label="serve LALM answer"),
        mpatches.Patch(color=COLOR_OUT_ABSTAIN, label="abstain (refusal)"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.06), frameon=False, ncol=4, fontsize=8.5)

    plt.tight_layout()
    fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
