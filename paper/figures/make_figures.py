"""Generate every figure in the paper.

All measured quantities live in NUMBERS below, sourced from the repo's own
measurement passes. When a run lands, edit NUMBERS and re-run this script --
no figure has a hardcoded value anywhere else.

Usage (from paper/figures/):
    <repo>/.venv/Scripts/python.exe make_figures.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Measured numbers. Sources named per block.
# ---------------------------------------------------------------------------

NUMBERS = {
    # scripts/report_negative_composition.py, 2026-08-20.
    "composition": {
        "train": {"n": 88214, "pos": 2.9, "hard": 25.7, "already": 7.2, "never": 64.3},
        "val": {"n": 20490, "pos": 2.8, "hard": 15.2, "already": 4.6, "never": 77.5},
        "test": {"n": 69875, "pos": 3.1, "hard": 39.2, "already": 4.3, "never": 53.4},
    },
    # Same pass: hard-temporal windows bucketed by time remaining until onset
    # (train split).
    "time_to_onset": [
        ("1.1-2.1", 2220),
        ("2.1-3.2", 1966),
        ("3.2-5.0", 2888),
        (">5.0", 15560),
    ],
    "train_positives": 2530,
    # METHODOLOGY.md "How far ahead the head looks", 2026-09-07.
    "lookahead": [(32, 2530), (60, 4481), (96, 6400), (150, 9604)],
    "reported_horizon": 32,
    "default_lookahead": 96,
    "bin_width": 4,
}

# Print-safe, colourblind-friendly. Positive is the only saturated colour.
C_POS = "#b2182b"
C_HARD = "#ef8a62"
C_ALREADY = "#999999"
C_NEVER = "#c7d9e8"
C_INK = "#222222"
C_MUTE = "#777777"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.dpi": 300,
    }
)

COL_W = 3.4      # IEEEtran single column, inches
FULL_W = 7.16    # IEEEtran double column, inches

BAR_Y, BAR_H = -0.30, 0.60


def fig_protocols() -> None:
    """Fig. 1 -- what each protocol samples from one pedestrian track."""
    fig, (ax_a, ax_s) = plt.subplots(
        2, 1, figsize=(FULL_W, 2.15), sharex=True,
        gridspec_kw={"hspace": 0.70, "left": 0.015, "right": 0.985,
                     "top": 0.88, "bottom": 0.19},
    )

    track_end, onset = 100.0, 76.0
    horizon = 11.0          # reported horizon H, in track units
    win_w, step = 8.5, 5.5  # window length and stride (drawn separated)

    for ax in (ax_a, ax_s):
        ax.set_xlim(-13, track_end + 2)
        ax.set_ylim(-1.05, 1.05)
        ax.axis("off")
        # the pedestrian track itself
        ax.add_patch(mpatches.Rectangle(
            (0, BAR_Y), track_end, BAR_H, facecolor="#f2f2f2",
            edgecolor="#d5d5d5", lw=0.6))
        ax.vlines(onset, BAR_Y - 0.16, 0.62, color=C_POS, lw=1.0,
                  ls=(0, (2.5, 2)))
        ax.text(onset, 0.66, "crossing onset", ha="center", va="bottom",
                fontsize=6.8, color=C_POS)

    # -- (a) anchored ----------------------------------------------------
    ax_a.text(-12.5, 0.66, "(a) Event-anchored", ha="left", va="bottom",
              fontsize=7.8, color=C_INK, fontweight="bold")
    w_lo = onset - 30
    ax_a.add_patch(mpatches.Rectangle(
        (w_lo, BAR_Y), win_w, BAR_H, facecolor=C_POS, edgecolor=C_POS, lw=0.7))
    ax_a.annotate("", xy=(onset, BAR_Y - 0.24), xytext=(w_lo + win_w, BAR_Y - 0.24),
                  arrowprops=dict(arrowstyle="<->", color=C_INK, lw=0.65,
                                  shrinkA=0, shrinkB=0))
    ax_a.text((onset + w_lo + win_w) / 2, BAR_Y - 0.34, "TTE 1-2 s",
              ha="center", va="top", fontsize=6.5, color=C_INK)
    ax_a.text(w_lo + win_w / 2, 0.36, "one window per track", ha="center",
              va="bottom", fontsize=6.5, color=C_POS)
    ax_a.text(w_lo / 2, 0, "never sampled", ha="center", va="center",
              fontsize=6.5, style="italic", color=C_MUTE)
    ax_a.text((onset + track_end) / 2 + 4, 0, "never sampled", ha="center",
              va="center", fontsize=6.5, style="italic", color=C_MUTE)

    # -- (b) streaming ---------------------------------------------------
    ax_s.text(-12.5, 0.66, "(b) Streaming", ha="left", va="bottom",
              fontsize=7.8, color=C_INK, fontweight="bold")

    for s in np.arange(0, track_end - win_w + 0.1, step):
        end = s + win_w
        if end > onset:
            face, edge = C_ALREADY, "#6f6f6f"       # crossing already underway
        elif end + horizon >= onset:
            face, edge = C_POS, C_POS               # positive
        elif end + 3.6 * horizon >= onset:
            face, edge = C_HARD, "#c1582f"          # hard temporal negative
        else:
            face, edge = C_NEVER, "#8fa9bf"         # easy negative
        ax_s.add_patch(mpatches.Rectangle(
            (s + 0.35, BAR_Y), win_w - 0.7, BAR_H, facecolor=face,
            edgecolor=edge, lw=0.6))

    ax_s.annotate("", xy=(onset, BAR_Y - 0.24), xytext=(onset - horizon, BAR_Y - 0.24),
                  arrowprops=dict(arrowstyle="<->", color=C_INK, lw=0.65,
                                  shrinkA=0, shrinkB=0))
    ax_s.text(onset - horizon / 2, BAR_Y - 0.34, "horizon $H$", ha="center",
              va="top", fontsize=6.5, color=C_INK)

    handles = [
        mpatches.Patch(facecolor=C_POS, edgecolor=C_POS,
                       label="positive: onset within $H$"),
        mpatches.Patch(facecolor=C_HARD, edgecolor="#c1582f",
                       label="hard temporal negative"),
        mpatches.Patch(facecolor=C_NEVER, edgecolor="#8fa9bf",
                       label="easy negative"),
        mpatches.Patch(facecolor=C_ALREADY, edgecolor="#6f6f6f",
                       label="already crossing (discarded)"),
    ]
    ax_s.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                ncol=4, frameon=False, handlelength=1.0, columnspacing=1.4,
                handletextpad=0.4)

    fig.savefig("fig1_protocols.pdf")
    plt.close(fig)


def fig_composition() -> None:
    """Fig. 2 -- what the streaming negatives are actually made of."""
    comp = NUMBERS["composition"]
    splits = ["train", "val", "test"]
    order = [("pos", "positive", C_POS),
             ("hard", "will cross, later", C_HARD),
             ("already", "already crossed", C_ALREADY),
             ("never", "never crosses", C_NEVER)]

    fig, ax = plt.subplots(figsize=(COL_W, 2.05))
    left = np.zeros(len(splits))
    for key, label, colour in order:
        vals = np.array([comp[s][key] for s in splits])
        ax.barh(splits, vals, left=left, color=colour, label=label,
                edgecolor="white", lw=0.6, height=0.60)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v >= 8:
                ax.text(l + v / 2, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if colour in (C_POS, C_HARD) else C_INK)
        left += vals

    # the positive sliver is too thin to label in place
    for i, s in enumerate(splits):
        ax.annotate(f"{comp[s]['pos']:.1f}", xy=(comp[s]["pos"], i - 0.30),
                    xytext=(6.5, i - 0.62), fontsize=6.2, color=C_POS,
                    ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color=C_POS, lw=0.45,
                                    shrinkA=0, shrinkB=1))

    ax.set_xlim(0, 100)
    ax.set_ylim(2.75, -0.75)
    ax.set_xlabel("share of windows (%)")
    ax.set_yticks(range(len(splits)))
    ax.set_yticklabels([f"{s}\n$N$={comp[s]['n']:,}" for s in splits], fontsize=7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.40), ncol=2,
              frameon=False, handlelength=1.1, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig("fig2_composition.pdf")
    plt.close(fig)


def fig_time_to_onset() -> None:
    """Fig. 3 -- most of the 'hard' mass is not hard."""
    buckets = NUMBERS["time_to_onset"]
    labels = [b[0] for b in buckets]
    counts = np.array([b[1] for b in buckets], dtype=float)
    pos = NUMBERS["train_positives"]

    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    colours = [C_HARD, C_HARD, C_NEVER, C_NEVER]
    bars = ax.bar(labels, counts, color=colours, edgecolor="#8a8a8a", lw=0.6,
                  width=0.68)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 380,
                f"{int(c):,}\n({c / counts.sum() * 100:.1f}%)", ha="center",
                va="bottom", fontsize=6.3, color=C_INK)

    ax.axhline(pos, color=C_POS, lw=1.0, ls=(0, (3, 2)),
               label=f"positives ({pos:,})")

    ax.set_ylabel("train windows")
    ax.set_xlabel("time remaining until crossing onset (s)")
    ax.set_ylim(0, counts.max() * 1.32)
    ax.legend(loc="upper left", frameon=False, fontsize=6.5,
              handlelength=1.4, bbox_to_anchor=(-0.02, 1.03))

    # bracket the confusable band, next to the bars it describes
    band_y = counts.max() * 0.36
    ax.annotate("", xy=(-0.34, band_y), xytext=(1.34, band_y),
                arrowprops=dict(arrowstyle="|-|,widthA=0.3,widthB=0.3",
                                color="#c1582f", lw=0.7))
    ax.text(0.5, band_y * 1.10, "confusable band", ha="center", va="bottom",
            fontsize=6.8, color="#c1582f")
    fig.tight_layout()
    fig.savefig("fig3_time_to_onset.pdf")
    plt.close(fig)


def fig_hazard_cases() -> None:
    """Fig. 4 -- the four supervision cases the binary label cannot express."""
    k = 9  # bins drawn; the real head uses K=24
    cases = [
        ("event in range", "onset lands in bin 4",
         [0, 0, 0, 0, 1, None, None, None, None], False),
        ("event beyond range", "crossing seen, past the head",
         [0] * k, False),
        ("censored", "footage ends after bin 3",
         [0, 0, 0, 0, None, None, None, None, None], False),
        ("already crossed", "not at risk of a first crossing",
         [None] * k, True),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(COL_W, 2.45),
                             gridspec_kw={"hspace": 0.95, "left": 0.32,
                                          "right": 0.99, "top": 0.88,
                                          "bottom": 0.14})

    for ax, (name, note, target, dropped) in zip(axes, cases):
        ax.set_xlim(-0.4, k + 1.9)
        ax.set_ylim(-0.5, 0.5)
        ax.axis("off")
        for i, t in enumerate(target):
            if t is None:
                face, edge, txt, tc = "white", "#b0b0b0", "--", "#b0b0b0"
            elif t == 1:
                face, edge, txt, tc = C_POS, C_POS, "1", "white"
            else:
                face, edge, txt, tc = C_NEVER, "#8fa9bf", "0", C_INK
            ax.add_patch(mpatches.Rectangle(
                (i, -0.30), 0.82, 0.60, facecolor=face, edgecolor=edge,
                lw=0.7, ls="--" if t is None else "-"))
            ax.text(i + 0.41, 0, txt, ha="center", va="center", fontsize=6.5,
                    color=tc)
        if dropped:
            ax.plot([-0.15, k - 0.05], [0, 0], color=C_MUTE, lw=0.8)
            ax.text(k + 0.25, 0, "window\ndiscarded", ha="left", va="center",
                    fontsize=5.8, color=C_MUTE, linespacing=1.15)
        ax.text(-0.7, 0.16, name, ha="right", va="center", fontsize=7,
                color=C_INK)
        ax.text(-0.7, -0.20, note, ha="right", va="center", fontsize=6,
                color=C_MUTE, style="italic")

    axes[-1].text(k / 2, -0.74, r"future bin $k \rightarrow$", ha="center",
                  va="top", fontsize=6.5, color=C_INK)
    axes[0].text(k / 2, 0.80,
                 "shaded = supervised   |   dashed = masked, zero gradient",
                 ha="center", va="bottom", fontsize=6.3, color=C_INK)

    fig.savefig("fig4_hazard_cases.pdf")
    plt.close(fig)


def main() -> None:
    plt.rcParams["text.usetex"] = False
    for fn in (fig_protocols, fig_composition, fig_time_to_onset, fig_hazard_cases):
        fn()
        print(f"wrote {fn.__name__}")


if __name__ == "__main__":
    main()
