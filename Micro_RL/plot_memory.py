"""
Memory vs Hidden Size — Re-forward vs Classic (full activations).

Setup fisso:
- Topologia: d_in -> h -> d_out  (d_in=4, d_out=2)
- Buffer episodio: T=500 step
- Tipi: float32=4B, uint32_t=4B, pointer=4B
- DenseLayer con float** per le matrici (row-pointers inclusi)
- Ottimizzatore: Adam (W, dW, mW, vW; b, db, mb, vb; e 'out' per layer)
- CLASSIC = salvataggio pre+post attivazioni per ogni layer a ogni step

================================================================================
                          PANNELLO DI CONTROLLO
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ------------------------------------------------------------------ #
# 1. PARAMETRI MODELLO
# ------------------------------------------------------------------ #
d_in  = 4
d_out = 2
T     = 500
h_min, h_max, h_step = 8, 350, 1

float_bytes = 4
ptr_bytes   = 4

use_state_contiguous = False    # True se state_buffer e' contiguo (no row-pointers)

SRAM_LINES = [                  # linee di riferimento (nome, KB)
    ("STM32F4 - 128 KB",  128),
    ("STM32H7 - 1 MB",   1024),
]

# ------------------------------------------------------------------ #
# 2. STILE / FIGURA
# ------------------------------------------------------------------ #
FIGSIZE     = (8.0, 6.0)
DPI         = 300
FONT_FAMILY = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]
FONT_SIZE   = 20

COLOR_REFORWARD = "#1B9E77"     # verde acqua - approccio on-device
COLOR_CLASSIC   = "#6A3D9A"     # viola - backprop classica
SRAM_COLORS     = ["#9A9A9A", "#6E6E6E"]

LINE_WIDTH  = 2.4
GRID_MAJOR  = dict(lw=0.6, alpha=0.45, color="0.6")
GRID_MINOR  = dict(lw=0.3, alpha=0.30, color="0.75")

TITLE       = "Memory vs Hidden Size — Re-forward vs Classic"
SHOW_TITLE  = True
TITLE_PAD   = 54

LEGEND_Y    = 1.02

OUTPUT_BASENAME  = "memory_vs_hidden"
SAVE_FORMATS     = ["pdf", "svg", "png"]
SHOW             = False

# ================================================================== #
#                       FINE PANNELLO DI CONTROLLO
# ================================================================== #


def model_bytes(h: int) -> int:
    """Model + Adam + out-buffer + row-pointers + struct overhead."""
    sum_inout = h * (d_in + d_out)
    sum_out   = h + d_out
    model_float = float_bytes * (4 * sum_inout * 4 + 5 * sum_out)
    row_ptr     = 16 * sum_out
    struct_ovh  = 72
    return int(model_float + row_ptr + struct_ovh)


def buffer_bytes(T: int, d_in: int, contiguous: bool) -> int:
    """Buffer episodio condiviso (state + action + reward + advantage)."""
    base    = T * (d_in * float_bytes + 3 * float_bytes)
    row_ptr = 0 if contiguous else T * ptr_bytes
    return int(base + row_ptr)


def activations_bytes_classic(h: int, T: int) -> int:
    """Attivazioni tenute in memoria dal trainer classico (pre+post per layer, T step)."""
    return int(8 * T * (h + d_out))


def workspace_bytes(h: int) -> int:
    return int(float_bytes * max(h, d_out))


def main():
    hs  = np.arange(h_min, h_max + 1, h_step)
    buf = buffer_bytes(T, d_in, contiguous=use_state_contiguous)

    ref_kb = np.array([model_bytes(h) + buf + workspace_bytes(h) for h in hs]) / 1024.0
    cls_kb = np.array([model_bytes(h) + buf + activations_bytes_classic(h, T) for h in hs]) / 1024.0

    families = FONT_FAMILY if isinstance(FONT_FAMILY, (list, tuple)) else [FONT_FAMILY]
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": list(families),
        "font.size": FONT_SIZE,
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
    })

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.plot(hs, cls_kb, color=COLOR_CLASSIC,   lw=LINE_WIDTH, solid_capstyle="round",
            zorder=4, label="Classic backprop")
    ax.plot(hs, ref_kb, color=COLOR_REFORWARD, lw=LINE_WIDTH, solid_capstyle="round",
            zorder=4, label="Re-forward (on-device)")

    for (name, limit_kb), c in zip(SRAM_LINES, SRAM_COLORS):
        ax.axhline(limit_kb, linestyle="--", color=c, linewidth=1.2, zorder=2)
        ax.text(h_max, limit_kb, f" {name}", va="center", ha="left",
                fontsize=FONT_SIZE * 0.55, color=c)

    ax.set_xlabel("Hidden layer size (units)")
    ax.set_ylabel("Memory (KB)")
    ax.set_xlim(h_min, h_max)
    ax.set_ylim(bottom=0)

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which="major", **GRID_MAJOR)
    ax.grid(which="minor", axis="y", **GRID_MINOR)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    if SHOW_TITLE:
        ax.set_title(TITLE, pad=TITLE_PAD)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, LEGEND_Y),
              ncol=2, frameon=False, handlelength=1.6,
              columnspacing=1.4, borderaxespad=0.0)

    fig.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for fmt in SAVE_FORMATS:
        out = os.path.join(out_dir, f"{OUTPUT_BASENAME}.{fmt}")
        fig.savefig(out, format=fmt, bbox_inches="tight", dpi=DPI)
        print(f"[=] salvato {out}")

    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()
