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
        (modifica solo questa sezione per fare tuning del grafico)
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ------------------------------------------------------------------ #
# 1. PARAMETRI MODELLO  +  RANGE / SCALA ASSE X
# ------------------------------------------------------------------ #
d_in  = 4
d_out = 2
T     = 500
h_min, h_max, h_step = 8, 12000, 1   # range di calcolo (dati generati fino a h_max)

float_bytes = 4
ptr_bytes   = 4

use_state_contiguous = False

# Scala dell'asse X:  True = logaritmica,  False = lineare
X_LOG = False

# Limite destro VISIBILE dell'asse X.
#   None  →  usa h_max (mostra tutto il range calcolato)
#   int   →  tronca la vista a quel valore (utile per zoom; i dati oltre
#             vengono comunque calcolati ma non mostrati)
X_PLOT_MAX = None

SRAM_LINES = [                          # (nome, KB)
    ("STM32F446 — 128 KB",   128),
    ("STM32H7 — 1 MB",      1024),
    ("STM32N6 — 4 MB",      4096),
]
SRAM_COLORS = ["#9A9A9A", "#6E6E6E", "#3A3A3A"]

# ------------------------------------------------------------------ #
# 2. STILE / FIGURA
# ------------------------------------------------------------------ #
FIGSIZE     = (9.0, 6.0)
DPI         = 300
FONT_FAMILY = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]
FONT_SIZE   = 20

COLOR_REFORWARD = "#1B9E77"     # verde acqua - approccio on-device
COLOR_CLASSIC   = "#6A3D9A"     # viola - backprop classica

LINE_WIDTH  = 2.4
GRID_MAJOR  = dict(lw=0.6, alpha=0.45, color="0.6")
GRID_MINOR  = dict(lw=0.3, alpha=0.30, color="0.75")

SHOW_TITLE  = False

# Legenda dentro il plot, centro-sinistra
LEGEND_LOC     = "center left"
LEGEND_ANCHOR  = (0.04, 0.72)
LEGEND_NCOL    = 1

# Asse Y: mostra solo fino a poco sopra la linea N6 (4096 KB).
# La curva Classic esce dal frame per h grandi; le 6 intersezioni restano visibili.
Y_MAX = 5000   # KB

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
    model_float = float_bytes * (4 * sum_inout + 5 * sum_out)
    row_ptr     = 16 * sum_out
    struct_ovh  = 72
    return int(model_float + row_ptr + struct_ovh)


def buffer_bytes(T: int, d_in: int, contiguous: bool) -> int:
    """Buffer episodio condiviso (state + action + reward + advantage)."""
    base    = T * (d_in * float_bytes + 3 * float_bytes)
    row_ptr = 0 if contiguous else T * ptr_bytes
    return int(base + row_ptr)


def activations_bytes_classic(h: int, T: int) -> int:
    """Attivazioni tenute in memoria dal trainer PyTorch-style (solo post-activation per layer, T step)."""
    return int(4 * T * (h + d_out))


def workspace_bytes(h: int) -> int:
    return int(float_bytes * max(h, d_out))


def find_intersection(hs: np.ndarray, curve_kb: np.ndarray, limit_kb: float):
    """Restituisce h dove curve_kb supera limit_kb per la prima volta (da sotto)."""
    idx = np.searchsorted(curve_kb, limit_kb)
    if idx == 0 or idx >= len(hs):
        return None
    h0, h1 = float(hs[idx - 1]), float(hs[idx])
    y0, y1 = curve_kb[idx - 1], curve_kb[idx]
    if y1 == y0:
        return None
    return h0 + (limit_kb - y0) * (h1 - h0) / (y1 - y0)


def main():
    x_plot_max = h_max if X_PLOT_MAX is None else min(int(X_PLOT_MAX), h_max)

    hs  = np.arange(h_min, h_max + 1, h_step, dtype=int)
    buf = buffer_bytes(T, d_in, contiguous=use_state_contiguous)

    ref_kb = np.array(
        [model_bytes(h) + buf + workspace_bytes(h)              for h in hs], dtype=float
    ) / 1024.0
    cls_kb = np.array(
        [model_bytes(h) + buf + activations_bytes_classic(h, T) for h in hs], dtype=float
    ) / 1024.0

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

    hs_f = hs.astype(float)

    ax.plot(hs_f, cls_kb, color=COLOR_CLASSIC,   lw=LINE_WIDTH,
            solid_capstyle="round", zorder=4, label="Classic backprop")
    ax.plot(hs_f, ref_kb, color=COLOR_REFORWARD, lw=LINE_WIDTH,
            solid_capstyle="round", zorder=4, label="Re-forward (on-device)")

    # ---- Linee orizzontali SRAM ----
    for (name, limit_kb), sram_color in zip(SRAM_LINES, SRAM_COLORS):
        ax.axhline(limit_kb, linestyle="--", color=sram_color, linewidth=1.2, zorder=2)
        ax.text(x_plot_max, limit_kb, f"  {name}", va="center", ha="left",
                fontsize=FONT_SIZE * 0.55, color=sram_color, clip_on=False)

    # ---- Evidenziazione intersezioni ----
    # Raccoglie (h_x, y_kb, sram_color, curve_color) per ogni (curva, linea SRAM)
    marker_list = []
    for (name, limit_kb), sram_color in zip(SRAM_LINES, SRAM_COLORS):
        for curve_kb, curve_color in [(cls_kb, COLOR_CLASSIC), (ref_kb, COLOR_REFORWARD)]:
            h_x = find_intersection(hs_f, curve_kb, limit_kb)
            if h_x is not None:
                marker_list.append((h_x, limit_kb, sram_color, curve_color))

    for h_x, limit_kb, sram_color, curve_color in marker_list:
        if h_x > x_plot_max:
            continue
        # linea tratteggiata verticale dall'asse X all'intersezione (colore curva)
        ax.plot([h_x, h_x], [0, limit_kb],
                linestyle="--", color=curve_color, linewidth=1.0, alpha=0.55, zorder=3)
        # punto marcatore all'intersezione
        ax.scatter([h_x], [limit_kb], s=60, color=curve_color, zorder=7,
                   edgecolors="white", linewidths=0.9)
        # etichetta h affiancata al punto, dentro il plot, con offset fisso in punti
        # se vicino al bordo destro, si sposta a sinistra per non uscire
        near_right = np.log10(h_x) > np.log10(x_plot_max) - 0.55
        ha_ann = "right" if near_right else "left"
        xt     = (-5, 4) if near_right else (5, 4)
        ax.annotate(f"h = {int(round(h_x))}",
                    xy=(h_x, limit_kb), xytext=xt,
                    textcoords="offset points",
                    ha=ha_ann, va="bottom",
                    fontsize=FONT_SIZE * 0.5, color=curve_color,
                    zorder=8, clip_on=False)

    # ---- Assi, griglia, limiti ----
    ax.set_xlabel("Hidden layer size  $h$")
    ax.set_ylabel("Memory (KB)")

    if X_LOG:
        ax.set_xscale("log")

    ax.set_xlim(h_min, x_plot_max)
    ax.set_ylim(0, Y_MAX)

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if not X_LOG:
        ax.xaxis.set_minor_locator(AutoMinorLocator())

    ax.grid(which="major", **GRID_MAJOR)
    ax.grid(which="minor", **GRID_MINOR)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.legend(loc=LEGEND_LOC, bbox_to_anchor=LEGEND_ANCHOR,
              ncol=LEGEND_NCOL, frameon=False, handlelength=1.6,
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
