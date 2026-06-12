"""
Learning curve — singolo gruppo PC.

Carica pc_learning_curve{1..N}.csv dalla stessa cartella e mostra
media + banda di incertezza (deviazione standard).

================================================================================
                          PANNELLO DI CONTROLLO
================================================================================
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ------------------------------------------------------------------ #
# 1. DATI
# ------------------------------------------------------------------ #
DATA_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_PATTERN = "pc_learning_curve{i}.csv"   # file PC
REWARD_COL  = "reward"
RUN_IDS     = list(range(1, 11))

# ------------------------------------------------------------------ #
# 2. AGGREGAZIONE
# ------------------------------------------------------------------ #
N_GRID     = 400
BAND       = "std"          # "std" | "sem" | "pct"
BAND_SCALE = 1.0
PCT_LO, PCT_HI = 25, 75

# ------------------------------------------------------------------ #
# 3. SMOOTHING
# ------------------------------------------------------------------ #
SMOOTH        = True
SMOOTH_WINDOW = 21

# ------------------------------------------------------------------ #
# 4. COLORI / STILE
# ------------------------------------------------------------------ #
COLOR      = "#6A3D9A"      # viola
LINE_ALPHA = 1.0
BAND_ALPHA = 0.18
LINE_WIDTH = 2.4

FIGSIZE     = (8.0, 6.0)
DPI         = 300
FONT_FAMILY = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]
FONT_SIZE   = 20

GRID_MAJOR = dict(lw=0.6, alpha=0.45, color="0.6")
GRID_MINOR = dict(lw=0.3, alpha=0.30, color="0.75")

XLIM     = None
YLIM     = None
Y_MARGIN = 0.05

# ------------------------------------------------------------------ #
# 5. TITOLO, LEGENDA e OUTPUT
# ------------------------------------------------------------------ #
TITLE       = "PC learning curve"
SHOW_TITLE  = True
TITLE_PAD   = 54
CURVE_LABEL = "PC"
LEGEND_Y    = 1.02

OUTPUT_BASENAME = "learning_curve_summary"
SAVE_FORMATS    = ["pdf", "svg", "png"]
SHOW            = False

# ================================================================== #
#                       FINE PANNELLO DI CONTROLLO
# ================================================================== #


def _rolling(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling-mean centrata con finestra che si accorcia ai bordi (no NaN)."""
    if not SMOOTH or window <= 1:
        return arr
    n = len(arr)
    half = window // 2
    csum = np.concatenate(([0.0], np.cumsum(arr, dtype=float)))
    idx = np.arange(n)
    lo = np.maximum(0, idx - half)
    hi = np.minimum(n, idx + half + 1)
    return (csum[hi] - csum[lo]) / (hi - lo)


def _read_csv(path: str) -> dict:
    """Legge un CSV con header e restituisce {colonna: ndarray float}."""
    with open(path) as fh:
        header = fh.readline().strip().split(",")
    data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float)
    data = np.atleast_2d(data)
    return {name: data[:, j] for j, name in enumerate(header)}


def main():
    runs = []
    for i in RUN_IDS:
        path = os.path.join(DATA_DIR, CSV_PATTERN.format(i=i))
        if not os.path.exists(path):
            print(f"[!] file mancante: {path}")
            continue
        df     = _read_csv(path)
        reward = _rolling(np.asarray(df[REWARD_COL], dtype=float), SMOOTH_WINDOW)
        x      = np.arange(1, len(reward) + 1, dtype=float)
        runs.append((x, reward))

    if not runs:
        raise SystemExit("Nessun file valido trovato.")

    print(f"[+] {len(runs)} run caricate")

    x_min = max(r[0][0]  for r in runs)
    x_max = min(r[0][-1] for r in runs)
    grid  = np.linspace(x_min, x_max, N_GRID)

    stacked = np.vstack([np.interp(grid, x, y) for x, y in runs])
    mean    = stacked.mean(axis=0)

    if BAND == "std":
        spread = stacked.std(axis=0) * BAND_SCALE
        lo, hi = mean - spread, mean + spread
    elif BAND == "sem":
        spread = stacked.std(axis=0) / np.sqrt(stacked.shape[0]) * BAND_SCALE
        lo, hi = mean - spread, mean + spread
    elif BAND == "pct":
        lo = np.percentile(stacked, PCT_LO, axis=0)
        hi = np.percentile(stacked, PCT_HI, axis=0)
    else:
        raise ValueError(f"BAND sconosciuto: {BAND!r}")

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

    ax.fill_between(grid, lo, hi, color=COLOR, alpha=BAND_ALPHA, linewidth=0, zorder=2)
    ax.plot(grid, mean, color=COLOR, lw=LINE_WIDTH, alpha=LINE_ALPHA,
            label=CURVE_LABEL, zorder=3, solid_capstyle="round")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which="major", **GRID_MAJOR)
    ax.grid(which="minor", **GRID_MINOR)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    if XLIM is not None:
        ax.set_xlim(*XLIM)
    if YLIM is not None:
        ax.set_ylim(*YLIM)
    else:
        y0, y1 = ax.get_ylim()
        pad = (y1 - y0) * Y_MARGIN
        ax.set_ylim(y0 - pad, y1 + pad)

    if SHOW_TITLE and TITLE:
        ax.set_title(TITLE, pad=TITLE_PAD)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, LEGEND_Y),
              ncol=1, frameon=False, handlelength=1.6, borderaxespad=0.0)

    fig.tight_layout()

    for fmt in SAVE_FORMATS:
        out = os.path.join(DATA_DIR, f"{OUTPUT_BASENAME}.{fmt}")
        fig.savefig(out, format=fmt, bbox_inches="tight", dpi=DPI)
        print(f"[=] salvato {out}")

    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()
