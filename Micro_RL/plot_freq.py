"""
Bar chart raggruppato: Network Complexity (16/32/64) x {MCU, PC}.
y = Execution Time (s), con IC 95% come error bar.

================================================================================
                          PANNELLO DI CONTROLLO
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ------------------------------------------------------------------ #
# 1. DATI (secondi)
# ------------------------------------------------------------------ #
mcu_times_16 = [60.38, 59.89, 59.64, 59.78, 60.00, 60.32, 60.30, 59.85]
mcu_times_32 = [71.09, 70.62, 71.44, 70.80, 70.53, 70.29, 70.10, 70.22]
mcu_times_64 = [76.85, 75.78, 76.35, 76.11, 77.91, 76.00, 78.00, 77.67]

pc_times_16  = [13.82, 12.96, 12.96, 13.08, 13.05, 13.05, 13.29, 13.56]
pc_times_32  = [19.09, 18.27, 18.69, 18.73, 18.84, 18.90, 18.75, 18.89]
pc_times_64  = [22.38, 21.70, 22.01, 21.89, 22.27, 22.45, 22.93, 22.67]

COMPLEXITIES = [16, 32, 64]

# ------------------------------------------------------------------ #
# 2. STILE / FIGURA
# ------------------------------------------------------------------ #
FIGSIZE     = (8.0, 6.0)
DPI         = 300
FONT_FAMILY = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]
FONT_SIZE   = 20

COLOR_MCU = "#1B9E77"           # verde acqua
COLOR_PC  = "#6A3D9A"           # viola

BAR_WIDTH  = 0.38
BAR_ALPHA  = 0.88
CAPSIZE    = 5

GRID_MAJOR = dict(lw=0.6, alpha=0.45, color="0.6")
GRID_MINOR = dict(lw=0.3, alpha=0.30, color="0.75")

TITLE      = "Execution Time by Network Complexity"
SHOW_TITLE = True
TITLE_PAD  = 54

LEGEND_Y   = 1.02

OUTPUT_BASENAME = "exec_time_mcu_vs_pc"
SAVE_FORMATS    = ["pdf", "svg", "png"]
SHOW            = False

# ================================================================== #
#                       FINE PANNELLO DI CONTROLLO
# ================================================================== #

_T_MAP = {
    1: float("nan"), 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
    7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201,
    13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110,
    19: 2.101, 20: 2.093, 21: 2.086, 22: 2.080, 23: 2.074, 24: 2.069,
    25: 2.064, 26: 2.060, 27: 2.056, 28: 2.052, 29: 2.048,
}


def ci95(values):
    """Ritorna (mean, half_width_95CI)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = v.size
    mean = float(np.mean(v)) if n else float("nan")
    if n < 2:
        return mean, float("nan")
    sd    = float(np.std(v, ddof=1))
    se    = sd / np.sqrt(n)
    tcrit = 1.96 if n >= 30 else _T_MAP.get(n, float("nan"))
    return mean, tcrit * se


def main():
    mcu_groups = [mcu_times_16, mcu_times_32, mcu_times_64]
    pc_groups  = [pc_times_16,  pc_times_32,  pc_times_64]

    mcu_means, mcu_err = zip(*(ci95(g) for g in mcu_groups))
    pc_means,  pc_err  = zip(*(ci95(g) for g in pc_groups))

    mcu_means = np.array(mcu_means)
    pc_means  = np.array(pc_means)
    mcu_err   = np.array(mcu_err)
    pc_err    = np.array(pc_err)

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

    x = np.arange(len(COMPLEXITIES))

    ax.bar(x - BAR_WIDTH / 2, mcu_means, BAR_WIDTH, yerr=mcu_err, capsize=CAPSIZE,
           color=COLOR_MCU, alpha=BAR_ALPHA, label="MCU (on-device)",
           error_kw=dict(elinewidth=1.2, ecolor="0.3"), zorder=3)
    ax.bar(x + BAR_WIDTH / 2, pc_means,  BAR_WIDTH, yerr=pc_err,  capsize=CAPSIZE,
           color=COLOR_PC,  alpha=BAR_ALPHA, label="PC (DQN)",
           error_kw=dict(elinewidth=1.2, ecolor="0.3"), zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in COMPLEXITIES])
    ax.set_xlabel("Network complexity (hidden units)")
    ax.set_ylabel("Execution time (s)")

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which="major", axis="y", **GRID_MAJOR)
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

    # riepilogo numerico
    print(f"\n{'Complexity':>12} {'MCU mean':>10} {'MCU CI95':>10} {'PC mean':>10} {'PC CI95':>10}")
    print("-" * 55)
    for c, mm, me, pm, pe in zip(COMPLEXITIES, mcu_means, mcu_err, pc_means, pc_err):
        print(f"{c:>12} {mm:>10.3f} {me:>10.3f} {pm:>10.3f} {pe:>10.3f}")


if __name__ == "__main__":
    main()
