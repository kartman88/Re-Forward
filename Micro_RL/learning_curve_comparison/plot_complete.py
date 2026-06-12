"""
Learning-curve comparison: on-MCU training vs on-PC training.

Confronta le learning curve di 10 esperimenti (run) per ciascun gruppo,
mostrando la media e una banda di incertezza (deviazione standard di default).

Linee guida grafiche seguite (vedi PLOT_INSTRUCTION.md):
  - dimensioni/aspect-ratio fissi e DPI alto, output vettoriale (PDF/SVG)
  - limiti degli assi controllati manualmente (zoom centrato sui dati)
  - tipografia formale (Times New Roman) e testo grande
  - griglia maggiore/minore sottile e trasparente
  - palette di colori personalizzabile, una tinta per curva
  - legenda compatta fuori dalla tela, senza bordo

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
# 1. DATI: quali run caricare e come sono fatti i CSV
# ------------------------------------------------------------------ #
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Per ogni curva: etichetta in legenda, pattern dei file e nomi colonne.
#   reward_col   -> colonna con la ricompensa per episodio
#   x_col        -> colonna usata come asse X "nativo" (vedi X_AXIS sotto)
#   step_col     -> colonna con il numero di step dell'episodio (per asse "samples")
# Lascia step_col = None se non disponibile.
CURVES = {
    "PC (PPO)": {
        "pattern": "training_ppo_{i}.csv",
        "reward_col": "reward",
        "step_col": "episode_steps",
        "cumstep_col": "step",          # step cumulativi gia' presenti nel CSV
    },
    "MCU (on-device)": {
        "pattern": "training_mcu_{i}.csv",
        "reward_col": "reward",
        "step_col": "steps",
        "cumstep_col": None,            # verra' calcolato come cumsum(steps)
    },
}

RUN_IDS = list(range(1, 11))            # i 10 esperimenti: 1..10

# ------------------------------------------------------------------ #
# 2. ASSE X e AGGREGAZIONE delle run
# ------------------------------------------------------------------ #
# "episode"  -> X = numero di episodio
# "samples"  -> X = step di ambiente cumulativi (sample efficiency)
X_AXIS = "episode"

X_LABEL = {
    "episode": "Episode",
    "samples": "Environment steps",
}[X_AXIS]
Y_LABEL = "Episode reward"

# Numero di punti della griglia comune su cui interpolare tutte le run
# prima di calcolare media e banda.
N_GRID = 400

# Estremo destro dell'asse X usato per l'aggregazione.
#   None -> usa il minimo tra i massimi delle run (tutte le run coprono il range)
#   float -> taglia a quel valore
X_MAX = None

# Forza lo STESSO intervallo X per TUTTE le curve, cosi' il confronto avviene
# esattamente sullo stesso range. Le run MCU (che hanno punti in piu' rispetto
# al budget delle run PC) vengono tagliate al range comune; in pratica
# l'intervallo finale e' l'intersezione tra i gruppi (lo decide il piu' corto,
# cioe' il PC). Metti False per lasciare a ogni gruppo il suo range massimo.
MATCH_X_RANGE = True

# Tipo di banda di incertezza attorno alla media:
#   "std" -> deviazione standard, "sem" -> errore standard,
#   "pct" -> intervallo percentile (vedi PCT_LO/PCT_HI)
BAND = "std"
BAND_SCALE = 1.0                       # moltiplicatore (es. 2.0 per +-2 std)
PCT_LO, PCT_HI = 25, 75                # usati solo se BAND == "pct"

# ------------------------------------------------------------------ #
# 3. SMOOTHING (le ricompense per-episodio sono molto rumorose)
# ------------------------------------------------------------------ #
SMOOTH = True
SMOOTH_WINDOW = 21                     # finestra rolling-mean (in episodi)

# ------------------------------------------------------------------ #
# 4. COLORI  -  una tinta per curva (cambia qui la palette)
# ------------------------------------------------------------------ #
# Palette minimalista: viola e verde-acqua (vedi PLOT_INSTRUCTION.md).
# Le chiavi DEVONO coincidere con le etichette in CURVES.
PALETTE = {
    "PC (PPO)":        "#6A3D9A",      # viola
    "MCU (on-device)": "#1B9E77",      # verde acqua
}
LINE_ALPHA = 1.0                       # opacita' della linea media
BAND_ALPHA = 0.18                      # opacita' della banda di incertezza

# ------------------------------------------------------------------ #
# 5. STILE LINEE / GRIGLIA / FIGURA
# ------------------------------------------------------------------ #
LINE_WIDTH = 2.4                       # spessore curva media
FIGSIZE = (8.0, 6.0)                   # pollici (aspect-ratio bloccato)
DPI = 300

# Catena di font serif in ordine di preferenza: il primo disponibile vince.
# "Times New Roman" e' preferito; "Liberation Serif"/"Nimbus Roman" sono
# sostituti metric-compatibili (stesse proporzioni) usati se Times non c'e'.
FONT_FAMILY = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]
FONT_SIZE = 20                         # testo grande (verra' rimpicciolito nel paper)

GRID_MAJOR = dict(lw=0.6, alpha=0.45, color="0.6")
GRID_MINOR = dict(lw=0.3, alpha=0.30, color="0.75")

# Limiti manuali degli assi (zoom). None -> automatico con margine.
XLIM = None                            # es. (0, 1000)
YLIM = None                            # es. (0, 300)
Y_MARGIN = 0.05                        # margine frazionario se YLIM is None

# ------------------------------------------------------------------ #
# 6. TITOLO, LEGENDA e OUTPUT
# ------------------------------------------------------------------ #
TITLE = "On-device vs On-PC training"
SHOW_TITLE = True
TITLE_PAD = 54                         # distanza titolo-grafico (lascia spazio alla legenda)

LEGEND_NCOL = 2                        # legenda orizzontale, compatta
LEGEND_Y = 1.02                        # ancoraggio sopra la tela

OUTPUT_BASENAME = "learning_curve_comparison"
SAVE_FORMATS = ["pdf", "svg", "png"]   # vettoriale + un'anteprima raster
SHOW = False                           # plt.show() interattivo

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


def load_run(path: str, cfg: dict):
    """Restituisce (x, reward_smoothed) per una singola run, secondo X_AXIS."""
    df = _read_csv(path)
    reward = _rolling(np.asarray(df[cfg["reward_col"]], dtype=float), SMOOTH_WINDOW)

    if X_AXIS == "episode":
        x = np.arange(1, len(reward) + 1, dtype=float)
    else:  # "samples"
        if cfg["cumstep_col"] and cfg["cumstep_col"] in df:
            x = np.asarray(df[cfg["cumstep_col"]], dtype=float)
        elif cfg["step_col"] and cfg["step_col"] in df:
            x = np.cumsum(np.asarray(df[cfg["step_col"]], dtype=float))
        else:
            raise ValueError(f"Nessuna colonna di step in {path} per X_AXIS='samples'")
    return x, reward


def run_bounds(runs):
    """(x_min, x_max) coperti da TUTTE le run del gruppo (intersezione)."""
    return max(r[0][0] for r in runs), min(r[0][-1] for r in runs)


def aggregate(runs, x_min, x_max):
    """Interpola le run su una griglia comune [x_min, x_max] e calcola media + banda.

    np.interp non estrapola: i valori oltre l'ultimo x di una run vengono
    "appiattiti" all'ultimo valore. Qui pero' x_max e' scelto entro il range
    comune, quindi ogni run copre l'intero intervallo: nessuna estrapolazione,
    e i punti MCU che superano il range PC restano semplicemente tagliati."""
    grid = np.linspace(x_min, x_max, N_GRID)

    stacked = np.vstack([np.interp(grid, x, y) for x, y in runs])
    mean = stacked.mean(axis=0)

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
    return grid, mean, lo, hi


def main():
    families = FONT_FAMILY if isinstance(FONT_FAMILY, (list, tuple)) else [FONT_FAMILY]
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": list(families),
        "font.size": FONT_SIZE,
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",        # mantieni il testo editabile in SVG
    })

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # ---- 1a passata: carica tutte le run e calcola i limiti X ----
    loaded = {}                                  # label -> lista di run
    for label, cfg in CURVES.items():
        runs = []
        for i in RUN_IDS:
            path = os.path.join(DATA_DIR, cfg["pattern"].format(i=i))
            if os.path.exists(path):
                runs.append(load_run(path, cfg))
        if not runs:
            print(f"[!] Nessun file trovato per '{label}' ({cfg['pattern']})")
            continue
        loaded[label] = runs
        print(f"[+] '{label}': {len(runs)} run caricate")

    bounds = {lbl: run_bounds(runs) for lbl, runs in loaded.items()}
    if MATCH_X_RANGE:
        # Intersezione tra i gruppi: stesso range per tutti -> taglia i punti
        # MCU che superano il range PC (lo decide il gruppo piu' corto).
        shared_min = max(b[0] for b in bounds.values())
        shared_max = min(b[1] for b in bounds.values())
        if X_MAX is not None:
            shared_max = min(shared_max, X_MAX)
        print(f"[i] range X condiviso: [{shared_min:.0f}, {shared_max:.0f}]")

    # ---- 2a passata: aggrega e disegna ----
    for z, label in enumerate(loaded):
        runs = loaded[label]
        if MATCH_X_RANGE:
            x_min, x_max = shared_min, shared_max
        else:
            x_min, x_max = bounds[label]
            if X_MAX is not None:
                x_max = min(x_max, X_MAX)

        grid, mean, lo, hi = aggregate(runs, x_min, x_max)
        color = PALETTE.get(label, None)

        # banda di incertezza (sullo sfondo, z-order basso)
        ax.fill_between(grid, lo, hi, color=color, alpha=BAND_ALPHA,
                        linewidth=0, zorder=2 + 2 * z)
        # curva media (in primo piano, z-order alto)
        ax.plot(grid, mean, color=color, lw=LINE_WIDTH, alpha=LINE_ALPHA,
                label=label, zorder=3 + 2 * z, solid_capstyle="round")

    # ---- assi, griglia, limiti ----
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)

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

    # ---- legenda compatta, fuori dalla tela, senza bordo ----
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, LEGEND_Y),
              ncol=LEGEND_NCOL, frameon=False, handlelength=1.6,
              columnspacing=1.4, borderaxespad=0.0)

    fig.tight_layout()

    for fmt in SAVE_FORMATS:
        out = os.path.join(DATA_DIR, f"{OUTPUT_BASENAME}.{fmt}")
        fig.savefig(out, format=fmt, bbox_inches="tight", dpi=DPI)
        print(f"[=] salvato {out}")

    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()
