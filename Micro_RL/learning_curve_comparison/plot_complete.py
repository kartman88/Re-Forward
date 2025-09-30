#!/usr/bin/env python3
# Plot di due gruppi di curve: MCU (verde) e PC (rosso) con media ± 95% CI.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
# Se sei su server/headless, sblocca la riga sotto:
# matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ====== CONFIG ======
FOLDER = Path("./")          # cartella con i CSV
N_FILES = 10                 # numero di run attese per ciascun gruppo
START_EPISODE = 1
MAX_EPISODE = 150
OUTPUT_IMG = "learning_curve_mcu_vs_pc.png"
TITLE = "Learning Curves (MCU vs PC): Mean ± 95% CI"
PLOT_INDIVIDUALS = False     # True per vedere tutte le run in trasparenza
# Colori & etichette
MCU_LABEL = "MCU Mean reward"
MCU_COLOR = "darkgreen"
PC_LABEL  = "PC Mean reward"
PC_COLOR  = "red"
# ====================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_lower = {c: c.lower() for c in df.columns}
    df = df.rename(columns=cols_lower)
    if 'episode' not in df.columns:
        for alt in ('episodes','ep','iter','iteration'):
            if alt in df.columns:
                df = df.rename(columns={alt: 'episode'})
                break
    if 'reward' not in df.columns:
        for alt in ('return','returns','rew','score'):
            if alt in df.columns:
                df = df.rename(columns={alt: 'reward'})
                break
    if not {'episode','reward'}.issubset(df.columns):
        raise ValueError("Servono colonne 'episode' e 'reward'.")
    return df[['episode','reward']]

def load_run(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_columns(df)
    df['episode'] = pd.to_numeric(df['episode'], errors='coerce').astype('Int64')
    df['reward']  = pd.to_numeric(df['reward'],  errors='coerce')
    df = df.dropna(subset=['episode'])
    df = df[(df['episode'] >= START_EPISODE) & (df['episode'] <= MAX_EPISODE)]
    return df.set_index('episode').sort_index()

def load_group(prefix: str) -> pd.DataFrame:
    """Carica un gruppo di run con pattern <prefix>{1..N_FILES}.csv e restituisce
    una matrice [episodes, n_runs] con colonne run_1..run_k."""
    idx = pd.Index(range(START_EPISODE, MAX_EPISODE + 1), name='episode')
    cols = []
    for i in range(1, N_FILES + 1):
        p = FOLDER / f"{prefix}{i}.csv"
        if not p.exists():
            print(f"[AVVISO] File mancante: {p.name} (gruppo {prefix})")
            continue
        s = load_run(p).reindex(idx)['reward'].rename(f"run_{i}")
        cols.append(s)
    if not cols:
        raise SystemExit(f"Nessun file valido per prefisso '{prefix}'")
    return pd.concat(cols, axis=1)

def t_critical_array(n_arr: np.ndarray) -> np.ndarray:
    """Restituisce t_{0.975, n-1} per ogni n (IC 95%). Usa 1.96 per n>=30."""
    t_map = {
        1: np.nan, 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
        8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160,
        15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093,
        21: 2.086, 22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064, 26: 2.060,
        27: 2.056, 28: 2.052, 29: 2.048
    }
    out = np.empty_like(n_arr, dtype=float)
    for i, k in enumerate(n_arr):
        if k >= 30:
            out[i] = 1.96
        elif k >= 2:
            out[i] = t_map[int(k)]
        else:
            out[i] = np.nan
    return out

def summarize(M: pd.DataFrame):
    """Dato M [episodes, n_runs], calcola mean, lower, upper dell'IC 95%."""
    mean = M.mean(axis=1)
    std  = M.std(axis=1, ddof=1)
    n    = M.count(axis=1)
    tcrit = t_critical_array(n.to_numpy())
    se = std / np.sqrt(n)
    hw = tcrit * se
    lower = mean - hw
    upper = mean + hw
    return mean, lower, upper, M

def main():
    # Carica i due gruppi
    M_mcu = load_group("learning_curve")
    M_pc  = load_group("pc_learning_curve")

    mean_mcu, low_mcu, up_mcu, M1 = summarize(M_mcu)
    mean_pc,  low_pc,  up_pc,  M2 = summarize(M_pc)

    x = mean_mcu.index.to_numpy()

    plt.figure(figsize=(10, 5.5))

    if PLOT_INDIVIDUALS:
        for col in M1.columns:
            plt.plot(x, M1[col].to_numpy(), alpha=0.15, linewidth=1, color=MCU_COLOR)
        for col in M2.columns:
            plt.plot(x, M2[col].to_numpy(), alpha=0.15, linewidth=1, color=PC_COLOR)

    # MCU (verde)
    plt.plot(x, mean_mcu.to_numpy(), color=MCU_COLOR, linewidth=2, label=MCU_LABEL)
    mask1 = ~np.isnan(low_mcu.to_numpy()) & ~np.isnan(up_mcu.to_numpy())
    plt.fill_between(x[mask1], low_mcu.to_numpy()[mask1], up_mcu.to_numpy()[mask1],
                     alpha=0.18, label="MCU 95% CI", color=MCU_COLOR)

    # PC (rosso)
    plt.plot(x, mean_pc.to_numpy(), color=PC_COLOR, linewidth=2, label=PC_LABEL)
    mask2 = ~np.isnan(low_pc.to_numpy()) & ~np.isnan(up_pc.to_numpy())
    #plt.fill_between(x[mask2], low_pc.to_numpy()[mask2], up_pc.to_numpy()[mask2],alpha=0.18, label="PC 95% CI", color=PC_COLOR)

    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.title(TITLE)
    plt.xlim(START_EPISODE, MAX_EPISODE)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=150)
    print(f"[OK] Salvato: {OUTPUT_IMG}")
    plt.show()

if __name__ == "__main__":
    main()
