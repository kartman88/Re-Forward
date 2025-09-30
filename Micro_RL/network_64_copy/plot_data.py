#!/usr/bin/env python3
"""
Plot media e intervallo di confidenza (95%) dei reward per episodio
a partire da file learning_curve{1..N}.csv con colonne (episode, reward).
Assume che i file siano già completi e allineati per episodio (es. 1..150).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ====== CONFIG ======
FOLDER = Path("./")   # cartella con i CSV
N_FILES = 10          # learning_curve1..learning_curve10
START_EPISODE = 1
MAX_EPISODE = 150
OUTPUT_IMG = "learning_curve_summary.png"
TITLE = "Learning Curves: Mean ± 95% CI"
PLOT_INDIVIDUALS = False  # True per plottare le singole curve in trasparenza
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
    # cast sicuro
    df['episode'] = pd.to_numeric(df['episode'], errors='coerce').astype('Int64')
    df['reward']  = pd.to_numeric(df['reward'], errors='coerce')
    df = df.dropna(subset=['episode'])
    # mantieni nel range atteso e indicizza per episodio
    df = df[(df['episode'] >= START_EPISODE) & (df['episode'] <= MAX_EPISODE)]
    return df.set_index('episode').sort_index()

def main():
    files = [FOLDER / f"learning_curve{i}.csv" for i in range(1, N_FILES + 1)]
    runs = []
    for p in files:
        if not p.exists():
            print(f"[AVVISO] File mancante: {p.name} (verrà ignorato)")
            continue
        runs.append(load_run(p))

    if len(runs) == 0:
        raise SystemExit("Nessun file valido trovato.")

    # costruisci DataFrame con colonne una per run (allineate sugli episodi)
    # index = episodi completi START..MAX
    idx = pd.Index(range(START_EPISODE, MAX_EPISODE + 1), name='episode')
    runs_aligned = []
    for i, df in enumerate(runs, start=1):
        s = df.reindex(idx)['reward']
        runs_aligned.append(s.rename(f"run_{i}"))
    M = pd.concat(runs_aligned, axis=1)  # shape: [episodes, n_runs]

    # statistica per episodio (ignora eventuali NaN)
    mean = M.mean(axis=1)
    std = M.std(axis=1, ddof=1)  # sample std
    n = M.count(axis=1)          # numero di run disponibili per quell'episodio

    # t-critico per 95% CI (two-sided). Per n>=2 usiamo t_{0.975, n-1}.
    # Per evitare dipendenze da scipy, usiamo:
    # - t=1.96 per n>=30 (approssimazione normale)
    # - mapping per n in [2..29], e 0 per n<2 (n=1 -> niente CI)
    def t_critical(ns: np.ndarray) -> np.ndarray:
        # tabella essenziale (0.975 quantile) df:2..29
        t_map = {
            1: np.nan,  # non definito (n=1 => no CI)
            2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
            8: 2.365,  9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179,
            14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101,
            20: 2.093, 21: 2.086, 22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064,
            26: 2.060, 27: 2.056, 28: 2.052, 29: 2.048
        }
        out = np.empty_like(ns, dtype=float)
        for i, k in enumerate(ns):
            if k >= 30:
                out[i] = 1.96
            elif k >= 2:
                out[i] = t_map[int(k)]
            else:
                out[i] = np.nan
        return out

    tcrit = t_critical(n.to_numpy())
    se = std / np.sqrt(n)  # standard error
    half_width = tcrit * se

    lower = mean - half_width
    upper = mean + half_width

    # Plot
    plt.figure(figsize=(9, 5))
    x = mean.index.to_numpy()

    if PLOT_INDIVIDUALS:
        # plottiamo le singole run in trasparenza
        for col in M.columns:
            plt.plot(x, M[col].to_numpy(), linewidth=1, alpha=0.2)

    plt.plot(x, mean.to_numpy(), linewidth=2, color="darkgreen", label="Mean reward")
    # banda IC (ignora punti senza CI)
    mask = ~np.isnan(lower.to_numpy()) & ~np.isnan(upper.to_numpy())
    plt.fill_between(
        x[mask],
        lower.to_numpy()[mask],
        upper.to_numpy()[mask],
        color="green", alpha=0.2,
        label="95% CI"
    )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
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
