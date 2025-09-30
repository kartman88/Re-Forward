#!/usr/bin/env python3
"""
Completa i CSV learning_curve{n}.csv (episode,reward) fino a MAX_EPISODE,
riempiendo qualsiasi episodio mancante con FILL_REWARD e sovrascrive i file.
"""

from pathlib import Path
import pandas as pd

# ====== CONFIG ======
FOLDER = Path("./")          # cartella dove sono i CSV (metti il path della tua copia)
N_FILES = 10                 # learning_curve1..learning_curve10
START_EPISODE = 1
MAX_EPISODE = 150
FILL_REWARD = 500.0
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
        raise ValueError("Colonne richieste non trovate. Servono 'episode' e 'reward'.")
    return df[['episode','reward']]

def fill_to_max(df: pd.DataFrame, start_ep: int, max_ep: int, fill_reward: float) -> pd.DataFrame:
    df = df.copy()
    df['episode'] = pd.to_numeric(df['episode'], errors='coerce').astype('Int64')
    df['reward']  = pd.to_numeric(df['reward'], errors='coerce')
    df = df.dropna(subset=['episode']).drop_duplicates(subset=['episode'])
    df = df[(df['episode'] >= start_ep) & (df['episode'] <= max_ep)]
    full_index = pd.Index(range(start_ep, max_ep + 1), name='episode')
    df = df.set_index('episode').sort_index()
    df = df.reindex(full_index)
    df['reward'] = df['reward'].fillna(fill_reward)
    return df.reset_index()

def process_one(path: Path) -> None:
    df = pd.read_csv(path)
    df = normalize_columns(df)
    out = fill_to_max(df, START_EPISODE, MAX_EPISODE, FILL_REWARD)
    out.to_csv(path, index=False)

def main():
    files = [FOLDER / f"learning_curve{i}.csv" for i in range(1, N_FILES + 1)]
    existing = [p for p in files if p.exists()]
    if not existing:
        raise SystemExit(f"Nessun file trovato in {FOLDER.resolve()} (attesi learning_curve1..{N_FILES}.csv)")
    print(f"Trovati {len(existing)} file. Completo episodi {START_EPISODE}..{MAX_EPISODE} con reward={FILL_REWARD}.")
    for p in existing:
        try:
            process_one(p)
            print(f"[OK] {p.name} aggiornato.")
        except Exception as e:
            print(f"[ERRORE] {p.name}: {e}")

if __name__ == "__main__":
    main()
