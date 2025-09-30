#!/usr/bin/env python3
# Bar chart raggruppato: Network Complexity (16/32/64) × {MCU, PC}
# y = Execution Time (s), con IC 95% come error bar.

import numpy as np
import matplotlib
# Sblocca la riga seguente se sei su server/headless:
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd  # solo per stampa tabella riassuntiva (facoltativo)

# ====== DATI (secondi) ======
mcu_times_16 = [60.38, 59.89, 59.64, 59.78, 60.00, 60.32, 60.30, 59.85]
mcu_times_32 = [71.09, 70.62, 71.44, 70.80, 70.53, 70.29, 70.10, 70.22]
mcu_times_64 = [76.85, 75.78, 76.35, 76.11, 77.91, 76.00, 78.00, 77.67]

pc_times_16  = [13.82, 12.96, 12.96, 13.08, 13.05, 13.05, 13.29, 13.56]
pc_times_32  = [19.09, 18.27, 18.69, 18.73, 18.84, 18.90, 18.75, 18.89]
pc_times_64  = [22.38, 21.70, 22.01, 21.89, 22.27, 22.45, 22.93, 22.67]
# ============================

complexities = [16, 32, 64]
MCU_COLOR = "darkgreen"
PC_COLOR  = "red"
TITLE = "Execution Time by Network Complexity (Mean ± 95% CI)"
OUTPUT_IMG = "exec_time_mcu_vs_pc.png"

def ci95(values):
    """Ritorna (mean, half_width_95CI). Usa t-critico; half_width=NaN se n<2."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = v.size
    mean = np.nanmean(v) if n else np.nan
    if n < 2:
        return mean, np.nan
    sd = np.nanstd(v, ddof=1)
    se = sd / np.sqrt(n)
    t_map = {
        1: np.nan, 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
        8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160,
        15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093,
        21: 2.086, 22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064, 26: 2.060,
        27: 2.056, 28: 2.052, 29: 2.048
    }
    tcrit = 1.96 if n >= 30 else t_map.get(n, np.nan)
    return mean, tcrit * se

# Aggrega gruppi (MCU e PC) nell’ordine 16/32/64
mcu_groups = [mcu_times_16, mcu_times_32, mcu_times_64]
pc_groups  = [pc_times_16,  pc_times_32,  pc_times_64]

# Calcola media e CI per ogni complessità
mcu_means, mcu_err = zip(*(ci95(g) for g in mcu_groups))
pc_means,  pc_err  = zip(*(ci95(g) for g in pc_groups))

mcu_means = np.array(mcu_means, dtype=float)
pc_means  = np.array(pc_means,  dtype=float)
mcu_err   = np.array(mcu_err,   dtype=float)
pc_err    = np.array(pc_err,    dtype=float)

# Plot
plt.rcParams.update({"font.size": 12})
x = np.arange(len(complexities))
width = 0.38

plt.figure(figsize=(8.8, 5.2))
plt.title(TITLE, fontsize=14)

bars1 = plt.bar(x - width/2, mcu_means, width, yerr=mcu_err, capsize=4,
                label="MCU", color=MCU_COLOR, alpha=0.9)
bars2 = plt.bar(x + width/2, pc_means,  width, yerr=pc_err,  capsize=4,
                label="PC",  color=PC_COLOR,  alpha=0.9)

plt.xticks(x, [str(c) for c in complexities], fontsize=14)
plt.xlabel("Network Complexity", fontsize=14)
plt.ylabel("Execution Time (s)", fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=150)
print(f"[OK] Salvato: {OUTPUT_IMG}")
plt.show()

# (facoltativo) stampa tabella riassuntiva
df = pd.DataFrame({
    "Complexity": complexities,
    "MCU_mean_s": mcu_means, "MCU_CI95_halfwidth": mcu_err,
    "PC_mean_s":  pc_means,  "PC_CI95_halfwidth":  pc_err
})
print(df.to_string(index=False, float_format=lambda v: f"{v:0.3f}"))
