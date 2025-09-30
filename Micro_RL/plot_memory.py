"""
Memory vs Hidden Size — Re-forward vs Classic (full activations)
Setup fisso dal tuo caso:
- Topologia: d_in -> h -> d_out  (d_in=4, d_out=2)
- Buffer episodio: T=500 step
- Tipi: float32=4B, uint32_t=4B, pointer=4B
- DenseLayer con float** per le matrici (row-pointers inclusi)
- Ottimizzatore: Adam (W, dW, mW, vW; b, db, mb, vb; e 'out' per layer)
- CLASSIC = salvataggio pre+post attivazioni per ogni layer a ogni step
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --------------------
# CONFIG
# --------------------
d_in = 4
d_out = 2
T = 500
h_min, h_max, h_step = 8, 350, 1
float_bytes = 4
ptr_bytes = 4
include_128kb_line = True
sram_limit_kb = 128
save_path = "memory_vs_hidden.png"

# Se gli stati sono contigui (niente row-pointers nello state_buffer), metti True
use_state_contiguous = False

# --------------------
# FORMULE
# --------------------
def model_bytes(h: int) -> int:
    """
    Model + optimizer (Adam) + 'out' per layer + row-pointers + piccolo overhead struct.
    Per topologia d_in -> h -> d_out:
      sum_inout = d_in*h + h*d_out = h*(d_in + d_out)
      sum_out   = h + d_out
    Floats per layer set (Adam): 4*in*out + 5*out
    Bytes float modello: 4B * (4*sum_inout + 5*sum_out)
    Row-pointers: 4 matrici 2D (W, dW, mW, vW) * out ptr/layer -> 4B * 4 * sum_out = 16 * sum_out
    Struct overhead (puntatori in DenseLayer): ~72 B (due layer)
    """
    sum_inout = h * (d_in + d_out)
    sum_out = h + d_out

    model_float_bytes = float_bytes * (4 * sum_inout * 4 + 5 * sum_out)  # = 4*(4*sum_inout + 5*sum_out)
    row_ptr_bytes = 16 * sum_out  # (4 matrici) * (out ptr) * (4B)
    struct_ptr_bytes = 72
    return int(model_float_bytes + row_ptr_bytes + struct_ptr_bytes)

def buffer_bytes(T: int, d_in: int, contiguous_states: bool) -> int:
    """
    Buffer episodio condiviso (re-forward e classic):
      state_buffer: T * d_in * 4B + (T * 4B se non contiguo)
      action/reward/advantage: T * 4B ciascuno
    """
    base = T * (d_in * float_bytes + 3 * float_bytes)  # s + r + adv; a ≈ 4B
    row_ptr = 0 if contiguous_states else T * ptr_bytes
    return int(base + row_ptr)

def activations_bytes_classic_full(h: int, T: int) -> int:
    """
    Classic 'full': salvi pre- e post-attivazioni per ciascun layer (hidden + output) ad ogni step.
      Per L=2: tot neuroni per step = (h + d_out)
      Floats per step = 2 * (h + d_out)  ->  bytes = 4B * 2 * (h + d_out)
      Su T step: 4B * 2 * T * (h + d_out) = 8 * T * (h + d_out)
    """
    return int(8 * T * (h + d_out))

def workspace_bytes(h: int) -> int:
    """Workspace piccolo; includiamo 4B * max(h, d_out) per sicurezza."""
    return int(float_bytes * max(h, d_out))

# --------------------
# CALCOLO E PLOT
# --------------------
hs = np.arange(h_min, h_max + 1, h_step)
buf = buffer_bytes(T, d_in, contiguous_states=use_state_contiguous)

ref_bytes = np.array([model_bytes(h) + buf + workspace_bytes(h) for h in hs])
cls_bytes = np.array([model_bytes(h) + buf + activations_bytes_classic_full(h, T) for h in hs])

ref_kb = ref_bytes / 1024.0
cls_kb = cls_bytes / 1024.0

plt.figure()
plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)
plt.yscale("log")
# Maggior controllo sui tick logaritmici
# Tick principali (etichette leggibili)
plt.gca().yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=[1.0, 2.0, 5.0], numticks=12))
plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter())  # numeri normali, non notazione scientifica

# Tick minori (senza etichetta, solo griglia)
plt.gca().yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0)*0.1, numticks=10))
plt.gca().yaxis.set_minor_formatter(ticker.NullFormatter())
plt.plot(hs, ref_kb, label="Re-forward", color="darkgreen")
plt.plot(hs, cls_kb, label="Classic", color="red")

if include_128kb_line:
    plt.axhline(sram_limit_kb, linestyle=":", label=f"{sram_limit_kb} KB SRAM", color="blue")

plt.xlabel("Hidden layer size")
plt.ylabel("Memory (KB)")
plt.title(f"Memory vs Hidden Size — Re-forward vs Classic")
plt.legend()
plt.tight_layout()
plt.savefig(save_path, dpi=200)
print(f"Saved plot to {save_path}")
