#!/usr/bin/env python3
"""
Memory consumption calculator for tinyRL DQN on STM32H7.

Analizza il consumo di RAM heap delle strutture allocate dinamicamente:
  - QNetwork (rete online): pesi W + gradienti dW + momenti Adam mW, vW
    + bias b, db, mb, vb + output buffer
  - TargetNetwork: solo pesi W + bias b + output buffer
  - ReplayBuffer: state/next_state pool, action, reward, done + array
    di puntatori

Assume:
  - float = 4 byte
  - uint32_t = 4 byte
  - uint8_t = 1 byte
  - puntatore = 4 byte (STM32 Cortex-M, 32-bit)
"""

SIZE_FLOAT   = 4
SIZE_UINT32  = 4
SIZE_UINT8   = 1
SIZE_PTR     = 4  # ARM Cortex-M 32-bit


def ask_int(prompt, default=None):
    while True:
        s = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
        if not s and default is not None:
            return default
        try:
            return int(s)
        except ValueError:
            print("  Inserisci un intero valido.")


def ask_topology():
    print("\n--- Struttura della rete ---")
    print("Inserisci la topologia come lista di dimensioni dei layer,")
    print("separate da spazio o virgola.")
    print("Esempio: '3 64 64 5' -> input=3, due hidden da 64, output=5")
    while True:
        s = input("Topologia: ").strip()
        if not s:
            print("  Inserire almeno 2 valori.")
            continue
        parts = s.replace(",", " ").split()
        try:
            topo = [int(p) for p in parts]
        except ValueError:
            print("  Valori non validi.")
            continue
        if len(topo) < 2:
            print("  Servono almeno input e output.")
            continue
        if any(d <= 0 for d in topo):
            print("  Dimensioni devono essere > 0.")
            continue
        return topo


def dense_layer_bytes(in_dim, out_dim):
    """Memoria di un DenseLayer dell'online net (W + dW + mW + vW +
    b + db + mb + vb + out + array puntatori riga)."""
    # 4 matrici out x in (W, dW, mW, vW): dati + puntatori di riga
    matrices_data = 4 * (out_dim * in_dim * SIZE_FLOAT)
    matrices_row_ptrs = 4 * (out_dim * SIZE_PTR)
    # 5 vettori out (b, db, mb, vb, out)
    vectors = 5 * (out_dim * SIZE_FLOAT)
    return matrices_data + matrices_row_ptrs + vectors


def target_layer_bytes(in_dim, out_dim):
    """Memoria di un TargetLayer (W + b + out + puntatori riga)."""
    W_data = out_dim * in_dim * SIZE_FLOAT
    W_row_ptrs = out_dim * SIZE_PTR
    b = out_dim * SIZE_FLOAT
    out = out_dim * SIZE_FLOAT
    return W_data + W_row_ptrs + b + out


def qnetwork_breakdown(topology):
    """Ritorna lista (label, bytes) per ogni layer + overhead array layers."""
    items = []
    n_weights = len(topology) - 1
    # array di DenseLayer struct: stima ~ sizeof(DenseLayer)
    # DenseLayer: 2 int + 9 ptr + 1 enum ~= 2*4 + 9*4 + 4 = 48 byte
    sizeof_dense = 2 * 4 + 9 * SIZE_PTR + 4
    items.append((f"  QNetwork.layers array ({n_weights} structs)",
                  n_weights * sizeof_dense))
    for i in range(n_weights):
        in_d, out_d = topology[i], topology[i + 1]
        b = dense_layer_bytes(in_d, out_d)
        items.append((f"  Layer online {i} ({in_d}->{out_d}): "
                      f"W+dW+mW+vW + b/db/mb/vb + out", b))
    return items


def target_breakdown(topology):
    items = []
    n_weights = len(topology) - 1
    # TargetLayer: 2 int + 3 ptr + 1 enum ~= 2*4 + 3*4 + 4 = 24
    sizeof_tgt = 2 * 4 + 3 * SIZE_PTR + 4
    items.append((f"  TargetNetwork.layers array ({n_weights} structs)",
                  n_weights * sizeof_tgt))
    for i in range(n_weights):
        in_d, out_d = topology[i], topology[i + 1]
        b = target_layer_bytes(in_d, out_d)
        items.append((f"  Layer target {i} ({in_d}->{out_d}): W + b + out",
                      b))
    return items


def replay_breakdown(capacity, obs_dim):
    items = []
    items.append(("  state_pool (capacity * obs_dim * float)",
                  capacity * obs_dim * SIZE_FLOAT))
    items.append(("  snext_pool (capacity * obs_dim * float)",
                  capacity * obs_dim * SIZE_FLOAT))
    items.append(("  state[] (capacity puntatori)",
                  capacity * SIZE_PTR))
    items.append(("  next_state[] (capacity puntatori)",
                  capacity * SIZE_PTR))
    items.append(("  action[] (capacity * uint32_t)",
                  capacity * SIZE_UINT32))
    items.append(("  reward[] (capacity * float)",
                  capacity * SIZE_FLOAT))
    items.append(("  done[] (capacity * uint8_t)",
                  capacity * SIZE_UINT8))
    return items


def activations_breakdown(topology, batch_size):
    """Memoria delle attivazioni del forward pass che un approccio classico
    (es. PyTorch / autograd su PC, dove la RAM non e' un vincolo)
    conserverebbe per fare la backpropagation.

    Il MIO approccio NON le conserva: ricalcola le attivazioni durante il
    backward, quindi paga 0 byte qui. Un approccio classico invece tiene in
    memoria, per OGNI sample del batch, l'intero grafo di attivazioni:
      - l'input del batch (gli stati, obs_dim) -> serve per dW del 1o layer
      - per ogni layer la pre-attivazione z = W.x + b -> serve per la
        derivata della funzione di attivazione (ReLU'/tanh') nel backward
      - per ogni layer l'output post-attivazione a = f(z) -> serve come
        input del layer successivo:  dW[l] = delta[l] (outer) a[l-1]
    Tutto materializzato sull'intero batch (batch_size * dim).
    """
    items = []
    n_weights = len(topology) - 1
    obs_dim = topology[0]
    # input del batch conservato per la backprop del primo layer
    items.append((f"  Input del batch (stati, {obs_dim}): batch*{obs_dim}*float",
                  batch_size * obs_dim * SIZE_FLOAT))
    for i in range(n_weights):
        in_d, out_d = topology[i], topology[i + 1]
        # pre-attivazione z del layer i, per tutti i sample del batch
        z = batch_size * out_d * SIZE_FLOAT
        items.append((f"  Pre-attivazione z layer {i} ({in_d}->{out_d}): "
                      f"batch*{out_d}*float", z))
        # output post-attivazione a del layer i, per tutti i sample del batch
        a = batch_size * out_d * SIZE_FLOAT
        items.append((f"  Output a layer {i} ({in_d}->{out_d}): "
                      f"batch*{out_d}*float", a))
    return items


def sum_items(items):
    return sum(b for _, b in items)


def compute_totals(topology, capacity, obs_dim, batch_size):
    """Calcola (silenziosamente) i totali per le due modalita' di training.

    Ritorna un dict con:
      - mine    : totale del mio approccio (NO attivazioni conservate)
      - classic : totale approccio classico (mine + attivazioni del batch)
      - activations : solo le attivazioni del batch (= risparmio)
      - sub_q, sub_tgt, sub_rb, sub_st : sottototali per eventuale uso
    """
    sub_q   = sum_items(qnetwork_breakdown(topology))
    sub_tgt = sum_items(target_breakdown(topology))
    sub_rb  = sum_items(replay_breakdown(capacity, obs_dim))
    n_actions = topology[-1]
    sub_st  = 3 * (n_actions * SIZE_FLOAT)  # q_online, q_tgt, q
    mine = sub_q + sub_tgt + sub_rb + sub_st
    activations = sum_items(activations_breakdown(topology, batch_size))
    return {
        "mine": mine,
        "classic": mine + activations,
        "activations": activations,
        "sub_q": sub_q, "sub_tgt": sub_tgt,
        "sub_rb": sub_rb, "sub_st": sub_st,
    }


def print_table(title, items):
    print(f"\n{title}")
    print("-" * 78)
    print(f"{'Componente':<60}{'Bytes':>10}{'KB':>8}")
    print("-" * 78)
    subtotal = 0
    for label, b in items:
        subtotal += b
        print(f"{label:<60}{b:>10}{b/1024:>8.2f}")
    print("-" * 78)
    print(f"{'  SUBTOTALE':<60}{subtotal:>10}{subtotal/1024:>8.2f}")
    return subtotal


def run_report():
    """Modalita' 1: chiede tutti i parametri e stampa il resoconto completo."""
    print("=" * 78)
    print(" tinyRL DQN — Calcolatore consumo di memoria (heap)")
    print("=" * 78)

    topology = ask_topology()
    print(f"\nTopologia: {topology} -> {len(topology) - 1} layer pesati")

    obs_dim = topology[0]
    print(f"obs_dim dedotto dall'input layer: {obs_dim}")
    override = input("Vuoi forzare un obs_dim diverso per il replay buffer? "
                     "(invio per no): ").strip()
    if override:
        try:
            obs_dim = int(override)
        except ValueError:
            print("  Valore non valido, uso input layer.")

    capacity = ask_int("Capacità del replay buffer (REPLAY_SIZE)", 1000)
    batch_size = ask_int("Batch size (per stima stack)", 32)

    # --- Calcoli ---
    q_items   = qnetwork_breakdown(topology)
    tgt_items = target_breakdown(topology)
    rb_items  = replay_breakdown(capacity, obs_dim)

    # Stack/locali principali (stima): q_online, q_tgt di N_ACTIONS float
    n_actions = topology[-1]
    stack_items = [
        (f"  q_online[{n_actions}] (dqn_train locale)",
         n_actions * SIZE_FLOAT),
        (f"  q_tgt[{n_actions}] (dqn_train locale)",
         n_actions * SIZE_FLOAT),
        (f"  q[{n_actions}] (dqn_select_action locale)",
         n_actions * SIZE_FLOAT),
    ]

    # --- Output tabelle ---
    sub_q   = print_table("[1] QNetwork ONLINE (pesi + Adam moments + grads)",
                          q_items)
    sub_tgt = print_table("[2] TargetNetwork (solo pesi + bias + out)",
                          tgt_items)
    sub_rb  = print_table(f"[3] ReplayBuffer (capacity={capacity}, "
                          f"obs_dim={obs_dim})", rb_items)
    sub_st  = print_table("[4] Stack/locali principali (stima)", stack_items)

    total = sub_q + sub_tgt + sub_rb + sub_st

    # --- Approccio classico: attivazioni del batch conservate per backprop ---
    act_items = activations_breakdown(topology, batch_size)
    sub_act = print_table(
        f"[5] Attivazioni del batch: input + z + output per layer "
        f"(SOLO approccio classico, batch_size={batch_size})", act_items)
    total_classic = total + sub_act

    print("\n" + "=" * 78)
    print(" RIEPILOGO")
    print("=" * 78)
    print(f"{'Sezione':<60}{'Bytes':>10}{'KB':>8}")
    print("-" * 78)
    print(f"{'QNetwork (online)':<60}{sub_q:>10}{sub_q/1024:>8.2f}")
    print(f"{'TargetNetwork':<60}{sub_tgt:>10}{sub_tgt/1024:>8.2f}")
    print(f"{'ReplayBuffer':<60}{sub_rb:>10}{sub_rb/1024:>8.2f}")
    print(f"{'Stack/locali':<60}{sub_st:>10}{sub_st/1024:>8.2f}")
    print("-" * 78)
    print(f"{'TOTALE (mio approccio)':<60}{total:>10}{total/1024:>8.2f}")
    print("=" * 78)
    print(f"\n>>> Consumo totale stimato: {total/1024:.2f} KB "
          f"({total} byte) <<<\n")

    # --- Confronto: mio approccio vs approccio classico (con attivazioni) ---
    print("=" * 78)
    print(" CONFRONTO: mio approccio (ricalcolo attivazioni) vs classico")
    print("=" * 78)
    print(f"{'Approccio':<60}{'Bytes':>10}{'KB':>8}")
    print("-" * 78)
    print(f"{'Mio approccio (NO attivazioni conservate)':<60}"
          f"{total:>10}{total/1024:>8.2f}")
    print(f"{'  + attivazioni del batch (classico)':<60}"
          f"{sub_act:>10}{sub_act/1024:>8.2f}")
    print(f"{'Approccio classico (PyTorch-like)':<60}"
          f"{total_classic:>10}{total_classic/1024:>8.2f}")
    print("-" * 78)
    overhead = sub_act
    pct = (overhead / total * 100) if total else 0.0
    print(f"{'Memoria risparmiata dal mio approccio':<60}"
          f"{overhead:>10}{overhead/1024:>8.2f}")
    print(f"  ovvero il classico consuma il {pct:.1f}% in piu' "
          f"(x{total_classic/total:.2f}) rispetto al mio approccio")
    print("=" * 78)
    print()

    # Conteggio parametri allenabili (info aggiuntiva)
    n_params = sum(topology[i] * topology[i+1] + topology[i+1]
                   for i in range(len(topology) - 1))
    print(f"Parametri allenabili (W + b): {n_params}")
    print(f"  - peso per copia (float):        "
          f"{n_params * SIZE_FLOAT} byte = "
          f"{n_params * SIZE_FLOAT / 1024:.2f} KB")
    print(f"  - online net mantiene 4 copie    "
          f"(W, dW, mW, vW) per ogni parametro")
    print()


MCU_LIMIT_BYTES = 1024 * 1024  # 1 MB (limite RAM del microcontrollore)

# ===================================================================== #
#   PANNELLO GRAFICO (stile scientifico, vedi PLOT_INSTRUCTION.md)
# ===================================================================== #
PLOT_FIGSIZE    = (8.0, 6.0)
PLOT_DPI        = 300
PLOT_FONT       = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]
PLOT_FONT_SIZE  = 20
PLOT_FORMATS    = ["pdf", "svg", "png"]

COLOR_MINE      = "#1B9E77"     # verde acqua - approccio on-device
COLOR_CLASSIC   = "#6A3D9A"     # viola       - backprop classica
COLOR_LIMIT     = "#9A9A9A"     # grigio neutro per la linea limite MCU

PLOT_LINE_WIDTH = 2.4
PLOT_BAND_ALPHA = 0.12          # riempimento tra le due curve
GRID_MAJOR      = dict(lw=0.6, alpha=0.45, color="0.6")
GRID_MINOR      = dict(lw=0.3, alpha=0.30, color="0.75")
SHOW_TITLE      = True
TITLE_PAD       = 54
# ===================================================================== #


def run_plot():
    """Modalita' 2: fissa i parametri dell'esperimento, varia la dimensione
    dell'hidden layer (rete con 1 solo hidden) e disegna il grafico del
    consumo di memoria dei due approcci, con la linea del limite a 1 MB.

    Perche' 1 solo hidden + larghezza variabile: con un unico hidden non
    esiste nessun layer hidden->hidden (l'unico costo O(W^2)), quindi sia i
    pesi sia le attivazioni scalano linearmente in W. Le attivazioni pero'
    sono moltiplicate per batch_size, percio' il loro peso (= il risparmio
    del mio approccio) cresce in proporzione e la differenza tra i due
    approcci risulta massima e ben visibile.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.ticker import AutoMinorLocator
    except ImportError:
        print("\n[!] matplotlib/numpy non installati. Installa con:")
        print("    pip install matplotlib numpy\n")
        return

    print("=" * 78)
    print(" tinyRL DQN — Grafico consumo di memoria (mio vs classico)")
    print("=" * 78)
    print("\nRete con 1 hidden layer; vario la larghezza dell'hidden (asse X).")

    obs_dim    = ask_int("obs_dim (input layer)", 3)
    n_actions  = ask_int("n_actions (output layer)", 5)
    capacity   = ask_int("Capacita' del replay buffer (REPLAY_SIZE)", 1000)
    batch_size = ask_int("Batch size", 64)
    h_min      = ask_int("Larghezza hidden minima", 8)
    h_max      = ask_int("Larghezza hidden massima", 6000)
    n_points   = ask_int("Numero di punti sull'asse X", 200)

    widths = np.unique(np.linspace(h_min, h_max, max(2, n_points)).astype(int))

    mine_kb, classic_kb = [], []
    for w in widths:
        t = compute_totals([obs_dim, int(w), n_actions], capacity, obs_dim,
                           batch_size)
        mine_kb.append(t["mine"] / 1024.0)
        classic_kb.append(t["classic"] / 1024.0)

    mine_kb    = np.array(mine_kb)
    classic_kb = np.array(classic_kb)
    limit_kb   = MCU_LIMIT_BYTES / 1024.0

    # --- stampa di riepilogo testuale ---
    print(f"\n{'hidden':>8}{'mio (KB)':>14}{'classico (KB)':>16}{'extra %':>10}")
    print("-" * 48)
    for w, m, c in zip(widths, mine_kb, classic_kb):
        pct = (c - m) / m * 100 if m else 0.0
        print(f"{w:>8}{m:>14.1f}{c:>16.1f}{pct:>9.1f}%")

    # --- impostazioni grafiche ---
    families = PLOT_FONT if isinstance(PLOT_FONT, (list, tuple)) else [PLOT_FONT]
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": list(families),
        "font.size": PLOT_FONT_SIZE,
        "xtick.labelsize": PLOT_FONT_SIZE * 0.6,
        "ytick.labelsize": PLOT_FONT_SIZE * 0.6,
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
    })

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)

    # curve principali
    ax.plot(widths, classic_kb, color=COLOR_CLASSIC, lw=PLOT_LINE_WIDTH,
            solid_capstyle="round", zorder=5,
            label=f"Classic backprop (batch={batch_size})")
    ax.plot(widths, mine_kb, color=COLOR_MINE, lw=PLOT_LINE_WIDTH,
            solid_capstyle="round", zorder=5,
            label="Re-forward (on-device, 1 sample)")

    # riempimento tra le due curve (risparmio di memoria)
    ax.fill_between(widths, mine_kb, classic_kb,
                    color=COLOR_CLASSIC, alpha=PLOT_BAND_ALPHA,
                    linewidth=0, zorder=2)

    # linea limite MCU
    ax.axhline(limit_kb, linestyle="--", color=COLOR_LIMIT,
               linewidth=1.2, zorder=3)
    ax.text(h_max, limit_kb, f"  MCU limit ({limit_kb/1024:.0f} MB)",
            va="center", ha="left",
            fontsize=PLOT_FONT_SIZE * 0.55, color=COLOR_LIMIT)

    # --- incroci curva-limite con marker e tick colorati sull'asse X ---
    crossings = []
    for ys, curve_color in [(mine_kb, COLOR_MINE), (classic_kb, COLOR_CLASSIC)]:
        if ys[0] <= limit_kb <= ys[-1]:
            x_cross = float(np.interp(limit_kb, ys, widths))
            ax.plot([x_cross], [limit_kb], "o", color=curve_color,
                    markersize=8, zorder=7)
            ax.vlines(x_cross, 0, limit_kb, color=curve_color,
                      linestyle=":", linewidth=1.3, alpha=0.8, zorder=4)
            crossings.append((x_cross, curve_color))
            print(f"  >>> supera il limite a hidden width ~ {x_cross:.0f}")

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
        ax.set_title(
            f"DQN memory: on-device vs classic  "
            f"(obs={obs_dim}, act={n_actions}, replay={capacity})",
            fontsize=PLOT_FONT_SIZE * 0.6, pad=TITLE_PAD)

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=1, frameon=False, handlelength=1.6,
              borderaxespad=0.0, fontsize=PLOT_FONT_SIZE * 0.7)

    # tick X colorati sugli incroci
    cross_ticks = {int(round(xc)): c for xc, c in crossings}
    thr = (h_max - h_min) * 0.035
    auto_ticks = [t for t in ax.get_xticks()
                  if h_min <= t <= h_max
                  and all(abs(t - xc) > thr for xc in cross_ticks)]
    ax.set_xticks(sorted(set(auto_ticks) | set(cross_ticks)))
    ax.set_xticklabels([str(int(t)) for t in ax.get_xticks()])
    for lbl in ax.get_xticklabels():
        val = int(lbl.get_text())
        if val in cross_ticks:
            lbl.set_color(cross_ticks[val])
            lbl.set_fontweight("bold")

    fig.tight_layout()

    base = input("\nNome base file output [memory_dqn]: ").strip() \
        or "memory_dqn"
    for fmt in PLOT_FORMATS:
        path = f"{base}.{fmt}"
        fig.savefig(path, format=fmt, bbox_inches="tight", dpi=PLOT_DPI)
        print(f"[=] salvato {path}")

    try:
        plt.show()
    except Exception:
        pass


def main():
    print("=" * 78)
    print(" tinyRL DQN — Calcolatore consumo di memoria")
    print("=" * 78)
    print(" Modalità disponibili:")
    print("   1) Resoconto dettagliato di una singola configurazione")
    print("   2) Grafico consumo al variare della dimensione della rete")
    mode = ask_int("Scegli modalità", 1)
    if mode == 2:
        run_plot()
    else:
        run_report()


if __name__ == "__main__":
    main()
