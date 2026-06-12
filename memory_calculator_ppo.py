"""Calcolatore del consumo di memoria heap/statica dell'implementazione PPO su STM32.

Riproduce, parametrizzato sulla topologia delle reti e sulle dimensioni dei buffer,
il consumo delle strutture dati definite in tinyRL/Core (dense_layer.h, neural_net.h,
ppo.h) e lo confronta con un approccio classico di backpropagation a minibatch.

Due modalita' d'uso:
  - interattiva (default): chiede i parametri e stampa il breakdown dettagliato.
  - grafico (--plot): traccia consumo memoria (mio approccio vs classico) al
    variare della dimensione dell'hidden layer, con i limiti RAM di 3 schede ST.
"""

import argparse
import os

# Tipi sul target ARM Cortex-M (STM32H7, 32-bit)
SIZE_FLOAT  = 4
SIZE_UINT32 = 4
SIZE_UINT8  = 1
SIZE_PTR    = 4    # puntatori a 32 bit

# sizeof(DenseLayer): 2 int + 9 puntatori + 1 enum (4 byte), nessun padding
DENSE_LAYER_STRUCT = 2 * 4 + 9 * SIZE_PTR + 4
# sizeof(Network): 1 ptr + uint8 num_layers + uint32 adam_t (con padding -> 12)
NETWORK_STRUCT = SIZE_PTR + 4 + SIZE_UINT32
# sizeof(RolloutBuffer): 8 ptr + 4 uint32
ROLLOUT_STRUCT = 8 * SIZE_PTR + 4 * SIZE_UINT32

LABEL_W = 60
BYTES_W = 10
KB_W    = 8


def ask_int(prompt, default):
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            v = int(raw)
            if v <= 0:
                print("  Inserire un intero positivo.")
                continue
            return v
        except ValueError:
            print("  Valore non valido, inserire un intero.")


def ask_str(prompt, default):
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw if raw != "" else default


def ask_mode():
    print("Modalita' d'uso:")
    print("  [1] Calcolo interattivo  (breakdown dettagliato di memoria)")
    print("  [2] Grafico comparativo  (mio approccio vs classico, limiti schede)")
    while True:
        raw = input("Scegli modalita' [1]: ").strip()
        if raw == "":
            return 1
        if raw in ("1", "2"):
            return int(raw)
        print("  Inserire 1 o 2.")


def ask_topology(label, default):
    """Legge una topologia come lista di interi separati da spazio o virgola."""
    default_str = " ".join(str(x) for x in default)
    while True:
        raw = input(f"{label} (es. {default_str}) [{default_str}]: ").strip()
        if raw == "":
            return list(default)
        parts = raw.replace(",", " ").split()
        try:
            topo = [int(p) for p in parts]
            if len(topo) < 2 or any(x <= 0 for x in topo):
                print("  Servono almeno 2 dimensioni positive (input...output).")
                continue
            return topo
        except ValueError:
            print("  Valori non validi, usare solo interi.")


def layer_bytes_adam(in_dim, out_dim):
    """Bytes di un DenseLayer con ottimizzatore Adam, come in dense_init().

    4 matrici out x in (W, dW, mW, vW): dati + array di puntatori riga (alloc_2d).
    5 vettori out (b, db, mb, vb, out).
    Piu' overhead della struct DenseLayer.
    """
    mat_data = 4 * (out_dim * in_dim * SIZE_FLOAT)
    mat_ptrs = 4 * (out_dim * SIZE_PTR)          # alloc_2d alloca un float* per riga
    vecs     = 5 * (out_dim * SIZE_FLOAT)
    return mat_data + mat_ptrs + vecs + DENSE_LAYER_STRUCT


def network_breakdown(topology, name):
    items = []
    for l in range(len(topology) - 1):
        in_dim, out_dim = topology[l], topology[l + 1]
        b = layer_bytes_adam(in_dim, out_dim)
        items.append((f"{name} layer {l}  ({in_dim} -> {out_dim})", b))
    items.append((f"{name} struct Network", NETWORK_STRUCT))
    return items


def rollout_breakdown(T, obs_dim, n_act_dims):
    """Buffer di rollout PPO (continuous): vedi rollout_buffer_init()."""
    items = [
        (f"states[T][obs_dim]  ({T} x {obs_dim})",  T * obs_dim * SIZE_FLOAT),
        (f"actions[T][act_dims] ({T} x {n_act_dims})", T * n_act_dims * SIZE_FLOAT),
        ("log_probs_old[T]",  T * SIZE_FLOAT),
        ("values[T]",         T * SIZE_FLOAT),
        ("rewards[T]",        T * SIZE_FLOAT),
        ("advantages[T]",     T * SIZE_FLOAT),
        ("returns[T]",        T * SIZE_FLOAT),
        ("dones[T] (uint8)",  T * SIZE_UINT8),
        ("struct RolloutBuffer", ROLLOUT_STRUCT),
    ]
    return items


def stack_breakdown(T, max_layer_dim, n_act_dims):
    """Buffer statici/locali principali coinvolti nel training.

    - idx[ROLLOUT_STEPS] (uint32): permutazione minibatch in ppo_update (static).
    - delta_buf (float): buffer di backprop, dimensionato al layer piu' grande (static).
    - delta_out[N_ACT_DIMS] (float): gradiente sui logits dell'actor.
    - prev_obs / prev_action nell'agente (gia' in PPOAgent, qui solo i temporanei).
    """
    items = [
        (f"idx[T] (uint32, static in ppo_update)", T * SIZE_UINT32),
        (f"delta_buf[max_dim] ({max_layer_dim})",  max_layer_dim * SIZE_FLOAT),
        (f"delta_out[act_dims] ({n_act_dims})",    n_act_dims * SIZE_FLOAT),
    ]
    return items


def print_table(index, title, items):
    print()
    print(f"[{index}] {title}")
    print("-" * (LABEL_W + BYTES_W + KB_W))
    print(f"{'Componente':<{LABEL_W}}{'Bytes':>{BYTES_W}}{'KB':>{KB_W}}")
    print("-" * (LABEL_W + BYTES_W + KB_W))
    subtotal = 0
    for label, b in items:
        subtotal += b
        print(f"  {label:<{LABEL_W - 2}}{b:>{BYTES_W}}{b / 1024:>{KB_W}.2f}")
    print("-" * (LABEL_W + BYTES_W + KB_W))
    print(f"  {'SUBTOTALE':<{LABEL_W - 2}}{subtotal:>{BYTES_W}}{subtotal / 1024:>{KB_W}.2f}")
    return subtotal


def trainable_params(topology):
    total = 0
    for l in range(len(topology) - 1):
        in_dim, out_dim = topology[l], topology[l + 1]
        total += out_dim * in_dim + out_dim   # W + b
    return total


# Limiti di RAM (KB) delle schede ST usate negli esperimenti.
BOARDS = [
    ("STM32F406 - 128 KB", 128),
    ("STM32H7 - 1 MB",     1024),
    ("STM32N6 - 4 MB",     4096),
]


def section_total(items):
    return sum(b for _, b in items)


def memory_totals(hidden, obs_dim, n_act_dims, T, batch_size, depth=1):
    """Totale memoria (byte) per mio approccio e classico, con `depth` hidden layer.

    actor  = [obs, hidden, ..., hidden, n_act]  (depth volte hidden)
    critic = [obs, hidden, ..., hidden, 1]       (depth volte hidden)
    Con depth=1 si ricade nel caso a un solo hidden layer.
    Le reti (W,b + grad + Adam), il rollout buffer e lo stack sono identici nei due
    approcci. La differenza e' solo nelle attivazioni: l'on-device ne tiene 1 copia
    (gia' inclusa nel campo .out di ogni DenseLayer), un trainer autograd a minibatch
    ne tiene batch_size copie -> overhead (batch_size - 1) per campione.
    """
    actor_topo  = [obs_dim] + [hidden] * depth + [n_act_dims]
    critic_topo = [obs_dim] + [hidden] * depth + [1]

    nets    = section_total(network_breakdown(actor_topo, "Actor")) + \
              section_total(network_breakdown(critic_topo, "Critic"))
    rollout = section_total(rollout_breakdown(T, obs_dim, n_act_dims))
    max_dim = max(hidden, obs_dim, n_act_dims)
    stack   = section_total(stack_breakdown(T, max_dim, n_act_dims))

    mine = nets + rollout + stack

    act_per_sample = sum(actor_topo[1:]) + sum(critic_topo[1:])
    classic_extra  = (batch_size - 1) * act_per_sample * SIZE_FLOAT
    classic = mine + classic_extra

    return mine, classic


# ===================================================================== #
#         PANNELLO GRAFICO (stile scientifico, vedi PLOT_INSTRUCTION.md)
#            modifica queste variabili per il tuning del grafico
# ===================================================================== #
PLOT_FIGSIZE   = (8.0, 6.0)            # pollici, aspect-ratio bloccato
PLOT_DPI       = 300                   # alta risoluzione (no immagini sgranate)
# catena di font serif: il primo disponibile vince (Times -> sostituti metric-compat.)
PLOT_FONT      = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]
PLOT_FONT_SIZE = 20                    # testo grande (verra' rimpicciolito nel paper)
# formati di salvataggio: vettoriale (PDF/SVG) + anteprima raster (PNG)
PLOT_SAVE_FORMATS = ["pdf", "svg", "png"]

# Palette minimalista: una tinta per curva (viola / verde-acqua).
COLOR_MINE     = "#1B9E77"             # verde acqua - approccio on-device
COLOR_CLASSIC  = "#6A3D9A"             # viola - backprop classica
# Linee di riferimento (limiti RAM schede) in grigio neutro, tonalita' diverse.
BOARD_COLORS   = ["#9A9A9A", "#6E6E6E", "#3F3F3F"]

PLOT_LINE_WIDTH = 2.4                  # spessore curve principali
MARKER_SIZE     = 8                    # marker sugli incroci curva-limite
GRID_MAJOR      = dict(lw=0.6, alpha=0.45, color="0.6")
GRID_MINOR      = dict(lw=0.3, alpha=0.30, color="0.75")
TITLE_PAD       = 54                   # spazio titolo-grafico (lascia posto alla legenda)
SHOW_TITLE      = True


def run_plot(args):
    import numpy as np
    import matplotlib
    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator

    families = PLOT_FONT if isinstance(PLOT_FONT, (list, tuple)) else [PLOT_FONT]
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": list(families),
        "font.size": PLOT_FONT_SIZE,
        "xtick.labelsize": PLOT_FONT_SIZE * 0.6,   # tick piu' piccoli delle label assi
        "ytick.labelsize": PLOT_FONT_SIZE * 0.6,   # (qui i tick X marcano gli incroci)
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",        # mantieni il testo editabile in SVG
    })

    hiddens = np.unique(np.linspace(args.hmin, args.hmax, args.points).astype(int))
    mine_kb, classic_kb = [], []
    for h in hiddens:
        mine, classic = memory_totals(int(h), args.obs, args.act,
                                      args.rollout, args.batch, args.depth)
        mine_kb.append(mine / 1024.0)
        classic_kb.append(classic / 1024.0)

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)

    # curve principali in primo piano (z-order alto)
    ax.plot(hiddens, classic_kb, color=COLOR_CLASSIC, lw=PLOT_LINE_WIDTH,
            solid_capstyle="round", zorder=6,
            label=f"Backprop classica (minibatch={args.batch})")
    ax.plot(hiddens, mine_kb, color=COLOR_MINE, lw=PLOT_LINE_WIDTH,
            solid_capstyle="round", zorder=6,
            label="Mio approccio (on-device, 1 campione)")

    # linee-limite delle schede: grigio neutro, sullo sfondo (z-order basso)
    for (name, limit_kb), c in zip(BOARDS, BOARD_COLORS):
        ax.axhline(limit_kb, linestyle="--", color=c, linewidth=1.2, zorder=2)
        ax.text(args.hmax, limit_kb, f" {name}", va="center", ha="left",
                fontsize=PLOT_FONT_SIZE * 0.55, color=c)

    # Incroci curva-limite: marker sul punto + linea verticale tratteggiata fino
    # all'asse X col valore esatto di H. I valori sono monotoni crescenti in H,
    # quindi np.interp inverte y->x. Niente incrocio se il limite e' sotto il
    # consumo minimo (es. il rollout buffer supera gia' i 128 KB della F406).
    crossings = []  # (x_cross, color) per ogni incrocio curva-limite
    for ys, curve_color in [(np.array(mine_kb), COLOR_MINE),
                            (np.array(classic_kb), COLOR_CLASSIC)]:
        for (_, limit_kb), c in zip(BOARDS, BOARD_COLORS):
            if ys[0] <= limit_kb <= ys[-1]:
                x_cross = float(np.interp(limit_kb, ys, hiddens))
                ax.plot([x_cross], [limit_kb], "o", color=curve_color,
                        markersize=MARKER_SIZE, zorder=7)
                ax.vlines(x_cross, 0, limit_kb, color=curve_color,
                          linestyle=":", linewidth=1.3, alpha=0.8, zorder=5)
                crossings.append((x_cross, curve_color))

    ax.set_xlabel("Dimensione hidden layer (unita')")
    ax.set_ylabel("Consumo memoria (KB)")
    ax.set_xlim(args.hmin, args.hmax)
    ax.set_ylim(bottom=0)

    # griglia maggiore + minore, sottile e trasparente, dietro ai dati.
    # Solo l'asse Y usa i minor tick automatici: sull'asse X i tick sono
    # personalizzati (segnano gli incroci) e una spaziatura irregolare manderebbe
    # in tilt AutoMinorLocator.
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which="major", **GRID_MAJOR)
    ax.grid(which="minor", axis="y", **GRID_MINOR)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    if SHOW_TITLE:
        ax.set_title(f"Memoria PPO vs backprop classica  "
                     f"(obs={args.obs}, act={args.act}, depth={args.depth}, "
                     f"rollout={args.rollout})",
                     fontsize=PLOT_FONT_SIZE * 0.6, pad=TITLE_PAD)

    # legenda compatta, fuori dalla tela (sopra), senza bordo
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1,
              frameon=False, handlelength=1.6, borderaxespad=0.0,
              fontsize=PLOT_FONT_SIZE * 0.7)

    # Valori di H degli incroci come tick sull'asse X. I tick automatici troppo
    # vicini vengono rimossi per fare spazio senza sovrapporre le etichette.
    cross_ticks = {int(round(xc)): color for xc, color in crossings}
    thr = (args.hmax - args.hmin) * 0.035
    auto_ticks = [t for t in ax.get_xticks()
                  if args.hmin <= t <= args.hmax
                  and all(abs(t - xc) > thr for xc in cross_ticks)]
    ax.set_xticks(sorted(set(auto_ticks) | set(cross_ticks)))
    ax.set_xticklabels([str(int(t)) for t in ax.get_xticks()])
    for lbl in ax.get_xticklabels():
        val = int(lbl.get_text())
        if val in cross_ticks:
            lbl.set_color(cross_ticks[val])
            lbl.set_fontweight("bold")

    fig.tight_layout()

    if args.out:
        base = os.path.splitext(args.out)[0]
        for fmt in PLOT_SAVE_FORMATS:
            out = f"{base}.{fmt}"
            fig.savefig(out, format=fmt, bbox_inches="tight", dpi=PLOT_DPI)
            print(f"[OK] Grafico salvato in {out}")
    else:
        plt.show()


def run_interactive():
    print("=" * (LABEL_W + BYTES_W + KB_W))
    print(" CALCOLATORE MEMORIA - PPO su STM32 (Hopper-v4 default)")
    print("=" * (LABEL_W + BYTES_W + KB_W))
    print("Invio vuoto = valore di default.\n")

    actor_topo  = ask_topology("Topologia ACTOR (policy)",  [11, 64, 64, 3])
    critic_topo = ask_topology("Topologia CRITIC (value)",  [11, 64, 64, 1])

    obs_dim     = actor_topo[0]
    n_act_dims  = actor_topo[-1]

    T = ask_int("Rollout buffer size (N step)", 2048)
    batch_size = ask_int("Batch size minibatch PPO", 64)
    ask_int("n_epochs PPO (non incide su RAM)", 8)

    actor_items  = network_breakdown(actor_topo,  "Actor")
    critic_items = network_breakdown(critic_topo, "Critic")
    rollout_items = rollout_breakdown(T, obs_dim, n_act_dims)

    max_layer_dim = max(max(actor_topo), max(critic_topo))
    stack_items = stack_breakdown(T, max_layer_dim, n_act_dims)

    s_actor   = print_table(1, "ACTOR NETWORK (policy + Adam)", actor_items)
    s_critic  = print_table(2, "CRITIC NETWORK (value + Adam)", critic_items)
    s_rollout = print_table(3, "ROLLOUT BUFFER (traiettoria PPO)", rollout_items)
    s_stack   = print_table(4, "STACK / LOCALI PRINCIPALI", stack_items)

    total = s_actor + s_critic + s_rollout + s_stack

    print()
    print("=" * (LABEL_W + BYTES_W + KB_W))
    print(" RIEPILOGO")
    print("=" * (LABEL_W + BYTES_W + KB_W))
    print(f"{'Sezione':<{LABEL_W}}{'Bytes':>{BYTES_W}}{'KB':>{KB_W}}")
    print("-" * (LABEL_W + BYTES_W + KB_W))
    for name, b in [("Actor network", s_actor), ("Critic network", s_critic),
                    ("Rollout buffer", s_rollout), ("Stack/locali", s_stack)]:
        print(f"{name:<{LABEL_W}}{b:>{BYTES_W}}{b / 1024:>{KB_W}.2f}")
    print("-" * (LABEL_W + BYTES_W + KB_W))
    print(f"{'TOTALE':<{LABEL_W}}{total:>{BYTES_W}}{total / 1024:>{KB_W}.2f}")
    print("=" * (LABEL_W + BYTES_W + KB_W))
    print(f"\n>>> Consumo totale stimato: {total / 1024:.2f} KB ({total} byte) <<<\n")

    # --- Confronto con backpropagation classica a minibatch ---
    # Un trainer classico autograd (stile PyTorch) processa il minibatch in modo
    # vettorializzato: il grafo tiene le attivazioni di batch_size campioni per ogni
    # layer (actor + critic) finche' non ha calcolato il backward, poi le libera.
    # L'implementazione on-device rifa' invece il forward un campione alla volta e
    # accumula subito il gradiente, tenendo le attivazioni di un solo campione.
    p_actor  = trainable_params(actor_topo)
    p_critic = trainable_params(critic_topo)

    # Attivazioni salvate per campione: somma delle dimensioni di output di ogni
    # layer (sia actor che critic).
    act_out_per_sample = sum(actor_topo[1:]) + sum(critic_topo[1:])
    classic_activations  = batch_size * act_out_per_sample * SIZE_FLOAT  # batch vettorializzato
    ondevice_activations = act_out_per_sample * SIZE_FLOAT               # 1 campione alla volta

    print("-" * (LABEL_W + BYTES_W + KB_W))
    print(f" CONFRONTO CON BACKPROP CLASSICA (minibatch = {batch_size})")
    print("-" * (LABEL_W + BYTES_W + KB_W))
    print(f"{'Parametri allenabili actor':<{LABEL_W}}{p_actor:>{BYTES_W}}")
    print(f"{'Parametri allenabili critic':<{LABEL_W}}{p_critic:>{BYTES_W}}")
    print(f"{'Pesi (1 copia) actor+critic [byte]':<{LABEL_W}}"
          f"{(p_actor + p_critic) * SIZE_FLOAT:>{BYTES_W}}")
    print(f"{'Pesi+Adam (x4 copie) actor+critic [byte]':<{LABEL_W}}"
          f"{(p_actor + p_critic) * SIZE_FLOAT * 4:>{BYTES_W}}")
    print("-" * (LABEL_W + BYTES_W + KB_W))
    print(f"{'Attivazioni on-device (1 campione)':<{LABEL_W}}"
          f"{ondevice_activations:>{BYTES_W}}{ondevice_activations / 1024:>{KB_W}.2f}")
    print(f"{f'Attivazioni classiche minibatch ({batch_size} campioni)':<{LABEL_W}}"
          f"{classic_activations:>{BYTES_W}}{classic_activations / 1024:>{KB_W}.2f}")
    extra = classic_activations - ondevice_activations
    print(f"{'Overhead extra backprop classica':<{LABEL_W}}"
          f"{extra:>{BYTES_W}}{extra / 1024:>{KB_W}.2f}")
    print("-" * (LABEL_W + BYTES_W + KB_W))
    classic_total = total + extra
    print(f"{'Totale stimato approccio classico':<{LABEL_W}}"
          f"{classic_total:>{BYTES_W}}{classic_total / 1024:>{KB_W}.2f}")
    if total:
        print(f"\n>>> Risparmio on-device: {extra / 1024:.2f} KB "
              f"({100.0 * extra / classic_total:.1f}% del totale classico) <<<\n")


def main():
    mode = ask_mode()
    print()
    if mode == 1:
        run_interactive()
        return

    # Modalita' grafico: parametri chiesti in modo interattivo (invio = default).
    cfg = argparse.Namespace()
    cfg.obs     = ask_int("obs_dim", 11)
    cfg.act     = ask_int("n_act_dims", 3)
    cfg.depth   = ask_int("Profondita' rete (n. hidden layer)", 2)
    cfg.rollout = ask_int("Rollout buffer T", 2048)
    cfg.batch   = ask_int("Batch size minibatch PPO", 64)
    cfg.hmin    = ask_int("Hidden layer minimo (asse X)", 16)
    cfg.hmax    = ask_int("Hidden layer massimo (asse X)", 8192)
    cfg.points  = ask_int("Punti campionati sull'asse X", 200)

    save = input("Output: [1] salva PNG  [2] mostra a schermo [1]: ").strip()
    if save == "2":
        cfg.out = ""
    else:
        cfg.out = ask_str("Nome file PNG", "memory_comparison.png")

    run_plot(cfg)


if __name__ == "__main__":
    main()
