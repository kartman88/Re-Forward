import time
import struct
import serial
import os
import csv
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# ------------------------------------------------------------
#  CONFIGURAZIONE SERIALE
# ------------------------------------------------------------
PORT            = "/dev/ttyACM0"
BAUDRATE        = 115200
SER_TIMEOUT_S   = 0.05
SEND_PERIOD_MS  = 30
MAX_WAIT_MS     = 100

# --- TOGGLE DISCRETO/CONTINUO ---
USE_CONTINUOUS_ACTIONS = True     # False = azione discreta (es. CartPole)

# ------------------------------------------------------------
#  CONFIGURAZIONE TASK  ← modifica solo qui quando cambi task
# ------------------------------------------------------------
ENV_NAME        = "Hopper-v4"      # Nome ambiente Gymnasium
OBS_DIM         = 11             # Dimensione spazio osservazione
ACTION_DIM      = 3              # Continuo: n. dims  |  Discreto: n. classi
ACTION_RANGE    = (-1.0, 1.0)    # (low, high) di ogni azione — usato nei plot
ENV_KWARGS      = {"terminate_when_unhealthy": False, "render_mode": "human"}  # kwargs extra per gym.make

# Parametri Grafici
PLOT_EVERY_N_EPISODES = 5
CONSOLE_LINES = 5

# ------------------------------------------------------------
#  LOGGING CSV  (stesso formato di pc_ppo_trainer.py per sovrapporre le curve)
# ------------------------------------------------------------
LOG_FOLDER   = "learning_curve_comparison"
LOG_FILENAME = "training_ppo_mcu.csv"

# Profiling: con TIME_LOG attivo la scheda invia UNA RIGA per ogni update di
# ppo_update, marcata con PROF_MARK e chiusa da \n. Il PC la appende subito al
# CSV, quindi il file e' leggibile durante il training (es. `tail -f`) e non si
# perde nulla se la sessione viene chiusa a meta'.
# Deve corrispondere al TIME_LOG di Core/Inc/main.h.
TIME_LOG = True
PROF_CSV_FILENAME = "mcu_timings.csv"
PROF_MARK   = b"<<<PROF>>>"
PROF_HEADER = "episode,steps,total_ms,forward_ms,backward_ms,adam_ms"


def compute_reward(obs, action):
    """Replica esatta di compute_reward() del micro (main.c, Hopper-v4).

    Necessaria per misurare la STESSA reward che il micro ottimizza: la curva
    diventa cosi' confrontabile con pc_ppo_trainer.py, che somma anch'esso questa
    reward custom (non quella nativa di gym).
    """
    healthy = 1.0 if (obs[0] >= 0.7 and -0.2 <= obs[1] <= 0.2) else 0.0
    ctrl_cost = float(np.sum(np.square(action)))
    return healthy + float(obs[5]) - 1e-3 * ctrl_cost

# ------------------------------------------------------------
#  STRUCT SERIALI (auto-calcolate, non toccare)
# ------------------------------------------------------------
_STATE_STRUCT   = f"<{OBS_DIM}f"

if USE_CONTINUOUS_ACTIONS:
    ACTION_BYTE_SIZE = 4 * ACTION_DIM
    _ACTION_STRUCT   = f"<{ACTION_DIM}f"   # N float little-endian
else:
    ACTION_BYTE_SIZE = 1
    _ACTION_STRUCT   = "<B"                # uint8 per azione discreta

_ACTION_FRAME_LEN = 1 + ACTION_BYTE_SIZE + 1 + 1   # STX + action_bytes + done + ETX

# ------------------------------------------------------------
#  FUNZIONI SERIALI
# ------------------------------------------------------------
def send_state(ser: serial.Serial, obs: np.ndarray):
    """Invia gli OBS_DIM float32 LE: STX + payload + checksum(XOR) + ETX.

    Il checksum + ETX permettono al micro di scartare i frame disallineati che
    si verificano dopo la lunga pausa di training (overrun UART): senza di essi
    un frame spazzatura passava i controlli e avvelenava il training.
    """
    payload = struct.pack(_STATE_STRUCT, *obs.astype(np.float32))
    chk = 0
    for b in payload:
        chk ^= b
    ser.write(b'\x02' + payload + bytes([chk]) + b'\x03')

# Buffer di ricezione persistente: sullo stesso stream arrivano sia i frame
# azione binari sia (una volta) il blocco di profiling testuale. Accumuliamo i
# byte e li interpretiamo senza perdere l'allineamento.
_rx = bytearray()
_prof_rows = 0      # righe di timing gia' scritte su file
_prof_last = None   # ultima riga ricevuta, mostrata nella dashboard


def _drain_prof_rows(prof_path: str):
    """Estrae dal buffer le righe di timing complete (PROF_MARK...\\n) e le
    appende subito al CSV, una per update PPO. Il file resta consultabile durante
    il training: la prima riga arriva dopo il primo update (~ROLLOUT_STEPS step),
    non a fine sessione."""
    global _prof_rows, _prof_last
    while True:
        b = _rx.find(PROF_MARK)
        if b < 0:
            return
        e = _rx.find(b"\n", b + len(PROF_MARK))
        if e < 0:
            return  # riga non ancora arrivata per intero
        row = bytes(_rx[b + len(PROF_MARK):e]).decode("ascii", errors="replace").strip()
        # Consuma solo la riga: eventuali frame azione arrivati PRIMA del
        # marcatore restano nel buffer e li raccoglie lo scanner dei frame.
        del _rx[b:e + 1]
        if row.count(",") != 5:
            continue  # riga corrotta (byte persi sul link): si salta, arrivera' la prossima
        with open(prof_path, "w" if _prof_rows == 0 else "a", newline="") as f:
            if _prof_rows == 0:
                f.write(PROF_HEADER + "\n")
            f.write(row + "\n")
        _prof_rows += 1
        _prof_last = row


def recv_action_done(ser: serial.Serial, prof_path: str = None):
    """Legge dallo stream. Ritorna (actions, done) se un frame azione valido e'
    pronto, altrimenti None. Cattura anche il blocco di profiling verso prof_path
    senza perdere l'allineamento dei frame binari.
    Per azioni continue: actions è un np.ndarray di shape (ACTION_DIM,).
    Per azioni discrete: actions è un int.
    """
    global _rx
    data = ser.read(ser.in_waiting or _ACTION_FRAME_LEN)
    if data:
        _rx.extend(data)

    # 1. Cattura le righe di profiling arrivate (una per update PPO).
    if prof_path is not None:
        _drain_prof_rows(prof_path)

    # 2. Cerca un frame azione valido: STX ... ETX alla posizione attesa.
    i = _rx.find(b'\x02')
    while i >= 0 and len(_rx) - i >= _ACTION_FRAME_LEN:
        frame = _rx[i:i + _ACTION_FRAME_LEN]
        if frame[-1] == 0x03:
            action_bytes = frame[1:1 + ACTION_BYTE_SIZE]
            if USE_CONTINUOUS_ACTIONS:
                # Spacchetta ACTION_DIM float — funziona per qualsiasi N
                actions = np.array(struct.unpack(_ACTION_STRUCT, action_bytes),
                                   dtype=np.float32)
            else:
                actions = struct.unpack(_ACTION_STRUCT, action_bytes)[0]  # int discreto
            done_flag = bool(frame[-2])
            del _rx[:i + _ACTION_FRAME_LEN]   # consuma fino a fine frame
            return (actions, done_flag)
        i = _rx.find(b'\x02', i + 1)          # falso STX, riprova dal successivo

    # 3. Nessun frame: evita crescita illimitata del buffer (ma non tagliare una
    #    riga di profiling in arrivo).
    if len(_rx) > 8192 and PROF_MARK not in _rx:
        del _rx[:-64]
    return None


def open_serial():
    """Apre la porta seriale. Solleva l'eccezione se non e' disponibile."""
    ser = serial.Serial(PORT, BAUDRATE, timeout=SER_TIMEOUT_S)
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    return ser


def reopen_serial(old_ser):
    """Il VCP dell'ST-Link puo' sparire e ri-enumerare (USB disconnect): in quel
    caso read/write falliscono con EIO. Chiude e riapre la porta finche' non
    torna disponibile, senza perdere il training gia' fatto sul micro."""
    try:
        old_ser.close()
    except Exception:
        pass
    _rx.clear()   # il buffer parziale non e' piu' allineato allo stream
    print("\n[PC] Seriale caduta (USB disconnect?): attendo che la porta torni...")
    while True:
        try:
            ser = open_serial()
            print("[PC] Seriale ripristinata, riprendo.")
            return ser
        except Exception:
            time.sleep(0.5)

def clear_console():
    """Pulisce il terminale"""
    os.system('cls' if os.name == 'nt' else 'clear')

# ------------------------------------------------------------
#  MAIN LOOP
# ------------------------------------------------------------
def main():
    print(f"[PC] Apertura seriale su {PORT} a {BAUDRATE} baud...")
    try:
        ser = open_serial()
    except Exception as e:
        print(f"Errore apertura seriale: {e}")
        return

    # Inizializza Ambiente
    env = gym.make(ENV_NAME, **ENV_KWARGS)
    
    # Setup Grafici (Matplotlib)
    plt.ion()
    n_plots = 1 + ACTION_DIM
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    ax_reward  = axes[0]
    ax_actions = list(axes[1:])   # un subplot per ogni dim di azione

    # --- Subplot Reward ---
    ax_reward.set_title(f"Training PPO su STM32 - {ENV_NAME}")
    ax_reward.set_xlabel("Episodi")
    ax_reward.set_ylabel("Reward")

    reward_history = []
    moving_avg_history = []
    moving_avg_queue = deque(maxlen=20)
    console_history = deque(maxlen=CONSOLE_LINES)
    all_actions = [[] for _ in range(ACTION_DIM)]  # Una lista per ogni dim di azione

    line_raw, = ax_reward.plot([], [], 'b-', alpha=0.3, label='Reward Grezzo')
    line_avg, = ax_reward.plot([], [], 'r-', linewidth=2, label='Media Mobile (20 ep)')
    ax_reward.legend()

    # --- Subplots Azioni (uno per dim) ---
    for i, ax in enumerate(ax_actions):
        ax.set_title(f"Azione a{i}")
        ax.set_xlabel("Valore")
        ax.set_ylabel("Frequenza")
        if USE_CONTINUOUS_ACTIONS:
            ax.set_xlim(*ACTION_RANGE)

    # Variabili di stato
    episode_count = 0
    global_step = 0
    update_count = 0
    last_update_time = 0.0

    # CSV nello stesso formato di pc_ppo_trainer.py
    os.makedirs(LOG_FOLDER, exist_ok=True)
    csv_path = os.path.join(LOG_FOLDER, LOG_FILENAME)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "step", "reward", "episode_steps"])

    if TIME_LOG:
        os.makedirs(LOG_FOLDER, exist_ok=True)
        prof_path = os.path.join(LOG_FOLDER, PROF_CSV_FILENAME)
    else:
        prof_path = None

    obs, _ = env.reset()

    try:
        while True:
            current_ep_reward = 0.0
            ep_steps = 0
            done = False
            state_sent = False
            last_tx_ms = 0.0
            episode_count += 1
            
            # Timer per rilevare il blocco della Backpropagation
            action_wait_start = time.time()

            while not done:
                env.render()
                now_ms = time.time() * 1000

                # 1. Ritrasmissione robusta + 2. lettura risposta. Il VCP puo'
                #    ri-enumerare a meta' sessione: in quel caso riapriamo la
                #    porta invece di far morire il training.
                try:
                    if (not state_sent) or (now_ms - last_tx_ms > MAX_WAIT_MS):
                        send_state(ser, obs)
                        last_tx_ms = now_ms
                        state_sent = True

                    pkt = recv_action_done(ser, prof_path)
                except (serial.SerialException, OSError):
                    ser = reopen_serial(ser)
                    state_sent = False
                    continue

                if pkt is None:
                    time.sleep(SEND_PERIOD_MS / 1000.0) 
                    continue

                # ---- FRAME VALIDO RICEVUTO ----
                actions, mcu_done = pkt
                state_sent = False

                # Raccogli azioni per dim (per i plot)
                if USE_CONTINUOUS_ACTIONS:
                    for i, a in enumerate(actions):
                        all_actions[i].append(float(a))
                else:
                    all_actions[0].append(actions)

                # 3. Controllo se il micro ha appena fatto training
                wait_time = time.time() - action_wait_start
                if wait_time > 0.5:
                    update_count += 1
                    last_update_time = wait_time

                # 4. Reward custom (identica a quella del micro) calcolata sullo
                #    stato INVIATO al micro e sull'azione ricevuta, PRIMA dello step.
                step_reward = compute_reward(obs, actions)

                # 5. Step nell'ambiente fisico
                # actions è già un np.ndarray (continuo) o int (discreto) — pronto per gym
                obs, r, terminated, truncated, _ = env.step(actions)

                # Se gym termina prima del micro (es. caduta Hopper), resettiamo
                # l'env internamente e inviamo il nuovo obs al micro come passo successivo.
                # Il done è comandato SOLO dal microcontrollore.
                if terminated or truncated:
                    obs, _ = env.reset()

                done = mcu_done

                current_ep_reward += step_reward
                ep_steps += 1
                global_step += 1
                
                # Resettiamo il cronometro per il prossimo step
                action_wait_start = time.time() 

            # ---- FINE EPISODIO ----
            reward_history.append(current_ep_reward)
            moving_avg_queue.append(current_ep_reward)
            current_avg = np.mean(moving_avg_queue)
            moving_avg_history.append(current_avg)

            # Log CSV (stesso formato di pc_ppo_trainer.py)
            csv_writer.writerow([episode_count, global_step,
                                 current_ep_reward, ep_steps])
            csv_file.flush()

            # Prepara riga per la Dashboard
            row = f"{episode_count:8d} | {current_ep_reward:7.1f} | {current_avg:10.1f} | {global_step:9d} | {update_count:9d}"
            console_history.append(row)

            # Stampa Dashboard Pulita
            clear_console()
            print("============================================================")
            print(f"       STM32 PPO - DASHBOARD ({'CONTINUO' if USE_CONTINUOUS_ACTIONS else 'DISCRETO'})")
            print("============================================================")
            if last_update_time > 0:
                print(f"🚀 Ultimo blocco di training ({update_count}) concluso in {last_update_time:.2f} sec")
            else:
                print("⏳ Accumulo dati nel buffer... In attesa del primo training.")
            if TIME_LOG:
                if _prof_rows:
                    print(f"⏱  Timing MCU: {_prof_rows} update -> {prof_path}")
                    print(f"   ultimo (ep,upd,tot,fwd,bwd,adam): {_prof_last}")
                else:
                    print("⏱  Timing MCU: in attesa del primo update...")
            print("------------------------------------------------------------")
            print(f"{'EPISODIO':>8} | {'REWARD':>7} | {'MEDIA (20)':>10} | {'STEP TOT.':>9} | {'UPDATE N°':>9}")
            print("------------------------------------------------------------")
            for line in console_history:
                print(line)
            print("============================================================")

            # Resetta ambiente per il nuovo episodio
            obs, _ = env.reset()
            state_sent = False

            # Aggiorna il Grafico ogni N episodi
            if episode_count % PLOT_EVERY_N_EPISODES == 0:
                line_raw.set_xdata(range(len(reward_history)))
                line_raw.set_ydata(reward_history)
                line_avg.set_xdata(range(len(moving_avg_history)))
                line_avg.set_ydata(moving_avg_history)
                ax_reward.relim()
                ax_reward.autoscale_view()
                
                # Aggiorna un istogramma per ogni dim di azione
                for i, ax in enumerate(ax_actions):
                    ax.clear()
                    ax.set_title(f"Azione a{i}")
                    ax.set_xlabel("Valore")
                    ax.set_ylabel("Frequenza")
                    if USE_CONTINUOUS_ACTIONS:
                        ax.hist(all_actions[i], bins=30, color='green',
                                alpha=0.7, edgecolor='black')
                        ax.set_xlim(*ACTION_RANGE)
                    else:
                        ax.hist(all_actions[0], bins=range(0, ACTION_DIM + 1),
                                color='green', alpha=0.7, edgecolor='black')
                
                fig.canvas.draw()
                fig.canvas.flush_events()

    except KeyboardInterrupt:
        clear_console()
        print("\n[PC] Training interrotto dall'utente. Risorse liberate.")
    finally:
        csv_file.close()
        print(f"[PC] Log salvato in {csv_path}")
        env.close()
        ser.close()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()