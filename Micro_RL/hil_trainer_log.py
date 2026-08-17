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
#  CONFIGURAZIONE
# ------------------------------------------------------------
PORT            = "/dev/ttyACM0"          # Sostituisci con la tua porta
BAUDRATE        = 115200
SER_TIMEOUT_S   = 0.05            # Timeout velocissimo per la read della seriale
SEND_PERIOD_MS  = 30              # Pausa per evitare busy-loop (30 ms)
MAX_WAIT_MS     = 100             # Ritrasmette l'osservazione se non arriva risposta

# --- TOGGLE DISCRETO/CONTINUO ---
USE_CONTINUOUS_ACTIONS = True     # Metti a False per tornare al CartPole Discreto!

# Parametri Grafici
RENDER = True
PLOT_EVERY_N_EPISODES = 5
CONSOLE_LINES = 5

# Logging CSV
LOG_FOLDER   = "learning_curve_comparison"
LOG_FILENAME = "training_mcu_x.csv"

# Profiling: con TIME_LOG attivo la scheda invia una volta, a buffer pieno, il
# CSV dei tempi di dqn_train racchiuso tra questi marcatori. Il PC lo cattura e
# lo salva a parte. Deve corrispondere al TIME_LOG di Core/Inc/main.h.
TIME_LOG = True
PROF_CSV_FILENAME = "mcu_timings.csv"
PROF_BEGIN = b"<<<PROF_BEGIN>>>"
PROF_END   = b"<<<PROF_END>>>"

# ------------------------------------------------------------
#  DIMENSIONI TASK
# ------------------------------------------------------------
ENV_NAME        = "Pendulum-v1" if USE_CONTINUOUS_ACTIONS else "CartPole-v1"
OBS_DIM         = 3 if USE_CONTINUOUS_ACTIONS else 4  # Aggiorna in base all'ambiente!

_STATE_STRUCT   = f"<{OBS_DIM}f"

# Calcolo automatico della dimensione del frame in base alla tipologia di azione
if USE_CONTINUOUS_ACTIONS:
    ACTION_DIM = 1  # <-- Metti il numero di azioni (es. 2 per un braccio)
    ACTION_BYTE_SIZE = 4 * ACTION_DIM
    _ACTION_STRUCT = f"<{ACTION_DIM}f"   # N float little-endian
else:
    ACTION_BYTE_SIZE = 1   # L'azione è un intero discreto (1 byte)
    _ACTION_STRUCT = "<B"  # Little-endian unsigned char (uint8)

_ACTION_FRAME_LEN = 1 + ACTION_BYTE_SIZE + 1 + 1   # 0x02 + action_bytes + done + 0x03

# ------------------------------------------------------------
#  FUNZIONI SERIALI
# ------------------------------------------------------------
def send_state(ser: serial.Serial, obs: np.ndarray):
    """Invia gli OBS_DIM float32 LE: STX + payload + checksum(XOR) + ETX.

    Il checksum + ETX permettono al micro (uart_recv_floats in Core/Src/uart.c)
    di scartare i frame disallineati che si accumulano durante la pausa di
    dqn_train (overrun UART): senza di essi un frame spazzatura passerebbe i
    controlli e finirebbe nel replay buffer.
    """
    payload = struct.pack(_STATE_STRUCT, *obs.astype(np.float32))
    chk = 0
    for b in payload:
        chk ^= b
    ser.write(b'\x02' + payload + bytes([chk]) + b'\x03')

def compute_reward(obs, action_val):
    """Replica di evaluate_reward_pendulum() del micro (main.c, Pendulum-v1).
    Garantisce parita' di misura tra le curve C e quelle Python."""
    theta = np.arctan2(obs[1], obs[0])
    omega = obs[2]
    u     = float(action_val)   # torque gia' ricevuto dal micro
    return -(theta * theta + 0.1 * omega * omega + 0.001 * u * u)

# Buffer di ricezione persistente: sullo stesso stream arrivano sia i frame
# azione binari sia (una volta) il blocco di profiling testuale. Accumuliamo i
# byte e li interpretiamo senza perdere l'allineamento.
_rx = bytearray()
_prof_saved = False


def _try_save_prof(prof_path: str):
    """Estrae dal buffer i blocchi di profiling completi (BEGIN...END) e li
    rimuove. Il micro ripete il dump per qualche episodio in caso di caduta del
    link: solo il primo blocco completo viene salvato su prof_path."""
    global _prof_saved
    while True:
        b = _rx.find(PROF_BEGIN)
        if b < 0:
            return
        e = _rx.find(PROF_END, b + len(PROF_BEGIN))
        nxt = _rx.find(PROF_BEGIN, b + len(PROF_BEGIN))
        if nxt >= 0 and (e < 0 or nxt < e):
            # Un nuovo dump e' iniziato prima che il precedente si chiudesse: il
            # primo e' troncato (link caduto a meta' trasmissione). Scarta solo
            # il marcatore orfano, il testo residuo lo smaltisce lo scanner dei
            # frame senza inghiottire i frame azione che stanno in mezzo.
            del _rx[b:b + len(PROF_BEGIN)]
            continue
        if e < 0:
            return  # blocco non ancora arrivato per intero
        block = bytes(_rx[b + len(PROF_BEGIN):e]).decode("ascii", errors="replace")
        del _rx[b:e + len(PROF_END)]
        if not _prof_saved:
            with open(prof_path, "w", newline="") as f:
                f.write(block.strip("\r\n") + "\n")
            _prof_saved = True
            n_rows = max(block.count("\n") - 1, 0)  # meno la riga di header
            print(f"\n[PC] Timing MCU salvati in {prof_path} ({n_rows} misure)")


def recv_action_done(ser: serial.Serial, prof_path: str = None):
    """Legge dallo stream. Ritorna (azione_decodificata, done) se un frame azione
    valido e' pronto, altrimenti None. Cattura anche il blocco di profiling verso
    prof_path senza perdere l'allineamento dei frame binari."""
    global _rx
    data = ser.read(ser.in_waiting or _ACTION_FRAME_LEN)
    if data:
        _rx.extend(data)

    # 1. Cattura eventuale blocco di profiling (CSV ASCII inviato una sola volta).
    if prof_path is not None:
        _try_save_prof(prof_path)

    # 2. Cerca un frame azione valido: STX ... ETX alla posizione attesa.
    i = _rx.find(b'\x02')
    while i >= 0 and len(_rx) - i >= _ACTION_FRAME_LEN:
        frame = _rx[i:i + _ACTION_FRAME_LEN]
        if frame[-1] == 0x03:
            action_bytes = frame[1:1 + ACTION_BYTE_SIZE]
            action_val = struct.unpack(_ACTION_STRUCT, action_bytes)[0]
            done_flag = bool(frame[-2])
            del _rx[:i + _ACTION_FRAME_LEN]   # consuma fino a fine frame
            return (action_val, done_flag)
        i = _rx.find(b'\x02', i + 1)          # falso STX, riprova dal successivo

    # 3. Nessun frame: evita crescita illimitata del buffer (ma non tagliare un
    #    blocco di profiling in arrivo).
    if len(_rx) > 8192 and PROF_BEGIN not in _rx:
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


def save_episode_log(episode_log):
    """Scrive la curva di apprendimento. Chiamata a ogni fine episodio cosi' un
    crash non porta via i dati gia' raccolti."""
    os.makedirs(LOG_FOLDER, exist_ok=True)
    csv_path = os.path.join(LOG_FOLDER, LOG_FILENAME)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward"])
        w.writerows(episode_log)


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
    if RENDER:
      env = gym.make(ENV_NAME, render_mode="human") 
    else:
      env = gym.make(ENV_NAME)
    
    # Setup Grafici (Matplotlib)
    plt.ion()
    fig, (ax_reward, ax_action) = plt.subplots(1, 2, figsize=(15, 5))
    
    # --- Subplot Reward ---
    ax_reward.set_title(f"Training PPO su STM32 - {ENV_NAME}")
    ax_reward.set_xlabel("Episodi")
    ax_reward.set_ylabel("Reward")
    
    reward_history = []
    moving_avg_history = []
    moving_avg_queue = deque(maxlen=20)
    console_history = deque(maxlen=CONSOLE_LINES)
    all_actions = []  # Raccoglie tutte le azioni
    
    line_raw, = ax_reward.plot([], [], 'b-', alpha=0.3, label='Reward Grezzo')
    line_avg, = ax_reward.plot([], [], 'r-', linewidth=2, label='Media Mobile (20 ep)')
    ax_reward.legend()
    
    # --- Subplot Action Distribution ---
    ax_action.set_title("Distribuzione Azioni")
    ax_action.set_xlabel("Valore Azione")
    ax_action.set_ylabel("Frequenza")
    if USE_CONTINUOUS_ACTIONS:
        ax_action.set_xlim(-2.0, 2.0)  # Tipico range per Pendulum
    hist_patches = ax_action.patches

    # Setup CSV
    episode_log = []
    # Path dove salvare i tempi di training che la scheda dumpa a buffer pieno.
    if TIME_LOG:
        os.makedirs(LOG_FOLDER, exist_ok=True)
        prof_path = os.path.join(LOG_FOLDER, PROF_CSV_FILENAME)
    else:
        prof_path = None

    # Variabili di stato
    episode_count = 0
    global_step = 0
    update_count = 0
    last_update_time = 0.0

    obs, _ = env.reset()

    try:
        while True:
            current_ep_reward = 0.0
            done = False
            state_sent = False
            last_tx_ms = 0.0
            episode_count += 1
            
            # Timer per rilevare il blocco della Backpropagation
            action_wait_start = time.time()

            while not done:
                if RENDER:
                    env.render()
                now_ms = time.time() * 1000

                # 1. Ritrasmissione robusta
                #    (una caduta della seriale non interrompe la sessione: si
                #     riapre la porta e si riparte dallo stato corrente)
                try:
                    if (not state_sent) or (now_ms - last_tx_ms > MAX_WAIT_MS):
                        send_state(ser, obs)
                        last_tx_ms = now_ms
                        state_sent = True

                    # 2. Prova a leggere la risposta (e cattura il blocco profiling)
                    pkt = recv_action_done(ser, prof_path)
                except (serial.SerialException, OSError):
                    ser = reopen_serial(ser)
                    state_sent = False
                    continue

                if pkt is None:
                    time.sleep(SEND_PERIOD_MS / 1000.0) 
                    continue

                # ---- FRAME VALIDO RICEVUTO ----
                action_val, mcu_done = pkt
                state_sent = False 
                all_actions.append(action_val)  # Raccogli l'azione
                
                # 3. Controllo se il micro ha appena fatto training
                wait_time = time.time() - action_wait_start
                if wait_time > 0.5: 
                    update_count += 1
                    last_update_time = wait_time

                # 4. Formattazione dell'Azione per Gymnasium
                if USE_CONTINUOUS_ACTIONS:
                    # Gli ambienti continui richiedono un array numpy
                    action_to_env = np.array([action_val], dtype=np.float32)
                else:
                    # Gli ambienti discreti richiedono un intero
                    action_to_env = int(action_val)
                
                # Reward calcolata sullo stato PRE-step (lo stesso che ha visto
                # il micro), replicando esattamente la formula del firmware.
                step_reward = compute_reward(obs, action_val)

                # Step nell'ambiente fisico
                obs, r, terminated, truncated, _ = env.step(action_to_env)
                #print("REWARD:", r)  # Debug: stampa il reward ricevuto
                #Divido per 8 l'obs[2] per normalizzare
                #obs[2] /= 8

                # LA VERA MAGIA: Ci fidiamo SOLO del microcontrollore!
                done = mcu_done

                current_ep_reward += step_reward
                global_step += 1
                
                # Resettiamo il cronometro per il prossimo step
                action_wait_start = time.time() 

            # ---- FINE EPISODIO ----
            episode_log.append((episode_count, current_ep_reward))
            save_episode_log(episode_log)   # salvataggio incrementale
            reward_history.append(current_ep_reward)
            moving_avg_queue.append(current_ep_reward)
            current_avg = np.mean(moving_avg_queue)
            moving_avg_history.append(current_avg)

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
                
                # Aggiorna l'istogramma delle azioni
                ax_action.clear()
                ax_action.set_title("Distribuzione Azioni")
                ax_action.set_xlabel("Valore Azione")
                ax_action.set_ylabel("Frequenza")
                if USE_CONTINUOUS_ACTIONS:
                    ax_action.hist(all_actions, bins=30, color='green', alpha=0.7, edgecolor='black')
                    ax_action.set_xlim(-2.0, 2.0)
                else:
                    ax_action.hist(all_actions, bins=range(0, 5), color='green', alpha=0.7, edgecolor='black')
                
                fig.canvas.draw()
                fig.canvas.flush_events()

    except KeyboardInterrupt:
        clear_console()
        print("\n[PC] Training interrotto dall'utente. Risorse liberate.")
    finally:
        env.close()
        try:
            ser.close()
        except Exception:
            pass
        save_episode_log(episode_log)
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()