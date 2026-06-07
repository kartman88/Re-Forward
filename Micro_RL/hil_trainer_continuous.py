import time
import struct
import serial
import os
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
PLOT_EVERY_N_EPISODES = 5
CONSOLE_LINES = 5

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
    """Invia gli OBS_DIM float32 in little-endian tra 0x02 e 0x03"""
    payload = struct.pack(_STATE_STRUCT, *obs.astype(np.float32))
    ser.write(b'\x02' + payload + b'\x03')

def compute_reward(obs, action_val):
    """Replica di evaluate_reward_pendulum() del micro (main.c, Pendulum-v1).
    Garantisce parita' di misura tra le curve C e quelle Python."""
    theta = np.arctan2(obs[1], obs[0])
    omega = obs[2]
    u     = float(action_val)   # torque gia' ricevuto dal micro
    return -(theta * theta + 0.1 * omega * omega + 0.001 * u * u)

def recv_action_done(ser: serial.Serial):
    """Legge il frame di risposta. Ritorna (azione_decodificata, done) o None se incompleto."""
    frame = ser.read(_ACTION_FRAME_LEN)
    if len(frame) == _ACTION_FRAME_LEN and frame[0] == 0x02 and frame[-1] == 0x03:
        
        # Estrae i byte dell'azione e li spacchetta nel tipo corretto (float o int)
        action_bytes = frame[1:1 + ACTION_BYTE_SIZE]
        action_val = struct.unpack(_ACTION_STRUCT, action_bytes)[0]
        
        # Estrae il done flag (penultimo byte)
        done_flag = bool(frame[-2])
        
        return (action_val, done_flag)
    return None

def clear_console():
    """Pulisce il terminale"""
    os.system('cls' if os.name == 'nt' else 'clear')

# ------------------------------------------------------------
#  MAIN LOOP
# ------------------------------------------------------------
def main():
    print(f"[PC] Apertura seriale su {PORT} a {BAUDRATE} baud...")
    try:
        ser = serial.Serial(PORT, BAUDRATE, timeout=SER_TIMEOUT_S)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except Exception as e:
        print(f"Errore apertura seriale: {e}")
        return

    # Inizializza Ambiente
    env = gym.make(ENV_NAME) #, render_mode="human"
    
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
                env.render()
                now_ms = time.time() * 1000

                # 1. Ritrasmissione robusta
                if (not state_sent) or (now_ms - last_tx_ms > MAX_WAIT_MS):
                    send_state(ser, obs)
                    last_tx_ms = now_ms
                    state_sent = True

                # 2. Prova a leggere la risposta
                pkt = recv_action_done(ser)
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
        ser.close()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()