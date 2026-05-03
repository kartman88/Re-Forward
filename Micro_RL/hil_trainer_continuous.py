import time
import struct
import serial
import os
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
ENV_NAME        = "Hopper-v4"    # Nome ambiente Gymnasium
OBS_DIM         = 11             # Dimensione spazio osservazione
ACTION_DIM      = 3              # Continuo: n. dims  |  Discreto: n. classi
ACTION_RANGE    = (-1.0, 1.0)    # (low, high) di ogni azione — usato nei plot
ENV_KWARGS      = {"terminate_when_unhealthy": False, "render_mode": "human"}  # kwargs extra per gym.make

# Parametri Grafici
PLOT_EVERY_N_EPISODES = 5
CONSOLE_LINES = 5

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
    """Invia gli OBS_DIM float32 in little-endian tra 0x02 e 0x03"""
    payload = struct.pack(_STATE_STRUCT, *obs.astype(np.float32))
    ser.write(b'\x02' + payload + b'\x03')

def recv_action_done(ser: serial.Serial):
    """Legge il frame di risposta. Ritorna (actions, done) o None se incompleto.
    Per azioni continue: actions è un np.ndarray di shape (ACTION_DIM,).
    Per azioni discrete: actions è un int.
    """
    frame = ser.read(_ACTION_FRAME_LEN)
    if len(frame) == _ACTION_FRAME_LEN and frame[0] == 0x02 and frame[-1] == 0x03:
        action_bytes = frame[1:1 + ACTION_BYTE_SIZE]
        if USE_CONTINUOUS_ACTIONS:
            # Spacchetta ACTION_DIM float — funziona per qualsiasi N
            actions = np.array(struct.unpack(_ACTION_STRUCT, action_bytes), dtype=np.float32)
        else:
            actions = struct.unpack(_ACTION_STRUCT, action_bytes)[0]  # int discreto
        done_flag = bool(frame[-2])
        return (actions, done_flag)
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

                # 4. Step nell'ambiente fisico
                # actions è già un np.ndarray (continuo) o int (discreto) — pronto per gym
                obs, r, terminated, truncated, _ = env.step(actions)

                # Se gym termina prima del micro (es. caduta Hopper), resettiamo
                # l'env internamente e inviamo il nuovo obs al micro come passo successivo.
                # Il done è comandato SOLO dal microcontrollore.
                if terminated or truncated:
                    obs, _ = env.reset()

                done = mcu_done
                
                current_ep_reward += r
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
        env.close()
        ser.close()
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()