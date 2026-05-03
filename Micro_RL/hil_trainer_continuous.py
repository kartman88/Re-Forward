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
USE_CONTINUOUS_ACTIONS = False    # Metti a False per tornare al CartPole Discreto!

# --- DEBUG DATA DUMP ---
DEBUG_MODE   = True               # Riceve e salva i dati di debug dal micro dopo il 1° episodio
NET_TOPOLOGY = [4, 64, 2]         # Deve corrispondere alla topologia in main.c

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
    ACTION_BYTE_SIZE = 4   # L'azione è un float (4 byte)
    _ACTION_STRUCT = "<f"  # Little-endian float
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

def recv_initial_weights(ser, topology):
    """Legge frame 0x05...0x06, ricostruisce i pesi layer by layer."""
    old_to = ser.timeout
    ser.timeout = 6.0
    while ser.read(1) != b'\x05':
        pass
    n_floats = struct.unpack('<I', ser.read(4))[0]
    data = np.frombuffer(ser.read(n_floats * 4), dtype=np.float32)
    ser.read(1)  # ETX 0x06
    ser.timeout = old_to
    result, idx = {}, 0
    for l in range(len(topology) - 1):
        in_d, out_d = topology[l], topology[l + 1]
        result[f'W{l}'] = data[idx:idx + out_d * in_d].reshape(out_d, in_d).copy()
        idx += out_d * in_d
        result[f'b{l}'] = data[idx:idx + out_d].copy()
        idx += out_d
    return result

def recv_debug_batch(ser, obs_dim):
    """Legge frame 0x07...0x08 con stati, returns, advantages, logits e loss."""
    old_to = ser.timeout
    ser.timeout = 6.0
    while ser.read(1) != b'\x07':
        pass
    step_count = struct.unpack('<I', ser.read(4))[0]
    out_dim    = struct.unpack('<I', ser.read(4))[0]
    states   = np.frombuffer(ser.read(step_count * obs_dim * 4), dtype=np.float32).reshape(step_count, obs_dim).copy()
    returns  = np.frombuffer(ser.read(step_count * 4), dtype=np.float32).copy()
    adv_norm = np.frombuffer(ser.read(step_count * 4), dtype=np.float32).copy()
    logits   = np.frombuffer(ser.read(step_count * out_dim * 4), dtype=np.float32).reshape(step_count, out_dim).copy()
    loss     = struct.unpack('<f', ser.read(4))[0]
    ser.read(1)  # ETX 0x08
    ser.timeout = old_to
    return {'states': states, 'raw_returns': returns,
            'norm_advantages': adv_norm, 'logits': logits, 'loss': loss}

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
    env = gym.make(ENV_NAME, render_mode="human")
    
    # Setup Grafici (Matplotlib)
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"Training PPO su STM32 - {ENV_NAME}")
    ax.set_xlabel("Episodi")
    ax.set_ylabel("Reward")
    
    reward_history = []
    moving_avg_history = []
    moving_avg_queue = deque(maxlen=20)
    console_history = deque(maxlen=CONSOLE_LINES)
    
    line_raw, = ax.plot([], [], 'b-', alpha=0.3, label='Reward Grezzo')
    line_avg, = ax.plot([], [], 'r-', linewidth=2, label='Media Mobile (20 ep)')
    ax.legend()

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
                
                # Step nell'ambiente fisico
                obs, r, terminated, truncated, _ = env.step(action_to_env)
                
                # LA VERA MAGIA: Ci fidiamo SOLO del microcontrollore!
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

            # Ricezione debug data dopo il primo episodio
            if DEBUG_MODE and episode_count == 1:
                print("[PC] Ricezione debug data dal micro...")
                weights = recv_initial_weights(ser, NET_TOPOLOGY)
                batch   = recv_debug_batch(ser, OBS_DIM)
                np.savez('debug_data.npz', **weights, **batch)
                print(f"[PC] debug_data.npz salvato  |  steps={batch['states'].shape[0]}  loss={batch['loss']:.4f}")

            # Resetta ambiente per il nuovo episodio
            obs, _ = env.reset()
            state_sent = False

            # Aggiorna il Grafico ogni N episodi
            if episode_count % PLOT_EVERY_N_EPISODES == 0:
                line_raw.set_xdata(range(len(reward_history)))
                line_raw.set_ydata(reward_history)
                line_avg.set_xdata(range(len(moving_avg_history)))
                line_avg.set_ydata(moving_avg_history)
                ax.relim()
                ax.autoscale_view()
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