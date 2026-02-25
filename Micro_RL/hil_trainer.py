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
PORT            = "/dev/ttyACM0"          # Sostituisci con la tua porta (es. /dev/ttyACM0 o COM3)
BAUDRATE        = 115200
SER_TIMEOUT_S   = 0.05            # Timeout velocissimo per la read della seriale
SEND_PERIOD_MS  = 30              # Pausa per evitare busy-loop (30 ms)
MAX_WAIT_MS     = 100             # Se in 100 ms non arriva risposta → ritrasmette l'osservazione

# Parametri Grafici
PLOT_EVERY_N_EPISODES = 5
CONSOLE_LINES = 5

# ------------------------------------------------------------
#  DIMENSIONI TASK
# ------------------------------------------------------------
OBS_DIM         = 4   # n° float32 (CartPole = 4)
ACTION_DIM      = 1   # n° byte per l'azione
_STATE_STRUCT   = f"<{OBS_DIM}f"
_ACTION_FRAME_LEN = 1 + ACTION_DIM + 1 + 1   # 0x02 + payload + done + 0x03

# ------------------------------------------------------------
#  FUNZIONI SERIALI (Dalla tua implementazione robusta)
# ------------------------------------------------------------
def send_state(ser: serial.Serial, obs: np.ndarray):
    """Invia gli OBS_DIM float32 in little-endian tra 0x02 e 0x03"""
    payload = struct.pack(_STATE_STRUCT, *obs.astype(np.float32))
    ser.write(b'\x02' + payload + b'\x03')

def recv_action_done(ser: serial.Serial):
    """Legge il frame di risposta. Ritorna (azione, done) o None se incompleto."""
    frame = ser.read(_ACTION_FRAME_LEN)
    if len(frame) == _ACTION_FRAME_LEN and frame[0] == 0x02 and frame[-1] == 0x03:
        action_bytes = frame[1:1 + ACTION_DIM]
        actions = list(action_bytes)
        done_flag = bool(frame[-2])
        return (actions[0] if ACTION_DIM == 1 else actions, done_flag)
    return None

def clear_console():
    """Pulisce il terminale"""
    os.system('cls' if os.name == 'nt' else 'clear')

def applica_fisica_personalizzata(env, m_pole=0.1, length=0.5):
    # Accediamo al core dell'ambiente
    u = env.unwrapped
    
    # Sovrascriviamo i parametri base
    u.masspole = m_pole
    u.length = length  # Ricorda: è la metà della lunghezza totale
    
    # Ricalcoliamo i parametri derivati necessari per le equazioni
    u.total_mass = u.masspole + u.masscart
    u.polemass_length = u.masspole * u.length
    
    print(f"Fisica aggiornata: Massa Asta={u.masspole}, Lunghezza={u.length*2}m")

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

    # Inizializza Ambiente. (Rimuovi render_mode="human" se vuoi farlo andare a velocità massima)
    env = gym.make("CartPole-v1", render_mode="human")
    applica_fisica_personalizzata(env, m_pole=0.5, length=1.0) #m_pole=0.1, length=0.5 default values
    
    # Setup Grafici (Matplotlib)
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Training PPO su STM32 - CartPole")
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

                # 1. Ritrasmissione robusta: invia lo stato ogni MAX_WAIT_MS
                if (not state_sent) or (now_ms - last_tx_ms > MAX_WAIT_MS):
                    send_state(ser, obs)
                    last_tx_ms = now_ms
                    state_sent = True

                # 2. Prova a leggere la risposta
                pkt = recv_action_done(ser)
                if pkt is None:
                    time.sleep(SEND_PERIOD_MS / 1000.0) # Freno per non impallare la CPU
                    continue

                # ---- FRAME VALIDO RICEVUTO ----
                actions, mcu_done = pkt
                state_sent = False 
                
                # 3. Controllo se il micro ha appena fatto training
                wait_time = time.time() - action_wait_start
                if wait_time > 0.5: # Se ha impiegato più di mezzo secondo, era in training!
                    update_count += 1
                    last_update_time = wait_time

                # 4. Step nell'ambiente fisico
                action_to_env = int(actions) if ACTION_DIM == 1 else np.array(actions)
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
            print("                 STM32 PPO - DASHBOARD")
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
        plt.show() # Lascia aperto il grafico alla fine

if __name__ == "__main__":
    main()