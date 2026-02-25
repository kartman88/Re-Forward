import serial
import struct
import time
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# --- CONFIGURAZIONE ---
SERIAL_PORT = '/dev/ttyACM0'  # <-- Modifica con la tua porta (es. COM3, /dev/ttyACM0)
BAUD_RATE = 115200
TIMEOUT = 1.0         # Timeout per il ciclo di Retry (1 secondo)
MAX_EPISODES = 2000
PLOT_EVERY_N_EPISODES = 5
CONSOLE_LINES = 5    # Quanti episodi mostrare nella dashboard

def send_observation(ser, obs):
    """Impacchetta 4 float in Little-Endian e invia con STX/ETX"""
    packed_obs = struct.pack('<4f', *obs)
    frame = bytes([0x02]) + packed_obs + bytes([0x03])
    ser.write(frame)
    ser.flush()

def receive_action(ser):
    """
    Legge la risposta del micro. 
    Se scatta il timeout (micro in training) o ci sono errori, ritorna None, None.
    """
    b = ser.read(1)
    if len(b) == 0:
        return None, None # Timeout scattato!
    
    if b[0] == 0x02: # STX trovato
        payload = ser.read(3)
        if len(payload) == 3 and payload[2] == 0x03:
            return payload[0], payload[1]
    
    return None, None

def clear_console():
    """Pulisce la console per creare l'effetto Dashboard"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    env = gym.make('CartPole-v1', render_mode="human")
    
    print(f"Apertura porta seriale {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except Exception as e:
        print(f"Errore apertura seriale: {e}")
        return

    print("Seriale aperta! Sincronizzazione in corso...\n")
    time.sleep(2)
    
    # Setup Grafici
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
    
    # --- VARIABILI DI STATO ---
    obs, _ = env.reset()
    current_ep_reward = 0
    episode_count = 0
    global_step = 0
    update_count = 0  
    last_update_time = 0.0
    
    try:
        while episode_count < MAX_EPISODES:
            
            t_start = time.time()
            action = None
            stm_done = None
            
            # ==========================================
            # LOOP ANTI-DEADLOCK (MECCANISMO DI RETRY)
            # ==========================================
            send_observation(ser, obs)
            while True:
                action, stm_done = receive_action(ser)
                
                if action is not None:
                    break # Ricevuto con successo, usciamo dal ciclo!
                
                # Se siamo qui, il receive_action è andato in TIMEOUT.
                # L'STM32 sta ignorando la seriale perché sta addestrando la rete.
                # Puliamo i buffer e reinviamo l'osservazione finché non si sveglia!
                ser.reset_input_buffer()
                send_observation(ser, obs)
            # ==========================================
            
            t_elapsed = time.time() - t_start
            
            # Se ha impiegato più di 1 secondo a rispondere, contiamo l'Update!
            if t_elapsed > 1.0:
                update_count += 1
                last_update_time = t_elapsed
                
            next_obs, reward, terminated, truncated, _ = env.step(action)
            current_ep_reward += reward
            global_step += 1
            
            done = terminated or truncated or (stm_done > 0)
            
            if done:
                episode_count += 1
                reward_history.append(current_ep_reward)
                moving_avg_queue.append(current_ep_reward)
                current_avg = np.mean(moving_avg_queue)
                moving_avg_history.append(current_avg)
                
                row = f"{episode_count:8d} | {current_ep_reward:7.1f} | {current_avg:10.1f} | {global_step:9d} | {update_count:9d}"
                console_history.append(row)
                
                # STAMPA DASHBOARD
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
                
                current_ep_reward = 0
                obs, _ = env.reset()
                
                if episode_count % PLOT_EVERY_N_EPISODES == 0:
                    line_raw.set_xdata(range(len(reward_history)))
                    line_raw.set_ydata(reward_history)
                    line_avg.set_xdata(range(len(moving_avg_history)))
                    line_avg.set_ydata(moving_avg_history)
                    ax.relim()
                    ax.autoscale_view()
                    fig.canvas.draw()
                    fig.canvas.flush_events()
            else:
                obs = next_obs
                
    except KeyboardInterrupt:
        clear_console()
        print("\nTraining interrotto dall'utente. Dati salvati.")
    finally:
        ser.close()
        env.close()
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    main()