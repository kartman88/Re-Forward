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
TIMEOUT = 1.0         # Timeout breve: ritenteremo silenziosamente
MAX_EPISODES = 2000
PLOT_EVERY_N_EPISODES = 5
CONSOLE_LINES = 15    # Quanti episodi mostrare nella dashboard

def send_observation(ser, obs):
    """Impacchetta 4 float in Little-Endian e invia con STX/ETX"""
    packed_obs = struct.pack('<4f', *obs)
    frame = bytes([0x02]) + packed_obs + bytes([0x03])
    ser.write(frame)
    ser.flush()

def receive_action(ser):
    """Aspetta silenziosamente l'STX. Ignora i timeout (il micro sta trainando)."""
    while True:
        b = ser.read(1)
        if len(b) > 0 and b[0] == 0x02: # STX Trovato!
            break
        # Se len(b) == 0, la seriale è andata in timeout. 
        # Il loop continua silenziosamente senza stampare errori.
            
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
    console_history = deque(maxlen=CONSOLE_LINES) # Memoria per la tabella
    
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
            send_observation(ser, obs)
            
            # Cronometriamo quanto ci mette a rispondere
            t_start = time.time()
            action, stm_done = receive_action(ser)
            t_elapsed = time.time() - t_start
            
            # Se ha impiegato più di mezzo secondo, era impegnato nella Backpropagation!
            if t_elapsed > 0.5:
                update_count += 1
                last_update_time = t_elapsed
            
            if action is None:
                continue # Pacchetto sporco, ritentiamo senza fare rumore
                
            next_obs, reward, terminated, truncated, _ = env.step(action)
            current_ep_reward += reward
            global_step += 1
            
            # Sincronizzazione dei fine-episodio tra PC e STM32
            done = terminated or truncated or (stm_done > 0)
            
            if done:
                episode_count += 1
                reward_history.append(current_ep_reward)
                moving_avg_queue.append(current_ep_reward)
                current_avg = np.mean(moving_avg_queue)
                moving_avg_history.append(current_avg)
                
                # Format riga tabella
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
                
                # Reset
                current_ep_reward = 0
                obs, _ = env.reset()
                
                # Aggiorna il grafico ogni N episodi
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