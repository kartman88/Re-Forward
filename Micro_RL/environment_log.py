import time
import struct
import serial
import gymnasium as gym
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------------------------------------------------------------
#  CONFIG
# ------------------------------------------------------------
PORT            = "/dev/ttyACM0"
BAUDRATE        = 115200
SER_TIMEOUT_S   = 0.05            # timeout per read della seriale
SEND_PERIOD_MS  = 30              # ogni quanto riprovo a spedire (30 ms)
MAX_WAIT_MS     = 100             # se in 150 ms non arriva risposta → ritrasmetto
GAMMA           = 0.99

# ------------------------------------------------------------
#  TASK DIMENSIONS (scalabilità)
# ------------------------------------------------------------
#  Modifica questi due valori se cambi environment o protocollo
OBS_DIM         = 4   # n° float32 da inviare al MCU (CartPole obs = 4)
ACTION_DIM      = 1   # n° byte che il MCU invia come azione

# Struct di pack per lo stato: "<4f" se OBS_DIM==4 ecc.
_STATE_STRUCT   = f"<{OBS_DIM}f"
_STATE_SIZE     = struct.calcsize(_STATE_STRUCT)

# Numero di byte da ricevere in frame azione: start + ACTION_DIM + done + end
_ACTION_FRAME_LEN = 1 + ACTION_DIM + 4 + 1 + 1   # 0x02 + payload + ret + done + 0x03

#LOG STRUCTURES
time_list = []
step_list = []

# ------------------------------------------------------------
#  SERIAL HELPERS
# ------------------------------------------------------------
def send_state(ser: serial.Serial, obs: np.ndarray):
    """
    Invia OBS_DIM float32 little‑endian racchiusi tra 0x02 … 0x03
    Frame: 0x02 <payload> 0x03
    """
    if obs.shape[-1] != OBS_DIM:
        raise ValueError(f"Expected obs dim {OBS_DIM}, got {obs.shape[-1]}")
    payload = struct.pack(_STATE_STRUCT, *obs.astype(np.float32))
    ser.write(b'\x02' + payload + b'\x03')


def recv_action_done(ser: serial.Serial):
    """
    Legge 0x02 <ACTION_DIM byte payload> <done flag> 0x03
    Restituisce (actions, done) dove:
        - actions è un int se ACTION_DIM==1, altrimenti una lista di int
        - done    è bool
    Se il frame è incompleto/errato → None
    """
    global step_list, time_list
    frame = ser.read(1)
    if len(frame) == 1 and frame[0] == 0x02:
        frame_len = ACTION_DIM + 1 + 1  #payload + done + 0x03
        frame = ser.read(frame_len)
        if len(frame) == frame_len and frame[-1] == 0x03:
            action_bytes = frame[0:ACTION_DIM]
            actions = list(action_bytes)
            done_flag = bool(frame[-2])
            return (actions[0] if ACTION_DIM == 1 else actions, done_flag)
    elif len(frame) == 1 and frame[0] == 0x01:
        frame_len = 4 + 4 + 1 #dt + step + 0x04
        frame = ser.read(frame_len)
        if len(frame) == frame_len and frame[-1] == 0x04:
            dt = struct.unpack('>I', frame[0:4])[0]
            dt = dt /1_000_000
            step = struct.unpack('>I', frame[4:8])[0]
            time_list.append(dt)
            step_list.append(step)
            #print(f'time: {dt}, step: {step}')

    return None

# ------------------------------------------------------------
#  MAIN LOOP
# ------------------------------------------------------------
def main():
    ser = serial.Serial(PORT, BAUDRATE, timeout=SER_TIMEOUT_S)
    print(f"[PC] Serial opened on {PORT} @ {BAUDRATE} baud")

    env = gym.make("CartPole-v1") #CartPole-v1 MountainCar-v0, , render_mode="human"

    episode = 0
    obs, _  = env.reset(seed=88) #88
    state_sent   = False
    last_tx_ms   = 0.0

    try:
        while True:                        # ----- episodi -----
            step = 0
            G  = 0.0
            done = False
            episode += 1

            while not done:                # ----- passi -----
                env.render()
                now_ms = time.time() * 1000

                # 1. invia lo stato al massimo ogni MAX_WAIT_MS
                if (not state_sent) or (now_ms - last_tx_ms > MAX_WAIT_MS):
                    send_state(ser, obs)
                    last_tx_ms = now_ms
                    state_sent = True

                # 2. prova a leggere la risposta
                pkt = recv_action_done(ser)
                if pkt is None:
                    time.sleep(SEND_PERIOD_MS / 1000.0)  # evita busy-loop
                    continue

                # ---- frame valido ----
                actions, mcu_done = pkt
                state_sent = False                      # pronto per il prossimo stato

                # Se l'ambiente richiede azione scalare usa actions altrimenti passa array
                action_to_env = int(actions) if ACTION_DIM == 1 else np.array(actions)
                obs, r, terminated, truncated, _ = env.step(action_to_env)
                done = mcu_done  # or terminated or truncated

                G  = r + GAMMA * G
                step += 1


            print(f"[PC] Episode {episode} finished in {step} steps, G≈{G:.2f}")

            obs, _      = env.reset(seed=88) #88
            state_sent  = False            # invierò subito il nuovo stato

    except KeyboardInterrupt:
        print("\n[PC] Interrupted by user")

    finally:
        global step_list, time_list
        env.close()
        ser.close()
        print("[PC] Resources released, bye!")
        np.savetxt(
            "finish_episode_time_complete.csv",
            np.column_stack([step_list, time_list]),
            delimiter=",",
            header="step,time",
            comments="",
            fmt=["%d", "%d"]
        )
        print(f'TEMPO MEDIO: {np.mean(time_list)}')

        idx = np.argsort(step_list)
        step_list = np.array(step_list)[idx]
        time_list = np.array(time_list)[idx]
        step_list, first_idx = np.unique(step_list, return_index=True)
        time_list = time_list[first_idx]
        np.savetxt(
            "finish_episode_time_filtered.csv",
            np.column_stack([step_list, time_list]),
            delimiter=",",
            header="step,time",
            comments="",
            fmt=["%d", "%d"]
        )

        plt.figure()
        plt.plot(step_list, time_list)
        plt.xlabel('Number of Steps')
        plt.ylabel('Time (ms)')
        plt.title('Finish Episode Time')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Salva l'immagine (PNG ad alta risoluzione)
        plt.savefig('finish_episode_time.png', dpi=300)

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
