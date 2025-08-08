#!/usr/bin/env python3
"""
Real‑time Acrobot runner (PC ↔ MCU) — the Gymnasium environment never
signals episode end; only the microcontroller decides when to reset.
The loop advances at a fixed frequency (STEP_PERIOD_MS) and re‑uses the
last action if the MCU hasn't sent a new one.

Key changes vs previous version
-------------------------------
* IgnoreDoneWrapper: when Gym would emit `terminated`/`truncated`, we
  immediately auto‑reset *internally* but return `terminated=False` to
  the caller.  This prevents Gym from blocking future `step()` calls
  yet lets the "world" continue running.
* The main loop now checks end‑of‑episode **solely** through the flag
  coming from the MCU.

Edit STEP_PERIOD_MS, PORT, etc. to match your setup.
"""

import time
import struct
import threading
import queue
import serial
import gymnasium as gym
import numpy as np

# ------------------------------------------------------------
#  CONFIGURATION
# ------------------------------------------------------------
PORT              = "/dev/ttyACM0"
BAUDRATE          = 115200
SER_TIMEOUT_S     = 0.0          # non‑bloccante
STEP_PERIOD_MS    = 20           # 20=50Hz
GAMMA             = 0.99         # solo per logging

# ------------------------------------------------------------
#  TASK DIMENSIONS (Acrobot‑v1)
# ------------------------------------------------------------
OBS_DIM    = 6
ACTION_DIM = 1   # torque {-1, 0, +1}

_STATE_STRUCT      = f"<{OBS_DIM}f"
_ACTION_FRAME_LEN  = 1 + ACTION_DIM + 1 + 1   # 0x02 + payload + done + 0x03

# ------------------------------------------------------------
#  SERIAL HELPERS
# ------------------------------------------------------------

def send_state(ser: serial.Serial, obs: np.ndarray):
    """Invia OBS_DIM float32 tra 0x02 e 0x03"""
    if obs.shape[-1] != OBS_DIM:
        raise ValueError(f"Expected obs dim {OBS_DIM}, got {obs.shape[-1]}")
    payload = struct.pack(_STATE_STRUCT, *obs.astype(np.float32))
    ser.write(b"\x02" + payload + b"\x03")


def parse_action_frame(frame: bytes):
    """Decodifica 0x02 <ACTION_DIM byte> <done> 0x03"""
    if (
        len(frame) == _ACTION_FRAME_LEN
        and frame[0] == 0x02
        and frame[-1] == 0x03
    ):
        action_bytes = frame[1 : 1 + ACTION_DIM]
        action = (
            action_bytes[0] if ACTION_DIM == 1 else list(action_bytes)
        )
        done_flag = bool(frame[-2])
        return action, done_flag
    return None

# ------------------------------------------------------------
#  WRAPPER: IGNORA IL "DONE" DI GYM
# ------------------------------------------------------------

class IgnoreDoneWrapper(gym.Wrapper):
    """Restituisce sempre terminated=False, truncated=False.

    Se Gym segnala fine episodio, facciamo subito reset interno così che
    la simulazione prosegua senza blocchi.  Dal punto di vista del
    micro, il mondo "salta" a un nuovo stato ma non si ferma.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            # auto‑reset e azzero flag
            obs, _ = self.env.reset()
            terminated = truncated = False
        return obs, reward, terminated, truncated, info

# ------------------------------------------------------------
#  SERIAL READER (THREAD)
# ------------------------------------------------------------

def serial_worker(ser: serial.Serial, action_q: queue.Queue, mcu_done: threading.Event):
    """Legge async e tiene l'ultima azione."""

    while True:
        frame = ser.read(_ACTION_FRAME_LEN)
        if not frame:
            time.sleep(0.001)
            continue
        pkt = parse_action_frame(frame)
        if pkt is None:
            continue
        action, done_flag = pkt
        # tieni solo l'ultima azione
        while not action_q.empty():
            try:
                action_q.get_nowait()
            except queue.Empty:
                break
        action_q.put(action)
        if done_flag:
            mcu_done.set()

# ------------------------------------------------------------
#  MAIN LOOP
# ------------------------------------------------------------

def main():
    ser = serial.Serial(PORT, BAUDRATE, timeout=SER_TIMEOUT_S)
    print(f"[PC] Serial opened on {PORT} @ {BAUDRATE} baud")

    base_env = gym.make("Acrobot-v1", render_mode="human")
    env = IgnoreDoneWrapper(base_env)

    action_q      = queue.Queue(maxsize=1)
    mcu_done_flag = threading.Event()

    threading.Thread(
        target=serial_worker,
        args=(ser, action_q, mcu_done_flag),
        daemon=True,
    ).start()

    obs, _ = env.reset()
    send_state(ser, obs)

    current_action = 0
    step_period_s  = STEP_PERIOD_MS / 1000.0
    next_tick      = time.perf_counter() + step_period_s

    episode, step, G = 0, 0, 0.0

    try:
        while True:
            # sincronizza clock
            now = time.perf_counter()
            sleep_s = next_tick - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            next_tick += step_period_s

            # aggiorna azione se disponibile
            try:
                current_action = action_q.get_nowait()
            except queue.Empty:
                pass

            # passo env
            obs, r, _, _, _ = env.step(int(current_action))
            env.render()
            send_state(ser, obs)

            G = r + GAMMA * G
            step += 1

            # check fine episodio controllato dal micro
            if mcu_done_flag.is_set():
                print(f"[PC] Ep {episode} finished in {step} steps, G≈{G:.2f}")
                episode += 1
                obs, _ = env.reset()
                send_state(ser, obs)
                mcu_done_flag.clear()
                step, G        = 0, 0.0
                current_action = 0
                next_tick      = time.perf_counter() + step_period_s
                while not action_q.empty():  # svuota coda
                    try:
                        action_q.get_nowait()
                    except queue.Empty:
                        break

    except KeyboardInterrupt:
        print("\n[PC] Interrupted by user")

    finally:
        env.close()
        ser.close()
        print("[PC] Resources released, bye!")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
