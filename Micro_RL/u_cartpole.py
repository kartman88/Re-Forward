import time
import struct
import serial
import gymnasium as gym
import numpy as np

# ------------------------------------------------------------
#  CONFIG
# ------------------------------------------------------------
PORT            = "/dev/ttyACM0"
BAUDRATE        = 115200
SER_TIMEOUT_S   = 0.05            # timeout per read della seriale
SEND_PERIOD_MS  = 30              # ogni quanto riprovo a spedire (30 ms)
MAX_WAIT_MS     = 150             # se in 150 ms non arriva risposta → ritrasmetto
GAMMA           = 0.99

# ------------------------------------------------------------
#  SERIAL HELPERS
# ------------------------------------------------------------
def send_state(ser: serial.Serial, obs: np.ndarray):
    """Spedisce 4 float32 little-endian racchiusi tra 0x02 … 0x03"""
    payload = struct.pack("<4f", *obs.astype(np.float32))
    ser.write(b'\x02' + payload + b'\x03')

def recv_action_done(ser: serial.Serial):
    """Legge 4 byte: 0x02 <action> <done> 0x03.  Ritorna (action, done) o None"""
    frame = ser.read(4)
    if len(frame) == 4 and frame[0] == 0x02 and frame[3] == 0x03:
        return frame[1], bool(frame[2])
    return None

# ------------------------------------------------------------
#  MAIN LOOP
# ------------------------------------------------------------
def main():
    ser = serial.Serial(PORT, BAUDRATE, timeout=SER_TIMEOUT_S)
    print(f"[PC] Serial opened on {PORT} @ {BAUDRATE} baud")

    env = gym.make("CartPole-v1", render_mode="human")

    episode = 0
    obs, _  = env.reset()
    state_sent   = False
    last_tx_ms   = 0.0

    try:
        while True:                        # ----- episodi -----
            step = 0
            G    = 0.0
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
                action, mcu_done = pkt
                state_sent = False                      # pronto per il prossimo stato

                obs, r, terminated, truncated, _ = env.step(int(action))
                done = mcu_done #or terminated or truncated

                G    = r + GAMMA * G
                step += 1

            print(f"[PC] Episode {episode} finished in {step} steps, G≈{G:.2f}")

            obs, _      = env.reset()      # nuovo episodio
            state_sent  = False            # invierò subito il nuovo stato

    except KeyboardInterrupt:
        print("\n[PC] Interrupted by user")

    finally:
        env.close()
        ser.close()
        print("[PC] Resources released, bye!")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
