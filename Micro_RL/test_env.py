#!/usr/bin/env python3
import gymnasium as gym
import numpy as np

ENV_NAME          = "Ant-v5"
NUM_EPISODES      = 5            # quante prove vuoi fare
MAX_STEPS         = 1000         # safety-cap per episodio

def main():
    env = gym.make(ENV_NAME, render_mode="human")   # rimuovi render_mode se non vuoi la finestra
    for ep in range(NUM_EPISODES):
        obs, _ = env.reset(seed=np.random.randint(0, 2**32 - 1))
        total_reward = 0.0

        for t in range(1, MAX_STEPS + 1):
            action = env.action_space.sample()            # pura random-policy
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:                   # Gym ha segnalato fine episodio
                print(f"Ep {ep} finished after {t} steps, R = {total_reward}")
                break
        else:
            print(f"Ep {ep} hit step-limit ({MAX_STEPS}), R = {total_reward}")

    env.close()

if __name__ == "__main__":
    main()
