# REINFORCE su CartPole-v1 con PyTorch + Gymnasium
# pip install torch gymnasium[classic-control]
import math
import random
from collections import deque
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from environment_log import episode_list

# --------------------
# Config
# --------------------
ENV_ID = "CartPole-v1"
GAMMA = 0.99
LR = 0.02
HIDDEN = 64
EPISODES = 150
EVAL_EVERY = 50
SEED = random.randint(1, 798456132) #88x, 89--9886, 90--99x, 80x, 376347--986, 88x, 666--987, 888-x, 887x, 111x
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
print(DEVICE)

episode_list = []
reward_list = []
n_experiment = 1

# --------------------
# Utils
# --------------------
def set_seed(env, seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

def discounted_returns(rewards, gamma=GAMMA):
    """Calcola i ritorni scontati G_t con normalizzazione (stabilizza il training)."""
    G = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        G.append(running)
    G.reverse()
    G = torch.tensor(G, dtype=torch.float32, device=DEVICE)
    # Normalizzazione (solo se var>0)
    if len(G) > 1:
        std = G.std(unbiased=False)
        if std > 0:
            G = (G - G.mean()) / (std + 1e-8)
    return G

# --------------------
# Policy Network
# --------------------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, act_dim),  # logits
        )

    def forward(self, x):
        logits = self.net(x)
        return logits

    def act(self, obs):
        """Restituisce azione ~ π(a|s) e log_prob corrispondente."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits = self.forward(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action).squeeze(0)

# --------------------
# Training
# --------------------
def train():
    env = gym.make(ENV_ID)
    set_seed(env, SEED)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNet(obs_dim, act_dim).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    recent_returns = deque(maxlen=EVAL_EVERY)
    best_eval = -float("inf")

    for ep in range(1, EPISODES + 1):
        step_count = 0
        obs, info = env.reset(seed=SEED)
        log_probs, rewards = [], []
        done = False

        while not done:
            action, logp = policy.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

            log_probs.append(logp)
            rewards.append(reward)
            obs = next_obs

        # REINFORCE loss: L = - E [ G_t * log π(a_t|s_t) ]
        G = discounted_returns(rewards, GAMMA)
        log_probs_t = torch.stack(log_probs)
        loss = -(log_probs_t * G).sum()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        ep_return = sum(rewards)
        recent_returns.append(ep_return)

        episode_list.append(ep)
        reward_list.append(step_count)

        if ep % 10 == 0:
            avg10 = np.mean(list(recent_returns)[-10:])
            print(f"Ep {ep:4d} | Return: {ep_return:6.1f} | Avg10: {avg10:6.1f}")

        # Valutazione periodica (greedy rispetto alla policy stocastica: azione argmax logits)
        if ep % EVAL_EVERY == 0:
            eval_return = evaluate(policy)
            print(f"==> Eval @ Ep {ep}: return medio {eval_return:.1f}")
            if eval_return > best_eval:
                best_eval = eval_return
                #torch.save(policy.state_dict(), "policy_cartpole_reinforce.pt")
                #print("Model salvato: policy_cartpole_reinforce.pt")

    env.close()
    print("Training terminato.")
    np.savetxt(
        f'pc_learning_curve{n_experiment}.csv',
        np.column_stack([episode_list, reward_list]),
        delimiter=",",
        header="episode,reward",
        comments="",
        fmt=["%d", "%d"]
    )

# --------------------
# Evaluation
# --------------------
def evaluate(policy, episodes=5):
    env = gym.make(ENV_ID)
    total = 0.0
    with torch.no_grad():
        for k in range(episodes):
            obs, info = env.reset(seed=SEED)
            done = False
            ep_ret = 0.0
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                logits = policy(obs_t)
                action = torch.argmax(logits, dim=-1).item()
                obs, r, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_ret += r
            total += ep_ret
    env.close()
    return total / episodes

if __name__ == "__main__":
    time_list = []
    for i in range(10):
        start_time = time.time()
        train()
        end_time = time.time() - start_time
        time_list.append(end_time)
        episode_list = []
        reward_list = []
        n_experiment += 1
    print(f'FINE TEMPO: {time_list}')
