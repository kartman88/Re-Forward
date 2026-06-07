"""Trainer PPO interamente in Python/NumPy, speculare all'implementazione C su STM32.

Serve da baseline di confronto: replica rete, inizializzazione, Adam, GAE, clipping,
gaussiana diagonale con tanh-squash e iperparametri definiti in tinyRL/Core
(neural_net.c, ppo.c, ppo.h, main.c) sulla task Hopper-v4. Salva una learning curve
in CSV con lo stesso formato dei trainer MCU/PC, cosi' da sovrapporre le curve e
dimostrare che il codice del micro impara come la controparte Python.
"""

import os
import csv
import argparse
import numpy as np
import gymnasium as gym
from collections import deque

# ------------------------------------------------------------
#  TASK (Hopper-v4) - speculare a neural_net.h / main.c
# ------------------------------------------------------------
ENV_NAME    = "Hopper-v4"
OBS_DIM     = 11
N_ACT_DIMS  = 3
MAX_STEPS_PER_EP = 1000          # MAX_STEPS_PER_EP in main.c

# ------------------------------------------------------------
#  IPERPARAMETRI PPO - da ppo.h
# ------------------------------------------------------------
PPO_LR_ACTOR   = 5e-4
PPO_LR_CRITIC  = 1e-3
PPO_GAMMA      = 0.99
PPO_LAMBDA     = 0.95
PPO_CLIP_EPS   = 0.2
PPO_EPOCHS     = 8
PPO_BATCH_SIZE = 64
ROLLOUT_STEPS  = 2048
PPO_C1         = 0.5
PPO_C2         = 0.01            # entropy coef (non usato nel ramo continuous del C)
PPO_GRAD_CLIP  = 0.5

# Adam - da neural_net.h
BETA1    = 0.9
BETA2    = 0.999
EPS_ADAM = 1e-8

# Sigma schedule (stato-indipendente) - da ppo.h
PPO_SIGMA_INIT    = 1.5
PPO_SIGMA_MIN     = 0.1
PPO_SIGMA_N_STEPS = 500000
PPO_SIGMA_DECAY   = np.log(PPO_SIGMA_INIT / PPO_SIGMA_MIN) / PPO_SIGMA_N_STEPS

PPO_USE_TANH_SQUASH = True       # PPO_USE_TANH_SQUASH in neural_net.h
LOG_2PI = 1.8378770664093453

# Precisione di calcolo. Il micro gira in float32; usare --float32 per un
# confronto fair (vedi piano: diagnostica precisione).
DTYPE = np.float64

# ------------------------------------------------------------
#  LOGGING CSV
# ------------------------------------------------------------
LOG_FOLDER   = "learning_curve_comparison"
LOG_FILENAME = "training_ppo_10.csv"


# ============================================================
#  RETE DENSA - speculare a dense_layer.c / neural_net.c
# ============================================================
class DenseLayer:
    # activation: "relu" | "tanh" | "none"
    def __init__(self, in_dim, out_dim, activation, rng):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        # Inizializzazione Xavier/Glorot uniforme, come init_layer_params()
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.W = rng.uniform(-limit, limit, size=(out_dim, in_dim)).astype(DTYPE)
        self.b = np.zeros(out_dim, dtype=DTYPE)
        self.out = np.zeros(out_dim, dtype=DTYPE)
        # gradienti e momenti Adam
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)


class Network:
    def __init__(self, topology, activations, rng):
        self.layers = [DenseLayer(topology[i], topology[i + 1], activations[i], rng)
                       for i in range(len(topology) - 1)]
        self.adam_t = 0

    def forward(self, x):
        cur = np.asarray(x, dtype=DTYPE)
        for ly in self.layers:
            z = ly.W @ cur + ly.b
            if ly.activation == "relu":
                ly.out = np.maximum(z, 0.0)
            elif ly.activation == "tanh":
                ly.out = np.tanh(z)
            else:
                ly.out = z
            cur = ly.out
        return cur

    def zero_grad(self):
        for ly in self.layers:
            ly.dW[:] = 0.0
            ly.db[:] = 0.0

    def backward_from_delta(self, x, delta):
        """Accumula dW/db e propaga delta all'indietro (come backward_from_delta in C)."""
        x = np.asarray(x, dtype=DTYPE)
        delta = np.asarray(delta, dtype=DTYPE)
        for l in range(len(self.layers) - 1, -1, -1):
            ly = self.layers[l]
            inp = x if l == 0 else self.layers[l - 1].out
            ly.db += delta
            ly.dW += np.outer(delta, inp)
            if l > 0:
                prev = self.layers[l - 1]
                acc = ly.W.T @ delta
                if prev.activation == "relu":
                    acc = np.where(prev.out > 0.0, acc, 0.0)
                elif prev.activation == "tanh":
                    acc = acc * (1.0 - prev.out * prev.out)
                delta = acc

    def clip_grad(self):
        gnorm_sq = 0.0
        for ly in self.layers:
            gnorm_sq += np.sum(ly.dW * ly.dW) + np.sum(ly.db * ly.db)
        gnorm = np.sqrt(gnorm_sq)
        if gnorm > PPO_GRAD_CLIP:
            s = PPO_GRAD_CLIP / gnorm
            for ly in self.layers:
                ly.dW *= s
                ly.db *= s

    def adam_update(self, lr):
        self.adam_t += 1
        b1t = 1.0 - BETA1 ** self.adam_t
        b2t = 1.0 - BETA2 ** self.adam_t
        for ly in self.layers:
            ly.mb = BETA1 * ly.mb + (1.0 - BETA1) * ly.db
            ly.vb = BETA2 * ly.vb + (1.0 - BETA2) * ly.db * ly.db
            ly.b -= lr * (ly.mb / b1t) / (np.sqrt(ly.vb / b2t) + EPS_ADAM)
            ly.mW = BETA1 * ly.mW + (1.0 - BETA1) * ly.dW
            ly.vW = BETA2 * ly.vW + (1.0 - BETA2) * ly.dW * ly.dW
            ly.W -= lr * (ly.mW / b1t) / (np.sqrt(ly.vW / b2t) + EPS_ADAM)


# ============================================================
#  REWARD e DONE - speculari a main.c (Hopper-v4)
# ============================================================
def compute_reward(obs, action):
    healthy = 1.0 if (obs[0] >= 0.7 and -0.2 <= obs[1] <= 0.2) else 0.0
    ctrl_cost = float(np.sum(action * action))
    return healthy + obs[5] - 1e-3 * ctrl_cost


def is_done(obs, step_in_ep):
    if step_in_ep >= MAX_STEPS_PER_EP:
        return True
    if obs[0] < 0.7 or obs[1] < -0.2 or obs[1] > 0.2:
        return True
    return False


# ============================================================
#  PPO - sampling, GAE, update (speculari a ppo.c)
# ============================================================
def actor_sample_action(actor, critic, obs, log_sigma, rng):
    mu = actor.forward(obs)
    log_prob = 0.0
    action = np.empty(N_ACT_DIMS)
    for i in range(N_ACT_DIMS):
        ls = log_sigma[i]
        eps = rng.standard_normal()
        z = mu[i] + np.exp(ls) * eps
        if PPO_USE_TANH_SQUASH:
            a = np.tanh(z)
            action[i] = a
            log_prob += -0.5 * (eps * eps + 2.0 * ls + LOG_2PI) \
                        - np.log(1.0 - a * a + 1e-6)
        else:
            action[i] = z
            log_prob += -0.5 * (eps * eps + 2.0 * ls + LOG_2PI)
    value = float(critic.forward(obs)[0])
    return action, log_prob, value


def actor_forward_continuous(actor, obs, action, log_sigma):
    mu = actor.forward(obs)
    log_prob = 0.0
    for i in range(N_ACT_DIMS):
        ls = log_sigma[i]
        if PPO_USE_TANH_SQUASH:
            a = np.clip(action[i], -1.0 + 1e-6, 1.0 - 1e-6)
            z = np.arctanh(a)
            diff = z - mu[i]
            log_prob += -0.5 * (diff * diff / np.exp(2.0 * ls) + 2.0 * ls + LOG_2PI) \
                        - np.log(1.0 - a * a + 1e-6)
        else:
            diff = action[i] - mu[i]
            log_prob += -0.5 * (diff * diff / np.exp(2.0 * ls) + 2.0 * ls + LOG_2PI)
    return log_prob


def actor_backward_continuous(actor, obs, action, log_sigma, advantage, ratio):
    mu = actor.layers[-1].out
    clipped = (advantage >= 0.0 and ratio > 1.0 + PPO_CLIP_EPS) or \
              (advantage < 0.0 and ratio < 1.0 - PPO_CLIP_EPS)
    w = 0.0 if clipped else ratio * advantage
    delta = np.empty(N_ACT_DIMS)
    for i in range(N_ACT_DIMS):
        if PPO_USE_TANH_SQUASH:
            a = np.clip(action[i], -1.0 + 1e-6, 1.0 - 1e-6)
            z = np.arctanh(a)
            delta[i] = -w * (z - mu[i]) / np.exp(2.0 * log_sigma[i])
        else:
            delta[i] = -w * (action[i] - mu[i]) / np.exp(2.0 * log_sigma[i])
    actor.backward_from_delta(obs, delta)


def critic_backward(critic, obs, value_target):
    delta = PPO_C1 * (critic.layers[-1].out[0] - value_target)
    critic.backward_from_delta(obs, np.array([delta]))


def compute_gae(states, rewards, values, dones, last_value):
    n = len(rewards)
    advantages = np.zeros(n)
    returns = np.zeros(n)
    gae = 0.0
    for t in range(n - 1, -1, -1):
        not_done = 0.0 if dones[t] else 1.0
        v_next = (last_value * not_done) if t == n - 1 else (values[t + 1] * not_done)
        delta = rewards[t] + PPO_GAMMA * v_next - values[t]
        gae = delta + PPO_GAMMA * PPO_LAMBDA * not_done * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
    return advantages, returns


def ppo_update(actor, critic, states, actions, log_probs_old, advantages, returns,
               log_sigma, rng):
    n = len(states)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    idx = np.arange(n)
    actor.zero_grad()
    critic.zero_grad()
    for _ in range(PPO_EPOCHS):
        rng.shuffle(idx)
        for start in range(0, n, PPO_BATCH_SIZE):
            batch = idx[start:start + PPO_BATCH_SIZE]
            inv_bsz = 1.0 / len(batch)
            actor.zero_grad()
            critic.zero_grad()
            for t in batch:
                lp_new = actor_forward_continuous(actor, states[t], actions[t], log_sigma)
                ratio = np.exp(np.clip(lp_new - log_probs_old[t], -10.0, 10.0))
                actor_backward_continuous(actor, states[t], actions[t], log_sigma,
                                          advantages[t], ratio)
                critic.forward(states[t])
                critic_backward(critic, states[t], returns[t])
            for ly in actor.layers:
                ly.dW *= inv_bsz
                ly.db *= inv_bsz
            for ly in critic.layers:
                ly.dW *= inv_bsz
                ly.db *= inv_bsz
            actor.clip_grad()
            critic.clip_grad()
            actor.adam_update(PPO_LR_ACTOR)
            critic.adam_update(PPO_LR_CRITIC)


def sigma_decay(log_sigma):
    min_ls = np.log(PPO_SIGMA_MIN)
    log_sigma -= PPO_SIGMA_DECAY
    np.maximum(log_sigma, min_ls, out=log_sigma)


# ============================================================
#  MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Trainer PPO PC-only (baseline Hopper)")
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--csv", type=str,
                        default=os.path.join(LOG_FOLDER, LOG_FILENAME))
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--float32", action="store_true",
                        help="Calcola in float32 come il micro (confronto fair).")
    args = parser.parse_args()

    if args.float32:
        global DTYPE
        DTYPE = np.float32
        print("[PC] Precisione: float32 (come il micro)")

    rng = np.random.default_rng(args.seed)
    env = gym.make(ENV_NAME, terminate_when_unhealthy=False)
    reset_kwargs = {"seed": args.seed} if args.seed is not None else {}

    actor = Network([OBS_DIM, 64, 64, N_ACT_DIMS], ["tanh", "tanh", "none"], rng)
    critic = Network([OBS_DIM, 64, 64, 1], ["relu", "relu", "none"], rng)
    log_sigma = np.full(N_ACT_DIMS, np.log(PPO_SIGMA_INIT), dtype=DTYPE)

    # Buffer di rollout
    b_states, b_actions, b_rewards = [], [], []
    b_values, b_logp, b_dones = [], [], []

    # Range delle azioni per i plot (tanh-squash -> [-1, 1])
    ACTION_RANGE = (-1.0, 1.0)
    all_actions = [[] for _ in range(N_ACT_DIMS)]   # una lista per ogni dim di azione

    if not args.no_plot:
        import matplotlib.pyplot as plt
        plt.ion()
        n_plots = 1 + N_ACT_DIMS
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        ax_reward = axes[0]
        ax_actions = list(axes[1:])   # un subplot per ogni dim di azione

        # --- Subplot Reward ---
        ax_reward.set_title(f"Learning Curve (PPO) - {ENV_NAME}")
        ax_reward.set_xlabel("Episodi")
        ax_reward.set_ylabel("Reward")
        line_raw, = ax_reward.plot([], [], 'b-', alpha=0.3, label="Reward Grezzo")
        line_avg, = ax_reward.plot([], [], 'r-', linewidth=2, label="Media Mobile (20 ep)")
        ax_reward.legend()

        # --- Subplots Azioni (uno per dim) ---
        for i, ax in enumerate(ax_actions):
            ax.set_title(f"Azione a{i}")
            ax.set_xlabel("Valore")
            ax.set_ylabel("Frequenza")
            ax.set_xlim(*ACTION_RANGE)

    csv_dir = os.path.dirname(args.csv)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    csv_file = open(args.csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "step", "reward", "episode_steps"])

    reward_history, avg_history = [], []
    avg_queue = deque(maxlen=20)
    global_step = 0
    rollout_count = 0

    obs, _ = env.reset(**reset_kwargs)
    step_in_ep = 0
    ep_reward = 0.0
    ep_steps = 0
    episode = 0

    try:
        while global_step < args.steps:
            done = is_done(obs, step_in_ep)
            action, log_prob, value = actor_sample_action(actor, critic, obs, log_sigma, rng)

            # Step nell'ambiente con azione clippata in [-1, 1] (come main.c)
            act_clipped = np.clip(action, -1.0, 1.0)
            next_obs, _, terminated, truncated, _ = env.step(act_clipped.astype(np.float32))

            # Reward calcolato sul micro (compute_reward), non quello di gym
            reward = compute_reward(obs, action)

            # Raccogli le azioni (clippate in [-1, 1]) per gli istogrammi
            for i in range(N_ACT_DIMS):
                all_actions[i].append(float(act_clipped[i]))

            b_states.append(obs.copy())
            b_actions.append(action.copy())
            b_rewards.append(reward)
            b_values.append(value)
            b_logp.append(log_prob)
            b_dones.append(1 if done else 0)

            ep_reward += reward
            ep_steps += 1
            global_step += 1
            rollout_count += 1

            if terminated or truncated:
                next_obs, _ = env.reset(**reset_kwargs)

            if done:
                episode += 1
                reward_history.append(ep_reward)
                avg_queue.append(ep_reward)
                cur_avg = float(np.mean(avg_queue))
                avg_history.append(cur_avg)
                csv_writer.writerow([episode, global_step, ep_reward, ep_steps])
                csv_file.flush()
                print(f"Ep {episode:4d} | step {global_step:7d} | reward {ep_reward:9.2f} "
                      f"| media(20) {cur_avg:9.2f} | sigma {np.exp(log_sigma[0]):.3f}")

                if not args.no_plot and episode % 5 == 0:
                    line_raw.set_data(range(1, len(reward_history) + 1), reward_history)
                    line_avg.set_data(range(1, len(avg_history) + 1), avg_history)
                    ax_reward.relim(); ax_reward.autoscale_view()

                    for i, ax in enumerate(ax_actions):
                        ax.clear()
                        ax.set_title(f"Azione a{i}")
                        ax.set_xlabel("Valore")
                        ax.set_ylabel("Frequenza")
                        ax.hist(all_actions[i], bins=30, color='green',
                                alpha=0.7, edgecolor='black')
                        ax.set_xlim(*ACTION_RANGE)

                    fig.canvas.draw(); fig.canvas.flush_events()

                obs, _ = env.reset(**reset_kwargs)
                step_in_ep = 0
                ep_reward = 0.0
                ep_steps = 0
            else:
                obs = next_obs
                step_in_ep += 1

            # Training quando il rollout e' pieno (come ppo_step in C)
            if rollout_count >= ROLLOUT_STEPS:
                last_val = 0.0 if (step_in_ep == 0) else float(critic.forward(obs)[0])
                states = np.array(b_states)
                actions = np.array(b_actions)
                advantages, returns = compute_gae(
                    states, np.array(b_rewards), np.array(b_values),
                    np.array(b_dones), last_val)
                if len(states) >= PPO_BATCH_SIZE:
                    ppo_update(actor, critic, states, actions, np.array(b_logp),
                               advantages, returns, log_sigma, rng)
                    sigma_decay(log_sigma)
                b_states, b_actions, b_rewards = [], [], []
                b_values, b_logp, b_dones = [], [], []
                rollout_count = 0

    except KeyboardInterrupt:
        print("\n[PC] Interrotto dall'utente.")
    finally:
        csv_file.close()
        env.close()
        print(f"[PC] Log salvato in {args.csv}")
        if not args.no_plot:
            import matplotlib.pyplot as plt
            plt.ioff()
            plt.show()


if __name__ == "__main__":
    main()
