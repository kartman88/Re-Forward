"""
pc_dqn_trainer.py
------------------------------------------------------------
Training DQN in puro Python/NumPy del Pendulum-v1, pensato per confrontare
la learning curve con quella ottenuta sul microcontrollore (../tinyRL).

La struttura replica il più fedelmente possibile l'implementazione C che gira
sul micro (tinyRL/Core/Src/{main,dqn,neural_net,dense_layer}.c):

  * Rete (neural_net.h / main.c):
        topology {3, 64, 64, 5}, attivazioni [ReLU, ReLU, None]
        init Xavier/Glorot uniforme: W ~ U(-L, L), L = sqrt(6/(in+out)); bias = 0
  * Azioni (main.c): 5 coppie discrete {-2, -1, 0, +1, +2} Nm
  * Reward (main.c evaluate_reward_pendulum):
        -(theta^2 + 0.1*omega^2 + 0.001*u^2)   (identica a Pendulum-v1)
  * Iperparametri (neural_net.h):
        LR=1e-3, Adam(0.9, 0.999, 1e-8), GAMMA=0.99
        epsilon LINEARE 1.0 -> 0.05 su 10000 step globali (calc_epsilon)
        TARGET_UPDATE=100, REPLAY_MIN=300, REPLAY_SIZE=1000, BATCH=32
        MAX_STEPS_PER_EP=200
  * Training (dqn.c dqn_train):
        td_error = Q_online[a] - (r if done else r + GAMMA*max_a' Q_target)
        gradiente accumulato sul batch con fattore 1/batch_size,
        gradient clipping a norma globale 0.5, poi UN passo Adam.
  * done = solo truncation a 200 step (il pendolo non "termina" mai).

Salva un CSV "episode, reward" così da poter confrontare la curva con quelle
del micro in learning_curve_comparison/.
"""

import os
import csv
import argparse
from collections import deque

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# ------------------------------------------------------------
#  CONFIGURAZIONE (specchio di neural_net.h + main.c)
# ------------------------------------------------------------
ENV_NAME        = "Pendulum-v1"
OBS_DIM         = 3
N_ACTIONS       = 5
TOPOLOGY        = [OBS_DIM, 64, 64, N_ACTIONS]   # main.c:117

# Coppie discrete associate alle azioni (main.c:66)
PENDULUM_TORQUES = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)

# Iperparametri (neural_net.h)
LR              = 1e-3
BETA1           = 0.9
BETA2           = 0.999
EPS_ADAM        = 1e-8
GAMMA           = 0.99
EPSILON_START   = 1.0
EPSILON_END     = 0.05
EPSILON_DECAY   = 10000      # step globali su cui epsilon scende linearmente
TARGET_UPDATE   = 100        # passi di train tra due copie verso la target net
REPLAY_MIN      = 300        # esperienze minime prima di iniziare a trainare
REPLAY_SIZE     = 1000
BATCH_SIZE      = 32
MAX_STEPS_PER_EP = 200
GRAD_CLIP        = 0.5       # dqn.c gradient_norm_q

# Training / Log
MAX_EPISODES          = 100      # ~ stessa durata delle run del micro (training_mcu_N.csv)
PLOT_EVERY_N_EPISODES = 5

# Nome del file CSV di output (cambia qui per le run successive: training_pc_2.csv, ...).
# Convenzione coerente con i csv del micro (training_mcu_N.csv).
LOG_FILENAME = "training_pc_10.csv"
LOG_FOLDER   = "learning_curve_comparison"


# ------------------------------------------------------------
#  epsilon lineare (specchio di calc_epsilon in dqn.c)
# ------------------------------------------------------------
def calc_epsilon(step):
    t = step / EPSILON_DECAY
    if t > 1.0:
        t = 1.0
    return EPSILON_START + t * (EPSILON_END - EPSILON_START)


# ------------------------------------------------------------
#  Layer denso (specchio di DenseLayer + dense_layer.c)
# ------------------------------------------------------------
class DenseLayer:
    # attivazioni
    ACT_NONE = 0
    ACT_RELU = 1

    def __init__(self, in_dim, out_dim, activation, rng):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation

        # Init Xavier/Glorot uniforme (dense_layer.c init_layer_params)
        limit = np.sqrt(6.0 / (in_dim + out_dim))
        self.W = rng.uniform(-limit, limit, size=(out_dim, in_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)

        # gradienti + momenti Adam
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

        self.out = np.zeros(out_dim, dtype=np.float32)

    def forward(self, x):
        acc = self.W @ x + self.b
        if self.activation == self.ACT_RELU:
            acc = np.maximum(acc, 0.0)
        self.out = acc.astype(np.float32)
        return self.out


# ------------------------------------------------------------
#  Q-Network (specchio di QNetwork + neural_net.c)
#     online: forward + backward + Adam
# ------------------------------------------------------------
class QNetwork:
    def __init__(self, topology, activations, rng):
        self.layers = [
            DenseLayer(topology[i], topology[i + 1], activations[i], rng)
            for i in range(len(topology) - 1)
        ]
        self.num_layers = len(self.layers)
        self.adam_t = 0

    def forward(self, x):
        cur = np.asarray(x, dtype=np.float32)
        for ly in self.layers:
            cur = ly.forward(cur)
        return cur

    def zero_grad(self):
        for ly in self.layers:
            ly.dW.fill(0.0)
            ly.db.fill(0.0)

    def backward(self, input_vec, action, td_error):
        """
        Specchio di dqn_backward: gradiente di output sparso (solo 'action'),
        propagato all'indietro. Accumula in dW/db (richiede forward(input) appena
        eseguito, così layer.out contiene le attivazioni correnti).
        """
        out_dim_last = self.layers[-1].out_dim
        delta = np.zeros(out_dim_last, dtype=np.float32)
        delta[action] = td_error

        for l in range(self.num_layers - 1, -1, -1):
            ly = self.layers[l]
            inp = input_vec if l == 0 else self.layers[l - 1].out

            # Accumula gradienti del layer
            ly.dW += np.outer(delta, inp)
            ly.db += delta

            # Propaga delta al layer sottostante
            if l > 0:
                acc = ly.W.T @ delta
                h_prev = self.layers[l - 1].out
                if self.layers[l - 1].activation == DenseLayer.ACT_RELU:
                    acc = acc * (h_prev > 0.0).astype(np.float32)
                delta = acc

    def gradient_norm(self):
        """Clipping a norma globale GRAD_CLIP (dqn.c gradient_norm_q)."""
        gnorm_sq = 0.0
        for ly in self.layers:
            gnorm_sq += float(np.sum(ly.db * ly.db))
            gnorm_sq += float(np.sum(ly.dW * ly.dW))
        gnorm = np.sqrt(gnorm_sq)
        if gnorm > GRAD_CLIP:
            s = GRAD_CLIP / gnorm
            for ly in self.layers:
                ly.db *= s
                ly.dW *= s

    def adam_step(self):
        """Un passo Adam su tutti i layer (dqn.c adam_optimizer_q)."""
        self.adam_t += 1
        b1t = 1.0 - BETA1 ** self.adam_t
        b2t = 1.0 - BETA2 ** self.adam_t
        for ly in self.layers:
            # bias
            ly.mb = BETA1 * ly.mb + (1.0 - BETA1) * ly.db
            ly.vb = BETA2 * ly.vb + (1.0 - BETA2) * ly.db * ly.db
            ly.b -= LR * (ly.mb / b1t) / (np.sqrt(ly.vb / b2t) + EPS_ADAM)
            # pesi
            ly.mW = BETA1 * ly.mW + (1.0 - BETA1) * ly.dW
            ly.vW = BETA2 * ly.vW + (1.0 - BETA2) * ly.dW * ly.dW
            ly.W -= LR * (ly.mW / b1t) / (np.sqrt(ly.vW / b2t) + EPS_ADAM)


class TargetNetwork:
    """Solo forward; pesi copiati dalla online (neural_net.c forward_target)."""
    def __init__(self, topology, rng):
        # attivazioni come l'online (verranno copiate insieme ai pesi)
        acts = [DenseLayer.ACT_RELU, DenseLayer.ACT_RELU, DenseLayer.ACT_NONE]
        self.layers = [
            DenseLayer(topology[i], topology[i + 1], acts[i], rng)
            for i in range(len(topology) - 1)
        ]
        self.num_layers = len(self.layers)

    def forward(self, x):
        cur = np.asarray(x, dtype=np.float32)
        for ly in self.layers:
            cur = ly.forward(cur)
        return cur

    def copy_from(self, online: QNetwork):
        for tl, sl in zip(self.layers, online.layers):
            tl.W = sl.W.copy()
            tl.b = sl.b.copy()
            tl.activation = sl.activation


# ------------------------------------------------------------
#  Replay Buffer circolare (specchio di ReplayBuffer in dqn.c)
# ------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity, obs_dim, rng):
        self.capacity = capacity
        self.rng = rng
        self.state      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_state = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action     = np.zeros(capacity, dtype=np.int64)
        self.reward     = np.zeros(capacity, dtype=np.float32)
        self.done       = np.zeros(capacity, dtype=np.uint8)
        self.head = 0
        self.size = 0

    def push(self, s, action, reward, s_next, done):
        idx = self.head
        self.state[idx]      = s
        self.next_state[idx] = s_next
        self.action[idx]     = action
        self.reward[idx]     = reward
        self.done[idx]       = done
        self.head = (idx + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1


# ------------------------------------------------------------
#  Selezione azione epsilon-greedy (dqn.c dqn_select_action)
# ------------------------------------------------------------
def dqn_select_action(net, obs, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(0, N_ACTIONS))
    q = net.forward(obs)
    return int(np.argmax(q))


# ------------------------------------------------------------
#  Un passo di training su un batch (dqn.c dqn_train)
# ------------------------------------------------------------
def dqn_train(online, target, buf, batch_size, rng):
    inv_batch = 1.0 / float(batch_size)
    online.zero_grad()

    idxs = rng.integers(0, buf.size, size=batch_size)
    for idx in idxs:
        s      = buf.state[idx]
        s_next = buf.next_state[idx]
        act    = int(buf.action[idx])
        rew    = float(buf.reward[idx])
        dn     = buf.done[idx]

        q_online = online.forward(s)          # popola le attivazioni per il backward
        q_tgt    = target.forward(s_next)
        max_qt   = float(np.max(q_tgt))

        target_val = rew if dn else rew + GAMMA * max_qt
        td_error   = q_online[act] - target_val

        online.backward(s, act, td_error * inv_batch)

    online.gradient_norm()
    online.adam_step()


# ------------------------------------------------------------
#  MAIN LOOP (specchio del while(1) in main.c, ma guidato dall'env locale)
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DQN PC trainer per Pendulum (mirror del micro)")
    parser.add_argument("--episodes", type=int, default=MAX_EPISODES)
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed per riproducibilità (default: casuale)")
    parser.add_argument("--csv", type=str,
                        default=os.path.join(LOG_FOLDER, LOG_FILENAME),
                        help="File CSV di output (episode, reward). "
                             "Default: LOG_FOLDER/LOG_FILENAME definiti in cima al file.")
    parser.add_argument("--no-plot", action="store_true", help="Disabilita il grafico live")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    env = gym.make(ENV_NAME)
    activations = [DenseLayer.ACT_RELU, DenseLayer.ACT_RELU, DenseLayer.ACT_NONE]
    online = QNetwork(TOPOLOGY, activations, rng)
    target = TargetNetwork(TOPOLOGY, rng)
    target.copy_from(online)
    replay = ReplayBuffer(REPLAY_SIZE, OBS_DIM, rng)

    # Setup CSV (stesse colonne dei csv del micro)
    csv_dir = os.path.dirname(args.csv)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    csv_file = open(args.csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "reward"])

    # Setup grafico
    if not args.no_plot:
        plt.ion()
        fig, ax_reward = plt.subplots(figsize=(10, 6))
    reward_history = []
    moving_avg_history = []
    moving_avg_queue = deque(maxlen=20)

    step_total = 0
    train_step = 0

    for episode in range(1, args.episodes + 1):
        reset_seed = (args.seed + episode) if args.seed is not None else None
        obs, _ = env.reset(seed=reset_seed)
        total_reward = 0.0

        for step_in_ep in range(MAX_STEPS_PER_EP):
            # epsilon basato sullo step globale (come il micro)
            epsilon = calc_epsilon(step_total)
            action = dqn_select_action(online, obs, epsilon, rng)
            torque = PENDULUM_TORQUES[action]

            next_obs, reward, terminated, truncated, _ = env.step(
                np.array([torque], dtype=np.float32))

            # done = truncation a fine episodio (il pendolo non termina mai da solo)
            done = 1 if (step_in_ep == MAX_STEPS_PER_EP - 1) else 0

            # Memorizza esperienza (azione discreta) e allena
            replay.push(obs, action, reward, next_obs, done)

            if replay.size >= REPLAY_MIN:
                dqn_train(online, target, replay, BATCH_SIZE, rng)
                train_step += 1
                if train_step % TARGET_UPDATE == 0:
                    target.copy_from(online)

            obs = next_obs
            total_reward += reward
            step_total += 1

        # Log episodio
        reward_history.append(total_reward)
        moving_avg_queue.append(total_reward)
        moving_avg_history.append(np.mean(moving_avg_queue))

        csv_writer.writerow([episode, total_reward])
        csv_file.flush()

        print(f"Ep {episode:4d} | Reward {total_reward:9.2f} | "
              f"MovAvg(20) {moving_avg_history[-1]:9.2f} | eps {calc_epsilon(step_total):.3f}")

        if not args.no_plot and episode % PLOT_EVERY_N_EPISODES == 0:
            ax_reward.clear()
            ax_reward.set_title(f"Learning Curve (PC DQN) - {ENV_NAME}")
            ax_reward.set_xlabel("Episodio")
            ax_reward.set_ylabel("Reward Totale")
            ax_reward.plot(reward_history, label="Reward")
            ax_reward.plot(moving_avg_history, label="Media Mobile (20)")
            ax_reward.legend()
            plt.pause(0.01)

    csv_file.close()
    env.close()

    print(f"\n[PC] Training completato. CSV salvato in: {args.csv}")
    if not args.no_plot:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
