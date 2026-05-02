# PPO Implementation Plan — tinyRL

## 1. Overview

Partendo dall'infrastruttura DQN esistente, l'obiettivo è implementare PPO (Proximal Policy Optimization) per il task Pendulum sull'STM32H7. PPO è un algoritmo on-policy Actor-Critic: raccoglie traiettorie con la policy corrente, calcola i vantaggi tramite GAE, e aggiorna policy e value function con loss clippeata.

---

## 2. DQN vs PPO — Differenze Chiave

| Aspetto | DQN (attuale) | PPO (target) |
|---|---|---|
| **Tipo** | Off-policy, value-based | On-policy, Actor-Critic |
| **Output rete** | Q(s,a) per ogni azione | π(a\|s) per Actor; V(s) per Critic |
| **Memory** | Replay buffer (campionamento casuale) | Rollout buffer (traiettorie ordinate) |
| **Esplorazione** | ε-greedy | Stocastica (sampling da softmax) |
| **Policy** | Implicita (argmax Q) | Esplicita (probabilità per azione) |
| **Target network** | Sì (per stabilità Q) | No |
| **Training** | Mini-batch da replay | K epoch su rollout completo |
| **Loss** | TD error (MSE su Q) | Clipped surrogate + Value loss + Entropy |

---

## 3. Algoritmo PPO (Discrete Actions)

### 3.1 Ciclo principale

```
Per ogni iterazione:
  1. Colleziona T step con policy corrente π_θ_old
     - Per ogni step: obs -> Actor -> sample action -> env -> reward
     - Salva: (s_t, a_t, r_t, done_t, log_π_old(a_t|s_t), V(s_t))
  
  2. Calcola ritorni e vantaggi (GAE)
     - δ_t = r_t + γ·V(s_{t+1})·(1-done_t) - V(s_t)   (TD residual)
     - A_t = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}             (GAE)
     - R_t = A_t + V(s_t)                                  (V target)
     - Normalizza A_t: (A_t - mean) / (std + ε)
  
  3. Per K epoch (shuffle mini-batch):
     a. Forward Actor: π_θ(·|s_t) -> log_prob_new(a_t), entropy
     b. Ratio: r_t(θ) = exp(log_prob_new - log_prob_old)
     c. Policy loss: L_clip = -mean[ min(r·A, clip(r, 1-ε_c, 1+ε_c)·A) ]
     d. Value loss: L_vf = 0.5 · mean[ (V_θ(s_t) - R_t)² ]
     e. Entropy bonus: L_ent = -mean[ H[π_θ(·|s_t)] ]
     f. Total loss: L = L_clip + c1·L_vf + c2·L_ent
     g. Backward Actor + Critic, gradient clipping, Adam update
```

### 3.2 Formule gradient

**Policy gradient (discrete softmax):**
```
dL_clip/dlogit_k = -(clipped_weight * A_t) * (I(k==a_t) - π_k)
```
dove `clipped_weight = min(r, clip(r)) * sign(A_t)` gestito con la maschera di clipping.

**Value gradient:**
```
dL_vf/dV = V(s_t) - R_t
```

**Entropy gradient:**
```
dH/dlogit_k = π_k * (log(π_k) + H[π])
```

---

## 4. Strutture Dati

### 4.1 ActorNetwork

Riusa `QNetwork` (stesse strutture `DenseLayer` + Adam) con output softmax invece di lineare.

```c
// In neural_net.h — riusa QNetwork, cambia solo topologia e attivazioni
// topology: [OBS_DIM, 64, 64, N_ACTIONS]
// activations: [ACT_RELU, ACT_RELU, ACT_SOFTMAX]
typedef QNetwork ActorNetwork;
```

### 4.2 CriticNetwork

Rete separata con output scalare V(s).

```c
// topology: [OBS_DIM, 64, 64, 1]
// activations: [ACT_RELU, ACT_RELU, ACT_NONE]
typedef QNetwork CriticNetwork;
```

> **Nota:** Non servirà più `TargetNetwork` (non usata in PPO).

### 4.3 RolloutBuffer

Sostituisce `ReplayBuffer`. Dati ordinati temporalmente, non campionamento casuale.

```c
// In ppo.h
typedef struct {
    float  *states;        // [T * obs_dim]  osservazioni
    float  *log_probs_old; // [T]            log π_old(a_t|s_t)
    float  *values;        // [T]            V(s_t) al momento della raccolta
    float  *rewards;       // [T]            reward ricevuto
    float  *advantages;    // [T]            GAE calcolato post-rollout
    float  *returns;       // [T]            R_t = A_t + V(s_t)
    uint32_t *actions;     // [T]            azione discreta eseguita
    uint8_t  *dones;       // [T]            flag terminazione episodio
    uint32_t head;         // indice di scrittura corrente
    uint32_t size;         // step raccolti (fino a ROLLOUT_STEPS)
    uint32_t obs_dim;
} RolloutBuffer;
```

---

## 5. Iperparametri PPO

```c
// In ppo.h
#define PPO_LR_ACTOR      3e-4f   // Learning rate Actor
#define PPO_LR_CRITIC     1e-3f   // Learning rate Critic
#define PPO_GAMMA         0.99f   // Discount factor
#define PPO_LAMBDA        0.95f   // GAE lambda
#define PPO_CLIP_EPS      0.2f    // Clipping range per il ratio
#define PPO_EPOCHS        4       // Epoch di update per rollout
#define PPO_BATCH_SIZE    64      // Mini-batch per epoch
#define ROLLOUT_STEPS     256     // Step da raccogliere per aggiornamento
#define PPO_C1            0.5f    // Coefficiente value loss
#define PPO_C2            0.01f   // Coefficiente entropy bonus
#define PPO_GRAD_CLIP     0.5f    // Max gradient norm
#define PPO_OBS_DIM       OBS_DIM // Riusa define esistente
#define PPO_N_ACTIONS     N_ACTIONS
```

---

## 6. File da Creare / Modificare

### 6.1 File Nuovi

| File | Contenuto |
|---|---|
| `Core/Inc/ppo.h` | Iperparametri PPO, struct `RolloutBuffer`, dichiarazioni funzioni PPO |
| `Core/Src/ppo.c` | Implementazione: `rollout_buffer_init`, `rollout_buffer_push`, `compute_gae`, `ppo_update` |

### 6.2 File Modificati

| File | Modifica |
|---|---|
| `Core/Inc/neural_net.h` | Aggiungere `actor_forward` (restituisce logits + log_prob + entropy), `critic_forward`, `actor_backward`, `critic_backward`, learning rate separati per actor/critic |
| `Core/Src/neural_net.c` | Implementare le nuove funzioni forward/backward; riusare `adam_optimizer_q` rinominandola o parametrizzandola |
| `Core/Src/main.c` | Rimpiazzare loop DQN con loop PPO: raccolta rollout → `compute_gae` → `ppo_update` |
| `Core/Inc/reinforce.h` | Può rimanere placeholder o essere rimosso |
| `Core/Src/reinforce.c` | Può rimanere placeholder o essere rimosso |

### 6.3 File Invariati (riusati as-is)

- `dense_layer.h / dense_layer.c` — nessuna modifica
- `utils.h / utils.c` — nessuna modifica
- Funzioni UART in `dqn.c` — `uart_recv_floats`, `uart_send_float_action` (spostate o lasciate)

---

## 7. Nuove Funzioni da Implementare

### 7.1 `neural_net.h / .c` — Estensioni Actor-Critic

```c
// Forward Actor: ritorna log-prob dell'azione scelta e l'entropia della distribuzione
void actor_forward(QNetwork *actor, float *obs, float *probs_out,
                   float *log_prob_out, uint32_t action, float *entropy_out);

// Forward Critic: ritorna V(s) scalare
float critic_forward(QNetwork *critic, float *obs);

// Backward Actor: gradiente della policy loss (clipped surrogate + entropy)
void actor_backward(QNetwork *actor, float *obs, uint32_t action,
                    float advantage, float ratio, float log_prob_old,
                    float entropy_coef);

// Backward Critic: gradiente della value loss (MSE)
void critic_backward(QNetwork *critic, float *obs, float value_target);

// Adam con LR parametrico (per usare LR diversi per actor e critic)
void adam_update(QNetwork *net, float lr);
```

### 7.2 `ppo.h / ppo.c` — Core PPO

```c
// Alloca il rollout buffer
int rollout_buffer_init(RolloutBuffer *buf, uint32_t T, uint32_t obs_dim);

// Salva un'esperienza nel buffer durante la raccolta
void rollout_buffer_push(RolloutBuffer *buf, float *obs, uint32_t action,
                         float reward, uint8_t done, float log_prob, float value);

// Calcola GAE e ritorni dopo T step (richiede valore finale V(s_T))
void compute_gae(RolloutBuffer *buf, float last_value, float gamma, float lambda);

// Normalizza i vantaggi (zero mean, unit variance)
void normalize_advantages(RolloutBuffer *buf);

// Un'iterazione PPO: K epoch di mini-batch update su actor e critic
void ppo_update(QNetwork *actor, QNetwork *critic, RolloutBuffer *buf);

// Sampling stocastico da distribuzione softmax
uint32_t actor_sample_action(QNetwork *actor, float *obs,
                              float *log_prob_out, float *value_out,
                              QNetwork *critic);
```

### 7.3 `main.c` — Nuovo Loop PPO

```c
// Pseudo-codice del nuovo main loop
while (1) {
    // --- Fase 1: Raccolta Rollout ---
    rollout_buf.head = 0;
    rollout_buf.size = 0;

    for (uint32_t t = 0; t < ROLLOUT_STEPS; t++) {
        uart_recv_floats(&huart3, obs, OBS_DIM, 100);

        float log_prob, value_critic;
        uint32_t action = actor_sample_action(&actor, obs, &log_prob,
                                               &value_critic, &critic);

        if (!first_step) {
            float reward = evaluate_reward_pendulum(obs, prev_action);
            uint8_t done = (step_in_ep >= MAX_STEPS_PER_EP);
            rollout_buffer_push(&rollout_buf, prev_obs, prev_action,
                                reward, done, prev_log_prob, prev_value);
            if (done) step_in_ep = 0;
            else      step_in_ep++;
        }

        uart_send_float_action(&huart3, PENDULUM_TORQUES[action], 0, 100);

        memcpy(prev_obs, obs, OBS_DIM * sizeof(float));
        prev_action   = action;
        prev_log_prob = log_prob;
        prev_value    = value_critic;
        first_step    = 0;
    }

    // Valore bootstrap finale per GAE
    float last_val = critic_forward(&critic, obs);
    compute_gae(&rollout_buf, last_val, PPO_GAMMA, PPO_LAMBDA);
    normalize_advantages(&rollout_buf);

    // --- Fase 2: Update PPO ---
    ppo_update(&actor, &critic, &rollout_buf);
}
```

---

## 8. Analisi Memoria (STM32H7)

| Componente | Dimensione |
|---|---|
| Actor weights (3→64→64→5, con W/mW/vW/dW) | ~74 KB |
| Actor biases | ~2 KB |
| Critic weights (3→64→64→1, con W/mW/vW/dW) | ~70 KB |
| Critic biases | ~2 KB |
| RolloutBuffer (256 step × (3+1+1+1+1+1+1) float) | ~7 KB |
| Stack/misc | ~5 KB |
| **Totale stimato** | **~160 KB** |

> STM32H7 ha 512 KB di AXI SRAM accessibile via `malloc`. Il DTCM (128 KB, più veloce) non basta da solo — le reti vanno allocate su AXI SRAM (comportamento di default di `malloc` con il linker script STM32CubeIDE standard). Non ci sono problemi di memoria.

---

## 9. Passi Implementativi Ordinati

| # | Passo | File | Note |
|---|---|---|---|
| 1 | Definire `RolloutBuffer` e iperparametri PPO | `ppo.h` | Struct + `#define` |
| 2 | Implementare `rollout_buffer_init` e `rollout_buffer_push` | `ppo.c` | Simile a `replay_buffer_*` |
| 3 | Implementare `compute_gae` e `normalize_advantages` | `ppo.c` | Loop inverso sul buffer |
| 4 | Aggiungere `actor_forward` con softmax e log-prob | `neural_net.c` | Riusa softmax esistente |
| 5 | Aggiungere `critic_forward` con output scalare | `neural_net.c` | Forward pass con 1 output |
| 6 | Aggiungere `actor_backward` con clipped surrogate | `neural_net.c` | Gradiente policy loss |
| 7 | Aggiungere `critic_backward` con MSE | `neural_net.c` | Gradiente value loss |
| 8 | Parametrizzare `adam_update(net, lr)` | `neural_net.c` | Riusa logica esistente |
| 9 | Implementare `actor_sample_action` | `ppo.c` | Sample da softmax + log-prob |
| 10 | Implementare `ppo_update` (K epoch, mini-batch) | `ppo.c` | Loop shuffle + update |
| 11 | Riscrivere `main.c` con loop PPO | `main.c` | Rollout → GAE → update |
| 12 | Testing: verificare loss decrescente, no NaN | — | Log su UART o breakpoint |

---

## 10. Rischi e Mitigazioni

| Rischio | Mitigazione |
|---|---|
| **Instabilità training** (ratio esplode) | Gradient clipping già presente; `PPO_CLIP_EPS=0.2` limita il ratio |
| **Advantages collassano a zero** | Normalizzazione con `ε=1e-8` nel denominatore |
| **NaN in log-prob** | Clamp probabilità softmax: `max(prob, 1e-8f)` prima del log |
| **Memoria insufficiente** | Reti allocate su AXI SRAM via `malloc`; profilo con `heap_stats` |
| **Troppo lento per ROLLOUT_STEPS** | Ridurre a 128 step se il loop UART diventa il bottleneck |
