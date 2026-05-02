# PPO Implementation Status

## Completati

### Passo 1 — `ppo.h`: Iperparametri e `RolloutBuffer`
**File:** `Core/Inc/ppo.h`
- `#define` per tutti gli iperparametri PPO (`PPO_LR_ACTOR`, `PPO_LR_CRITIC`, `PPO_GAMMA`, `PPO_LAMBDA`, `PPO_CLIP_EPS`, `PPO_EPOCHS`, `PPO_BATCH_SIZE`, `ROLLOUT_STEPS`, `PPO_C1`, `PPO_C2`, `PPO_GRAD_CLIP`)
- Struct `RolloutBuffer` con tutti i campi: `states`, `log_probs_old`, `values`, `rewards`, `advantages`, `returns`, `actions`, `dones`, `head`, `size`, `obs_dim`
- Dichiarazioni di tutte le funzioni PPO (implementate in fasi successive)

### Passo 2 — `ppo.c`: Buffer management
**File:** `Core/Src/ppo.c`
- `rollout_buffer_init(buf, T, obs_dim)` — alloca tutti gli array con `malloc`, ritorna 0 se fallisce
- `rollout_buffer_push(buf, obs, action, reward, done, log_prob, value)` — scrive al `head` corrente e incrementa `head` e `size`

### Passo 3 — `ppo.c`: GAE e normalizzazione vantaggi
**File:** `Core/Src/ppo.c`
- `compute_gae(buf, last_value, gamma, lambda)` — loop inverso sul buffer; gestisce correttamente i confini episodio con la maschera `not_done`; usa `last_value` come bootstrap all'ultimo step
- `normalize_advantages(buf)` — zero-mean, unit-variance con `ε = 1e-8`

---

### Passo 4 — `actor_forward`: softmax + log-prob + entropy
**File:** `Core/Inc/neural_net.h`, `Core/Src/neural_net.c`
- Riusa `forward_q` (il last layer ha già `ACT_SOFTMAX`)
- `log_prob_out = log(probs[action])` — il softmax clampea a 1e-7 quindi no NaN
- `entropy_out = -Σ p·log(p)` sul vettore probs

### Passo 5 — `critic_forward`: output scalare V(s)
**File:** `Core/Inc/neural_net.h`, `Core/Src/neural_net.c`
- Riusa `forward_q` con `q_out=NULL` (output non copiato)
- Ritorna `critic->layers[last].out[0]` — il critic ha 1 neurone di output con `ACT_NONE`

### Passo 8 — `adam_update(net, lr)`: Adam con LR parametrico
**File:** `Core/Inc/neural_net.h`, `Core/Src/neural_net.c`
- `adam_update_single_layer` ora accetta `lr` come parametro (al posto del `#define LR`)
- `adam_update(net, lr)` — implementazione principale con LR parametrico
- `adam_optimizer_q(net)` — wrapper backward-compatibile che chiama `adam_update(net, LR)`
- Il codice DQN non cambia comportamento

---

### Passo 6 — `actor_backward`: clipped surrogate + entropy gradient
**File:** `Core/Inc/neural_net.h`, `Core/Src/neural_net.c`
- Firma: `actor_backward(actor, obs, action, advantage, ratio, clip_eps, entropy_coef)`
- `clip_eps` passato dal caller (= `PPO_CLIP_EPS`) per mantenere `neural_net.c` indipendente da `ppo.h`
- Condizione di clipping: gradiente = 0 se `(A≥0 && r>1+ε)` oppure `(A<0 && r<1-ε)`
- Gradiente logits: `−w·(I(k==a)−π_k) + c2·π_k·(log(π_k)+H)` dove `w = r·A` se non clipped, 0 altrimenti
- Usa il nuovo helper condiviso `backward_from_delta`

### Passo 7 — `critic_backward`: MSE value loss
**File:** `Core/Inc/neural_net.h`, `Core/Src/neural_net.c`
- Firma: `critic_backward(critic, obs, value_target)`
- Delta output: `V(s) − R_t` (derivata di `0.5·(V−R)²`)
- Una sola uscita scalare; usa `backward_from_delta` con puntatore a singolo float

### Refactoring `dqn_backward`
**File:** `Core/Src/neural_net.c`
- Estratto `backward_from_delta(net, input, delta_out)` come helper statico condiviso
- `dqn_backward` ora imposta solo il delta sparso e chiama il helper — nessun cambio di comportamento

---

### Passo 9 — `actor_sample_action`: sampling stocastico
**File:** `Core/Src/ppo.c`
- Chiama `forward_q` direttamente (senza azione) per ottenere il vettore `probs`
- Sampling categorico via inverse CDF con Fisher-Yates su float; fallback all'ultima azione se floating-point non raggiunge 1.0
- Calcola `log_prob_out = log(probs[action])` dopo il sample
- Chiama `critic_forward` per popolare `value_out` (opzionale, se non NULL)

### Passo 10 — `ppo_update`: ciclo K epoch mini-batch
**File:** `Core/Src/ppo.c`
- Arrays `idx[ROLLOUT_STEPS]` e `probs[PPO_N_ACTIONS]` dichiarati `static` per evitare pressione sullo stack embedded
- `zero_grad_q` su actor e critic all'inizio (sicurezza prima del primo mini-batch)
- Per ogni epoch: Fisher-Yates shuffle degli indici
- Per ogni mini-batch:
  1. Accumulo gradiente: `actor_forward` → ratio → `actor_backward`; `critic_forward` → `critic_backward`
  2. Media gradiente: divisione per `bsz` su tutti i layer di actor e critic separatamente
  3. `gradient_norm_q` su entrambi (clip = 0.5)
  4. `adam_update(actor, PPO_LR_ACTOR)` e `adam_update(critic, PPO_LR_CRITIC)` — azzera anche i gradienti
- I loop di averaging su actor e critic sono separati per gestire reti con numero di layer diverso

---

### Passo 11 — `main.c`: loop PPO completo
**File:** `Core/Src/main.c`

**Reti:**
- Actor: `[OBS_DIM, 64, 64, N_ACTIONS]` con `[ACT_RELU, ACT_RELU, ACT_SOFTMAX]`
- Critic: `[OBS_DIM, 64, 64, 1]` con `[ACT_RELU, ACT_RELU, ACT_NONE]`
- Rimossi: `TargetNetwork`, `ReplayBuffer`, `epsilon`, `train_step`, `num_episode`
- `dqn.h` mantenuto per `uart_recv_floats` e `uart_send_float_action`

**Loop principale (struttura invariata rispetto a DQN per compatibilità UART):**
1. Fase raccolta: `for (t < ROLLOUT_STEPS)` — riceve obs, computa `done`, pusha se `!first_step`, campiona azione con `actor_sample_action`, salva `prev_*`, segnala `done` all'ambiente via UART
2. Reset episodio: `first_step = 1` dopo un `done` (esattamente come DQN) — il primo step del nuovo episodio non viene pushato al buffer, evitando transizioni spurie cross-episodio
3. Fase update: bootstrap `last_val = first_step ? 0 : critic_forward(obs)` → `compute_gae` → `normalize_advantages` → `ppo_update` (solo se `buf.size >= PPO_BATCH_SIZE`)
4. LED LD2 (arancio) attivo durante `ppo_update` come in DQN

---

## Implementazione Completata

Tutti i 12 passi del piano sono implementati. Riepilogo file modificati/creati:

| File | Stato |
|------|-------|
| `Core/Inc/ppo.h` | Creato — iperparametri, `RolloutBuffer`, dichiarazioni |
| `Core/Src/ppo.c` | Creato — buffer, GAE, `actor_sample_action`, `ppo_update` |
| `Core/Inc/neural_net.h` | Modificato — aggiunte dichiarazioni PPO (passi 4,5,6,7,8) |
| `Core/Src/neural_net.c` | Modificato — `backward_from_delta`, `actor/critic_forward/backward`, `adam_update` |
| `Core/Src/main.c` | Modificato — loop DQN sostituito con loop PPO |

---

## Da Fare

| # | Passo | Note |
|---|-------|------|
| 12 | Testing: loss decrescente, no NaN | Verificare con breakpoint o UART su `rollout_buf.advantages`, ratio, policy loss |

---

## Note

- I passi 4-8 estendono `neural_net.h/c` senza rompere il codice DQN esistente — le funzioni DQN rimangono invariate.
- `adam_update(net, lr)` sostituirà `adam_optimizer_q(net)` usando LR come parametro invece del `#define LR`.
- `actor_sample_action` e `ppo_update` sono già dichiarate in `ppo.h` ma non ancora implementate.
- Il codice DQN in `main.c`, `dqn.h/c` rimane intatto fino al Passo 11.
