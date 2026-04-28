# Piano Implementativo DQN — STM32H7

## Contesto

Migrazione da PPO (actor-critic con shared backbone) a DQN su STM32H743ZI.
Base di partenza: `dense_layer.h/c` e `utils.h/c` riutilizzati intatti.
Solo azioni **discrete** (DQN standard).

---

## Architettura DQN

```
Osservazione s
      │
      ▼
┌─────────────────┐      ┌─────────────────┐
│  Online Network │      │  Target Network │
│  Q(s, a) per    │      │  Q(s, a) per    │
│  ogni azione a  │      │  ogni azione a  │
└────────┬────────┘      └─────────────────┘
         │                        ▲
         │   copy_weights()       │
         └────────────────────────┘
                ogni TARGET_UPDATE step di training

Output: vettore [Q(s,a0), Q(s,a1), ..., Q(s,aN)]
Azione: argmax (greedy) oppure random (esplorazione ε-greedy). Usiamo ε-greedy con decadimento dell'ε
Loss:   MSE( Q_online(s,a) − [ r + γ · max_a' Q_target(s',a') ] )²
```

La rete target **non ha Adam moments** (solo W e b): risparmio ~60% di memoria
rispetto al tenerla come rete completa con ottimizzatore.

---

## Struttura File

| File | Stato | Note |
|---|---|---|
| `dense_layer.h/c` | **Invariato** | Riusato al 100% |
| `utils.h/c` | **Invariato** | Riusato al 100% |
| `neural_net.h/c` | **Riscritto** | PPO → QNetwork |
| `reinforce.h/c` | **Rimosso** | Sostituito da `dqn.h/c` |
| `dqn.h/c` | **Nuovo** | ReplayBuffer + training loop |
| `main.c` | **Modificato** | Loop DQN |

---

## Tecnica Re-Forward (fulcro del progetto)

Il problema centrale del training on-device è la memoria per le attivazioni
intermedie. Esistono due approcci:

**Approccio naive (inapplicabile su MCU):**
```
Per ogni sample del mini-batch → forward_q() → salva layer[i].out in cache
                                                          ↓
                                 cache[batch_size][num_layers][max_dim]
Poi per ogni sample → backward()  ← legge dalla cache

Memoria extra: batch_size × Σ(layer_dim) × 4 byte
Con batch=32, rete [4,64,64,2]: 32 × 130 × 4 ≈ 16 KB extra (accettabile qui,
ma su reti più grandi o batch più grossi scala male ed è concettualmente sbagliato)
```

**Re-Forward (approccio adottato):**
```
Per ogni sample del mini-batch:
    1. forward_q(online, s)         → scrive layer[i].out (sovrascrive il precedente)
    2. forward_target(target, s')   → scrive target.layers[i].out (struct separata)
    3. calcola td_error
    4. dqn_backward(online, s, a, td_error)  → legge layer[i].out (ancora validi)
    5. accumula dW, db
    → le attivazioni vengono BUTTATE (sovrascritte dal sample successivo)

Memoria per le attivazioni: max(layer_dim) × 4 byte  (un solo slot, riusato)
```

Le attivazioni del sample `t` vengono ricalcolate al volo appena prima del suo
backward, poi sovrascritte da quelle del sample `t+1`. Il costo è un forward pass
extra per sample rispetto al caso ideale, ma su MCU questo è il vincolo giusto
da accettare: si paga in cicli di clock, non in RAM.

Il campo `layer->out` in `DenseLayer` è il singolo slot di attivazione riusato.
Il vettore di stato `s` viene letto direttamente da `buf->state[idx]` (puntatore
stabile nel replay buffer) — serve come input al layer 0 durante il backward.

La rete target ha i propri `TargetLayer.out` separati: il suo forward non
interferisce mai con le attivazioni della rete online.

---

## Ottimizzazioni Memoria

### 1. Target Network senza Adam moments
La target network serve solo per l'inferenza (forward pass).
Non si allena mai direttamente → nessun gradiente, nessun momento Adam.
Struttura alleggerita: solo `W`, `b`, `out` per ogni layer.

```c
// Online:  W, b, out, mW, vW, dW, mb, vb, db  → 9 array per layer
// Target:  W, b, out                           → 3 array per layer
```

Per una rete [4, 64, 64, 2]: online ~77 KB, target ~19 KB (vs 77 KB se full).

### 2. Replay Buffer con indici invece di copia stato
Il replay buffer usa array `float **state` e `float **next_state` come
puntatori a righe pre-allocate in blocchi contigui. Questo evita la
frammentazione heap che su MCU porta a fallimenti malloc silenziosi.

```c
// Allocazione contigua (un unico malloc per tutte le righe):
float *state_pool  = malloc(capacity * obs_dim * sizeof(float));
float *snext_pool  = malloc(capacity * obs_dim * sizeof(float));
// poi state[i] = state_pool + i * obs_dim
```

### 3. Replay Buffer dimensionato per RAM disponibile
Il buffer è il componente che scala peggio con la dimensione.
Formula per stimare l'occupazione:

```
bytes = capacity × (2 × obs_dim + 1 + 1 + 1) × 4  (float per tutto tranne done)
```

Esempio: capacity=1000, obs_dim=4 → 1000 × 10 × 4 = 40 KB.

### 4. Backprop su un solo output (azione scelta)
Durante il training, il gradiente della loss MSE è non-zero solo per
l'output corrispondente all'azione scelta (Q(s,a_taken)).
Gli altri output del vettore Q non contribuiscono al gradiente.
Questo dimezza il lavoro al layer di uscita rispetto a una loss full-vector.

### 5. Buffer temporanei statici per backprop
I buffer `delta_buf` per la backpropagation vengono dichiarati `static`
nel corpo della funzione (come già nel codice PPO). Evitano malloc/free
ad ogni step di training e mantengono il worst-case fisso in RAM.

### 6. Stack allocation per Q-values (VLA o array fisso)
I vettori Q di output sono piccoli (numero di azioni, tipicamente < 16).
Si allocano sullo stack con dimensione fissa a compile-time tramite `#define N_ACTIONS`.

---

## Stima Memoria per Configurazione Tipica

Rete: `[obs_dim=4, 64, 64, n_actions=2]`

| Componente | Floats | Bytes |
|---|---|---|
| Online net (W+b+momenti+grad) | ~19.700 | ~77 KB |
| Target net (solo W+b+out) | ~4.800 | ~19 KB |
| Replay buffer (cap=1000, obs=4) | ~10.000 | ~40 KB |
| Stack temporanei backprop | ~200 | ~0.8 KB |
| **Totale** | | **~137 KB** |

STM32H743 ha 512 KB DTCM + 288 KB AXI SRAM → abbondante margine.
Con rete più piccola `[3, 32, 32, n_actions]` si scende a ~40 KB totali.

---

---

## TASK 1 — Semplificare `neural_net.h/c`: QNetwork

**Obiettivo:** Sostituire `SharedBackbone` + `Head` con `QNetwork`.
Nessuna dipendenza da `dqn.c` ancora — questo task è autonomo.

### 1.1 — `neural_net.h`

Rimuovere:
- `SharedBackbone`, `Head`, `action_t`, `USE_CONTINUOUS_ACTIONS`
- Tutti gli `#define` PPO-specifici (`EPS_CLIPPING`, `PPO_EPSILON`, `CRITIC_COEFF`,
  `ENT_BETA`, `CRIT_LOSS`, `N_EPOCHS`, `ROLLOUT`, `STARTING_ACTION_SIGMA`, `TOTAL_ADAM_STEPS`)

Aggiungere:
```c
// Iperparametri DQN
#define LR              0.001f
#define BETA1           0.9f
#define BETA2           0.999f
#define EPS_ADAM        1e-8f
#define GAMMA           0.99f
#define EPSILON_START   1.0f
#define EPSILON_END     0.05f
#define EPSILON_DECAY   5000    // step totali per il decadimento lineare
#define TARGET_UPDATE   200     // ogni quanti step di training aggiornare target
#define REPLAY_MIN      500     // campioni minimi per iniziare il training
#define REPLAY_SIZE     1000    // capacità del replay buffer
#define BATCH_SIZE      32
#define MAX_EPISODE     500
#define N_ACTIONS       2       // numero di azioni discrete

// Struttura rete (una sola, niente actor/critic)
typedef struct {
    DenseLayer *layers;
    uint8_t     num_layers;
    uint32_t    adam_t;
} QNetwork;

// Struttura target network (solo pesi, niente Adam)
typedef struct {
    float **W;    // [out_dim][in_dim] per ogni layer
    float  *b;    // [out_dim] per ogni layer
    float  *out;  // [out_dim] per ogni layer
    int     in_dim;
    int     out_dim;
} TargetLayer;

typedef struct {
    TargetLayer *layers;
    uint8_t      num_layers;
} TargetNetwork;

// API
int   init_qnetwork(QNetwork *net, int num_layers, int *topology,
                    ActivationType *activations);
int   init_target_network(TargetNetwork *tgt, int num_layers, int *topology);
int   forward_q(QNetwork *net, float *input, float *q_out);
int   forward_target(TargetNetwork *tgt, float *input, float *q_out);
void  copy_weights_to_target(QNetwork *src, TargetNetwork *dst);
void  adam_optimizer_q(QNetwork *net);
void  gradient_norm_q(QNetwork *net);
void  zero_grad_q(QNetwork *net);
```

### 1.2 — `neural_net.c`

Riscrivere mantenendo:
- `softmax` (invariata)
- `forward_dense_layer` static inline (invariata)
- `adam_update_single_layer` (invariata)

Nuove implementazioni:
- `init_qnetwork`: loop semplice su `topology`, senza fork actor/critic.
  Aggiungere scaling finale layer (×0.01 come per l'actor PPO) opzionale.
- `init_target_network`: alloca solo `W`, `b`, `out` senza Adam moments.
- `forward_q`: loop lineare su `net->layers[]`, copia risultato in `q_out`.
- `forward_target`: identico ma su `TargetLayer`.
- `copy_weights_to_target`: copia `W[i][j]` e `b[i]` da ogni `DenseLayer` al
  corrispondente `TargetLayer`. Chiamata periodica ogni `TARGET_UPDATE` step.
- `adam_optimizer_q`, `gradient_norm_q`, `zero_grad_q`: come le versioni PPO
  ma con il loop solo su `net->layers[]` (niente actor/critic).

### Test Task 1
Compilare il progetto con `main.c` modificato minimalmente:
```c
QNetwork online;
TargetNetwork target;
int topology[] = {4, 32, 32, N_ACTIONS};
ActivationType acts[] = {ACT_RELU, ACT_RELU, ACT_NONE};
init_qnetwork(&online, 4, topology, acts);
init_target_network(&target, 4, topology);
float obs[4] = {0.1f, -0.2f, 0.3f, -0.1f};
float q[N_ACTIONS];
forward_q(&online, obs, q);         // deve produrre valori finiti
copy_weights_to_target(&online, &target);
forward_target(&target, obs, q);    // deve produrre gli stessi valori
```
Verificare via UART o LED che non ci siano crash / valori NaN.

---

## TASK 2 — Backpropagation DQN: `dqn_backward`

**Obiettivo:** Implementare il gradiente della loss MSE per il DQN.
Dipende da Task 1 (QNetwork deve esistere).

### 2.1 — Aggiungere in `neural_net.h`

```c
void dqn_backward(QNetwork *net, float *input, uint32_t action,
                  float td_error);
```

### 2.2 — Implementazione in `neural_net.c`

La loss per un campione è:
```
L = 0.5 * td_error²   dove  td_error = Q_online(s,a) - target_val
```

Il gradiente al layer di uscita è un vettore di zeri tranne in posizione `action`:
```
dL/dQ[i] = td_error   se i == action
dL/dQ[i] = 0.0f       altrimenti
```

Poi backprop standard layer per layer (stessa logica di `backward_core` PPO):
- Accumula `dW[i][j] += delta[i] * input[j]`
- Accumula `db[i] += delta[i]`
- Propaga `delta = W^T * delta` applicando derivata attivazione

Nota: i gradienti vengono **accumulati** (non azzerati qui).
`zero_grad_q` viene chiamata prima del mini-batch, `adam_optimizer_q` dopo.

### Test Task 2
Partire da una rete già inizializzata (Task 1).
Fare forward, costruire un `td_error` manuale, chiamare `dqn_backward`.
Verificare che i pesi cambino dopo `adam_optimizer_q`:
```c
float q_before = q[action];
dqn_backward(&online, obs, action, td_error);
adam_optimizer_q(&online);
forward_q(&online, obs, q);
// q[action] deve essersi avvicinato a (q_before - lr * td_error)
```

---

## TASK 3 — Replay Buffer: `dqn.h/c`

**Obiettivo:** Implementare il replay buffer circolare con allocazione contigua.
Nessuna dipendenza da Task 1/2 — può essere sviluppato in parallelo.

### 3.1 — `dqn.h`

```c
#ifndef DQN_H
#define DQN_H

#include "neural_net.h"
#include "stm32h7xx_hal.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    float    *state_pool;     // blocco contiguo [capacity * obs_dim]
    float    *snext_pool;     // blocco contiguo [capacity * obs_dim]
    float   **state;          // puntatori riga → state_pool
    float   **next_state;     // puntatori riga → snext_pool
    uint32_t *action;
    float    *reward;
    uint8_t  *done;
    uint32_t  head;           // indice di scrittura (circolare)
    uint32_t  size;           // campioni presenti (max capacity)
    uint32_t  capacity;
    uint32_t  obs_dim;
} ReplayBuffer;

int   replay_buffer_init(ReplayBuffer *buf, uint32_t capacity, uint32_t obs_dim);
void  replay_buffer_push(ReplayBuffer *buf, float *s, uint32_t action,
                         float reward, float *s_next, uint8_t done);

// UART (spostate da reinforce.c)
int uart_recv_floats(UART_HandleTypeDef *huart, float *dst, size_t dim,
                     uint32_t timeout);
int uart_send_action_discrete(UART_HandleTypeDef *huart, uint32_t action,
                              uint8_t done, uint32_t timeout);

// ε-greedy
float calc_epsilon(uint32_t step);
uint32_t dqn_select_action(QNetwork *net, float *obs, float epsilon,
                            uint32_t n_actions);

// Training step
void dqn_train(QNetwork *online, TargetNetwork *target,
               ReplayBuffer *buf, uint32_t batch_size, uint32_t n_actions);

#endif
```

### 3.2 — `dqn.c`

**`replay_buffer_init`:**
- Un unico `malloc` per `state_pool` (capacity × obs_dim floats)
- Un unico `malloc` per `snext_pool`
- Assegna `state[i] = state_pool + i * obs_dim`
- head=0, size=0

**`replay_buffer_push`:**
- Scrive in `buf->state[buf->head]` copiando obs con `memcpy`
- Incrementa `head = (head + 1) % capacity`
- Incrementa `size` fino a `capacity`

**`calc_epsilon`:**
```c
float calc_epsilon(uint32_t step) {
    float t = (float)step / EPSILON_DECAY;
    if (t > 1.0f) t = 1.0f;
    return EPSILON_START + t * (EPSILON_END - EPSILON_START);
}
```

**`dqn_select_action`:**
```c
// Se rand < epsilon → azione random
// Altrimenti → argmax Q_online(obs)
```

**`dqn_train`:**
1. Campiona `batch_size` indici random dal buffer (senza Fisher-Yates completo:
   basta estrarre indici `rand() % buf->size` — accettabile per DQN)
2. `zero_grad_q(online)`
3. Per ogni campione del mini-batch:
   - `forward_q(online, s, q_online)`
   - `forward_target(target, s_next, q_target)`
   - `target_val = done ? reward : reward + GAMMA * max(q_target)`
   - `td_error = q_online[action] - target_val`
   - `dqn_backward(online, s, action, td_error / batch_size)`
4. `gradient_norm_q(online)`
5. `adam_optimizer_q(online)`

### Test Task 3
Test standalone del buffer:
```c
ReplayBuffer buf;
replay_buffer_init(&buf, 100, 4);
float s[4] = {1,2,3,4}, sn[4] = {5,6,7,8};
replay_buffer_push(&buf, s, 0, 1.0f, sn, 0);
replay_buffer_push(&buf, sn, 1, -1.0f, s, 1);
// Verificare buf.size == 2
// Verificare sovrascrittura circolare dopo 100 push
```

---

## TASK 4 — Integrazione in `main.c`

**Obiettivo:** Collegare tutto nel loop principale.
Dipende da Task 1, 2, 3 completati.

### 4.1 — Struttura del loop

```c
// Inizializzazione
QNetwork online;
TargetNetwork target;
ReplayBuffer replay;
uint32_t step_total  = 0;  // step ambiente totali
uint32_t train_step  = 0;  // step di training (per TARGET_UPDATE)

// init rete, target, buffer...
copy_weights_to_target(&online, &target);  // target = online al t=0

float obs[OBS_DIM], prev_obs[OBS_DIM];
uint32_t action = 0;
float epsilon;
uint8_t first_step = 1;

while (1) {
    if (!uart_recv_floats(&huart3, obs, OBS_DIM, 50)) continue;

    // Reward e done calcolati sull'osservazione CORRENTE
    // (come nel PPO attuale: i dati di t-1 si chiudono a t)
    manual_done   = done_check(...);
    manual_reward = evaluate_reward(...);

    // Aggiunge transizione al buffer (se non è il primissimo step)
    if (!first_step) {
        replay_buffer_push(&replay, prev_obs, action,
                           manual_reward, obs, manual_done);
    }
    first_step = 0;

    // Training online se buffer abbastanza pieno
    if (replay.size >= REPLAY_MIN) {
        dqn_train(&online, &target, &replay, BATCH_SIZE, N_ACTIONS);
        train_step++;
        if (train_step % TARGET_UPDATE == 0)
            copy_weights_to_target(&online, &target);
    }

    // Selezione azione
    epsilon = calc_epsilon(step_total);
    action  = dqn_select_action(&online, obs, epsilon, N_ACTIONS);

    // Salva obs corrente per il prossimo giro
    memcpy(prev_obs, obs, OBS_DIM * sizeof(float));

    // Invia azione
    uart_send_action_discrete(&huart3, action, manual_done, 50);

    if (manual_done) {
        memset(prev_obs, 0, sizeof(prev_obs));
        first_step = 1;
        num_episode++;
    }
    step_total++;
}
```

### 4.2 — Reward e done
Stesse funzioni `evaluate_reward_*` e `done_check_*` già in `main.c`.
Non cambiano — sono indipendenti dall'algoritmo RL.

### Test Task 4
Test end-to-end con ambiente simulato lato PC.
Verificare:
- L'epsilon scende correttamente nel tempo
- Il target network viene aggiornato ogni `TARGET_UPDATE` step
- La rete non crasha e i Q-values restano finiti
- Dopo ~1000 step il comportamento è meno random (Q-values differenziati)

---

## Ordine di Sviluppo Consigliato

```
Task 1 (QNetwork struttura + forward)
    │
    ├── Task 2 (dqn_backward)     ← dipende da Task 1
    │
    └── Task 3 (ReplayBuffer)     ← indipendente, parallelizzabile
         │
         └── Task 4 (main.c)     ← dipende da Task 1 + 2 + 3
```

Task 3 può essere scritto e testato su PC (`gcc` + `main` di test) senza
toccare il progetto STM32 — il buffer non ha dipendenze HAL.

---

## Note su Debugging su MCU

- Usare i LED (LD1/LD2/LD3 già in `main.c`) per segnalare stati:
  - LD1 = rete inizializzata OK
  - LD2 = training in corso
  - LD3 = errore allocazione
- Inviare via UART i Q-values raw durante i test per verificare che
  convergano (aggiungere un frame di log opzionale come `uart_send_log`).
- Su STM32H7 il `malloc` fallisce silenziosamente se la heap è piena:
  aggiungere controllo esplicito su tutti i return value di `malloc`
  (già presente nel PPO — mantenere la stessa prassi).
