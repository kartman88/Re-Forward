# tinyRL — Piano di Refactoring e Correzioni

Documento di pianificazione per la pulizia, correzione e miglioramento del codice.
Ordinato per priorità: prima i bug di correttezza, poi dead code, poi riorganizzazione.

---

## FASE 1 — Bug di Correttezza Algoritmica

Questi problemi possono causare training errato su task diverse da Swimmer.

---

### BUG-1 · Bootstrap mancante quando il buffer si riempie a metà episodio

**File:** `Core/Src/reinforce.c` — `step()`, Phase 2 (~riga 232)

**Problema:**
Quando `step()` ritorna `2` (buffer pieno), `terminal_value_buffer[step_count-1]`
rimane `0.0f` (inizializzato in Phase 3 del passo precedente). Se il buffer si riempie
mentre l'episodio è ancora in corso (`done==0`), il calcolo GAE in
`evaluate_advantages_and_returns` tratta quel troncamento come una terminazione
reale (V=0), sottostimando sistematicamente i return dell'ultimo batch di step.

**Perché non si vede su Swimmer:**
Con `MAX_STEPS=2000` e `ROLLOUT=1000`, il timing è tale che l'episodio termina
(`done=1` da Phase 1) **prima** che Phase 2 possa ritornare `2`. Ma se `ROLLOUT`
non è un divisore esatto di `MAX_STEPS`, o per task senza terminazione esplicita
(es. HalfCheetah), il bug è attivo.

**Fix:**
In Phase 2, prima di `return 2`, aggiungere un forward pass su `obs` corrente
e salvare il valore di bootstrap:

```c
// In step(), Phase 2:
if (*step_count >= MAX_STEPS) {
    // Bootstrap: il buffer è pieno ma l'episodio non è terminato.
    // Usiamo V(s_t) come valore terminale per il GAE dell'ultimo step.
    if (buffer->done_buffer[*step_count - 1] == 0) {
        float tmp_actor[out_dim_actor], tmp_critic[out_dim_critic];
        forward(net, obs, tmp_actor, tmp_critic);
        buffer->terminal_value_buffer[*step_count - 1] = tmp_critic[0];
        buffer->done_buffer[*step_count - 1] = 1; // troncamento, non terminazione
    }
    return 2;
}
```

---

### BUG-2 · Azione stale inviata all'ambiente dopo `finish_episode`

**File:** `Core/Src/main.c` (~riga 194-220)

**Problema:**
Quando `step()` ritorna `2`, Phase 3 non viene eseguita e `out_action` non viene
aggiornato. `action[]` contiene ancora l'ultima azione calcolata dal Phase 3 del
passo precedente (step 999 dell'episodio 2 nel caso Swimmer). Questa azione stale
viene inviata all'ambiente come primo passo del nuovo rollout, senza essere calcolata
dall'osservazione corrente.

**Fix:**
Dopo `finish_episode`, eseguire un forward pass sull'osservazione corrente per
calcolare la prima azione valida del nuovo rollout prima di inviare all'ambiente:

```c
if (status == 2) {
    finish_episode(&buffer, &net, MAX_STEPS, 1);
    step_count = 0;
    ep_step = 0;
    // Calcola azione valida per l'osservazione corrente
    // invece di usare quella stale dell'ultimo Phase 3
    // (la prossima chiamata a step() la ricalcolerà correttamente)
}
```

Alternativa più pulita: lasciare che la prima chiamata a `step()` dopo il reset
calcoli l'azione (come già fa), ma non inviare l'azione stale. Invece, inviare
un'azione neutra (zero) e attendere la prossima osservazione prima di inviare
un'azione significativa.

---

### BUG-3 · Off-by-one di `ep_step` dopo status==2

**File:** `Core/Src/main.c` (~riga 209-215)

**Problema:**
```c
if (status == 2) {
    ...
    ep_step = 0;  // reset
}
ep_step++;        // → ep_step=1 immediatamente dopo il reset!
```
La prima iterazione del nuovo ciclo dopo il training parte con `ep_step=1`, non `0`.
`done_check` vedrà lo step 1 invece dello step 0. Per Swimmer, l'episodio viene
troncato a 999 step invece di 1000 una volta dopo ogni training update.

**Fix:**
Strutturare la logica in modo che `ep_step++` non sia eseguito dopo un reset:

```c
if (status == 2) {
    finish_episode(...);
    step_count = 0;
    ep_step = 0;
    // NON fare ep_step++ qui: il reset azera già il contatore
    // e l'osservazione corrente è il punto di partenza (step 0)
}
// ep_step++ solo se NON abbiamo appena resettato
else {
    ep_step++;
}
```

---

### BUG-4 · `done_buffer` e `terminal_value_buffer` non controllati nel fail path di `buffer_init`

**File:** `Core/Src/reinforce.c` — `buffer_init()` (~riga 36-41)

**Problema:**
```c
buf->done_buffer = calloc((n_steps + 1), sizeof(uint8_t));
buf->terminal_value_buffer = calloc((n_steps + 1), sizeof(float));
// ...
if (!buf->state_buffer || !buf->action_buffer || !buf->log_prob_old_buffer ||
    !buf->advantage_buffer || !buf->critic_buffer || !buf->sigma_buffer)
    goto fail;  // ← done_buffer e terminal_value_buffer non inclusi!
```
Se queste `calloc` falliscono, il programma continua con puntatori NULL causando
un crash immediato alla prima scrittura.

**Fix:**
Aggiungere entrambi alla condizione di fallimento:
```c
if (!buf->state_buffer || !buf->action_buffer || !buf->log_prob_old_buffer ||
    !buf->advantage_buffer || !buf->critic_buffer || !buf->sigma_buffer ||
    !buf->done_buffer || !buf->terminal_value_buffer)
    goto fail;
```

---

## FASE 2 — Naming Confuso e Semantica Invertita

Questi problemi non causano bug ma rendono il codice difficile da leggere e manutenere.

---

### NAME-1 · `advantage_buffer` e `critic_buffer` hanno contenuto scambiato dopo GAE

**File:** `Core/Src/reinforce.c` — `evaluate_advantages_and_returns()` e `finish_episode()`

**Problema:**
Dopo `evaluate_advantages_and_returns`, i buffer contengono:
- `advantage_buffer[t]` → **Return target** `G_t = GAE + V(s_t)` (usato dal Critic)
- `critic_buffer[t]` → **Advantage normalizzato** `A_t_norm` (usato dall'Actor)

I nomi sono **semanticamente invertiti** rispetto al contenuto effettivo.
In `finish_episode` si legge:
```c
float normalized_advantage = buf->critic_buffer[t];   // in realtà è l'advantage
float return_target = buf->advantage_buffer[t];        // in realtà è il return
```

**Fix:**
Rinominare i campi del Buffer:
```c
typedef struct {
    float *return_buffer;    // ex advantage_buffer: contiene G_t (target del critic)
    float *advantage_buffer; // ex critic_buffer: contiene A_t normalizzato (per l'actor)
    ...
} RolloutBuffer;
```
E aggiornare tutti i siti di utilizzo di conseguenza.

---

### NAME-2 · Variabile `clipped` semanticamente invertita

**File:** `Core/Src/reinforce.c` — `backward_actor_critic()` (~riga 739)

**Problema:**
```c
float clipped = 0; // 0 = gradiente NON fluisce, 1 = gradiente fluisce
if (norm_adv > 0.0f) {
    if (ratio < 1.0f + EPS_CLIPPING) {
        clipped = 1; // ← ma questo significa "NON clippato" in senso PPO!
    }
}
if (clipped) {
    grad_ppo_i = -(ratio * norm_adv * d_log_prob_i);
}
```
Nel gergo PPO, "clipped" significa che il gradiente è **zero**. Qui `clipped=1`
significa che il gradiente **fluisce** (ratio nella zona safe, NON clippato).
La logica è corretta ma il nome è l'opposto del significato convenzionale.

**Fix:**
```c
int ppo_gradient_active = 0;
// ... stessa logica ...
if (ppo_gradient_active) {
    grad_ppo_i = -(ratio * norm_adv * d_log_prob_i);
}
```

---

### NAME-3 · Convenzione `num_layers` conta i nodi, non i layer di pesi

**File:** `Core/Src/neural_net.c` — `init_network()`, `Core/Src/main.c`

**Problema:**
```c
int num_layers = 2;            // in main.c
int net_topology[] = {8, 64};  // 2 elementi = 1 layer di pesi (8→64)
```
La variabile si chiama `num_layers` ma conta i **nodi** (incluso l'input), non i
layer di peso. `init_network` poi fa `num_layers--` per ottenere il numero reale
di layer, modificando silenziosamente il parametro in ingresso. 
Questo è controintuitivo.

**Fix:**
Rinominare per chiarire:
- `num_layers` → `topology_size` (numero di entry nell'array topologia, incluso input)
- Il decremento interno rimane ma il nome del parametro è ora chiaro
- Aggiungere un commento: `// topology_size entries = topology_size-1 weight layers`

---

### NAME-4 · `SharedBackbone` e `Buffer` hanno nomi troppo generici

**File:** `Core/Inc/neural_net.h`, `Core/Inc/reinforce.h`

**Rinominazioni proposte:**

| Attuale | Proposto | Motivazione |
|---|---|---|
| `SharedBackbone` | `PPONetwork` | chiarisce il contesto algoritmo |
| `Buffer` | `RolloutBuffer` | specifica il tipo di buffer (rollout PPO) |
| `new_episode` (static in step) | `is_episode_start` | variabile booleana, nome descrittivo |
| `manual_reward` (in main) | `reward` | "manual" è ridondante con il design attuale |
| `manual_done` (in main) | `done` | idem |
| `ep_step` (in step(), static) | rimuovere | dead variable (mai letta) |
| `step_count` (in main) | `buffer_idx` | chiarisce che è un indice nel buffer |

---

## FASE 3 — Dead Code da Rimuovere

Codice mai utilizzato che aumenta la dimensione e la confusione.

---

### DEAD-1 · `backward_pg()` e `backward_core()` in `neural_net.c`

**File:** `Core/Src/neural_net.c` (~riga 213 e 276)

`backward_pg` era per l'algoritmo REINFORCE (versione precedente). Ora PPO usa
`backward_actor_critic` in `reinforce.c`. `backward_core` è chiamata solo da
`backward_pg`, quindi è anch'essa dead code.

**Azione:** Rimuovere entrambe le funzioni e le loro dichiarazioni da `neural_net.h`.

---

### DEAD-2 · `evaluate_mean_std()` in `reinforce.c`

**File:** `Core/Src/reinforce.c` (~riga 386)

Funzione mai chiamata. Era probabilmente un'utility per il calcolo separato di
media/std degli advantage, poi sostituita da `evaluate_advantages_and_returns`.

**Azione:** Rimuovere la funzione e la sua dichiarazione (se presente in `reinforce.h`).

---

### DEAD-3 · `EnvType` enum in `reinforce.h`

**File:** `Core/Inc/reinforce.h` (~riga 31)

```c
typedef enum {
    CART_POLE,
    ACROBOT
} EnvType;
```
Non usato in nessun punto del codice.

**Azione:** Rimuovere. Quando verranno aggiunte nuove task, la selezione avviene
via function pointer o condizionale in `main.c`, non tramite questo enum.

---

### DEAD-4 · `CRIT_LOSS` e `PPO_EPSILON` in `neural_net.h`

**File:** `Core/Inc/neural_net.h` (~riga 15-29)

```c
#define CRIT_LOSS 0.5      // mai usato
#define PPO_EPSILON 0.2    // mai usato (al suo posto c'è EPS_CLIPPING)
```

**Azione:** Rimuovere entrambi. `EPS_CLIPPING` è il macro corretto e attivo.

---

### DEAD-5 · `static ep_step` dentro `step()` in `reinforce.c`

**File:** `Core/Src/reinforce.c` (~riga 185)

```c
static uint32_t ep_step = 0; // incrementato ma mai letto
```
Viene incrementato in Phase 3 e resettato in Phase 1 ma il suo valore non è mai
utilizzato per nessun calcolo o condizione.

**Azione:** Rimuovere la variabile e tutti i siti di incremento/reset interni a `step()`.

---

### DEAD-6 · Branch epsilon-greedy morto in `sample_action()`

**File:** `Core/Src/reinforce.c` (~riga 145)

```c
const float EPSILON = 0.0f;
float r = (float)rand() / (float)RAND_MAX;
if (r < EPSILON)               // sempre falso: 0.0f > qualsiasi r
    return (uint8_t)(rand() % dim);
```
Con `EPSILON = 0.0f`, il branch non viene mai preso.

**Azione:** Rimuovere le righe relative a EPSILON e il branch di explorazione random.
Se in futuro si vorrà aggiungere epsilon-greedy, lo si farà in modo controllato.

---

### DEAD-7 · Parametro `done` inutilizzato in `finish_episode()`

**File:** `Core/Src/reinforce.c` (~riga 828)

```c
uint32_t finish_episode(Buffer *buf, SharedBackbone *net,
                        uint32_t step_count, uint8_t done) {
    // 'done' non viene mai usato nel corpo della funzione
    (void)0; // placeholder
```

**Azione:** Rimuovere il parametro `done` dalla firma e aggiornare i siti di chiamata.

---

### DEAD-8 · Codice commentato per MountainCar e CartPole in `reinforce.c`

**File:** `Core/Src/reinforce.c` (fondo del file, ~riga 952-977)

Blocchi commentati con le funzioni `done_check` e `evaluate_reward` delle task
precedenti. Il codice storico appartiene a git history, non al sorgente.

**Azione:** Rimuovere i blocchi commentati.

---

### DEAD-9 · `STARTING_ACTION_SIGMA` definita due volte

**File:** `Core/Inc/neural_net.h:8` e `Core/Inc/reinforce.h:10`

```c
// In neural_net.h:
#define STARTING_ACTION_SIGMA 0.8f

// In reinforce.h (che include neural_net.h):
#define STARTING_ACTION_SIGMA 0.8f  // ← ridefinizione → warning del compilatore
```

**Azione:** Rimuovere la definizione da `reinforce.h`. Mantenere solo in `neural_net.h`
(o nel futuro `config.h`).

---

## FASE 4 — Riorganizzazione File e Architettura

Da fare **dopo** che le fasi precedenti sono complete e testate.

---

### ARCH-1 · Creare `config.h` con tutti gli iperparametri

**Problema attuale:**
Gli `#define` sono sparsi tra `neural_net.h` e `reinforce.h`. Cambiare un
iperparametro richiede sapere in quale file si trova.

**Proposta:**
```
Core/Inc/config.h   ← NUOVO
```
Contenuto:
```c
// Iperparametri rete
#define LR              0.0003f
#define BETA1           0.9f
#define BETA2           0.999f
#define EPS_ADAM        1e-8f
#define GAMMA           0.99f
#define ENT_BETA        0.0f

// Iperparametri PPO
#define EPS_CLIPPING    0.2f
#define CRITIC_COEFF    2.0f
#define N_EPOCHS        5
#define BATCH_SIZE      64

// Iperparametri ambiente/training
#define MAX_STEPS       2000
#define MAX_EPISODE     1000
#define ROLLOUT         1000

// Iperparametri esplorazione
#define STARTING_ACTION_SIGMA  0.8f
#define TOTAL_ADAM_STEPS       (MAX_EPISODE * (MAX_STEPS / BATCH_SIZE) * N_EPOCHS)

// Selezione tipo azione
#define USE_CONTINUOUS_ACTIONS 1
```

---

### ARCH-2 · Separare `comm.c/.h` dalla logica RL

**Problema attuale:**
Le funzioni UART (`uart_recv_floats`, `uart_send_action`, `uart_send_log`) vivono
in `reinforce.c`, mescolando il protocollo di comunicazione con l'algoritmo RL.

**Proposta:**
```
Core/Inc/comm.h    ← NUOVO (estratto da reinforce.h)
Core/Src/comm.c    ← NUOVO (estratto da reinforce.c)
```
`reinforce.c` non includerà più `stm32h7xx_hal.h` (quella dipendenza passerà a `comm.c`).

---

### ARCH-3 · Estrarre `optimizer.c/.h` da `neural_net.c`

**Problema attuale:**
`neural_net.c` contiene sia la logica di rete (forward, init) sia l'ottimizzatore
(adam, gradient norm, zero_grad). Sono responsabilità distinte.

**Proposta:**
```
Core/Inc/optimizer.h   ← NUOVO
Core/Src/optimizer.c   ← NUOVO
```
Contenuto: `adam_optimizer`, `adam_update_single_layer`, `gradient_norm_l2`, `zero_grad`.

---

### ARCH-4 · Struttura file finale proposta

```
Core/
├── Inc/
│   ├── config.h          ← NUOVO: tutti gli #define
│   ├── dense_layer.h     ← invariato
│   ├── network.h         ← ex neural_net.h (solo strutture + forward + init)
│   ├── optimizer.h       ← NUOVO: adam, gradient_norm, zero_grad
│   ├── ppo.h             ← ex reinforce.h (Buffer, step, GAE, backward)
│   ├── comm.h            ← NUOVO: uart functions
│   └── utils.h           ← invariato
└── Src/
    ├── dense_layer.c     ← invariato
    ├── network.c         ← ex neural_net.c (rimossi backward_pg, backward_core)
    ├── optimizer.c       ← NUOVO
    ├── ppo.c             ← ex reinforce.c (rimosso evaluate_mean_std, dead code)
    ├── comm.c            ← NUOVO
    ├── utils.c           ← invariato
    └── main.c            ← pulizia nomi locali, nessuna logica RL
```

---

### ARCH-5 · Interface per task-specific logic in `main.c`

**Problema attuale:**
Le funzioni `done_check_swimmer` e `evaluate_reward_swimmer` sono dichiarate come
`uint8_t (*done_check)(...)` e `float (*reward_fn)(...)` direttamente in `main.c`.
Per cambiare task, bisogna modificare il codice e ricompilare.

**Proposta (leggera, senza overhead):**
Definire una struct `EnvConfig` con function pointer in `main.c`:
```c
typedef struct {
    uint8_t (*done_fn)(uint32_t ep_step, float *obs);
    float   (*reward_fn)(float *obs, action_t *prev_actions);
    uint32_t rollout_steps;
    uint32_t obs_dim;
    uint32_t action_dim;
} EnvConfig;
```
Questo permette di definire in cima a `main.c` le configurazioni per ogni task e
switchare semplicemente quale `EnvConfig` passare alla main loop, senza toccare
la logica RL.

---

## Checklist Riepilogativa

### Priorità Alta (correttezza algoritmica)
- [ ] **BUG-1** Bootstrap mancante quando buffer pieno a metà episodio
- [ ] **BUG-2** Azione stale dopo `finish_episode`
- [ ] **BUG-3** Off-by-one `ep_step` dopo status==2
- [ ] **BUG-4** `done_buffer`/`terminal_value_buffer` non controllati in `buffer_init`

### Priorità Media (naming e leggibilità)
- [ ] **NAME-1** Rinominare `advantage_buffer` → `return_buffer`, `critic_buffer` → `advantage_buffer`
- [ ] **NAME-2** Rinominare `clipped` → `ppo_gradient_active`
- [ ] **NAME-3** Rinominare `num_layers` → `topology_size` e chiarire la convenzione
- [ ] **NAME-4** Rinominare `SharedBackbone` → `PPONetwork`, `Buffer` → `RolloutBuffer`

### Priorità Media (dead code)
- [ ] **DEAD-1** Rimuovere `backward_pg` e `backward_core` da `neural_net.c`
- [ ] **DEAD-2** Rimuovere `evaluate_mean_std` da `reinforce.c`
- [ ] **DEAD-3** Rimuovere `EnvType` enum da `reinforce.h`
- [ ] **DEAD-4** Rimuovere `CRIT_LOSS` e `PPO_EPSILON` da `neural_net.h`
- [ ] **DEAD-5** Rimuovere `static ep_step` da `step()`
- [ ] **DEAD-6** Rimuovere branch epsilon-greedy morto da `sample_action`
- [ ] **DEAD-7** Rimuovere parametro `done` da `finish_episode`
- [ ] **DEAD-8** Rimuovere codice commentato MountainCar/CartPole
- [ ] **DEAD-9** Rimuovere `STARTING_ACTION_SIGMA` duplicata da `reinforce.h`

### Priorità Bassa (architettura)
- [ ] **ARCH-1** Creare `config.h` con tutti gli iperparametri
- [ ] **ARCH-2** Separare `comm.c/.h` dalla logica RL
- [ ] **ARCH-3** Estrarre `optimizer.c/.h` da `neural_net.c`
- [ ] **ARCH-4** Rinominare file (`neural_net` → `network`, `reinforce` → `ppo`)
- [ ] **ARCH-5** Strutturare task-specific logic con `EnvConfig` in `main.c`
