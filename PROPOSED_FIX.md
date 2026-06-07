# PROPOSED_FIX — Portare i fix del PPO su DQN e REINFORCE

## Scopo

Durante il lavoro sul branch **PPO** sono stati individuati e risolti alcuni
problemi che **non sono specifici del PPO** ma riguardano la base comune
(`neural_net.c`, `dense_layer.c`, la UART, il loop HIL in `main.c`, i driver
Python). Questo documento elenca quei fix e dice, per ciascuno, **se e come**
applicarli al **DQN** (branch `DQN`, codice C/STM32) e al **REINFORCE**, così da
avere implementazioni coerenti tra loro.

Riferimento: l'implementazione canonica di ogni fix è sul branch `PPO`
(`tinyRL/Core/Src/rng.c`, `ppo.c`, `neural_net.c`, `dense_layer.c`, `uart.c`,
`main.c`; `Micro_RL/hil_trainer_continuous.py`, `hil_trainer_log.py`).

---

## Tabella di applicabilità (quick reference)

| # | Fix | DQN (C) | REINFORCE PyTorch | REINFORCE su MCU (se mai portato in C) |
|---|---|:---:|:---:|:---:|
| 1 | RNG di qualità (PCG32) al posto di `rand()` | ✅ | ❌ (torch ok) | ✅ |
| 2 | Guard sui gradienti non finiti | ✅ | ❌ (ha `clip_grad_norm_`) | ✅ |
| 3 | Sanity guard sull'osservazione (HIL) | ✅ | ➖ (non HIL) | ✅ |
| 4 | Hardening transport UART (checksum+ETX, flush) | ⚠️ consigliato (priorità minore) | ➖ | ✅ |
| 5 | Clamp della media della policy continua (`mu`) | ❌ (discreto) | ❌ (discreto) | ✅ se continuo |
| 6 | Convenzione `done` | ➖ già standard, verificare | ➖ | dipende |
| 7 | Parità di misura reward + diagnostica float32 | ✅ (driver) | ❌ | ✅ (driver) |

Legenda: ✅ applicare · ⚠️ consigliato · ➖ verificare/contesto · ❌ non serve.

> **REINFORCE PyTorch ([Micro_RL/reinforce.py](../Micro_RL/reinforce.py)) è già a
> posto**: ha seeding completo (`set_seed`), normalizzazione dei ritorni,
> `nn.utils.clip_grad_norm_(max_norm=1.0)` e l'Adam di torch. Nessun fix C-level
> serve lì. Le righe ❌/➖ della sua colonna sono quindi attese.

---

## FIX 1 — RNG di qualità (PCG32) al posto di `rand()`  ·  **DQN: SÌ**

### Perché
`rand()` di newlib è un LCG debole. Sul DQN influenza **esplorazione
ε-greedy**, **campionamento del replay buffer** e **inizializzazione dei pesi** —
tutte cose che vogliamo statisticamente sane e coerenti col PPO.

### Cosa fare
1. **Copiare i due file** dal branch PPO nel progetto DQN (drop-in, nessuna
   dipendenza, ~16 byte di stato):
   - `tinyRL/Core/Inc/rng.h`
   - `tinyRL/Core/Src/rng.c`
   (espongono `rng_seed`, `rng_u32`, `rng_uniform` → `[0,1)`, `rng_normal`).

2. **Sostituire gli usi di `rand()`** (posizioni sul branch DQN):

   - `dense_layer.c:12` — init pesi:
     ```c
     // prima
     static inline float frand(void) { return (float)rand() / RAND_MAX; }
     // dopo  (aggiungi #include "rng.h")
     static inline float frand(void) { return rng_uniform(); }
     ```

   - `dqn.c` → `dqn_select_action` (ε-greedy):
     ```c
     // prima:  float r = (float)rand() / ((float)RAND_MAX + 1.0f);
     //         return (uint32_t)((uint32_t)rand() % n_actions);
     // dopo:
     float r = rng_uniform();
     if (r < epsilon)
         return rng_u32() % n_actions;
     ```

   - `dqn.c` → `dqn_train` (campionamento replay):
     ```c
     // prima:  uint32_t idx = (uint32_t)rand() % buf->size;
     // dopo:
     uint32_t idx = rng_u32() % buf->size;
     ```

   - `main.c:111` — seed:
     ```c
     // prima:  srand(HAL_GetTick());
     // dopo:   rng_seed(HAL_GetTick());   // aggiungi #include "rng.h"
     ```

3. Verifica: `grep -rn "rand()\|RAND_MAX\|srand" Core/Src` non deve restituire
   più chiamate reali (solo eventuali commenti).

> Nota: PCG32 ≠ PCG64 di NumPy, quindi le curve C non coincideranno bit-per-bit
> con le baseline Python — l'obiettivo è la **qualità** del rumore, non la
> riproducibilità esatta.

---

## FIX 2 — Guard sui gradienti non finiti  ·  **DQN: SÌ**

### Perché
Rete di sicurezza: se per qualsiasi motivo un gradiente diventa `Inf`/`NaN`, lo
scarto evita che venga scritto nei pesi (corruzione permanente). A costo ~0 e
zero cambiamenti di comportamento quando va tutto bene.

### Cosa fare
Nel DQN il clipping è in `neural_net.c` → **`gradient_norm_q`** (sul branch DQN,
~riga 242, con `CLIP = 0.5f`). Subito dopo aver calcolato `gnorm`, inserire:

```c
float gnorm = sqrtf(gnorm_sq);
if (!isfinite(gnorm)) {
    // gradiente non finito: scarta l'update azzerando i gradienti
    for (int l = 0; l < net->num_layers; l++) {
        DenseLayer *ly = &net->layers[l];
        memset(ly->db, 0, ly->out_dim * sizeof(float));
        for (int i = 0; i < ly->out_dim; i++)
            memset(ly->dW[i], 0, ly->in_dim * sizeof(float));
    }
    return;
}
if (gnorm > CLIP) { ... }   // resto invariato
```

Assicurarsi che `neural_net.c` includa `<math.h>` (per `isfinite`) e `<string.h>`
(per `memset`).

---

## FIX 3 — Sanity guard sull'osservazione ricevuta (HIL)  ·  **DQN: SÌ**

### Perché
Un frame UART disallineato/corrotto può consegnare un'osservazione con valori
**NaN/Inf** o **finiti ma assurdi** (es. `1e30`). Un obs simile va dritto nel
replay buffer e avvelena il training. Scartare il frame è gratis e robusto.

### Cosa fare
Nel loop HIL di `main.c` (DQN), subito dopo la `uart_recv_floats`:

```c
if (!uart_recv_floats(&huart3, obs, OBS_DIM, 100))
    continue;

int obs_ok = 1;
for (int i = 0; i < OBS_DIM; i++)
    if (!isfinite(obs[i]) || fabsf(obs[i]) > 1000.0f) { obs_ok = 0; break; }
if (!obs_ok)
    continue;   // frame sporco: il PC lo ritrasmette
```

La soglia `1000.0f` è molto sopra i valori legittimi di Pendulum/Hopper; alzala
se la tua task ha osservazioni di magnitudine maggiore. Richiede `<math.h>`.

---

## FIX 4 — Hardening del transport UART  ·  **DQN: consigliato (priorità minore)**

### Perché / differenza col PPO
Sul PPO il training è un **blocco lungo** (rollout da 2048 step → backprop di
qualche secondo): durante quella pausa la UART va in **overrun** e, al recupero,
un frame disallineato passava i controlli (l'ETX **non veniva validato** e non
c'era checksum). Sul **DQN il training è un mini-batch per step** (millisecondi),
quindi **non c'è una pausa lunga** e il rischio di overrun è molto più basso.
→ Per il DQN questi fix sono **igiene consigliata**, non urgenti come sul PPO.

### Cosa fare (se lo applichi)
1. **Checksum + validazione ETX** nel frame stato.

   Lato MCU — `uart_recv_floats` (sul branch DQN è dentro `dqn.c`):
   ```c
   int uart_recv_floats(UART_HandleTypeDef *huart, float *dst, size_t dim,
                        uint32_t timeout) {
       uint8_t stx;
       if (HAL_UART_Receive(huart, &stx, 1, timeout) != HAL_OK || stx != 0x02)
           return 0;
       const size_t nbytes = dim * sizeof(float);
       if (HAL_UART_Receive(huart, (uint8_t *)dst, nbytes, timeout) != HAL_OK)
           return 0;
       uint8_t chk;
       if (HAL_UART_Receive(huart, &chk, 1, timeout) != HAL_OK) return 0;
       uint8_t etx;
       if (HAL_UART_Receive(huart, &etx, 1, timeout) != HAL_OK || etx != 0x03)
           return 0;
       uint8_t calc = 0;
       const uint8_t *p = (const uint8_t *)dst;
       for (size_t i = 0; i < nbytes; i++) calc ^= p[i];
       if (calc != chk) return 0;   // frame corrotto/disallineato
       return 1;
   }
   ```

   Lato PC — `send_state` in **tutti** i driver HIL del progetto DQN
   (`hil_trainer_continuous.py`, `hil_trainer_log.py`):
   ```python
   def send_state(ser, obs):
       payload = struct.pack(_STATE_STRUCT, *obs.astype(np.float32))
       chk = 0
       for b in payload:
           chk ^= b
       ser.write(b'\x02' + payload + bytes([chk]) + b'\x03')
   ```

   ⚠️ **Firmware e driver vanno aggiornati INSIEME**: il frame stato ha ora un
   byte in più. Vecchio + nuovo non comunicano.

2. **Flush RX + clear overrun dopo il training** — **solo se** il tuo DQN
   accumula una pausa lunga (es. training a fine episodio invece che per-step).
   Se il DQN allena per-step, **salta questo punto**. Altrimenti, dopo il blocco
   di training:
   ```c
   __HAL_UART_CLEAR_OREFLAG(&huart3);
   __HAL_UART_SEND_REQ(&huart3, UART_RXDATA_FLUSH_REQUEST);
   ```

---

## FIX 5 — Clamp della media della policy continua (`mu`)  ·  **DQN: NO**

Specifico per policy **ad azione continua** (Gaussiana diagonale): limita
l'uscita dell'actor (pre-tanh) a `±MU_CLAMP` per impedire il feedback che fa
esplodere i pesi in float32. Il **DQN è discreto** (Q-values, niente `mu`):
**non applicabile**. La protezione equivalente per il DQN contro la divergenza è
già coperta dal **FIX 2** (guard sui gradienti) + target network + grad clip.

Da applicare solo a un eventuale **REINFORCE/PPO continuo su MCU**.

---

## FIX 6 — Convenzione `done`  ·  **DQN: verificare (probabilmente già ok)**

Sul PPO è stato allineato *quando* il flag `done` viene associato alla
transizione (schema a rollout). Sul **DQN il meccanismo è diverso**: il replay
salva esplicitamente `(s, a, r, s', done)` e il target usa
`target = done ? r : r + γ·max Q(s')`. Questa è **già la convenzione standard**
(`done` = "`s'` è terminale"), quindi di norma **non va cambiata**.

Verifica solo che nel push `replay_buffer_push(prev_obs, action, reward, obs, done)`
il `done` si riferisca a `obs` (= `s'`, lo stato successivo), e non allo stato di
partenza. Nota a parte (non legata ai nostri fix): se la tua task termina per
**time-limit** (`step_in_ep >= MAX_STEPS`), per correttezza DQN quel caso andrebbe
trattato come **troncamento** (continuare a fare bootstrap), non come stato
terminale — ma è una scelta di design separata.

---

## FIX 7 — Parità di misura + diagnostica float32  ·  **DQN: SÌ (lato driver)**

Per confrontare onestamente le curve C vs Python:

1. **Stessa reward nelle curve.** Nei driver HIL del DQN, accumulare la **stessa
   `compute_reward` che calcola il micro** (replicata in Python) invece della
   reward nativa di gym, e loggare un CSV con header coerente. Vedi
   `compute_reward()` e il logging in
   [hil_trainer_continuous.py](../Micro_RL/hil_trainer_continuous.py) /
   [hil_trainer_log.py](../Micro_RL/hil_trainer_log.py) sul branch PPO.

2. **Diagnostica precisione.** Se hai una baseline Python in NumPy del DQN, falla
   girare in **float32** (come il micro) prima di concludere che una differenza
   è "colpa del micro": spesso non lo è. Sul micro **non** alzare a `double`
   (raddoppia la RAM di reti+buffer).

---

## Checklist rapida per il branch DQN

- [ ] Copiare `rng.h` / `rng.c`; sostituire `rand()` in `dense_layer.c`, `dqn.c`
      (ε-greedy + sampling), `srand` in `main.c` → **FIX 1**
- [ ] Guard `!isfinite(gnorm)` in `gradient_norm_q` → **FIX 2**
- [ ] Guard `isfinite` + range sull'obs nel loop HIL di `main.c` → **FIX 3**
- [ ] (Consigliato) checksum+ETX in `uart_recv_floats` e nei `send_state` Python;
      flush RX solo se c'è una pausa di training lunga → **FIX 4**
- [ ] (Driver) reward custom nelle curve + eventuale baseline float32 → **FIX 7**
- [ ] Verificare la convenzione `done` del replay (di norma già corretta) → **FIX 6**
- [ ] NON serve: clamp `mu` (discreto) → **FIX 5**

## Nota su REINFORCE
- **PyTorch** ([Micro_RL/reinforce.py](../Micro_RL/reinforce.py)): nessun
  intervento — già robusto (seeding, return-normalization, grad-clip, Adam torch).
- **Eventuale porting su MCU (C)**: condividerebbe `neural_net.c`/`dense_layer.c`/
  UART, quindi applicare **FIX 1, 2, 3, 4, 7**; se l'azione è continua anche il
  **FIX 5**; la gestione `done`/ritorni segue lo schema Monte-Carlo a fine
  episodio (allinearla al PPO se si vuole coerenza).
