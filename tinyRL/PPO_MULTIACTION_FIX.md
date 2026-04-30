# Fix: PPO Multi-Action Bug — Storico

---

## Fix 1 — Ratio PPO congiunto → per-dimensione

### Problema osservato
Con Pendulum (1 azione) il training funziona. Con Swimmer (2 azioni) e Hopper (3 azioni)
solo **action[0]** esplora bene; le altre collassano su un valore fisso (spike a ±1.0
negli istogrammi).

### Root cause
`log_prob_old_buffer` salvava un **singolo float** per timestep = somma delle log-prob
di tutte le N dimensioni:

```c
// VECCHIO (sbagliato)
log_prob_old = Σ_i log N(a_i | μ_i, σ)   ← scalar
ratio = exp(log_prob_new - log_prob_old) = r_0 * r_1 * r_2   ← ratio CONGIUNTO
```

Conseguenza: se **una sola** dimensione aveva ratio > 1+ε, il clip scattava e
azzerava il gradiente di **tutte** le dimensioni.

Con N=3 azioni il clip effettivo per singola dimensione è 1.2^(1/3) ≈ 1.063
invece di 1.2 → **3× più restrittivo**.

### File modificati
- `Core/Inc/reinforce.h`
- `Core/Src/reinforce.c`

### Cambiamenti

#### `Buffer` struct in `reinforce.h`
```c
// Prima:
float *log_prob_old_buffer;

// Dopo:
#if USE_CONTINUOUS_ACTIONS
  float **log_prob_per_dim_buffer; // [n_steps][action_dim]
#else
  float *log_prob_old_buffer;
#endif
```

#### `buffer_init()` in `reinforce.c`
Allocazione 2D `log_prob_per_dim_buffer[n_steps][action_dim]`
(come già fatto per `action_buffer`) + cleanup corretto nel blocco `fail:`.

#### `step()` in `reinforce.c`
```c
// Prima: salvava somma in log_prob_old_buffer[t]
log_prob_sum += gaussian_log_prob(a_clipped_i, mu_i, sigma);
buffer->log_prob_old_buffer[t] = log_prob_sum;

// Dopo: salva per-dimensione
buffer->log_prob_per_dim_buffer[t][i] = gaussian_log_prob(a_clipped_i, mu_i, sigma);
```

#### `backward_actor_critic()` in `reinforce.c`
Firma: `float old_log_prob` → `float *old_log_prob_per_dim`.
Ratio e clip check indipendenti per ogni dimensione:

```c
// Prima: ratio congiunto, un solo clip check per tutti
float ratio = exp(log_prob_new - old_log_prob);  // ratio congiunto!
uint8_t clipped = (ratio < 1+ε) ? 1 : 0;        // blocca TUTTE le dim

// Dopo: ratio e clip check indipendenti per ogni dimensione
for (int i = 0; i < out_dim; i++) {
    float log_r_i = gaussian_log_prob(a_i, mu_i_new, sigma) - old_log_prob_per_dim[i];
    float r_i = expf(clip(log_r_i, -4.0f, 4.0f));
    uint8_t active_i = (norm_adv > 0) ? (r_i < 1+ε) : (r_i > 1-ε);
    grad_i = active_i ? -(r_i * A * d_log_p_i) : 0.0f;
}
```

#### `finish_episode()` in `reinforce.c`
```c
// Prima:
float old_log_prob = buf->log_prob_old_buffer[t];

// Dopo:
#if USE_CONTINUOUS_ACTIONS
  float *old_log_prob = buf->log_prob_per_dim_buffer[t];
#else
  float *old_log_prob = &buf->log_prob_old_buffer[t];
#endif
```

### Stato dopo Fix 1
Il ratio per-dimensione è corretto e compilabile. Tuttavia i test su Hopper-v5
mostrano che action[1] e action[2] **rimangono saturate a +1.0** (spike massiccio
negli istogrammi), mentre action[0] ha una distribuzione ragionevole.
→ Fix 1 necessario ma non sufficiente. Causa residua: **saturation trap**.

---

## Fix 2 — Saturation trap: azione raw nel buffer invece della clippata

### Problema osservato
Dopo Fix 1, running Hopper-v5 (~1800 episodi):
- action[0]: distribuzione decente centrata intorno a 0, con spike moderato a +1
- action[1] e action[2]: spike enorme a +1.0 (frequenza ~5×10⁴), distribuzione quasi piatta altrove

### Root cause

Dopo Fix 1 rimane un bug sottile in `step()`:

```c
float a_clipped_i = fmaxf(fminf(a_raw_i, 1.0f), -1.0f);
buffer->action_buffer[t][i]          = a_clipped_i;   // ← CLIPPATA
buffer->log_prob_per_dim_buffer[t][i] = gaussian_log_prob(a_clipped_i, mu_i, sigma); // ← log prob CLIPPATA
```

Quando `μ_i → +1.0` (tanh saturo) le azioni vengono clippate a `+1.0`.
In `backward_actor_critic()`:

```c
float d_log_prob_i = (action_buf[i] - mu_i) / (old_sigma * old_sigma);
//                    = (1.0 - 1.0) / σ²  =  0   ← GRADIENTE ZERO PERMANENTE
```

Il ratio rimane ≈ 1 (entrambe le log-prob al picco della Gaussiana), il clip check
è sempre attivo, ma il gradiente è zero → la policy non può mai uscire dalla
saturazione.

### Perché action[0] sopravvive
action[0] riceve un segnale di reward più forte (correlato con la velocità forward)
e in fase iniziale del training viene aggiornato prima che μ_0 saturi, accumulando
gradienti sufficienti. action[1] e action[2] entrano nella trappola prima.

### Fix applicato

**In `step()`, salvare l'azione grezza (pre-clip) nel buffer:**

```c
// Prima:
buffer->action_buffer[*step_count][i] = a_clipped_i;
buffer->log_prob_per_dim_buffer[*step_count][i] = gaussian_log_prob(a_clipped_i, mu_i, sigma);

// Dopo:
buffer->action_buffer[*step_count][i] = a_raw_i;     // ← RAW
buffer->log_prob_per_dim_buffer[*step_count][i] = gaussian_log_prob(a_raw_i, mu_i, sigma); // ← log prob RAW
```

`out_action[i] = a_clipped_i` rimane invariato (l'ambiente riceve ancora l'azione clippata).

**Nessuna modifica in `backward_actor_critic()`**: usa già `action_buf[i]` per
log prob e gradiente; ora `action_buf[i] = a_raw_i` automaticamente.

### Perché è corretto

La policy è definita come "campiona `a_raw ~ N(μ, σ)`, clippa per l'ambiente".
Il ratio PPO e il gradiente devono riferirsi al **campione effettivo** `a_raw`:

```
ratio_i = π_new(a_raw | μ_new, σ) / π_old(a_raw | μ_old, σ)   ← corretto
d_log_prob_i = (a_raw - μ) / σ²                                 ← mai zero in distribuzione
```

Anche con `μ → 1.0`, il campione `a_raw ~ N(1.0, σ)` è tipicamente `1.0 ± σ`,
quindi il gradiente `(a_raw - μ)/σ²` ha media 0 ma varianza `1/σ²` → la policy
riceve segnale e può aggiornarsi.

### File modificati
- `Core/Src/reinforce.c` — due righe nel loop di `step()` (righe ~326-328)

---

## Stato attuale
- Fix 1 (ratio per-dimensione): **implementato**
- Fix 2 (azione raw nel buffer): **implementato**
- Da testare: rieseguire Hopper-v5 e confrontare istogrammi delle azioni

## Possibili miglioramenti futuri
- **SAC-style squashed Gaussian**: campionare in spazio non bounded,
  applicare `tanh` come squashing, correggere log-prob con
  `log π(a) = log N(a_raw | μ, σ) - Σ log(1 - tanh²(a_raw))`.
  Più corretto teoricamente, richiede refactor dell'output layer (rimuovere tanh finale).
- **Entropy bonus per dimensione**: `ENT_BETA * σ²` per scoraggiare il collasso
  della distribuzione verso sigma piccolo (agisce direttamente sulla sigma, non su mu).
- **Gradient clipping per-dimensione**: attualmente `gradient_norm_l2` normalizza
  il gradiente globale della rete; un clipping separato per l'actor head potrebbe
  aiutare le dimensioni con segnale debole.
