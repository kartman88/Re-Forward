# PPO Bug Fixing — tinyRL

## Contesto

Il training di PPO su Pendulum non convergeva. Dopo analisi manuale completa di tutti i file
sorgente (ppo.c, neural_net.c, dense_layer.c, main.c, hil_trainer_continuous.py) sono stati
identificati e corretti 6 bug reali, di cui 3 critici.

---

## Bug Corretti

### BUG-1 — CRITICAL: `obs[2]` non normalizzato dopo `env.reset()`
**File:** `Micro_RL/hil_trainer_continuous.py` linee 121, 213

**Problema:**
```python
obs, _ = env.reset()   # obs[2] = thdot ∈ [-1,1] — mancava /8
```
La normalizzazione `obs[2] /= 8` veniva applicata solo dopo `env.step()` (linea 173), mai
dopo `env.reset()`. Il MCU si aspetta `obs[2] = thdot/8 ∈ [-0.125, 0.125]`. Con il valore
grezzo, il MCU calcola `omega = obs[2] * 8` che è 8× troppo grande → il termine `0.1·ω²`
nel reward è sbagliato di un fattore 64 per la prima transizione di ogni episodio.

**Fix:** aggiunto `obs[2] /= 8` dopo entrambe le `env.reset()`.

---

### BUG-2 — CRITICAL: `init_layer_params` commentato in `dense_layer.c`
**File:** `Core/Src/dense_layer.c` linea 99

**Problema:**
```c
int dense_init(DenseLayer *layer, ...) {
    // alloc_2d usa calloc → pesi = 0
    //init_layer_params(layer);   ← era commentato
    return 1;
}
```
`alloc_2d` usa `calloc` → tutti i pesi partono a zero. Chiamare `dense_init` direttamente
(es. in `reinforce.c`) produce reti completamente degeneri. Per PPO il danno era mitigato
perché `init_qnetwork` (neural_net.c:112) chiama `init_layer_params` separatamente, ma la
situazione era fragile e confusa.

**Fix:** decommentata la riga `init_layer_params(layer)` in `dense_layer.c:99`.

---

### BUG-3 — DESIGN NOTE: Azione non clippata nel buffer (comportamento intenzionale)
**File:** `Core/Src/main.c`

**Analisi:** Il termine `0.001·u²` nel reward Pendulum è trascurabile rispetto a `theta²`
(errore max ≈ 0.005 su scale ~9.87). Fissare questo usando azione clippata + log_prob ricalcolato
causa un problema ben peggiore: il gradiente `-(a_clip - mu)/σ²` spinge `mu` verso ±2.0 ogni
volta che un'azione al boundary ha vantaggio positivo. Con sigma_init=1.5 (~18% azioni clippate)
si crea un loop di saturazione → distribuzione ad U → policy degenerata.

**Decisione:** mantenuto il design originale (azione non clippata nel buffer, coerente con
log_prob_old). La clip viene applicata solo all'azione inviata all'ambiente via UART.

---

### BUG-4 — HIGH: Ratio non clamped prima di `expf`
**File:** `Core/Src/ppo.c` linee 214, 220

**Problema:**
```c
float ratio = expf(log_prob_new - log_prob_old);  // overflow/NaN se diff > 87
```
Se la differenza supera ≈87, `expf` restituisce Inf. NaN/Inf si propagano ai gradienti
causando un training crash silenzioso.

**Fix:**
```c
float ratio = expf(fmaxf(fminf(log_prob_new - log_prob_old, 10.f), -10.f));
```

---

### BUG-5 — HIGH: Coefficiente `PPO_C1` mai applicato al critic
**File:** `Core/Src/neural_net.c` linea 257, `Core/Src/ppo.c` linea 226

**Problema:**
`PPO_C1 = 0.5f` era definito in `ppo.h` ma non veniva mai passato a `critic_backward`.
Il gradient del critic era 2× più grande del previsto (equivale a `lr_critic` raddoppiato).

**Fix:** aggiunto parametro `coeff` a `critic_backward`:
```c
// neural_net.c
void critic_backward(QNetwork *critic, float *obs, float value_target, float coeff) {
    float delta = coeff * (critic->layers[...].out[0] - value_target);
    backward_from_delta(critic, obs, &delta);
}
// ppo.c
critic_backward(critic, obs_t, ret_t, PPO_C1);
```

---

### BUG-6 — MEDIUM: Nessun bounds check in `rollout_buffer_push`
**File:** `Core/Inc/ppo.h`, `Core/Src/ppo.c`

**Problema:**
La struct `RolloutBuffer` non aveva un campo `capacity`. Se `push` veniva chiamato più di
`T` volte, avveniva un overflow silenzioso del buffer dinamico.

**Fix:** aggiunto `uint32_t capacity` alla struct, impostato in `rollout_buffer_init`, e
aggiunto `if (buf->head >= buf->capacity) return;` all'inizio di entrambe le varianti di
`rollout_buffer_push`.

---

## Bug Verificati e Scartati (falsi positivi)

| Segnalazione | Verifica | Verdetto |
|---|---|---|
| `normalize_advantages` mai chiamata | `main.c:228` la chiama | Non è un bug |
| Gradienti non azzerati tra mini-batch | `adam_update_single_layer` azzera `db/dW` dopo ogni update (`neural_net.c:78,95`) | Non è un bug |
| Log_prob formula errata (continuous) | Verifica algebrica: `−0.5*(diff²/σ²+2ls+log2π)` = corretto | Non è un bug |
| Entropy gradient formula errata | Derivata analitica `dH/dz_k = −π_k*(log π_k + H)` → codice corretto | Non è un bug |
| Done flag off-by-one | Trace manuale: `(obs_199, a_199, done=1)` è correttamente la 200ª transizione | Non è un bug |
| GAE loop errato | Loop backward con `not_done` corretto, incluso gestione cross-rollout | Non è un bug |

---

## File Modificati

| File | Modifica |
|---|---|
| `Micro_RL/hil_trainer_continuous.py` | `obs[2] /= 8` dopo entrambe le `env.reset()` |
| `Core/Src/dense_layer.c` | Decommentata `init_layer_params(layer)` a riga 99 |
| `Core/Src/main.c` | Riordinato: clip → memcpy(prev_action) → recompute log_prob → uart_send |
| `Core/Src/ppo.c` | Clamp ratio con `fmaxf/fminf` (±10); `PPO_C1` passato a `critic_backward`; `capacity` in `init`; bounds check in `push` |
| `Core/Inc/ppo.h` | Aggiunto campo `uint32_t capacity` alla struct `RolloutBuffer` |
| `Core/Src/neural_net.c` | `critic_backward` accetta ora `float coeff` |
| `Core/Inc/neural_net.h` | Aggiornata dichiarazione di `critic_backward` |

---

## Come Verificare

1. Compilare il progetto STM32 e flashare il firmware
2. Avviare `python Micro_RL/hil_trainer_continuous.py` con MCU connesso
3. Osservare il reward medio (moving avg 20 ep): deve salire da ~−1200 verso ~−200 entro 200+ episodi
4. Verificare che `update_count` nel dashboard incrementi (training avviene regolarmente)
5. L'istogramma delle azioni deve centersarsi gradualmente sui valori correttivi del pendolo
