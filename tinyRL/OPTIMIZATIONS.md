# Ottimizzazioni tinyRL — porting su REINFORCE (STM32H743)

Porting su REINFORCE delle ottimizzazioni descritte in `../OPTIMIZATION.md`,
fatte originariamente su PPO. Stesso vincolo autoimposto: **solo trasformazioni
esatte** — nessun cambio di iperparametri, nessuna approssimazione numerica.

Bug e sprechi trovati: `BUG_FIXING.md`. Test: `test/README.md`.

## Configurazione di riferimento

| voce | valore |
|---|---|
| MCU | STM32H743ZI (Cortex-M7, FPU **scalare**), 480 MHz, I/D-cache attive |
| Toolchain | `arm-none-eabi-gcc` 13.3, `-Ofast -mfpu=fpv5-d16 -mfloat-abi=hard`, newlib-nano |
| Rete (discreta) | policy `[4, 64, 2]` — ReLU + softmax |
| Rete (continua) | policy `[4, 64, 64, 1]` — ReLU/ReLU/lineare |
| Update | Monte-Carlo, **un update per episodio**, fino a 500 step |
| Ambiente | CartPole-v1 in HIL su UART @115200 |

## File toccati

```
Core/Inc/dense_layer.h   Core/Src/dense_layer.c   arene, layout piatto
Core/Inc/neural_net.h    Core/Src/neural_net.c    kernel forward/backward, Adam, clipping
Core/Inc/reinforce.h     Core/Src/reinforce.c     arena buffer, 1/T ripiegato, inv_var, profiling
Core/Inc/uart.h          Core/Src/uart.c          send unificata + bounds check
Core/Inc/main.h          Core/Src/main.c          BENCH_KERNELS, chiamata uart
Core/Inc/utils.h         Core/Src/utils.c         include guard, clip() morta rimossa
test/                                             NUOVO: equivalenza + smoke + ASan
```

---

# Parte A — le ottimizzazioni generiche

| # | Tecnica | Stato |
|---|---|---|
| A1 | Arena contigua per layer | ✅ applicata |
| A2 | Arena unica per il buffer di esperienza | ✅ applicata al buffer di episodio |
| A3 | Blocking 2×1, accumulatori indipendenti | ✅ forward, accumulo dW, `W^T·δ` |
| A4 | Loop swap in `W^T·δ` | ✅ applicata |
| A5 | Ping-pong sul buffer del delta | ✅ **bug latente** — vedi BUG-7 |
| A6 | Attivazione in passata separata | ✅ applicata |
| A7 | Adam in forma efficiente | ✅ applicata |
| A8 | Clipping fuso nell'ottimizzatore | ✅ applicata |
| A9 | `restrict`/`const`, tipi dimensionati, doppia init | ✅ applicata |
| A10 | Macro di profiling + `BENCH_KERNELS` | ✅ applicata |
| A11 | Test di equivalenza + ASan | ✅ portato |
| A12 | API unificata, codice morto | ✅ applicata |

## A1 — Arena per layer: qui il caso peggiore del repo

Un layer intero in **una** `malloc`, con `[W|dW|mW|vW|b|db|mb|vb|out]` come
viste. L'effetto e' piu' marcato che su PPO e DQN per una ragione di topologia:
il primo layer ha `in_dim = 4`, cioe' **righe da 16 byte**, che newlib-nano
arrotonda a 24 con l'header — **+50% di spreco sulla matrice piu' grande della
rete**, ripetuto per `W`, `dW`, `mW` e `vW` su 64 righe.

Sul totale: overhead dell'allocatore **15,6% → 0,14%** dei dati utili.

⚠️ L'ordine dei blocchi non si puo' cambiare senza aggiornare anche le due
`memset` di `init_layer_params`.

## A2 — Arena unica per il buffer di episodio

Da 4 `malloc` a 1, con i campi a 4 byte prima di quelli a 1 byte:

```
[ states | rewards | returns | actions ]
```

Sparisce il **percorso di fallimento parziale**: prima, se la terza `malloc`
falliva, le prime due restavano allocate e la init tornava 0 — leak silenzioso
in un contesto senza OS. Aggiunta `episode_buffer_free`.

## A3/A4/A6 — kernel

Applicate alla lettera: blocking 2×1 con la coda per `out_dim` dispari, loop
swap in `W^T·δ`, attivazione e sua derivata in passate separate dal
matrice-vettore.

## A5 — Ping-pong sul buffer del delta ⚠️

Il bug c'era, ma **latente**: con 2 layer di pesi la topologia discreta non lo
attiva. Diventa reale con `USE_CONTINUOUS_ACTION 1` (3 layer). Dettagli in
`BUG_FIXING.md`. Le learning curve discrete raccolte finora **restano valide**.

## A7/A8 — Adam e clipping

Adam nella forma dell'Algoritmo 2 di Kingma & Ba: **una divisione e una sqrt per
peso**. `network_clip_grad` ritorna il **fattore di scala** invece di riscrivere
i gradienti — e su REINFORCE il clip scatta praticamente sempre (norma misurata
~47 contro `GRAD_CLIP` 0,5), quindi la passata risparmiata e' reale a ogni
episodio, non un caso raro.

Il `CLIP = 0.5f` era una costante locale invisibile: ora e' `GRAD_CLIP` in
`neural_net.h`.

## A9 — `restrict`, `const`, tipi dimensionati

- `restrict`/`const` su tutti i puntatori dei kernel.
- **`uint8_t` per l'azione** nel buffer, era `uint32_t`: −1,5 KB su
  `MAX_STEPS_PER_EP = 500`, con `_Static_assert(N_ACTIONS <= 255)`.
- **Nessuna doppia inizializzazione**: `dense_init` chiama gia'
  `init_layer_params`, quindi la chiamata in `network_init` e' stata tolta.

## A10 — Profiling che sparisce, `BENCH_KERNELS`

Macro `PROF_T`/`PROF_ADD` che si riducono a `((void)0)` con `TIME_LOG = 0`.
Gli accumulatori restano a **64 bit**: qui servono davvero, perche' un episodio
da 500 step puo' superare gli ~8,9 s in cui il DWT a 32 bit wrappa.

`BENCH_KERNELS` misura `expf`/`logf`/`tanhf`/`sqrtf` e il forward completo.
Su REINFORCE discreto e' particolarmente informativo: le uniche trascendenti del
percorso caldo sono gli `expf` del softmax nel forward e i `logf` dell'entropia
nel backward, e questo dice quanto pesano davvero.

## A11 — Test host + sanitizer

Portato per intero. `reinforce.c` era gia' libero dall'HAL (le UART stanno in
`uart.c` da prima), quindi non ha richiesto la separazione che era servita su
DQN. Il test gira su **entrambe** le topologie, discreta e continua.

## A12 — Pulizia dell'API e codice morto

- **`uart_send_action` unificata**: `uart_send_float_action` (n = 1, mai usata)
  e `uart_send_floats_action` fuse in una sola, con il bounds check su `n` che
  mancava.
- ⚠️ **`uart_send_action_discrete` NON e' stata fusa.** Il suo payload e' un
  byte, non un float: e' il formato che il trainer PC si aspetta in modalita'
  discreta (frame da 4 byte). Fonderla cambierebbe il protocollo sul filo.
- **`clip()` rimossa** da `utils.c` (non usata) e include guard di `utils.h`
  sistemato: gli `#include` stavano *prima* della guard.

---

# Parte B — le regole di PPO applicate a REINFORCE

## B4 → `1/T` ripiegato nel delta: il guadagno strutturale principale

E' il caso piu' netto di tutto il porting. REINFORCE accumula un contributo di
gradiente per **ogni step** dell'episodio, poi mediava con:

```c
network_scale_grad(policy, 1.f / (float)buf->size);   // passata su TUTTI i parametri
```

cioe' una read-modify-write su tutti i pesi della rete dopo ogni episodio.
Entrambi i termini della loss sono **lineari nel proprio coefficiente**:

```
L = -log π(a|s)·G  -  ent_coef·H[π]
```

quindi basta passare `G/T` e `ent_coef/T` e la media si ottiene dentro il delta
di uscita — un vettore di `N_ACTIONS = 2` elementi invece di ~400 parametri.
`network_scale_grad` **e' sparita dal codice**.

Verificato che nessun ramo condizionale dipenda dal valore scalato (in REINFORCE
non c'e' il clipping del ratio di PPO: `policy_backward` non ha rami).

> Equivalenza **algebrica, non bit-per-bit**: `(Σ x_t)·s` e `Σ (x_t·s)`
> riassociano diversamente. Verificato contro un riferimento in doppia
> precisione: la forma ripiegata e' leggermente **piu'** accurata (errore L2
> 2,58e-8 contro 3,11e-8).

## B3 → costanti per update fuori dal loop dei campioni

- **`inv_var = 1/σ²` precalcolato** (modalita' continua): σ e' fissa per tutto
  il training, eppure `s*s` e la divisione venivano rifatte a ogni step della
  traiettoria. E' l'esatto analogo del `var`/`log_var` di PPO §B3.
- **`mu` clampato in place** nel forward, cosi' il backward lo rilegge gia'
  clampato — era gia' cosi', confermato.
- **`policy_sample_action`** legge le probabilita' direttamente dal buffer `out`
  dell'ultimo layer invece di copiarle in un array locale.

## B2 → lavoro che non entra nel risultato

Due casi, entrambi puro spreco:

1. **La log-probabilita' al campionamento.** `policy_forward_continuous`
   calcolava sempre `log_prob` — `D` divisioni e `D` `logf` per campione — e poi
   la buttava via: `policy_sample_action` passa `NULL`, e **nessuno in tutto il
   branch legge quel valore**. In PPO serve (finisce nel buffer per il ratio),
   in REINFORCE no: il gradiente la ricava da `(a − μ)`. Ora e' dentro
   `if (log_prob_out)`.

2. **`logf(probs[k])` calcolato due volte** in `policy_backward`, una per
   l'entropia e una per il gradiente: `2·n_acts` `logf` per step, tutte in
   software su M7. Calcolato una volta e riusato — scostamento verificato
   **esattamente 0**.

## B1 → quantita' primarie nel buffer: gia' corretto

Il buffer contiene `states`, `actions`, `rewards`, tutte primarie. Il **re-forward**
nell'update (ricalcolare invece di cachare le attivazioni) e' la scelta
deliberata gia' documentata in `../OPTIMIZATION.md`: cachare le attivazioni degli
hidden per 500 step costerebbe 500×64×4 = **128 KB**, contro i 12 KB dell'intero
buffer.

> Nota: in REINFORCE i pesi **non cambiano** durante l'episodio, quindi il
> forward dell'update riproduce esattamente quello del campionamento. E' un
> ricalcolo evitabile solo in linea di principio — evitarlo richiederebbe le
> attivazioni di tutti i layer, cioe' i 128 KB di cui sopra.

---

# Risultati misurati

## Memoria e binario

Heap e allocazioni calcolate sulla topologia discreta `[4,64,2]` +
`MAX_STEPS_PER_EP = 500`, con il modello di newlib-nano (header 8 B,
arrotondamento a 8 B) e **puntatori a 32 bit**, come sul target.
`.text`/`.bss` da `arm-none-eabi-size` sulla build reale.

| | prima | dopo | Δ |
|---|---|---|---|
| allocazioni heap | 286 | **3** | **−99%** |
| heap totale | 24.808 B | **19.992 B** | **−19,4%** |
| overhead allocatore | 3344 B | **28 B** | **−99,2%** |
| overhead in % dei dati utili | 15,6% | **0,14%** | — |
| `.text` | 24.864 B | **24.752 B** | −112 B |
| `.bss` | 2120 B | **2112 B** | −8 B |

Scomposizione dei 4816 B di heap risparmiati:

| voce | B |
|---|---|
| header + padding delle righe delle matrici | 2112 |
| array di puntatori di riga (pura indirezione) | 1120 |
| azioni `uint32_t` → `uint8_t` nel buffer | 1500 |
| header dei vettori e del buffer (4 malloc → 1) | 104 |
| *(dato utile in meno rispetto a prima)* | *1500* |

Il grosso non e' dato eliminato ma **spreco dell'allocatore**: le righe del
layer 0 sono da 16 byte (`in_dim = 4`) e newlib le arrotonda a 24 — **8 byte
sprecati su 16 utili, il 50%**, moltiplicato per 64 righe e 4 matrici. E' il
motivo per cui qui §A1 rende piu' che su PPO e DQN.

Qui il `.text` **cala**, al contrario del porting su DQN: `network_scale_grad`,
`alloc_2d`/`free_2d` e `init_layer_params` spariscono, e quel che si guadagna
compensa lo srotolamento dei kernel.

## Tempo

⚠️ **Non ancora misurato sulla scheda.** Serve una run HIL tracciata con
`TIME_LOG = 1`.

Misura **host (x86-64, gcc -O2)** di `reinforce_update` su un episodio da 500
step, media su 2000 update:

| | prima | dopo | Δ |
|---|---|---|---|
| `reinforce_update` (x86) | 227,8 µs | **156,7 µs** | **−31%** |

Va letta per quello che e': cattura il lavoro **eliminato** (la passata di
`network_scale_grad`, i `logf` doppi, l'indirezione dei puntatori di riga, la
passata del clipping) ma **non** l'effetto principale di §A3 — le catene di
accumulo indipendenti, che contano perche' la FMA del Cortex-M7 ha latenza ~3
cicli su FPU scalare, mentre un x86 con vettorizzazione e riordino non ha lo
stesso collo di bottiglia. Sull'H743 il guadagno dovrebbe essere **maggiore**.

## Verifiche eseguite

- `make -C test` — tutti i controlli passati, scostamento relativo massimo
  **7,6e-6** (riassociazione float su §B4, gia' permessa da `-Ofast`); i
  controlli sul riuso di `logf` sono a scostamento **0**.
- `make -C test asan` — AddressSanitizer + UBSan puliti.
- Build ARM completa, **zero warning** con `-Wall -Wextra`, in **tutte e 8** le
  combinazioni di `TIME_LOG` × `BENCH_KERNELS` × `USE_CONTINUOUS_ACTION`.
- **Protocollo UART invariato**, verificato byte per byte: frame continuo
  `02 31 08 3C 3F 01 03` identico a prima, frame discreto `02 01 00 03`
  intatto (ed e' la modalita' di default).
- Regressione BUG-7 riprodotta: reintroducendo il buffer singolo il test fallisce
  sul layer 0 della topologia a 3 layer e passa su quella a 2.
- **Apprendimento verificato end-to-end**: il corpo del `while(1)` di `main.c` e'
  stato girato contro la dinamica vera di CartPole-v1, con lo stesso seed, sia
  sul codice originale sia su quello ottimizzato — 120.000 step, zero frame
  rifiutati:

| | prima | dopo |
|---|---|---|
| primi 20 episodi | 27,6 step | **27,6 step** |
| ultimi 20 episodi | 500,0 step | **500,0 step** |
| episodi in 120k scambi | 320 | 316 |

  I 500,0 step finali sono esattamente il tetto di CartPole-v1 (vedi sotto la
  convenzione di conteggio: uno step di ambiente non e' uno scambio UART).
  I primi 20 episodi coincidono **esattamente**: stesso seed, stesso stream RNG,
  stessi pesi iniziali (nessuna doppia inizializzazione introdotta). Entrambe le
  versioni convergono al tetto dei 500 step di CartPole-v1. Il piccolo scarto nel
  numero di episodi e' la divergenza attesa delle traiettorie: dopo qualche
  update la riassociazione in virgola mobile sposta un campionamento, e da li' in
  poi le due run vedono stati diversi.

---

# Convenzione di conteggio degli step

Verificata su 400 episodi (troncamenti e cadute), perche' e' facile leggere male
i numeri: **uno scambio UART non e' una transizione di ambiente.**

Su un episodio troncato al massimo:

| | valore |
|---|---|
| chiamate a `reinforce_step_discrete` (= scambi UART) | 501 |
| transizioni di ambiente effettive | **500** |
| transizioni nel buffer di episodio all'update | **500** |
| `capacity` del buffer | 500 |

Il frame in piu' e' il **handshake terminale**: quando `is_done(s_T)` scatta, il
micro deve comunque rispondere al PC con un frame per trasmettere `done = 1`.
L'azione che ci mette dentro e' campionata in uno stato gia' finito ed e'
**scartata**: `reinforce_step_discrete` non la memorizza (`first_step` torna a 1)
e il trainer fa `break` su `mcu_done` **prima** di contare la reward e prima di
steppare l'ambiente.

Invariante verificata a ogni fine episodio, su tutti e 400:

```
buffer.size != transizioni di ambiente : 0 volte
buffer.size > capacita'                : 0 volte
massimo buffer.size osservato          : 500 (capacita' 500)
```

Il buffer arriva quindi a essere **esattamente pieno**, mai oltre: e' garantito
per costruzione da `episode_buffer_init(&agent->buf, MAX_STEPS_PER_EP, OBS_DIM)`,
che lega capacita' e soglia di troncamento alla stessa costante. Se qualcuno le
disaccoppiasse, `episode_buffer_push` scarterebbe le transizioni in eccesso **in
silenzio** e i ritorni Monte-Carlo verrebbero calcolati su una traiettoria
troncata senza alcun segnale: e' il motivo per cui le due grandezze devono
restare legate a `MAX_STEPS_PER_EP`.

---

# Cosa resta da fare

- [ ] **Misurare i tempi sulla scheda** con `TIME_LOG = 1` e aggiornare il CSV.
- [ ] **Girare `BENCH_KERNELS = 1`** una volta, per sapere quanto pesano gli
      `expf` del softmax e i `logf` dell'entropia sul percorso caldo.
- [ ] Le learning curve **discrete** restano valide (BUG-7 non si attivava);
      riacquisire solo quelle eventualmente raccolte in modalita' continua.
- [ ] **Alzare il baud rate UART**: il wall-clock di una sessione HIL e'
      dominato dal round-trip seriale, non dal training.
- [ ] Valutare la fusione `rewards`/`returns` in un array solo (−2 KB) sul
      branch F446, dove la RAM e' il vincolo.
