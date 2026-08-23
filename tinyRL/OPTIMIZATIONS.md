# Ottimizzazioni tinyRL — porting su DQN (STM32H743)

Porting su DQN delle ottimizzazioni descritte in `../OPTIMIZATION.md`, fatte
originariamente su PPO. Stesso vincolo autoimposto: **solo trasformazioni
esatte** — nessun cambio di iperparametri, nessuna approssimazione numerica, in
modo che ogni regressione sia imputabile al codice e non all'algoritmo.

Bug trovati durante il lavoro: `BUG_FIXING.md`. Test: `test/README.md`.

## Configurazione di riferimento

| voce | valore |
|---|---|
| MCU | STM32H743ZI (Cortex-M7, FPU **scalare**), 480 MHz, I/D-cache attive |
| Toolchain | `arm-none-eabi-gcc` 13.3, `-Ofast -mfpu=fpv5-d16 -mfloat-abi=hard`, newlib-nano |
| Reti | online `[3, 32, 32, 5]` (ReLU/ReLU/lineare) + target identica |
| Update | `BATCH_SIZE = 32`, `REPLAY_SIZE = 1000`, `TARGET_UPDATE = 100` |
| Ambiente | Pendulum-v1 discretizzato a 5 coppie, HIL su UART @115200 |

## File toccati

```
Core/Inc/dense_layer.h   Core/Src/dense_layer.c   arene, layout piatto
Core/Inc/neural_net.h    Core/Src/neural_net.c    kernel forward/backward, Adam, clipping, arena target
Core/Inc/dqn.h           Core/Src/dqn.c           arena replay, tipi, profiling, forward saltato
Core/Inc/uart.h          Core/Src/uart.c          NUOVI: UART fuori da dqn.c, send unificata
Core/Inc/main.h          Core/Src/main.c          BENCH_KERNELS, chiamata uart unificata
Core/Inc/utils.h         Core/Src/utils.c         include guard, clip() morta rimossa
Core/Inc/reinforce.h     Core/Src/reinforce.c     RIMOSSI
test/                                             NUOVO: equivalenza + smoke + ASan
Debug/Core/Src/subdir.mk Debug/objects.list       lista sorgenti aggiornata
```

---

# Parte A — le ottimizzazioni generiche, applicate

| # | Tecnica | Stato su DQN |
|---|---|---|
| A1 | Arena contigua per layer | ✅ applicata, **estesa anche alla rete target** |
| A2 | Arena unica per il buffer di esperienza | ✅ applicata al replay buffer |
| A3 | Blocking 2×1, accumulatori indipendenti | ✅ forward, accumulo dW, `W^T·δ` |
| A4 | Loop swap in `W^T·δ` | ✅ applicata |
| A5 | Ping-pong sul buffer del delta | ✅ **il bug c'era** — vedi BUG-7 |
| A6 | Attivazione in passata separata | ✅ applicata (forward e derivata) |
| A7 | Adam in forma efficiente | ✅ applicata |
| A8 | Clipping fuso nell'ottimizzatore | ✅ applicata |
| A9 | `restrict`/`const`, tipi dimensionati, doppia init | ✅ applicata |
| A10 | Macro di profiling + `BENCH_KERNELS` | ✅ applicata |
| A11 | Test di equivalenza + ASan | ✅ portato, ha trovato BUG-7 |
| A12 | API unificata, codice morto | ✅ applicata |

## A1 — Un'unica arena per layer, **rete target inclusa**

Oltre al `DenseLayer` della rete online (identico a PPO: `[W|dW|mW|vW|b|db|mb|vb|out]`
in una `malloc`), qui c'e' una seconda rete: il `TargetLayer`, che essendo di
sola lettura ha bisogno solo di `[W|b|out]`. Anche quello passa da
`1 + out_dim + 2` allocazioni a **una sola**.

⚠️ L'ordine dei blocchi non si puo' cambiare senza aggiornare anche le due
`memset` di `init_layer_params`, che azzerano `dW|mW|vW` e `b|db|mb|vb` in blocco.

Effetto collaterale: `copy_weights_to_target` passa da `out_dim + 1` memcpy per
layer a **due** (`W` intero, `b` intero) — e con arene piatte da entrambi i lati
la copia e' bit-per-bit, verificata in `test_update`.

## A2 — Arena unica per il replay buffer

Il replay buffer era **7 malloc**, due delle quali (`state[]` e `next_state[]`)
solo array di puntatori di riga sopra pool gia' contigui: **8 KB di pura
indirezione**, piu' una load in ogni accesso. Sostituiti da due `static inline`
che calcolano l'offset:

```c
replay_state(buf, i)      →  buf->states      + i*obs_dim
replay_next_state(buf, i) →  buf->next_states + i*obs_dim
```

Layout con i campi a 4 byte prima di quelli a 1 byte, cosi' l'aritmetica dei
puntatori resta allineata senza padding esplicito:

```
[ states | next_states | rewards | actions | dones ]
```

Sparisce anche il **percorso di fallimento parziale**: prima, se la sesta malloc
falliva, le prime cinque restavano allocate e la init tornava 0 — leak silenzioso
in un contesto senza OS.

> **Differenza rispetto a PPO:** li' `head` e `size` erano uguali per costruzione
> e sono stati fusi. Qui **no**: il buffer e' circolare, `head` torna a 0 mentre
> `size` si ferma a `capacity`. Restano due campi.

## A3/A4/A6 — kernel di forward e backward

Applicate alla lettera. In piu', **rete online e rete target ora condividono lo
stesso kernel**: `forward_dense_layer` e `forward_target_layer` erano due copie
identiche a meno del tipo di struct, con l'ovvio rischio di divergere. Ora c'e'
un solo `dense_matvec(W, b, in, out, in_dim, out_dim, act)` che prende i
puntatori sciolti, quindi il blocking 2×1 e' scritto una volta sola.

## A5 — Ping-pong sul buffer del delta ⚠️

**Il bug c'era.** La rete DQN ha 3 layer di pesi, cioe' esattamente la condizione
descritta in `../OPTIMIZATION.md`: dal secondo layer in poi sorgente e
destinazione del delta erano lo stesso buffer, e il gradiente del **layer 0**
usciva sistematicamente sbagliato — senza NaN, senza crash, senza alcun sintomo
osservabile dal training. Dettagli e riproduzione in `BUG_FIXING.md`.

Corretta anche la `malloc` a meta' passata (use-after-free latente): la capacita'
si dimensiona ora una volta sola, sul layer piu' largo, prima di iniziare.

## A7/A8 — Adam e clipping

Adam in forma dell'Algoritmo 2 di Kingma & Ba: **una divisione e una sqrt per
peso** invece di tre divisioni e una sqrt. `gradient_norm_q` ritorna ora il
**fattore di scala** invece di riscrivere tutti i gradienti, e Adam lo applica al
volo mentre li carica: una passata completa sui parametri in meno ogni volta che
il clip scatta — cioe' quasi sempre, visto che `GRAD_CLIP` e' 0.5 e la norma
misurata sul batch e' tipicamente ~2.

Il `CLIP = 0.5f` era una costante locale invisibile da fuori: ora e' `GRAD_CLIP`
in `neural_net.h`.

## A9 — `restrict`, `const`, tipi dimensionati

- `restrict`/`const` su tutti i puntatori dei kernel. Senza, il compilatore deve
  assumere che scrivere `dst[j]` possa modificare `w_row[j]`: tutto il lavoro di
  §A3 non si vedrebbe.
- **`uint8_t` per l'azione nel replay buffer**, era `uint32_t`: −3 KB su
  `REPLAY_SIZE = 1000`. Con `_Static_assert(N_ACTIONS <= 255)` che lega il tipo
  alla costante che lo giustifica. E' il caso che `../OPTIMIZATION.md` §A9 indica
  esplicitamente come "su DQN pesa di piu'".
- **Nessuna doppia inizializzazione**: `dense_init` chiama gia'
  `init_layer_params`, quindi la chiamata in `init_qnetwork` e' stata tolta.
  (Nella versione precedente la chiamata dentro `dense_init` era commentata, per
  cui il bug non c'era: e' una regressione **evitata**, non corretta.)

## A10 — Profiling che sparisce, `BENCH_KERNELS`

Macro `PROF_T`/`PROF_ADD` che si riducono a `((void)0)` con `TIME_LOG = 0`, senza
lasciare riferimenti a variabili inesistenti: il loop di `dqn_train` resta
leggibile invece di essere spezzato da una dozzina di `#if`. Le tre
configurazioni (`TIME_LOG` 0/1 × `BENCH_KERNELS` 0/1) compilano tutte pulite con
`-Wall -Wextra`.

`BENCH_KERNELS` misura il costo per chiamata di `tanhf`/`expf`/`logf`/`sqrtf` e
dei due forward completi. Su DQN ci si aspetta un esito **diverso da PPO**: qui
gli hidden sono ReLU, non tanh, quindi il forward dovrebbe risultare quasi tutto
MAC e il "tetto delle trascendenti" che limitava PPO al −11% non dovrebbe esserci.
E' il primo numero da guardare per confermarlo sulla scheda.

I contatori di profiling restano a **32 bit**: il DWT wrappa ogni ~8,9 s a
480 MHz e un update DQN (batch 32) sta nei millisecondi, quindi non serve
l'accumulo a 64 bit che serviva a PPO.

## A11 — Test host di equivalenza + sanitizer

Portato per intero, ed e' **il pezzo che ha trovato BUG-7**. Ha richiesto un
prerequisito: `dqn.c` conteneva le funzioni UART, cioe' l'unico punto in cui il
file dell'algoritmo toccava l'HAL. Spostate in `uart.c` (§A12), `dqn.c`,
`neural_net.c`, `dense_layer.c` e `rng.c` compilano in nativo con due header
stub vuoti.

## A12 — Pulizia dell'API e codice morto

- **`uart_send_action` unificata**: `uart_send_float_action` (scalare) e
  `uart_send_action_discrete` (mai chiamata) fuse nella firma di PPO
  `uart_send_action(huart, actions, n, done, timeout)`; per il caso scalare si
  passa `&val, 1`. Con il **bounds check su `n`** che nella variante vettoriale
  di PPO mancava.
- **`clip()` rimossa** da `utils.c` (non usata da nessuno) e include guard di
  `utils.h` sistemato: gli `#include` stavano *prima* della guard, quindi
  venivano riprocessati ad ogni inclusione.
- **`reinforce.c/h` rimossi**: erano due file con dentro un commento
  ("Replaced by dqn.c in Task 3"), compilati e linkati a vuoto.

---

# Parte B — le regole di PPO applicate a DQN

Le §B di `../OPTIMIZATION.md` nascono dalla matematica di PPO. Ecco cosa hanno
prodotto qui, seguendo la regola trasferibile di ciascuna.

## B1 → quantita' primarie nel buffer: **gia' corretto**

La regola dice: cercare nel buffer le quantita' *derivate* al posto di quelle
*primarie*. Il caso analogo indicato per DQN e' "salvare l'indice dell'azione
anziche' il Q-value da cui e' stata scelta" — ed e' gia' cosi'. Nessun intervento.

## B2 → termini che si cancellano: **forward della rete target saltato**

L'analogo del Jacobiano che si cancella nel ratio e' il termine che **non entra
mai nel risultato**:

```c
float target_val = dn ? rew : rew + GAMMA * max_qt;
```

Su una transizione terminale `max_qt` viene scartato — ma `forward_target()`,
un **forward di rete intero**, era stato eseguito comunque per calcolarlo. Ora:

```c
float target_val = rew;
if (!dn) {
    forward_target(target, s_next, NULL);
    ...
    target_val = rew + GAMMA * max_qt;
}
```

Esatta per costruzione. Con `MAX_STEPS_PER_EP = 200` risparmia ~0,5% dei forward;
su task a episodi corti (o con terminazione anticipata) vale molto di piu'.

## B2-bis → **backward sparso**: il vero guadagno specifico di DQN

Il gradiente della loss DQN e' non nullo **solo su `Q(s,a)`**: 1 elemento su
`N_ACTIONS = 5`. La versione precedente costruiva comunque un vettore denso
`delta_out[5]` con quattro zeri e lo dava in pasto al backward, che per
l'ultimo layer moltiplicava 4 righe su 5 per zero:

- `db[i] += 0` per 4 valori di `i`;
- `dW[i][:] += 0 * inp[:]` per 4 righe da 32 pesi;
- `dst[j] = Σ_i δ[i]·W[i][j]` con 4 termini nulli su 5.

`backward_engine` accetta ora un `sparse_idx`: quando e' ≥ 0 tocca **una sola
riga** e assegna `dst[j] = d·W[idx][j]` direttamente, senza `memset` e senza
sommare i termini nulli. E' **esatta**, non un'approssimazione: sommare `0.0f` a
un accumulatore inizializzato a zero non cambia nulla, e il test lo verifica con
tolleranza **0** (sezione [3] di `test_equiv`, scostamento 0 su tutti i layer).

Toglie l'80% del lavoro sull'ultimo layer, ~10% del backward totale.

> La via densa resta viva come `dqn_backward_delta()`: e' il riferimento contro
> cui il test confronta quella sparsa. E' lo stesso principio di §A11 — la
> versione che si sostituisce va tenuta viva in un test.

## B3 → costante per update fuori dal loop dei campioni

`inv_batch` era gia' fuori. In piu': i puntatori alle uscite delle due reti
(`q_online`, `q_tgt`) sono ora presi **una volta** prima del loop invece di
essere ricalcolati per campione, e i due `memcpy` di `forward_q`/`forward_target`
verso array locali sono spariti — passando `NULL` il chiamante legge direttamente
il buffer `out` dell'ultimo layer. Stessa cosa in `dqn_select_action`.

Il `powf` della bias correction di Adam era gia' una volta per update.

## B4 → fattore di scala ripiegato nel delta: **gia' corretto**

`dqn_backward(online, s, act, td_error * inv_batch)`: il `1/batch_size` viaggia
gia' dentro il delta di uscita (un solo float) invece di scalare a valle tutti i
parametri. Verificato che il ramo condizionale del clipping dipenda da `gnorm` e
non dall'advantage, quindi la scala non attraversa un ramo che dipende dal valore
scalato. Nessun intervento.

---

# Risultati misurati

## Memoria e binario

Heap e allocazioni calcolate su `[3,32,32,5]` + target + `REPLAY_SIZE = 1000`,
con il modello di newlib-nano (header 8 B, arrotondamento a 8 B).
`.text`/`.bss` da `arm-none-eabi-size` sulla build reale.

| | prima | dopo | Δ |
|---|---|---|---|
| allocazioni heap | 388 | **7** | **−98%** |
| heap totale | 75,1 KB | **57,6 KB** | **−23,3%** |
| overhead allocatore | 3772 B | **60 B** | −98% |
| `.bss` | 3304 B | **3280 B** | −0,7% |
| `.text` | 25.884 B | 26.180 B | **+296 B** |

Il `.text` **cresce**, al contrario di PPO (che ne guadagnava rimuovendo
`reinforce.c` e fondendo due funzioni UART). Il dettaglio per simbolo dice dove:

```
backward_engine     0 -> 1092   (assorbe dqn_backward, 420, + percorso sparso + code)
forward_q         356 ->  492   (blocking 2x1 con coda + attivazione in passata separata)
forward_target    352 ->  492   (idem)
init_layer_params 258 ->    0   (collassata nel layout piatto)
alloc_2d          110 ->    0   (rimossa)
gradient_norm_q   240 ->  136   (non riscrive piu' i gradienti)
replay_buffer_init 214 ->  84   (una malloc invece di sette)
```

Cioe': i 296 B in piu' sono **esattamente il prezzo dello srotolamento**, pagato
per il tempo. Su 2 MB di flash non e' un vincolo.

## Tempo

⚠️ **Non ancora misurato sulla scheda.** Serve una run HIL tracciata con
`TIME_LOG = 1` per aggiornare il CSV dei timing.

Misura **host (x86-64, gcc -O2)** di `dqn_train` sullo stesso replay buffer,
media su 20.000 update:

| | prima | dopo | Δ |
|---|---|---|---|
| `dqn_train` (x86) | 74,6 µs | **40,0 µs** | **−46%** |

Va letta per quello che e': cattura il lavoro **eliminato** (forward target
saltati, righe sparse, passate su tutti i parametri rimosse, indirezione dei
puntatori di riga) ma **non** l'effetto principale di §A3 — le catene di
accumulo indipendenti, che contano perche' la FMA del Cortex-M7 ha latenza ~3
cicli su FPU scalare, mentre un x86 con vettorizzazione e riordino non ha lo
stesso collo di bottiglia. Sull'H743 il guadagno dovrebbe essere **maggiore**,
non minore.

Ci si aspetta inoltre un miglioramento relativo **superiore a quello di PPO
(−11%)**, perche' DQN non ha il tetto delle trascendenti: gli hidden sono ReLU
e non ci sono `tanhf`/`expf`/`logf`/`atanhf` nel percorso caldo. `BENCH_KERNELS`
lo conferma sulla scheda.

## Verifiche eseguite

- `make -C test` — tutti i controlli passati, scostamento relativo massimo
  **1,75e-6** (solo sul ramo tanh, riassociazione float gia' permessa da `-Ofast`);
  il confronto sparso/denso e' a scostamento **0**.
- `make -C test asan` — AddressSanitizer + UBSan puliti su entrambi i test.
- Build ARM completa, **zero warning** con `-Wall -Wextra`, nelle tre
  configurazioni `TIME_LOG`/`BENCH_KERNELS`.
- Regressione BUG-7 riprodotta e verificata: reintroducendo il buffer singolo il
  test fallisce sul layer 0 e passa sui layer 1 e 2.

---

# Cosa resta da fare

- [ ] **Riacquisire le learning curve** — quelle esistenti vengono da codice con
      il gradiente del primo layer sbagliato (BUG-7).
- [ ] **Misurare i tempi sulla scheda** con `TIME_LOG = 1` e aggiornare il CSV.
- [ ] **Girare `BENCH_KERNELS = 1`** una volta, per confermare che senza tanh il
      forward sia dominato dalle MAC.
- [ ] **Alzare il baud rate UART**: a 115200 il wall-clock di una sessione HIL e'
      dominato dal round-trip seriale, non dal training. E' l'intervento piu'
      efficace sul tempo totale di un esperimento e non tocca l'algoritmo.
- [ ] Riportare le stesse modifiche su `DQN_F446`, dove il risparmio di heap
      (−17,5 KB su 128 KB di SRAM) conta molto di piu' che sull'H7.
