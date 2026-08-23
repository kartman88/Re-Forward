# Ottimizzazioni tinyRL — sessione di ottimizzazione PPO su STM32H743

Questo documento raccoglie **le ottimizzazioni dell'ultima sessione di lavoro** su
`tinyRL/` (PPO su STM32H743), cioè tutto ciò che al momento della scrittura è ancora
**non committato nel working tree**.

È scritto per essere **riusato sui porting con altri algoritmi** (DQN, REINFORCE, SAC,
TD3, …), quindi è diviso in due parti:

- **[Parte A — Generiche](#parte-a--ottimizzazioni-generiche)**: valgono per qualsiasi
  algoritmo che addestri una rete densa on-device. Si copiano quasi alla lettera.
- **[Parte B — Specifiche di PPO](#parte-b--ottimizzazioni-specifiche-di-ppo)**: nascono
  dalla matematica di PPO, ma ognuna chiude con la **regola trasferibile**, cioè cosa
  cercare nell'algoritmo che si sta portando.

Cosa era **già** nel codebase prima di questa sessione è elencato in
[§ Fuori perimetro](#fuori-perimetro--cosa-cera-gi%C3%A0) — serve saperlo, perché su un
nuovo target va rifatto anche quello.

Dettaglio con i frammenti prima/dopo: `tinyRL/OPTIMIZATIONS.md`.
Bug trovati durante il lavoro: `tinyRL/BUG_FIXING.md`.

---

## Configurazione di riferimento

| voce | valore |
|---|---|
| MCU | STM32H743ZI (Cortex-M7, FPU **scalare**), 480 MHz, I/D-cache attive |
| Toolchain | `arm-none-eabi-gcc`, `-Ofast -mfpu=fpv5-d16 -mfloat-abi=hard`, newlib-nano |
| Reti | actor `[11, 32, 32, 3]`, critic `[11, 32, 32, 1]` |
| Rollout | `ROLLOUT_STEPS = 512`, `PPO_EPOCHS = 8`, `PPO_BATCH_SIZE = 64` |
| Ambiente | Hopper-v4 in HIL su UART @115200 (PC = simulatore, MCU = agente) |

**Vincolo autoimposto per tutta la sessione: solo trasformazioni esatte.** Nessun cambio
di iperparametri, nessuna approssimazione numerica — così le learning curve prima/dopo
restano confrontabili e ogni regressione è imputabile al codice, non all'algoritmo.

---

## File toccati

```
Core/Inc/dense_layer.h   Core/Src/dense_layer.c    arene, layout piatto
Core/Inc/neural_net.h    Core/Src/neural_net.c     kernel forward/backward, Adam, clipping
Core/Inc/ppo.h           Core/Src/ppo.c            arena buffer, z, var/log_var, inv_bsz, profiling
Core/Inc/main.h          Core/Src/main.c           BENCH_KERNELS, chiamate uart
Core/Inc/uart.h          Core/Src/uart.c           uart_send_action unificata
Core/Inc/utils.h         Core/Src/utils.c          pulizia (clip() morta, include guard)
Core/Inc/reinforce.h     Core/Src/reinforce.c      rimossi
test/                                              nuovo: equivalenza + ASan
```

---

# Parte A — Ottimizzazioni generiche

## A1 — Un'unica arena contigua per layer

**File:** `dense_layer.c/h`

La versione a `float **` allocava, per ognuna delle quattro matrici (`W`, `dW`, `mW`,
`vW`), un array di puntatori di riga più una `calloc` per riga:

```c
*matrix = malloc(rows * sizeof(float *));
for (int i = 0; i < rows; ++i)
    (*matrix)[i] = calloc(cols, sizeof(float));   // rows allocazioni
```

Per actor + critic sono **592 allocazioni** e 94,1 KB di heap per 86,4 KB di dati utili.
Lo spreco è quasi tutto padding dell'allocatore: una riga da 11 float occupa 44 byte, che
newlib-nano arrotonda a 48 più 8 di header = **56 byte, +27%** sulle matrici del primo
layer. I soli array di puntatori di riga sono 2,1 KB.

Ora ogni layer vive in **una sola `malloc`**, con le 4 matrici e i 5 vettori come viste
dentro l'arena, matrici in row-major (`W[i*in_dim + j]`):

```
arena = [ W | dW | mW | vW | b | db | mb | vb | out ]
```

| | prima | dopo |
|---|---|---|
| allocazioni | 592 | 9 |
| heap | 94,1 KB | 86,4 KB |
| overhead allocatore | 7,8 KB | 0,1 KB |

Il beneficio non è solo di spazio: sparisce **un livello di indirezione da ogni loop
interno** (prima ogni riga costava una load del puntatore prima di leggere i dati) e le
righe sono contigue, quindi il prefetcher della D-cache lavora. Su H743 i 7,7 KB non si
notano fra 512 KB; su F446 — 128 KB per ~96 KB di working set — sono la differenza fra
starci e non starci.

Effetto collaterale gradito: l'ordine fisso dei blocchi permette di azzerare gradienti e
momenti con **due `memset`** invece di 9 loop (`memset(dW, 0, 3*n_w*4)` copre `dW|mW|vW`).
⚠️ Vincolo da documentare nel codice: quell'ordine non si può cambiare senza aggiornare
anche le `memset`.

> **Regola generale:** ogni struttura 2D con dimensioni note a init-time va in un unico
> blocco piatto con indicizzazione `i*stride + j`. L'array di puntatori di riga costa
> memoria, frammenta l'heap e aggiunge una dipendenza di load nel loop più caldo.

**Trasferibilità:** immediata. Vale identica per la Q-network di DQN (con la sua target
network, che raddoppia il conto), per la policy di REINFORCE, per i due critic di TD3/SAC.

---

## A2 — Un'unica arena anche per il buffer di esperienza

**File:** `ppo.c` → `rollout_buffer_init`, `ppo.h`

Stesso trattamento per il rollout buffer: **8 `malloc` → 1**. Oltre all'overhead, sparisce
il **percorso di fallimento parziale**: prima, se la settima `malloc` falliva, le prime sei
restavano allocate e la init tornava 0 — leak silenzioso in un contesto senza OS.

Nota di layout: i campi a 4 byte vanno raggruppati prima di quelli a 1 byte (`dones`), così
l'aritmetica dei puntatori resta allineata senza padding esplicito.

Semplificato anche lo stato: `size` fa ora insieme da contatore di transizioni valide e da
testa di scrittura — i due campi separati (`head` e `size`) erano uguali per costruzione, e
due contatori che devono restare sincronizzati sono un bug in attesa.

> **Regola generale:** un buffer di N vettori paralleli è una sola allocazione con N viste.
> E due contatori sempre uguali per costruzione sono un contatore solo.

**Trasferibilità:** il replay buffer di DQN/SAC/TD3 è esattamente questo caso, con in più
`next_states`. Lì l'arena unica pesa molto di più, perché il replay buffer è la struttura
dominante in RAM.

---

## A3 — Blocking 2×1 e catene di accumulo indipendenti

**File:** `neural_net.c` → `forward_dense_layer`, `backward_from_delta`

È la singola ottimizzazione di tempo con il rendimento più alto. Il prodotto scalare
scritto nel modo ovvio è una catena FMA **seriale**:

```c
for (int j = 0; j < in_dim; ++j)
    acc += w_row[j] * in_vec[j];    // ogni iterazione dipende dalla precedente
```

Sul Cortex-M7 la `VFMA.F32` ha throughput 1/ciclo ma **latenza ~3 cicli**: una riduzione
seriale costa ~3 cicli per MAC di sola latenza, tre volte il floor teorico. In più servono
due load per MAC e la M7 ne ritira una per ciclo.

**GCC non lo sistema da solo.** Lo splitting di una riduzione in accumulatori multipli
normalmente arriva dalla vettorizzazione, ma la M7 ha FPU **scalare**: niente vettori,
niente splitting. `-Ofast` autorizza la riassociazione ma non riscrive la catena di
dipendenza.

Elaborare **due righe di output per iterazione** risolve entrambi i problemi insieme:

```c
for (; i + 1 < out_dim; i += 2) {
    const float *restrict w0 = W + (size_t)i * in_dim;
    const float *restrict w1 = w0 + in_dim;
    float acc0 = b[i], acc1 = b[i + 1];
    for (int j = 0; j < in_dim; ++j) {
        float x = in_vec[j];          // una load per DUE MAC
        acc0 += w0[j] * x;            // due catene indipendenti
        acc1 += w1[j] * x;
    }
    out_vec[i] = acc0; out_vec[i + 1] = acc1;
}
/* + coda per out_dim dispari */
```

Applicato in tre punti: il matrice-vettore del forward, l'accumulo
`dW[i][:] += delta[i]*inp[:]` del backward, e `W^T·δ` (§A4). Serve sempre la **coda** per
`out_dim` dispari — è lì che si annidano i bug.

> **Regola generale:** una riduzione in virgola mobile scritta con un solo accumulatore
> gira alla *latenza* della FMA, non al suo *throughput*. Su FPU scalare il compilatore non
> lo aggiusta: servono accumulatori multipli scritti a mano.

**Trasferibilità:** totale — è dentro il forward/backward denso, che ogni algoritmo
condivide. Con più registri liberi si può salire a 4×1; qui 2×1 era il punto di equilibrio.

---

## A4 — Loop swap in `W^T·δ`

**File:** `neural_net.c` → `backward_from_delta`

La propagazione del delta era scritta con `j` esterno e `i` interno:

```c
for (int j = 0; j < in_dim; ++j) {
    float acc = 0.f;
    for (int i = 0; i < out_dim; ++i)
        acc += delta[i] * ly->W[i][j];   // colonna j: stride in_dim
    delta_buf[j] = acc;
}
```

Cioè un accesso **per colonna** su una matrice row-major: elementi consecutivi distano
`in_dim` float, e con il vecchio layout a puntatori di riga stavano anche in blocchi heap
diversi. In più l'accumulo su `acc` è di nuovo seriale.

Invertendo i loop e accumulando in un buffer, l'accesso diventa sequenziale sulla riga e
ogni `j` ha il proprio accumulatore:

```c
memset(dst, 0, in_dim * sizeof(float));
for (int i = 0; i < out_dim; ++i) {
    float di = delta[i];
    const float *restrict w_row = ly->W + (size_t)i * in_dim;
    for (int j = 0; j < in_dim; ++j)
        dst[j] += di * w_row[j];
}
```

> **Regola generale:** nei prodotti `W^T·v` iterare sempre sulle **righe** di W nel loop
> esterno, accumulando in un buffer. Mai leggere una matrice row-major per colonna.

**Trasferibilità:** totale. È il cuore del backward di qualunque MLP.

---

## A5 — Ping-pong sul buffer del delta *(correttezza, non velocità)*

**File:** `neural_net.c` → `backward_from_delta` — vedi **BUG-7** in `BUG_FIXING.md`

Conseguenza diretta di §A4, ed è la voce più importante di tutto il documento. Il delta del
layer precedente si **scrive** mentre quello corrente è ancora in **lettura**, e dal secondo
layer in poi i due sono lo stesso buffer (`delta = delta_buf` a fine iterazione). Con un
buffer solo, scrivere l'elemento `j` corrompe il `delta[j]` che serve ancora alle
iterazioni `j` successive.

Il buffer è quindi un pool a **due metà usate a ping-pong**, dimensionato una volta sul
layer più largo **prima** di iniziare la passata: una `realloc` a metà strada
invaliderebbe il delta in uso (secondo difetto, latente sulle topologie attuali perché
`in_dim` non cresce mai andando indietro, ma reale su una rete tipo `[11, 64, 32, 1]`).

Il bug era **reale e silenzioso**: niente NaN, niente crash, solo il gradiente del primo
layer sistematicamente sbagliato — la matrice più grande delle due, 32×11. Si manifesta su
qualsiasi rete con ≥3 layer di pesi; con 2 layer non c'è aliasing e non si vede.
**È stato trovato dal test di equivalenza (§A11), non dal training.**

⚠️ **Impatto sulle misure:** tutte le learning curve raccolte prima di questo fix sono
state prodotte con il gradiente del primo layer sbagliato. Vanno riacquisite prima di
qualsiasi confronto.

> **Regola generale:** quando un buffer temporaneo è insieme sorgente e destinazione fra due
> iterazioni, serve il ping-pong. E se la capacità è variabile, dimensionarla prima della
> passata: riallocare mentre un puntatore vecchio è ancora vivo è un uso-dopo-free
> silenzioso.

**Trasferibilità:** ⚠️ **da verificare per primo in ogni porting che riusi questo
backward.** Se il codice di partenza deriva da questa base e ha ≥3 layer, il bug c'è.

---

## A6 — Attivazione e derivata in passata separata

**File:** `neural_net.c` → `forward_dense_layer`, `backward_from_delta`

Lo `switch` sull'attivazione stava dentro il loop sugli output: una branch per neurone,
imprevedibile per il predictor e ostile all'unrolling. Spostandolo fuori, il loop di
matrice-vettore resta stretto e l'attivazione diventa un loop separato senza rami.

Stessa cosa nel backward, dove la derivata dell'attivazione è ora una passata a sé,
distinta dal prodotto `W^T·δ`.

> **Regola generale:** una `switch` invariante dentro un loop caldo va portata fuori, anche
> al costo di una passata in più sui dati: la passata extra è sequenziale e cache-friendly,
> la branch per elemento no.

**Trasferibilità:** totale.

---

## A7 — Adam in forma efficiente

**File:** `neural_net.c` → `adam_update_single_layer`, `network_adam_update`

La forma testuale costa per ogni peso **tre divisioni e una radice**:

```c
float m_hat = m / b1t;
float v_hat = v / b2t;
w -= lr * m_hat / (sqrtf(v_hat) + EPS_ADAM);
```

Su Cortex-M7 `VDIV.F32` e `VSQRT.F32` costano ~14 cicli l'una e **non sono pipelined**:
sono ~56 dei ~78 cicli/parametro misurati. L'**Algoritmo 2 di Kingma & Ba** dà una forma
algebricamente identica con **una sola divisione**:

```c
float sb2t  = sqrtf(b2t);              // una volta per update
float lr_t  = lr * sb2t / b1t;
float eps_t = EPS_ADAM * sb2t;
w -= lr_t * m / (sqrtf(v) + eps_t);    // per peso: 1 div, 1 sqrt
```

L'equivalenza è verificata numericamente in `test/test_equiv.c` (scarto 0).
Il layout piatto (§A1) collassa anche il doppio loop `i/j` su un unico loop lineare sui
`out_dim*in_dim` pesi.

> **Regola generale:** divisioni e radici sono l'unica operazione FPU che su M7 non si
> pipeline. Contarle **per elemento**, e portare fuori dal loop tutto ciò che non dipende
> dall'elemento.

**Trasferibilità:** totale — Adam è lo stesso ovunque. Vale identico per RMSProp (stessa
`sqrt` per peso) e per qualsiasi ottimizzatore con bias correction.

---

## A8 — Gradient clipping fuso nell'ottimizzatore

**File:** `neural_net.c` → `network_clip_grad`, `network_adam_update`

`network_clip_grad` faceva una passata per la norma e, se il clip scattava, una **seconda
read-modify-write su tutti i parametri**; poi Adam li rileggeva ancora. Ora ritorna il solo
**fattore di scala** (`float` invece di `void`), che Adam applica al volo mentre carica
`db`/`dW`: una passata completa in meno per ogni minibatch in cui il clip scatta.

Anche il ramo `!isfinite(gnorm)` ritorna `0.0f` invece di azzerare i gradienti a mano:
l'effetto è identico (Adam li moltiplica per zero e li ripulisce comunque) e risparmia
un'altra passata.

> **Regola generale:** se due passate consecutive toccano lo stesso array, chiedersi se la
> prima possa produrre uno **scalare** invece di riscrivere i dati.

**Trasferibilità:** totale — il grad clipping c'è in quasi tutti gli algoritmi.

---

## A9 — `restrict`, `const` e tipi dimensionati sul problema

**File:** `neural_net.c`, `ppo.c/h`

Tre interventi piccoli con effetti sproporzionati:

- **`restrict` su ogni puntatore dei kernel** (`W`, `b`, `dW`, `in_vec`, `out_vec`, `inp`,
  `dst`). Senza, il compilatore deve assumere che scrivere `dst[j]` possa modificare
  `w_row[j]`, e ricarica ad ogni iterazione: tutto il lavoro di §A3 non si vede.
  `const` sugli input serve allo stesso scopo e documenta l'intento.
- **`uint16_t idx[ROLLOUT_STEPS]`** per la permutazione dello shuffle, era `uint32_t`:
  **2 KB → 1 KB di `.bss`**. Con uno `_Static_assert(ROLLOUT_STEPS <= 65535)` che rende
  impossibile alzare la costante senza accorgersene.
- **Doppia inizializzazione rimossa**: `network_init` chiamava `init_layer_params` dopo
  `dense_init`, che la chiama già. Rifaceva il lavoro e consumava il doppio di estrazioni
  RNG — cioè cambiava i pesi iniziali rispetto a qualsiasi baseline con lo stesso seed.

> **Regola generale:** su MCU il tipo si sceglie dal range reale del dato, con uno
> `_Static_assert` che leghi il tipo alla costante che lo giustifica. E `restrict` non è
> decorazione: senza, le ottimizzazioni sui loop non producono l'effetto atteso.

**Trasferibilità:** totale. Su DQN pesa di più: gli indici del replay buffer e le azioni
discrete sono spesso `uint32_t` senza motivo.

---

## A10 — Profiling che sparisce, e micro-benchmark dei kernel

**File:** `ppo.c` (macro `PROF_T`/`PROF_ADD`), `main.c`/`main.h` (`BENCH_KERNELS`)

Il profiling DWT c'era già; questa sessione ha aggiunto due cose:

- **Macro `PROF_T`/`PROF_ADD`** che si riducono a `((void)0)` quando `TIME_LOG = 0`, senza
  lasciare riferimenti a variabili inesistenti. Il loop caldo di `ppo_update` resta
  leggibile invece di essere spezzato da una dozzina di `#if`.
- **`BENCH_KERNELS`**: micro-benchmark all'avvio che misura il costo **per chiamata** di
  `tanhf`, `expf`, `logf`, `atanhf`, `sqrtf` (4096 iterazioni ciascuno, con un loop vuoto
  come riferimento da sottrarre e un `volatile sink` che impedisce a `-Ofast` di eliminare
  tutto) e dei **due forward completi**. Actor e critic hanno la stessa topologia a meno
  dell'ultimo layer e differiscono solo per l'attivazione: la differenza fra i due tempi
  **è** il costo delle `tanh`.

  Serve ad **attribuire** il tempo prima di decidere dove intervenire — è il numero che
  dice se valga la pena riaprire la questione dell'approssimazione polinomiale (§Il tetto).
  Emette righe `<<<BENCH>>>nome,cicli` che il trainer PC tollera senza modifiche, quindi
  basta una run normale con il flag attivo. Al termine ripristina l'overrun UART come fa
  `on_train_end`, perché durante i suoi ~0,4 s il PC sta già trasmettendo.

> **Regola generale:** misurare prima di ottimizzare, poi **attribuire**: sapere che il
> forward costa 250 ms non dice se il problema sono le MAC o le trascendenti. E la
> strumentazione deve poter sparire senza lasciare `#if` nel codice caldo.

**Trasferibilità:** totale, le macro e la funzione `bench_kernels` si copiano come sono.

---

## A11 — Test host di equivalenza + sanitizer

**File:** `tinyRL/test/` (`Makefile`, `test_equiv.c`, `test_update.c`, `stub/`, `README.md`)
— **nuovo in questa sessione**

I file che implementano l'algoritmo (`dense_layer.c`, `neural_net.c`, `ppo.c`, `rng.c`)
**non usano l'HAL**: con due header stub vuoti compilano in nativo, quindi i kernel si
verificano su PC senza flashare nulla. È una proprietà da preservare deliberatamente in
ogni porting — è ciò che rende possibile tutto il resto.

- **`test_equiv`** confronta i kernel ottimizzati contro un'implementazione di riferimento
  **ingenua, scritta riga per riga come il codice prima delle ottimizzazioni** (forward per
  riga con lo switch dentro il loop, `W^T·δ` per colonna, Adam in forma testuale a tre
  divisioni). Serve a dimostrare che le trasformazioni dichiarate esatte lo siano davvero.
  Gli scarti residui (~1e-5 relativo) sono riassociazione in virgola mobile, già permessa
  da `-Ofast`.
- **`test_update`** è uno smoke test end-to-end: rollout sintetico, un `ppo_update`
  completo, verifica di capienza del buffer, bounds check della push, normalizzazione degli
  advantage, assenza di NaN/Inf nei pesi.
- **`make asan`** rigira tutto sotto AddressSanitizer + UBSan: è il controllo che
  l'aritmetica delle arene (§A1, §A2) non scriva fuori dai limiti.

**È questo che ha trovato BUG-7** (§A5): i layer 1 e 2 combaciavano esattamente, il layer 0
no. Nessuna learning curve lo avrebbe mai segnalato.

> **Regola generale:** ogni ottimizzazione "esatta" va accompagnata dalla versione ingenua
> che sostituisce, tenuta viva in un test. Il costo è un file; il ritorno è la possibilità
> di ottimizzare senza avere paura.

**Trasferibilità:** totale, ed è **il primo pezzo da portare** su un nuovo algoritmo, prima
di qualsiasi ottimizzazione.

---

## A12 — Pulizia dell'API e codice morto

**File:** `uart.c/h`, `utils.c/h`, `reinforce.c/h`

- **`uart_send_action` unificata**: `uart_send_action_discrete` (scalare) e la variante
  vettoriale sono fuse in `uart_send_action(huart, actions, n, done, timeout)`; per il caso
  scalare si passa `&val, 1`. Una funzione in meno da tenere allineata fra i branch — e
  soprattutto **il bounds check su `n` che prima mancava**: con `n > UART_MAX_ACTION_DIMS`
  la vecchia versione scriveva oltre il buffer di TX.
- **`clip()` rimossa** da `utils.c` (non più usata) e include guard di `utils.h` sistemato:
  gli `#include` stavano *prima* della guard, quindi venivano processati ad ogni inclusione.
- **`reinforce.c/h` rimossi**: residui di un altro algoritmo, non compilati ma pronti a
  divergere silenziosamente.

> **Regola generale:** due funzioni che differiscono solo per l'arità di un parametro sono
> una funzione. E il codice morto in un repo multi-branch non è neutro: è la prossima cosa
> che qualcuno copia.

---

# Parte B — Ottimizzazioni specifiche di PPO

## B1 — Salvare `z` invece di ricostruirlo con `atanhf`

**File:** `ppo.c` → `actor_sample_action`, `rollout_buffer_push`; `neural_net.c` → kernel continui

Con il tanh-squashing, `actor_sample_action` calcola `z = mu + exp(ls)·eps` e poi
`a = tanh(z)`. Nel rollout buffer finiva solo `a`, e i kernel dell'update ricostruivano
`z = atanhf(a)` ad ogni epoca: **24.576 `atanhf` per update** (8 epoche × 512 campioni × 3
dim, più 3 al campionamento) per riottenere un valore che era già stato calcolato a monte.

Ora nel buffer va `z` — **stessa occupazione di memoria, contenuto diverso** — e `a` resta
nell'agente (`prev_action`), dove serve solo al reward e all'ambiente.

Oltre a togliere le `atanhf`, il risultato è **più corretto**: il round-trip
`atanhf(tanhf(z))` perde precisione, e il clamp a `|a| ≤ 1−1e−6` satura `z` a ±7,25 — un
taglio che con `sigma_init = 1.5` e `mu` limitato a ±8 scattava davvero.

> **Regola trasferibile:** cercare nel buffer di esperienza le quantità **derivate** invece
> di quelle **primarie**. Se l'update deve invertire una trasformazione per tornare a un
> valore che il rollout aveva già in mano, salvare il valore originale: costa uguale in RAM,
> è più veloce, ed è più preciso. Una ricostruzione che passa per una funzione non
> invertibile in virgola mobile non è nemmeno esatta.
>
> **Dove guardare:** SAC (stesso identico squashing), TD3, qualsiasi policy con
> trasformazione dell'azione. In DQN il caso analogo è salvare l'indice dell'azione anziché
> il Q-value da cui è stata scelta.

---

## B2 — Termini che si cancellano nel ratio

**File:** `ppo.c`, `neural_net.c`

Il termine Jacobiano del tanh, `−log(1 − a²)`, dipende **solo dall'azione**, che è fissa nel
buffer. Compare quindi identico in `log_prob_old` e `log_prob_new`, e nel rapporto
`exp(lp_new − lp_old)` **si cancella esattamente**. Rimuoverlo da entrambi i siti toglie
3 `logf` per campione più 3 al campionamento, **senza cambiare di una virgola il gradiente**.

Stessa logica per l'**entropia analitica della gaussiana**,
`H = 0.5·D·(1 + log 2π) + Σ log σ`: era calcolata nel forward continuo ma è **costante
rispetto ai pesi dell'actor** (σ non è un parametro appreso, decade per schedule), quindi
non entra nel gradiente. Rimossa.

Condizione da rispettare: il termine va tolto da **entrambi** i lati, e vale solo perché
sigma non dipende dallo stato e l'azione è congelata nel buffer.

> **Regola trasferibile:** in PPO conta solo la **differenza** fra le log-probabilità. Ogni
> termine additivo che non dipende dai parametri della policy è lavoro sprecato. Prima di
> implementare una formula da paper, scriverla per esteso e cancellare a mano quello che si
> semplifica.
>
> **Dove guardare:** ogni algoritmo con un rapporto o una differenza di log-prob —
> importance sampling, off-policy correction, il termine di entropia di SAC. Nella loss di
> DQN il caso analogo sono le costanti additive del target che spariscono nella derivata.

---

## B3 — `var` e `log_var` precalcolati per update

**File:** `ppo.c` → `ppo_update`, `neural_net.c` → kernel continui

`g_ppo_log_sigma` è costante per l'intera durata di un update: `ppo_sigma_decay()` gira una
sola volta, in fondo. Eppure `expf(2·ls)` veniva ricalcolata dentro **ogni campione**, sia
nel forward sia nel backward: **24.576 `expf` per update** per ottenere sempre gli stessi
tre numeri. Ora `var` e `log_var` si calcolano una volta in cima a `ppo_update` e viaggiano
come parametri.

Nello stesso spirito, `mu` viene clampato **in place** nel forward (`mu[i] = m`), così il
backward lo rilegge già clampato invece di rifare `fminf/fmaxf` su ogni dimensione.

> **Regola trasferibile:** identificare cosa è costante **per update** e cosa varia **per
> campione**, e spostare tutto il primo gruppo fuori dal loop dei campioni. Su un update da
> 8 epoche × 512 campioni il moltiplicatore è 4096. Il compilatore spesso lo fa da solo, ma
> **non attraverso una chiamata a una funzione di libreria matematica che non può dimostrare
> pura**: ogni `expf`/`logf`/`powf` in un loop caldo va guardata con sospetto.
>
> **Dove guardare:** temperatura di SAC, epsilon di DQN, coefficienti di scheduling, target
> network — tutto ciò che cambia una volta per update.

---

## B4 — `inv_bsz` ripiegato nel delta

**File:** `ppo.c` → `ppo_update`

Il gradiente medio del minibatch si otteneva con una passata su **tutti** i parametri di
actor e critic dopo ogni minibatch — 64 volte per update:

```c
for (ogni layer) for (i) { la->db[i] *= inv_bsz;
                           for (j) la->dW[i][j] *= inv_bsz; }
```

Basta passare `adv_t * inv_bsz` (e `PPO_C1 * inv_bsz` al critic, `PPO_C2 * inv_bsz`
all'entropia nel ramo discreto) alle funzioni di backward: la passata sparisce del tutto.

La verifica che serviva: il **ramo di clipping dipende da `ratio`, non da `advantage`**,
quindi scalare l'advantage non lo sposta.

> **Regola trasferibile:** un fattore di scala applicato a valle su N parametri si può quasi
> sempre ripiegare a monte su un valore solo — qui, su un vettore di 1–3 elementi. Verificare
> soltanto che non attraversi un ramo condizionale che dipende dal valore scalato.
>
> **Dove guardare:** il `1/batch_size` di qualsiasi loss mediata; in REINFORCE il fattore di
> normalizzazione del return.

---

# Risultati misurati

## Memoria e binario *(verificati con `arm-none-eabi-size` e conteggio allocazioni)*

| | prima | dopo | Δ |
|---|---|---|---|
| allocazioni heap | 592 | **9** | −98% |
| heap (actor+critic) | 94,1 KB | **86,4 KB** | −8,2% |
| overhead allocatore | 7,8 KB | **0,1 KB** | −99% |
| `.bss` | 4216 B | **3176 B** | −24,7% |
| `.text` | 29.948 B | **28.612 B** | −4,5% |

## Tempo di `ppo_update` *(media su 106 update prima, 282 dopo)*

| voce | prima<br>`mcu_timings_1.csv` | dopo<br>`mcu_timings.csv` | Δ |
|---|---|---|---|
| **totale** | 790,3 ms | **703,0 ms** | **−11,0%** |
| forward | 312,6 ms | **255,9 ms** | −18,1% |
| backward | 443,3 ms | **415,7 ms** | −6,2% |
| adam + clipping | 31,4 ms | **28,0 ms** | −11,0% |

> ⚠️ **Da confermare prima di pubblicare.** `mcu_timings.csv` è la run più recente
> (2026-08-23 14:40), successiva alla build del firmware ottimizzato (14:17), e i numeri
> sono coerenti con le trasformazioni fatte — ma `TODO.md` riporta ancora la riga tempo come
> stima, quindi va riconfermata con una run tracciata. La stima a priori era 450–500 ms: il
> divario rispetto ai 703 ms misurati è spiegato dal paragrafo seguente.

---

# Il tetto: cosa non è stato ottimizzato, e perché

**Le funzioni trascendenti dominano.** Le due reti usano attivazioni diverse: `ACT_TANH` sui
due hidden dell'actor, `ACT_RELU` su quelli del critic. Sono **64 chiamate `tanhf` per
campione**, **262.144 per update**, e newlib le implementa in software. È il motivo per cui
il backward migliora solo del 6%: le sue MAC sono state sistemate, ma il forward che lo
precede paga un costo che nessuna riorganizzazione di loop tocca.

Sotto il vincolo "**solo trasformazioni esatte**" quel costo non si tocca. Le leve
disponibili cambiano i numeri e vanno validate sulle learning curve, non solo sull'errore
massimo:

1. **Approssimazione polinomiale di `tanh`** (minimax su range ridotto + riduzione
   dell'argomento);
2. **Actor a ReLU come il critic**: cambia l'apprendimento, non solo il tempo;
3. **Lookup table + interpolazione lineare**: scambia flash con cicli.

`BENCH_KERNELS = 1` (§A10) dice quanto vale davvero sulla scheda: è il primo numero da
guardare se si vuole riaprire la questione.

**Fuori dal perimetro di `ppo_update`, ma decisivo per gli esperimenti:** il wall-clock di
una sessione HIL è dominato dalla **UART**, non dal training. A 115200 baud un round-trip è
~5,4 ms, quindi 512 step costano **~2,8 s** contro gli **0,70 s** di un update. **Alzare il
baud rate è il singolo intervento più efficace sul tempo totale di un esperimento**, e non
tocca una riga di algoritmo.

---

# Fuori perimetro — cosa c'era già

Non fa parte di questa sessione, ma su un **nuovo target o un nuovo algoritmo va rifatto lo
stesso**. Elencato per non dare per scontato ciò che scontato non è:

| voce | dove |
|---|---|
| **Re-Forward**: forward ricalcolato prima di ogni backward invece di cachare le attivazioni (~19 KB risparmiati con batch 64) | `neural_net.c`, `ppo.c` |
| **PCG32** al posto di `rand()` di newlib, il cui LCG correla le coppie (u1,u2) del Box-Muller e falsa il rumore di esplorazione | `rng.c` |
| **Clock 480 MHz** (VOS0 + overdrive), **I/D-cache abilitate**, `-Ofast`, FPU hardware, varianti `f` delle funzioni math | `main.c`, `.cproject` |
| **DWT cycle counter** con accumulo a 64 bit: a 480 MHz il CYCCNT wrappa ogni ~8,9 s, meno di un update, quindi il totale va accumulato a pezzi per minibatch | `utils.c/h`, `ppo.c` |
| **Clamp del ratio** a ±10 prima di `expf` e **`MU_CLAMP = 8`** sull'uscita dell'actor (BUG-4) | `ppo.c`, `neural_net.h` |
| **Softmax stabile** (sottrazione del max) e `inv = 1/sum` al posto della divisione | `neural_net.c` |
| **Bounds check** su `rollout_buffer_push` con campo `capacity` (BUG-6) | `ppo.c`, `ppo.h` |
| **`PPO_C1` passato a `critic_backward`** — prima non arrivava mai al gradiente (BUG-5) | `neural_net.c` |
| **Schedule di sigma riscalato sul rollout** (`PPO_SIGMA_REF_ROLLOUT`): accorciando il rollout 2048→512 gli update diventano 4× più frequenti e sigma collasserebbe 4× prima in step di ambiente | `ppo.h` |
| **Frame UART con checksum** e ripristino dell'overrun dopo il training | `uart.c`, `main.c` |

---

# Checklist di porting su un nuovo algoritmo

**Prima di ottimizzare**
- [ ] Tenere i file dell'algoritmo **liberi dall'HAL**, così compilano in nativo (§A11)
- [ ] Portare `tinyRL/test/` e scrivere l'implementazione di riferimento ingenua (§A11)
- [ ] Strumentare l'update e misurare la baseline; poi `BENCH_KERNELS` per attribuirla (§A10)

**Correttezza**
- [ ] ⚠️ Verificare l'aliasing del buffer del delta se la rete ha ≥3 layer (§A5 / BUG-7)
- [ ] `restrict`/`const` su tutti i puntatori dei kernel, o §A3 non produce effetto (§A9)
- [ ] Nessuna doppia inizializzazione dei pesi: cambia i pesi iniziali a parità di seed (§A9)
- [ ] Bounds check su ogni push del buffer e su ogni frame UART (§A12)

**Memoria**
- [ ] Arena contigua per layer, row-major (§A1)
- [ ] Arena unica per replay/rollout buffer, campi a 4 byte prima di quelli a 1 (§A2)
- [ ] Tipi dimensionati sul range reale + `_Static_assert` (§A9)

**Tempo**
- [ ] Blocking 2×1 nel forward e nell'accumulo dei gradienti, con la coda dispari (§A3)
- [ ] Loop swap in `W^T·δ` (§A4)
- [ ] Attivazione e sua derivata in passata separata (§A6)
- [ ] Adam in forma efficiente, 1 divisione per peso (§A7)
- [ ] Grad clipping che ritorna uno scalare invece di riscrivere i gradienti (§A8)
- [ ] Sollevare fuori dal loop dei campioni tutto ciò che è costante per update (§B3)
- [ ] Ripiegare i fattori di scala sul delta di output (§B4)
- [ ] Cercare quantità derivate nel buffer al posto di quelle primarie (§B1)
- [ ] Scrivere per esteso la loss e cancellare i termini che si semplificano (§B2)

**Prima di pubblicare i numeri**
- [ ] `make -C test` e `make -C test asan`
- [ ] Riacquisire le learning curve (§A5: quelle vecchie vengono da codice buggato)
- [ ] Allineare topologia e rollout fra le schede che si vogliono confrontare
- [ ] Alzare il baud rate dell'UART (§Il tetto)

---

## Riepilogo

| # | Tecnica | File | Categoria | Trasferibile |
|---|---|---|---|---|
| A1 | Arena contigua per layer | `dense_layer.c/h` | Memoria | ✅ diretta |
| A2 | Arena unica per il rollout buffer | `ppo.c/h` | Memoria | ✅ diretta |
| A3 | Blocking 2×1, accumulatori indipendenti | `neural_net.c` | Tempo | ✅ diretta |
| A4 | Loop swap in `W^T·δ` | `neural_net.c` | Tempo | ✅ diretta |
| A5 | Ping-pong sul buffer del delta (BUG-7) | `neural_net.c` | **Correttezza** | ⚠️ verificare |
| A6 | Attivazione in passata separata | `neural_net.c` | Tempo | ✅ diretta |
| A7 | Adam in forma efficiente (1 divisione) | `neural_net.c` | Tempo | ✅ diretta |
| A8 | Clipping fuso nell'ottimizzatore | `neural_net.c` | Tempo | ✅ diretta |
| A9 | `restrict`/`const`, `uint16_t idx`, doppia init | `neural_net.c`, `ppo.c` | Tempo + memoria | ✅ diretta |
| A10 | Macro di profiling + `BENCH_KERNELS` | `ppo.c`, `main.c` | Strumentazione | ✅ diretta |
| A11 | Test di equivalenza + ASan | `test/` | Verifica | ✅ diretta |
| A12 | `uart_send_action` unificata, codice morto | `uart.c`, `utils.c` | Manutenibilità | ✅ diretta |
| B1 | `z` salvato invece di `atanhf` | `ppo.c` | Tempo + precisione | 🔁 via regola |
| B2 | Jacobiano ed entropia costante rimossi | `ppo.c`, `neural_net.c` | Tempo | 🔁 via regola |
| B3 | `var`/`log_var` per update, `mu` clampato in place | `ppo.c` | Tempo | 🔁 via regola |
| B4 | `inv_bsz` ripiegato nel delta | `ppo.c` | Tempo | 🔁 via regola |

---

## Documenti collegati

| file | contenuto |
|---|---|
| `tinyRL/OPTIMIZATIONS.md` | dettaglio con i frammenti prima/dopo |
| `tinyRL/BUG_FIXING.md` | i 7 bug trovati, e i falsi positivi scartati |
| `tinyRL/TODO.md` | misure ancora da rifare e decisioni aperte |
| `tinyRL/test/README.md` | come girare i test di equivalenza e ASan |
| `memory_calculator_ppo.py` | Re-Forward vs cache classica al variare della topologia |
| `Micro_RL/learning_curve_comparison/` | timing e learning curve, MCU vs baseline PC |
