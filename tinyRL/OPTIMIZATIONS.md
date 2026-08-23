# Ottimizzazioni di Memoria e Tempo — tinyRL (PPO su STM32H743)

Catalogo delle ottimizzazioni presenti nel codebase, con la misura di partenza
e quella di arrivo. L'obiettivo è che ogni scelta sia replicabile su qualsiasi
porting dello stesso algoritmo su MCU con memoria limitata, quindi ogni voce
chiude con la regola generale che ne sta dietro.

Configurazione di riferimento: STM32H743 a 480 MHz (VOS0), actor e critic
`[11, 32, 32, ·]`, `ROLLOUT_STEPS = 512`, `PPO_EPOCHS = 8`, `PPO_BATCH_SIZE = 64`,
compilato `-Ofast -mfpu=fpv5-d16 -mfloat-abi=hard` con newlib-nano.
Topologia e rollout sono allineati al branch NUCLEO-F446RE: i tempi delle due
schede sono confrontabili solo a parità di rete e di step per update.

## Baseline

Media su 106 update, da `Micro_RL/learning_curve_comparison/mcu_timings_1.csv`:

| voce | ms | costo unitario implicito |
|---|---|---|
| `ppo_update` totale | 790 | 8 epoche × 512 campioni = 4096 iterazioni |
| forward | 312 | ~12,7 cicli per MAC |
| backward | 443 | ~10,3 cicli per MAC |
| adam | 31 | ~78 cicli per parametro |

Nessuno di questi numeri era vicino al floor dell'hardware: su Cortex-M7 la
`VFMA.F32` ha throughput di 1/ciclo. Il resto del documento spiega dove
finivano gli altri cicli.

---

## Parte 1 — Struttura dei dati

### 1.1 — Un'unica arena contigua per layer

**File:** `dense_layer.c`, `dense_layer.h`

La versione precedente usava `float **` per ognuna delle quattro matrici
(`W`, `dW`, `mW`, `vW`): un array di puntatori di riga più una `calloc`
separata per ogni riga.

```c
*matrix = malloc(rows * sizeof(float *));
for (int i = 0; i < rows; ++i)
    (*matrix)[i] = calloc(cols, sizeof(float));   // rows allocazioni
```

Per actor + critic sono **592 allocazioni**, 94,1 KB di heap per 86,4 KB di
dati utili. Il grosso dello spreco è il padding: una riga da 11 float occupa
44 byte, che newlib-nano arrotonda a 48 più 8 di header = **56 byte, il 27% di
overhead** sulle matrici del primo layer. Gli array di puntatori di riga da
soli sono 2,1 KB.

Ora ogni layer vive in una sola `malloc`, con le quattro matrici e i cinque
vettori come viste dentro l'arena e le matrici in row-major (`W[i*in_dim + j]`):

| | prima | dopo |
|---|---|---|
| allocazioni | 592 | 9 |
| heap | 94,1 KB | 86,4 KB |
| overhead | 7,8 KB | 0,1 KB |

Il beneficio non è solo di spazio: sparisce un livello di indirezione da ogni
loop interno (prima ogni riga costava una load del puntatore prima di poter
leggere i dati) e le righe sono contigue, quindi il prefetcher della D-cache
lavora. Su H743 i 7,7 KB non si notano fra 512 KB, ma sul branch F446 — 128 KB
totali per ~96 KB di working set — sono la differenza fra starci e non starci.

Il rollout buffer ha ricevuto lo stesso trattamento: 8 `malloc` → 1, il che
elimina anche il percorso di fallimento parziale (se la settima `malloc`
falliva, le prime sei restavano allocate e la init tornava 0).

> **Regola generale:** ogni struttura 2D con dimensioni note a init-time va in
> un unico blocco piatto con indicizzazione `i*stride + j`. L'array di
> puntatori di riga costa memoria, frammenta l'heap e aggiunge una
> dipendenza di load nel loop più caldo del programma.

---

### 1.2 — Re-Forward: zero cache delle attivazioni

**File:** `neural_net.c`, `ppo.c` → `ppo_update`

L'approccio naïve al training su mini-batch memorizza le attivazioni
intermedie di ogni campione prima della backpropagation:

```
for b in batch: forward(s[b]) -> salva le attivazioni in cache[b]
for b in batch: backward()    <- legge da cache[b]

Memoria extra: batch_size × Σ(dim_layer) × 4 byte
```

Il Re-Forward esegue invece un forward immediatamente prima di ogni backward,
sfruttando il fatto che `DenseLayer.out` è uno slot fisso riscritto ad ogni
chiamata. Il costo è un forward extra per campione, pagato in cicli e non in
RAM. Per `[11,32,32,3]` con batch 64 sono ~19 KB di RAM risparmiati.

> **Regola generale:** su MCU preferire il ricalcolo al caching quando la rete
> è piccola. Il break-even dipende da clock e RAM disponibile.

---

## Parte 2 — Il loop caldo

### 2.1 — Blocking 2×1 e catene di accumulo indipendenti

**File:** `neural_net.c` → `forward_dense_layer`, `backward_from_delta`

Il prodotto scalare scritto nel modo ovvio è una catena FMA **seriale**:

```c
for (int j = 0; j < in_dim; ++j)
    acc += w_row[j] * in_vec[j];    // ogni iterazione dipende dalla precedente
```

Sul Cortex-M7 la `VFMA.F32` ha throughput 1/ciclo ma **latenza ~3 cicli**: una
riduzione seriale costa quindi ~3 cicli per MAC di sola latenza, tre volte il
floor. In più servono due load per MAC e la M7 ne ritira una per ciclo.

GCC non lo sistema da solo. Lo splitting di una riduzione in accumulatori
multipli arriva normalmente dalla vettorizzazione, ma la M7 ha una FPU
**scalare**: nessun vettore, nessuno splitting. `-Ofast` autorizza la
riassociazione ma non riscrive la catena di dipendenza.

Elaborare due righe di output per iterazione risolve entrambi i problemi:

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
    ...
}
```

Stesso trattamento per l'accumulo `dW[i][:] += delta[i]*inp[:]` nel backward.

> **Regola generale:** una riduzione in virgola mobile scritta con un solo
> accumulatore gira alla latenza della FMA, non al suo throughput. Su FPU
> scalare il compilatore non lo aggiusta: servono accumulatori multipli
> scritti a mano.

---

### 2.2 — Loop swap in `W^T·δ`

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

Cioè un accesso **per colonna** su una matrice row-major: elementi consecutivi
distano `in_dim` float, e con il vecchio layout a puntatori di riga ognuno
stava anche in un blocco heap diverso. In più l'accumulo su `acc` è di nuovo
seriale.

Invertendo i loop e accumulando in un buffer, l'accesso diventa sequenziale
sulla riga e ogni `j` ha il proprio accumulatore:

```c
memset(dst, 0, in_dim * sizeof(float));
for (int i = 0; i < out_dim; ++i) {
    float di = delta[i];
    const float *restrict w_row = ly->W + (size_t)i * in_dim;
    for (int j = 0; j < in_dim; ++j)
        dst[j] += di * w_row[j];
}
```

La derivata dell'attivazione va in una passata separata, il che toglie anche
uno `switch` dall'interno del loop.

> **Regola generale:** nei prodotti `W^T·v` iterare sempre sulle righe di W
> nel loop esterno, accumulando in un buffer. Mai leggere una matrice
> row-major per colonna.

---

### 2.3 — Ping-pong sul buffer del delta

**File:** `neural_net.c` → `backward_from_delta`

Conseguenza diretta del punto precedente, e la ragione per cui va menzionato:
il delta del layer precedente si **scrive** mentre quello corrente è ancora in
**lettura**, e dal secondo layer in poi i due sono lo stesso buffer
(`delta = delta_buf` alla fine di ogni iterazione). Con un buffer solo,
scrivere l'elemento `j` corrompe il `delta[j]` che serve ancora alle
iterazioni successive.

Il buffer è quindi un pool a due metà usate a ping-pong, dimensionato una
volta sul layer più largo **prima** di iniziare la passata: una `realloc` a
metà strada invaliderebbe il delta in uso.

Questo era un bug reale — vedi `BUG_FIXING.md`, BUG-7.

> **Regola generale:** quando un buffer temporaneo è insieme sorgente e
> destinazione fra due iterazioni, serve il ping-pong. E se la sua capacità è
> variabile, dimensionarla prima della passata: riallocare mentre un puntatore
> vecchio è ancora vivo è un uso-dopo-free silenzioso.

---

### 2.4 — Costanti sollevate fuori dal loop dei campioni

**File:** `ppo.c` → `ppo_update`, `neural_net.c` → i kernel continui

`g_ppo_log_sigma` è costante per l'intera durata di un update: `ppo_sigma_decay()`
gira una sola volta, in fondo. Eppure `expf(2*ls)` veniva ricalcolata dentro
ogni campione, sia nel forward sia nel backward: **24.576 `expf` per update**
per ottenere sempre gli stessi tre numeri.

Ora `var` e `log_var` si calcolano una volta in cima a `ppo_update` e viaggiano
come parametri.

> **Regola generale:** un valore costante dentro un loop va sollevato fuori
> esplicitamente. Il compilatore lo fa spesso, ma non attraverso una chiamata
> di funzione a una libreria matematica che non può dimostrare pura.

---

### 2.5 — Salvare `z` invece di ricostruirlo

**File:** `ppo.c` → `actor_sample_action`, `rollout_buffer_push`

Con il tanh-squashing, `actor_sample_action` calcola `z = mu + exp(ls)*eps` e
poi `a = tanh(z)`. Nel rollout buffer finiva solo `a`, e i kernel dell'update
ricostruivano `z = atanhf(a)` ad ogni epoca: **24.576 `atanhf` per update** per
riottenere un valore che era già stato calcolato.

Ora nel buffer va `z` (stessa dimensione, contenuto diverso) e `a` resta
nell'agente, dove serve solo al reward. Oltre a togliere le `atanhf`, il
risultato è più **corretto**: il round-trip `atanhf(tanhf(z))` perde
precisione, e il clamp a `|a| ≤ 1−1e−6` satura `z` a ±7,25, taglio che con
`sigma_init = 1.5` e `mu` limitato a ±8 scattava davvero.

> **Regola generale:** se una quantità è già stata calcolata a monte,
> memorizzarla costa meno che ricostruirla — e una ricostruzione che passa per
> una funzione non invertibile in virgola mobile non è nemmeno esatta.

---

### 2.6 — Termini che si cancellano

**File:** `ppo.c`, `neural_net.c`

Il termine Jacobiano del tanh, `−log(1 − a²)`, dipende **solo dall'azione**,
che è fissa nel buffer. Compare quindi identico in `log_prob_old` e
`log_prob_new`, e nel rapporto `exp(lp_new − lp_old)` si cancella
esattamente. Rimuoverlo da entrambi i siti toglie 3 `logf` per campione più 3
al campionamento, senza cambiare di una virgola il gradiente.

> **Regola generale:** in PPO solo la *differenza* fra le log-probabilità
> conta. Ogni termine che non dipende dai parametri della policy è lavoro
> sprecato, purché lo si tolga da entrambi i lati.

---

### 2.7 — `inv_bsz` ripiegato nel delta

**File:** `ppo.c` → `ppo_update`

Il gradiente medio del minibatch si otteneva con una passata su **tutti** i
parametri di actor e critic dopo ogni minibatch — 64 volte per update:

```c
for (ogni layer) for (i) { la->db[i] *= inv_bsz;
                           for (j) la->dW[i][j] *= inv_bsz; }
```

Basta passare `adv_t * inv_bsz` e `PPO_C1 * inv_bsz` alle funzioni di backward:
la passata sparisce. Il ramo di clipping dipende da `ratio`, non da
`advantage`, quindi scalare l'advantage non lo sposta.

> **Regola generale:** un fattore di scala applicato a valle su N parametri si
> può quasi sempre ripiegare a monte su un valore solo.

---

### 2.8 — Adam in forma efficiente

**File:** `neural_net.c` → `adam_update_single_layer`, `network_adam_update`

La forma testuale costa per ogni peso tre divisioni e una radice:

```c
float m_hat = m / b1t;
float v_hat = v / b2t;
w -= lr * m_hat / (sqrtf(v_hat) + EPS_ADAM);
```

Su Cortex-M7 `VDIV.F32` e `VSQRT.F32` costano ~14 cicli l'una e **non sono
pipelined**: sono ~56 dei ~78 cicli per parametro misurati. L'Algoritmo 2 di
Kingma & Ba dà una forma algebricamente identica con una sola divisione:

```c
float sb2t  = sqrtf(b2t);              // una volta per update
float lr_t  = lr * sb2t / b1t;
float eps_t = EPS_ADAM * sb2t;
w -= lr_t * m / (sqrtf(v) + eps_t);    // per peso
```

L'equivalenza è verificata numericamente in `test/test_equiv.c` (scarto 0).

> **Regola generale:** divisioni e radici sono l'unica operazione FPU che su
> M7 non si pipeline. Contarle per elemento, e portare fuori dal loop tutto
> ciò che non dipende dall'elemento.

---

### 2.9 — Gradient clipping fuso in Adam

**File:** `neural_net.c` → `network_clip_grad`

`network_clip_grad` faceva una passata per la norma e, se il clip scattava, una
seconda read-modify-write su tutti i parametri; poi Adam li rileggeva. Ora
ritorna il solo fattore di scala, che Adam applica al volo al caricamento di
`db`/`dW`: una passata completa in meno per ogni minibatch in cui il clip
scatta.

Il ramo `!isfinite(gnorm)` ritorna `0.0f` invece di azzerare i gradienti a
mano: l'effetto è identico (Adam li moltiplica per zero e li ripulisce
comunque) e risparmia un'altra passata.

> **Regola generale:** se due passate consecutive toccano lo stesso array,
> chiedersi se la prima possa produrre uno scalare invece di riscrivere i dati.

---

### 2.10 — Softmax numericamente stabile

**File:** `neural_net.c` → `softmax` *(solo modalità discreta)*

La versione naïve va in overflow per logit sopra ~88. Sottrarre il massimo
prima di `exp` lo evita, e la divisione finale si fa una volta sola
calcolando `inv = 1/sum` e moltiplicando.

> **Regola generale:** sostituire sempre `x / sum` con `x * (1/sum)` nei loop.

---

### 2.11 — Profiling con il DWT cycle counter

**File:** `utils.c/h`, `ppo.c` (macro `PROF_T`/`PROF_ADD`), `main.c`

Il Data Watchpoint and Trace del Cortex-M7 ha un contatore di cicli a 32 bit
leggibile senza overhead rilevante — molto meglio di `HAL_GetTick()`, che ha
risoluzione 1 ms.

Attenzione all'ampiezza: a 480 MHz il contatore wrappa ogni ~8,9 s, meno della
durata di un update intero. Il totale si accumula quindi a pezzi (uno per
minibatch) su un `uint64_t`, non con un unico delta start/stop.

`BENCH_KERNELS` in `main.h` attiva un micro-benchmark all'avvio che misura il
costo per chiamata di `tanhf`, `expf`, `logf`, `atanhf`, `sqrtf` e dei due
forward completi. Serve ad attribuire il tempo del forward prima di decidere
dove intervenire. Il trainer lato PC tollera le righe `<<<BENCH>>>` senza
modifiche — cerca `<<<PROF>>>` per il profiling e `0x02` per i frame azione, e
il testo ASCII non contiene nessuno dei due — quindi basta una run normale con
il flag attivo. Al termine il benchmark ripristina l'overrun UART come fa
`on_train_end`, perché durante i suoi ~0,4 s il PC sta già trasmettendo.

> **Regola generale:** misurare prima di ottimizzare, e misurare con lo
> strumento che ha la risoluzione giusta.

---

## Parte 3 — Il tetto

Le due reti usano attivazioni diverse: `ACT_TANH` sui due hidden layer
dell'actor, `ACT_RELU` su quelli del critic. Sono **64 chiamate `tanhf` per
campione**, 262.144 per update, e newlib le implementa in software.

Sotto il vincolo "solo trasformazioni esatte" quel costo non si tocca. Le
uniche leve sarebbero un'approssimazione polinomiale — che cambia i numeri e
va validata sulle learning curve — oppure passare l'actor a ReLU come il
critic, che cambia l'apprendimento. Il micro-benchmark di `BENCH_KERNELS` dice
quanto vale davvero sulla scheda: è il primo numero da guardare se si vuole
riaprire la questione.

Fuori dal perimetro di `ppo_update`, ma vale la pena saperlo: il wall-clock di
una sessione HIL è dominato dalla UART, non dal training. A 115200 baud un
round-trip è ~5,4 ms, quindi 512 step costano ~2,8 s contro gli 0,79 s di un
update. Alzare il baud rate è il singolo intervento più efficace sul tempo
totale di un esperimento, e non tocca una riga di algoritmo.

---

## Riepilogo

| # | Tecnica | File | Categoria |
|---|---|---|---|
| 1.1 | Arena contigua per layer e per rollout buffer | `dense_layer.c`, `ppo.c` | Memoria |
| 1.2 | Re-Forward: nessuna cache delle attivazioni | `neural_net.c`, `ppo.c` | Memoria |
| 2.1 | Blocking 2×1, accumulatori indipendenti | `neural_net.c` | Tempo |
| 2.2 | Loop swap in `W^T·δ` | `neural_net.c` | Tempo |
| 2.3 | Ping-pong sul buffer del delta | `neural_net.c` | Correttezza |
| 2.4 | `var`/`log_var` sollevate fuori dal loop | `ppo.c` | Tempo |
| 2.5 | `z` salvato invece che ricostruito con `atanhf` | `ppo.c` | Tempo + precisione |
| 2.6 | Termine Jacobiano rimosso (si cancella nel ratio) | `ppo.c`, `neural_net.c` | Tempo |
| 2.7 | `inv_bsz` ripiegato nel delta | `ppo.c` | Tempo |
| 2.8 | Adam in forma efficiente (1 divisione) | `neural_net.c` | Tempo |
| 2.9 | Gradient clipping fuso in Adam | `neural_net.c` | Tempo |
| 2.10 | Softmax stabile con `inv = 1/sum` | `neural_net.c` | Tempo |
| 2.11 | Profiling DWT + micro-benchmark dei kernel | `utils.c`, `ppo.c`, `main.c` | Strumentazione |

### Risultati misurati

| | prima | dopo |
|---|---|---|
| heap | 94,1 KB | 86,4 KB |
| allocazioni | 592 | 9 |
| `.bss` | 4216 B | 3176 B |
| `.text` | 29.948 B | 28.612 B |
| `ppo_update` | 790 ms | *da misurare sulla scheda* |

I numeri di memoria e di dimensione del binario sono verificati
(`arm-none-eabi-size`). Il tempo di `ppo_update` va rimisurato sulla scheda con
`TIME_LOG = 1`: la stima a priori è 450–500 ms, ma è una stima.

### Verifica

`make -C test` confronta i kernel ottimizzati contro un'implementazione di
riferimento ingenua scritta come il codice originale, e `make -C test asan`
ricontrolla l'aritmetica delle arene sotto AddressSanitizer. Vedi
`test/README.md`.
