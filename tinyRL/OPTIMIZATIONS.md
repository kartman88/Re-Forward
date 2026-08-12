# Ottimizzazioni di Memoria e Codice — tinyRL su STM32H7

Questo documento cataloga tutte le ottimizzazioni presenti nel codebase e quelle
proposte per versioni future. L'obiettivo è rendere ogni scelta replicabile su
qualsiasi porting dello stesso algoritmo (DQN, PPO, REINFORCE) su MCU con memoria
limitata.

> **Nota sul branch `PPO_F446`.** L'analisi qui sotto è scritta sul target
> originale (NUCLEO-H743ZI2, Cortex-M7 @ 480 MHz, con cache L1). Su questo branch
> il firmware gira su NUCLEO-F446RE (Cortex-M4 @ 84 MHz, 128 KB di RAM, **senza
> cache L1 né MPU**): le considerazioni algoritmiche restano valide, ma i numeri
> di tempo e i ragionamenti su cache line e D-cache non si applicano. Le
> differenze di piattaforma e il nuovo bilancio di memoria sono in
> [`PORTING_F446.md`](PORTING_F446.md).

---

## Parte 1 — Ottimizzazioni già implementate

### 1.1 — Re-Forward: zero overhead di cache delle attivazioni

**File:** `neural_net.c` → `dqn_backward`, `dqn.c` → `dqn_train`

L'approccio naive al training su mini-batch richiede di memorizzare le attivazioni
intermedie di ogni sample del batch prima di eseguire la backpropagation:

```
// Approccio naive (inapplicabile su MCU)
for b in batch:
    forward(s[b])  → salva layer[0].out, layer[1].out, ... in cache[b]
for b in batch:
    backward()     ← legge da cache[b]

Memoria extra: batch_size × Σ(dim_layer) × 4 byte
Esempio: batch=32, rete [3,64,64,5] → 32 × 133 × 4 ≈ 17 KB extra
```

Il Re-Forward esegue un forward pass immediatamente prima di ogni backward,
sfruttando il fatto che `layer->out` è uno slot fisso riscritto ad ogni chiamata:

```
// Re-Forward (approccio adottato)
for b in batch:
    forward_q(online, s[b])      → scrive layer[i].out  (sovrascrive il precedente)
    forward_target(target, s[b]) → scrive target.out     (struttura separata)
    calcola td_error
    dqn_backward(online, s[b], a, td_error)  ← legge layer[i].out ancora validi
    accumula dW, db
    → le attivazioni vengono BUTTATE al prossimo campione
```

Il costo è un forward pass extra per sample (pagato in cicli), non in RAM.
Il campo `DenseLayer.out` (`out_dim` float) è l'unico slot di attivazione,
riusato per ogni sample e per ogni layer in sequenza.

**Regola generale:** su MCU preferire sempre ricalcolo a caching quando la rete
è piccola (< 128 neuroni per layer). Il break-even dipende dal clock e dalla RAM
disponibile; su Cortex-M7 a 480 MHz il ricalcolo di un layer 64×64 costa ~13 µs.

---

### 1.2 — Target network senza momenti Adam

**File:** `neural_net.h` → `TargetLayer`, `neural_net.c` → `init_target_network`

La target network viene usata solo per l'inferenza (forward pass). Non viene
mai allenata direttamente, quindi non ha bisogno dei tensori per l'ottimizzatore.

```c
// DenseLayer (online): 9 array per layer
//   W, dW, mW, vW  (matrici out×in)
//   b, db, mb, vb, out  (vettori out)

// TargetLayer (target): 3 array per layer
//   W, b, out
```

Per la topologia `[3, 64, 64, 5]`:

| Struttura | Floats | Byte |
|---|---|---|
| Online (DenseLayer × 3) | ~19.700 | ~77 KB |
| Target (TargetLayer × 3) | ~4.800 | ~19 KB |
| Target se fosse DenseLayer | ~19.700 | ~77 KB |

Risparmio: ~58 KB su questa configurazione. Il risparmio scala con
`6 × Σ(out_dim × in_dim + out_dim)` (i 6 array eliminati: dW, mW, vW, db, mb, vb).

**Regola generale:** qualsiasi rete che svolge solo inferenza (actor frozen,
target network, ensemble member read-only) va dichiarata senza ottimizzatore.

---

### 1.3 — Replay buffer con allocazione contigua

**File:** `dqn.c` → `replay_buffer_init`

Un replay buffer con `capacity` transizioni e osservazione di dimensione `obs_dim`
naïvemente allocherebbe ogni stato con una `malloc` separata:

```c
// Approccio naïve: capacity × 2 malloc separate per gli stati
for (int i = 0; i < capacity; i++) {
    buf->state[i]      = malloc(obs_dim * sizeof(float));  // frammentazione
    buf->next_state[i] = malloc(obs_dim * sizeof(float));
}
```

Su MCU con heap piccola, dopo centinaia di `malloc`/`free` l'heap si frammenta
e le allocazioni successive falliscono anche se la memoria totale sarebbe sufficiente.

La soluzione adottata è un blocco contiguo con array di puntatori:

```c
// Un unico malloc per tutti gli stati
buf->state_pool = malloc(capacity * obs_dim * sizeof(float));
buf->snext_pool = malloc(capacity * obs_dim * sizeof(float));

// Array di puntatori alle righe
buf->state      = malloc(capacity * sizeof(float *));
buf->next_state = malloc(capacity * sizeof(float *));

for (uint32_t i = 0; i < capacity; i++) {
    buf->state[i]      = buf->state_pool + i * obs_dim;
    buf->next_state[i] = buf->snext_pool + i * obs_dim;
}
```

Vantaggi:
- Solo `2 + 2 = 4` allocazioni totali invece di `2 × capacity`
- Accesso sequenziale ai dati durante il campionamento (cache-friendly)
- `free` è sempre riuscita: non si può liberare parzialmente un blocco contiguo

**Regola generale:** qualsiasi struttura dati con array 2D di dimensione nota a
init-time va allocata con un unico blocco piatto + array di puntatori di riga.

---

### 1.4 — Gradiente sparso al layer di uscita

**File:** `neural_net.c` → `dqn_backward`

La loss del DQN è `L = 0.5 × (Q_online(s,a) − target_val)²`. Il gradiente
rispetto all'output è non-zero solo per l'azione scelta:

```
dL/dQ[i] = td_error   se i == action
dL/dQ[i] = 0.0f       altrimenti
```

Il vettore `delta_out` viene inizializzato a zero e poi impostato solo in
posizione `action`, eliminando il lavoro per tutti gli altri output:

```c
static float delta_out[N_ACTIONS];
for (int i = 0; i < out_dim; i++) delta_out[i] = 0.0f;
delta_out[action] = td_error;
```

Per `N_ACTIONS = 5` questo dimezza il lavoro di accumulo gradienti al layer finale
rispetto a una loss full-vector (dove tutti gli output contribuiscono).

**Regola generale:** nelle reti DQN con output discreto, la sparsità del gradiente
di output è strutturale e non va sprecata.

---

### 1.5 — Buffer temporanei backprop con allocazione `static` e lazy

**File:** `neural_net.c` → `dqn_backward`

Il vettore `delta` che si propaga verso l'input cambia dimensione ad ogni layer.
Invece di allocarlo e liberarlo ad ogni chiamata, si usa un buffer statico con
riallocazione lazy solo quando la dimensione cresce:

```c
static float *delta_buf = NULL;
static uint32_t delta_cap = 0;

if ((uint32_t)in_dim > delta_cap) {
    free(delta_buf);
    delta_buf = malloc(in_dim * sizeof(float));
    delta_cap = (uint32_t)in_dim;
}
```

Il buffer viene riallocato al massimo una volta per la rete più grande incontrata.
In pratica, su topologia fissa nota a compile-time, non viene mai riallocato dopo
il primo step.

**Regola generale:** buffer temporanei a dimensione variabile-ma-limitata vanno
dichiarati `static` con capacità tracciata. Evita `malloc`/`free` nel loop di
training critico.

---

### 1.6 — Q-values su stack con dimensione fissa a compile-time

**File:** `dqn.c` → `dqn_train`, `dqn_select_action`

I vettori di output della rete (Q-values) sono piccoli e di dimensione nota:

```c
float q_online[N_ACTIONS];  // stack, 5 float = 20 byte
float q_tgt[N_ACTIONS];     // stack
```

Usando `#define N_ACTIONS` come costante di compilazione si ottiene un array
di dimensione fissa sullo stack, senza `malloc`. Il compilatore può tenere questi
valori in registri se `N_ACTIONS` è abbastanza piccolo (< 8 circa).

**Regola generale:** tutti i vettori intermedi la cui dimensione è nota a
compile-time vanno dichiarati sullo stack con dimensione costante.

---

### 1.7 — `restrict` e `const` per aiutare il compilatore

**File:** `neural_net.c` → `forward_dense_layer`, `adam_update_single_layer`

```c
static inline void forward_dense_layer(const float *restrict in_vec,
                                        float *restrict out_vec,
                                        const DenseLayer *restrict layer)
```

`restrict` garantisce al compilatore che `in_vec`, `out_vec` e `layer` non si
sovrappongono in memoria. Questo abilita l'auto-vectorizzazione su Cortex-M7
(che ha FPU con pipeline a 2 stage) e rimuove i memory-aliasing barriers
nell'inner loop.

`const` sui puntatori input permette al compilatore di caricare il valore una
volta sola in registro invece di rileggerlo dalla memoria ad ogni iterazione.

**Regola generale:** aggiungere `restrict` a tutti i puntatori di funzione che
non si sovrappongono mai. Non ha costo runtime, solo benefici.

---

### 1.8 — Softmax numericamente stabile

**File:** `neural_net.c` → `softmax`

La versione naïve di softmax (`exp(x[i]) / Σ exp(x[j])`) causa overflow float
per logit grandi (> ~88). La versione stabile sottrae il massimo prima di `exp`:

```c
float max = in[0];
for (int i = 1; i < n; ++i)
    if (in[i] > max) max = in[i];

float sum = 0.f;
for (int i = 0; i < n; ++i) {
    out[i] = expf(in[i] - max);   // sempre ≤ 1, mai overflow
    sum += out[i];
}

float inv = 1.f / sum;            // una divisione, n moltiplicazioni
for (int i = 0; i < n; ++i)
    out[i] *= inv;
```

La divisione (costosa su Cortex-M7, ~14 cicli) viene eseguita una volta sola;
la normalizzazione usa moltiplicazioni per `inv`.

**Regola generale:** sostituire sempre `x / sum` con `x * (1/sum)` nei loop.

---

### 1.9 — Bias correction Adam precalcolata fuori dal loop

**File:** `neural_net.c` → `adam_optimizer_q`

I fattori di bias correction `b1t = 1 - β1^t` e `b2t = 1 - β2^t` sono
costanti per tutti i pesi durante un singolo passo di ottimizzazione. Vengono
calcolati una volta e passati come argomento:

```c
void adam_optimizer_q(QNetwork *net) {
    net->adam_t++;
    float b1t = 1.f - powf(BETA1, (float)net->adam_t);  // calcolato UNA volta
    float b2t = 1.f - powf(BETA2, (float)net->adam_t);
    for (int l = 0; l < net->num_layers; l++)
        adam_update_single_layer(&net->layers[l], b1t, b2t);
}
```

Senza questo, `powf` verrebbe chiamata `2 × num_weights` volte invece di 2.
Per la rete `[3,64,64,5]` sono circa 9.000 chiamate `powf` risparmiate per step.

**Regola generale:** qualsiasi valore costante all'interno di un loop va
sollevato fuori dal loop (loop invariant code motion — il compilatore lo fa
spesso, ma è meglio farlo esplicitamente per chiarezza e sicurezza).

---

### 1.10 — `inv_batch` per normalizzare il gradiente senza divisioni nel loop

**File:** `dqn.c` → `dqn_train`

Il gradiente medio del mini-batch è `dL/dθ = (1/batch_size) × Σ grad_i`.
Invece di dividere ogni `td_error` per `batch_size` dentro il loop:

```c
float inv_batch = 1.0f / (float)batch_size;  // una divisione fuori dal loop
for (uint32_t b = 0; b < batch_size; b++) {
    ...
    dqn_backward(online, s, act, td_error * inv_batch);  // moltiplicazione
}
```

**Regola generale:** stessa logica del punto 1.8 — le divisioni costano più
delle moltiplicazioni su FPU embedded.

---

### 1.11 — Profiling con DWT cycle counter

**File:** `utils.c/h` → `dwt_init`, `dwt_ticks`, `dwt_delta`

Il Data Watchpoint and Trace (DWT) del Cortex-M7 ha un contatore di cicli a 32
bit incrementato ogni ciclo di clock, accessibile senza overhead rilevante:

```c
uint32_t t0 = dwt_ticks();
dqn_train(...);
uint32_t cycles = dwt_delta(t0, dwt_ticks());
// cycles_to_us(cycles) → durata in microsecondi
```

L'overflow a 32 bit avviene ogni ~8.9 secondi a 480 MHz; `dwt_delta` gestisce
l'overflow automaticamente con l'aritmetica modulare `(stop - start)`.

**Regola generale:** usare DWT invece di `HAL_GetTick()` (risoluzione 1 ms) per
misurare porzioni di codice critiche. Non richiede timer hardware aggiuntivi.

---

## Parte 2 — Ottimizzazioni proposte

### 2.1 — Allocazione 2D contigua in `alloc_2d`

**File da modificare:** `dense_layer.c` → `alloc_2d`, `free_2d`

**Problema attuale:**

```c
*matrix = malloc(rows * sizeof(float *));
for (int i = 0; i < rows; ++i)
    (*matrix)[i] = calloc(cols, sizeof(float));  // rows allocazioni separate
```

Per una rete `[3,64,64,5]`, le 3 matrici peso W (più dW, mW, vW) generano
`4 × (64 + 64 + 5) = 532` allocazioni separate solo per i pesi. Ogni riga di
ogni matrice è in un indirizzo heap arbitrario.

**Effetti negativi:**
- Durante il forward pass `acc += w_row[j] * in_vec[j]` l'accesso a `w_row` è
  sequenziale (buono), ma passare da una riga all'altra richiede di caricare un
  nuovo indirizzo dal pointer array (una indirezione + potenziale cache miss).
- Con molte allocazioni piccole, la heap si frammenta e il `malloc` interno
  (newlib nano su STM32) degrada in complessità.
- `free_2d` richiede `rows + 1` chiamate `free`.

**Soluzione — blocco dati contiguo:**

```c
int alloc_2d(float ***matrix, int rows, int cols) {
    if (rows <= 0 || cols <= 0 || !matrix) return 0;

    // Un unico blocco per tutti i dati
    float *data = calloc(rows * cols, sizeof(float));
    if (!data) return 0;

    // Array di puntatori di riga
    *matrix = malloc(rows * sizeof(float *));
    if (!*matrix) { free(data); return 0; }

    for (int i = 0; i < rows; ++i)
        (*matrix)[i] = data + i * cols;

    return 1;
}

int free_2d(float ***matrix, int rows) {
    (void)rows;                  // non più necessario
    if (!matrix || !*matrix) return 0;
    free((*matrix)[0]);          // libera il blocco dati contiguo
    free(*matrix);               // libera l'array di puntatori
    *matrix = NULL;
    return 1;
}
```

**Effetti positivi:**
- Le righe di W sono contigue in memoria: l'inner loop del forward pass
  (`acc += W[i][j] * in[j]`) tocca indirizzi sequenziali per ogni riga.
- Da `rows + 1` allocazioni a esattamente **2 allocazioni** per matrice.
- `free_2d` non ha più bisogno del parametro `rows` (backward-compatible
  poiché il valore viene ignorato).
- Il D-cache del Cortex-M7 (32 KB, 8-way associative, line da 32 byte) può
  prefetchare le righe successive.

**Attenzione:** `(*matrix)[0]` deve essere il puntatore al blocco base. Questa
invariante è garantita dal loop `(*matrix)[i] = data + i * cols` dove `i=0`
dà esattamente `data`. Non mischiare questa `alloc_2d` con allocazioni create
in modo diverso.

---

### 2.2 — Loop swap nel backward pass per W^T·δ

**File da modificare:** `neural_net.c` → `dqn_backward`

**Problema attuale:**

```c
for (int j = 0; j < in_dim; ++j) {
    float acc = 0.0f;
    for (int i = 0; i < out_dim; ++i)
        acc += delta[i] * ly->W[i][j];   // accesso a colonna j di W
    // applica derivata attivazione...
    delta_buf[j] = acc;
}
```

Il prodotto matrice-vettore `W^T · delta` è scritto con `j` in outer loop e
`i` in inner loop. Questo accede a `W[0][j], W[1][j], ..., W[out_dim-1][j]`:
elemento `j` di righe diverse, cioè un accesso per colonna su una matrice
row-major. Ogni `W[i]` è un puntatore diverso (con la `alloc_2d` attuale,
anche a un indirizzo heap diverso) → cache miss ad ogni iterazione di `i`.

Per un layer 64×64: 64 × 64 = 4096 accessi, ognuno con stride di 64 float
(256 byte), molto maggiore della cache line (32 byte) → quasi 0% hit rate
sulla dimensione `i`.

**Soluzione — outer loop su righe di W:**

```c
// Accumula W^T · delta con accesso per riga (cache-friendly)
memset(delta_buf, 0, in_dim * sizeof(float));
for (int i = 0; i < out_dim; ++i) {
    float di = delta[i];
    const float *restrict w_row = ly->W[i];   // w_row è sequenziale
    for (int j = 0; j < in_dim; ++j)
        delta_buf[j] += di * w_row[j];         // accesso sequenziale a w_row
}

// Applica derivata attivazione (separata dal prodotto)
const float    *h_prev = net->layers[l - 1].out;
ActivationType  act    = net->layers[l - 1].activation;

switch (act) {
case ACT_RELU:
    for (int j = 0; j < in_dim; ++j)
        if (h_prev[j] <= 0.f) delta_buf[j] = 0.f;
    break;
case ACT_TANH:
    for (int j = 0; j < in_dim; ++j) {
        float hp = h_prev[j];
        delta_buf[j] *= (1.f - hp * hp);
    }
    break;
default: break;
}
```

Ora l'inner loop accede a `w_row[0], w_row[1], ..., w_row[in_dim-1]`:
accesso sequenziale per riga → il prefetcher del Cortex-M7 può precaricare
le cache line successive.

Beneficio aggiuntivo: separare il prodotto dalla derivata di attivazione
permette al compilatore di vettorizzare i due loop indipendentemente.

**Regola generale:** nei prodotti matrice-vettore `W^T · v`, iterare sempre
sulle righe di W nell'outer loop e accumulare in un buffer separato.

---

### 2.3 — Unificazione di `forward_dense_layer` e `forward_target_layer`

**File da modificare:** `neural_net.c`

Le due funzioni sono identiche riga per riga. L'unica differenza è il tipo
del parametro (`DenseLayer *` vs `TargetLayer *`), ma i campi usati
(`in_dim`, `out_dim`, `activation`, `W`, `b`, `out`) esistono in entrambi.

**Soluzione A — funzione con puntatori espliciti:**

```c
static inline void forward_layer_generic(
    const float    *restrict in_vec,
    float          *restrict out_vec,
    const float   **restrict W,
    const float    *restrict b,
    int in_dim, int out_dim,
    ActivationType activation)
{
    for (int i = 0; i < out_dim; ++i) {
        float acc = b[i];
        const float *restrict w_row = W[i];
        for (int j = 0; j < in_dim; ++j)
            acc += w_row[j] * in_vec[j];
        switch (activation) {
        case ACT_RELU: out_vec[i] = (acc > 0.f) ? acc : 0.f; break;
        case ACT_TANH: out_vec[i] = tanhf(acc);               break;
        default:       out_vec[i] = acc;                       break;
        }
    }
    if (activation == ACT_SOFTMAX)
        softmax(out_vec, out_vec, out_dim);
}
```

I wrapper specifici diventano:

```c
static inline void forward_dense_layer(const float *in, float *out,
                                        const DenseLayer *ly) {
    forward_layer_generic(in, out,
        (const float **)ly->W, ly->b,
        ly->in_dim, ly->out_dim, ly->activation);
}

static inline void forward_target_layer(const float *in, float *out,
                                         const TargetLayer *ly) {
    forward_layer_generic(in, out,
        (const float **)ly->W, ly->b,
        ly->in_dim, ly->out_dim, ly->activation);
}
```

Poiché entrambi i wrapper sono `static inline`, il compilatore li inlina
eliminando il call overhead — il codice macchina risultante è identico
alla versione con duplicazione.

**Regola generale:** codice duplicato con signature diversa ma logica identica
va unificato in una funzione su tipi primitivi (puntatori flat) e wrappato.

---

### 2.4 — Fusione di `gradient_norm_q` e `adam_optimizer_q`

**File da modificare:** `neural_net.c`

Nel loop di training, `dqn_train` chiama:

```c
gradient_norm_q(online);   // 1° passata completa su tutti i pesi
adam_optimizer_q(online);  // 2° passata completa su tutti i pesi
```

Le due funzioni iterano entrambe su `net->num_layers × out_dim × in_dim` pesi.
Se il gradient clipping scatta, tutti i pesi vengono letti tre volte (calcolo
norma, scala, aggiornamento Adam) invece di due.

**Soluzione — funzione unificata:**

```c
void clip_and_adam_q(QNetwork *net) {
    // Passata 1: calcola norma quadratica
    float gnorm_sq = 0.f;
    for (int l = 0; l < net->num_layers; l++) {
        DenseLayer *ly = &net->layers[l];
        for (int i = 0; i < ly->out_dim; i++) {
            gnorm_sq += ly->db[i] * ly->db[i];
            for (int j = 0; j < ly->in_dim; j++)
                gnorm_sq += ly->dW[i][j] * ly->dW[i][j];
        }
    }

    // Fattore di clip (= 1 se non serve clipping)
    const float CLIP = 0.5f;
    float gnorm = sqrtf(gnorm_sq);
    float scale = (gnorm > CLIP) ? (CLIP / gnorm) : 1.0f;

    // Passata 2: scala i gradienti e applica Adam nella stessa iterazione
    net->adam_t++;
    float b1t = 1.f - powf(BETA1, (float)net->adam_t);
    float b2t = 1.f - powf(BETA2, (float)net->adam_t);

    for (int l = 0; l < net->num_layers; l++) {
        DenseLayer *ly = &net->layers[l];
        for (int i = 0; i < ly->out_dim; i++) {
            float db = ly->db[i] * scale;
            float mb = BETA1 * ly->mb[i] + (1.f - BETA1) * db;
            float vb = BETA2 * ly->vb[i] + (1.f - BETA2) * db * db;
            ly->mb[i] = mb; ly->vb[i] = vb;
            ly->b[i] -= LR * (mb / b1t) / (sqrtf(vb / b2t) + EPS_ADAM);
            ly->db[i] = 0.f;

            float *restrict dw = ly->dW[i], *restrict mw = ly->mW[i];
            float *restrict vw = ly->vW[i], *restrict  w = ly->W[i];
            for (int j = 0; j < ly->in_dim; j++) {
                float dw_s = dw[j] * scale;
                float mwj  = BETA1 * mw[j] + (1.f - BETA1) * dw_s;
                float vwj  = BETA2 * vw[j] + (1.f - BETA2) * dw_s * dw_s;
                mw[j] = mwj; vw[j] = vwj;
                w[j] -= LR * (mwj / b1t) / (sqrtf(vwj / b2t) + EPS_ADAM);
                dw[j] = 0.f;
            }
        }
    }
}
```

La funzione esistente `zero_grad_q` diventa ridondante (il reset a zero è
integrato). Nella chiamata in `dqn_train` si rimuove `zero_grad_q` dal fondo
e si chiama `clip_and_adam_q` al posto delle due funzioni separate.

**Nota:** il risparmio assoluto è modesto (una passata su ~9.700 float per
`[3,64,64,5]` a 480 MHz ≈ 2–3 µs). Diventa rilevante su topologie più grandi
o batch più piccoli dove il rapporto compute/overhead è sfavorevole.

---

### 2.5 — Eliminazione dell'indirezione doppia su `TargetLayer.W`

**File da modificare:** `neural_net.h`, `neural_net.c`

Attualmente `TargetLayer.W` è `float **` (puntatore a array di puntatori di
riga), allocato con `alloc_2d`. Se si applica anche l'ottimizzazione 2.1, le
righe diventano già contigue, ma rimane comunque un livello di indirezione
(leggi pointer array → poi leggi dati).

Per la target network, che ha solo forward pass e nessun aggiornamento
incrementale, si può usare un layout flat con accesso indicizzato:

```c
typedef struct {
    float   *W_flat;    // [out_dim * in_dim] — righe contigue senza pointer array
    float   *b;
    float   *out;
    int      in_dim;
    int      out_dim;
    ActivationType activation;
} TargetLayer;
```

Il forward pass diventa:

```c
static inline void forward_target_layer(...) {
    for (int i = 0; i < out_dim; ++i) {
        float acc = b[i];
        const float *w_row = W_flat + i * in_dim;  // calcolo indirizzo, no deref
        for (int j = 0; j < in_dim; ++j)
            acc += w_row[j] * in_vec[j];
        ...
    }
}
```

`copy_weights_to_target` diventa una singola `memcpy` per layer invece di
`out_dim` copie per riga:

```c
void copy_weights_to_target(QNetwork *src, TargetNetwork *dst) {
    for (int l = 0; l < src->num_layers; l++) {
        DenseLayer  *sl = &src->layers[l];
        TargetLayer *tl = &dst->layers[l];
        tl->activation = sl->activation;
        // Una sola memcpy per tutta la matrice
        memcpy(tl->W_flat, sl->W[0], sl->out_dim * sl->in_dim * sizeof(float));
        memcpy(tl->b, sl->b, sl->out_dim * sizeof(float));
    }
}
```

**Prerequisito:** questa ottimizzazione richiede che `sl->W[0]` punti all'inizio
del blocco contiguo — garantito dall'ottimizzazione 2.1.

**Regola generale:** strutture read-only usate solo in forward pass non hanno
bisogno della flessibilità del pointer array. Un layout flat è più semplice,
usa meno memoria (si eliminano `out_dim` puntatori per layer) e permette
`memcpy` bulk.

---

## Riepilogo

### Ottimizzazioni già presenti

| # | Tecnica | File | Categoria |
|---|---|---|---|
| 1.1 | Re-Forward: no cache attivazioni | `neural_net.c`, `dqn.c` | Memoria |
| 1.2 | Target network senza Adam moments | `neural_net.h/c` | Memoria |
| 1.3 | Replay buffer con blocchi contigui | `dqn.c` | Memoria |
| 1.4 | Gradiente sparso all'output (DQN) | `neural_net.c` | Computazione |
| 1.5 | `delta_buf` statico con lazy realloc | `neural_net.c` | Memoria |
| 1.6 | Q-values su stack con `N_ACTIONS` | `dqn.c` | Memoria |
| 1.7 | `restrict`/`const` per il compilatore | `neural_net.c` | Computazione |
| 1.8 | Softmax stabile con `inv = 1/sum` | `neural_net.c` | Computazione |
| 1.9 | Bias correction Adam fuori dal loop | `neural_net.c` | Computazione |
| 1.10 | `inv_batch` per normalizzazione batch | `dqn.c` | Computazione |
| 1.11 | Profiling DWT cycle counter | `utils.c/h` | Strumentazione |

### Ottimizzazioni proposte

| # | Tecnica | File da modificare | Impatto |
|---|---|---|---|
| 2.1 | `alloc_2d` contigua + `free_2d` semplificata | `dense_layer.c` | **Alto** |
| 2.2 | Loop swap `W^T·δ` in backward | `neural_net.c` | **Alto** |
| 2.3 | Unifica `forward_dense/target_layer` | `neural_net.c` | Basso |
| 2.4 | Fonde `gradient_norm_q` + `adam_optimizer_q` | `neural_net.c` | Medio |
| 2.5 | `TargetLayer` con layout flat `W_flat` | `neural_net.h/c` | Medio |

Le ottimizzazioni **2.1** e **2.2** sono indipendenti tra loro e possono essere
applicate separatamente. Le ottimizzazioni **2.4** e **2.5** dipendono
concettualmente da **2.1** (richiedono che W sia contigua per sfruttare
`memcpy` bulk e la `free_2d` semplificata).
