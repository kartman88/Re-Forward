# Test host

I file di `Core/Src` che implementano l'algoritmo (`dense_layer.c`,
`neural_net.c`, `reinforce.c`, `rng.c`) non usano l'HAL: con due header stub
vuoti compilano in nativo, quindi i kernel si verificano su PC senza flashare
nulla. Le funzioni UART stanno in `uart.c`, che questi test non compilano — e'
quella separazione a rendere possibile tutto il resto, e va preservata.

`make` esegue:

**`test_equiv`** — confronta i kernel ottimizzati contro un'implementazione di
riferimento ingenua, scritta riga per riga come il codice prima delle
ottimizzazioni. Copre:

1. forward con blocking 2x1 e attivazione in passata separata;
2. backward con loop swap e **ping-pong sul buffer del delta**, su DUE
   topologie: quella discreta `[4,64,2]` (2 layer di pesi, immune
   all'aliasing) e quella continua `[4,64,64,1]` (3 layer, dove l'aliasing si
   manifesta e il layer 0 esce sbagliato del 100%);
3. `policy_backward` con `logf` riusato invece che ricalcolato due volte
   (scostamento atteso: esattamente 0);
4. `1/T` ripiegato nel delta invece che applicato a valle da
   `network_scale_grad`;
5. Adam in forma efficiente contro la forma testuale;
6. `network_clip_grad` che ritorna il fattore di scala invece di riscrivere i
   gradienti.

Il controllo [4] e' un'equivalenza **algebrica, non bit-per-bit**:
`(sum x_t)*s` e `sum (x_t*s)` riassociano diversamente in virgola mobile. E'
stato verificato a parte contro un riferimento in doppia precisione, e la forma
ripiegata risulta leggermente **piu'** accurata (errore L2 2,58e-8 contro
3,11e-8).

L'errore relativo usa un denominatore con pavimento a 1e-3 volte la scala del
vettore: un elemento di gradiente che nasce da cancellazione puo' valere ~1e-6
su un vettore di scala ~1e-1, e li' l'errore relativo per elemento misura il
rumore, non la qualita' del kernel.

**`test_update`** — smoke test end-to-end: campiona una traiettoria vera con
`policy_sample_action`, gira un `reinforce_update` completo e verifica capienza
del buffer, rifiuto delle azioni fuori range, correttezza analitica dei ritorni
(`G_t = (1-gamma^(T-t))/(1-gamma)` con reward costante), normalizzazione,
azzeramento dei gradienti dopo Adam, no-op su episodio vuoto e assenza di
NaN/Inf.

`make asan` rigira entrambi sotto AddressSanitizer e UBSan: e' il controllo che
l'aritmetica delle arene (una allocazione per layer, una per il buffer di
episodio) non scriva fuori dai limiti.

Da rieseguire dopo ogni modifica a forward, backward o all'ottimizzatore.
