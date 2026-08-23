# Test host

I file di `Core/Src` che implementano l'algoritmo (`dense_layer.c`,
`neural_net.c`, `dqn.c`, `rng.c`) non usano l'HAL: con due header stub vuoti
compilano in nativo, quindi i kernel si verificano su PC senza flashare nulla.

E' una proprieta' da preservare deliberatamente: le funzioni UART stavano in
`dqn.c` ed erano l'unico punto di contatto con l'HAL, sono state spostate in
`Core/Src/uart.c`, che questi test non compilano.

`make` esegue:

**`test_equiv`** — confronta i kernel ottimizzati contro un'implementazione di
riferimento ingenua, scritta riga per riga come il codice prima delle
ottimizzazioni (forward per riga con lo switch dentro il loop, `W^T*delta` con
accesso per colonna, Adam in forma testuale con tre divisioni). Serve a
dimostrare che le trasformazioni dichiarate esatte lo siano davvero. Copre:

1. forward con blocking 2x1 e attivazione in passata separata;
2. backward con loop swap e **ping-pong sul buffer del delta** — e' il controllo
   che ha trovato BUG-7: con un buffer solo i layer 1 e 2 combaciano e il layer
   0 esce sbagliato del 100%;
3. `dqn_backward` sparso contro il gradiente denso con 4 zeri su 5
   (scostamento atteso: esattamente 0);
4. Adam in forma efficiente contro la forma testuale;
5. `gradient_norm_q` che ritorna il fattore di scala invece di riscrivere i
   gradienti;
6. rete target e rete online che condividono lo stesso kernel.

Gli scarti residui (~1e-6 relativo, solo sul ramo tanh) sono riassociazione in
virgola mobile, gia' permessa da `-Ofast`.

**`test_update`** — smoke test end-to-end: riempie un replay buffer sintetico,
gira un `dqn_train` completo e verifica capienza e circolarita' del buffer, il
rifiuto delle azioni fuori range, l'azzeramento dei gradienti dopo Adam, la
copia esatta verso la rete target e l'assenza di NaN/Inf nei pesi.

`make asan` rigira entrambi sotto AddressSanitizer e UBSan: e' il controllo che
l'aritmetica delle arene (una sola allocazione per layer, una per la rete
target, una per il replay buffer) non scriva fuori dai limiti.

Da rieseguire dopo ogni modifica a forward, backward o all'ottimizzatore.
