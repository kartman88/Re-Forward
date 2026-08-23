# Test host

I file di `Core/Src` che implementano l'algoritmo (`dense_layer.c`,
`neural_net.c`, `ppo.c`, `rng.c`) non usano l'HAL: con due header stub vuoti
compilano in nativo, quindi i kernel si verificano su PC senza flashare nulla.

`make` esegue:

**`test_equiv`** — confronta i kernel ottimizzati contro un'implementazione di
riferimento ingenua, scritta riga per riga come il codice prima delle
ottimizzazioni (forward per riga con lo switch dentro il loop, `W^T*delta` con
accesso per colonna, Adam in forma testuale con tre divisioni). Serve a
dimostrare che le trasformazioni dichiarate esatte lo siano davvero. Gli scarti
residui (~1e-5 relativo) sono riassociazione in virgola mobile, gia' permessa
da `-Ofast`.

**`test_update`** — smoke test end-to-end: riempie un rollout sintetico, gira
un `ppo_update` completo e verifica capienza del buffer, bounds check della
push, normalizzazione degli advantage e assenza di NaN/Inf nei pesi.

`make asan` rigira entrambi sotto AddressSanitizer e UBSan: e' il controllo che
l'aritmetica delle arene (una sola allocazione per layer e una per il rollout
buffer) non scriva fuori dai limiti.

Da rieseguire dopo ogni modifica a forward, backward o all'ottimizzatore.
