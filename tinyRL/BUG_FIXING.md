# Bug e difetti trovati nel porting delle ottimizzazioni su REINFORCE

## BUG-7 — aliasing del buffer del delta *(latente su questo branch)*

**File:** `Core/Src/neural_net.c` → `backward_from_delta`
**Trovato da:** `test/test_equiv.c`, sezione [2].

Il backward propagava il delta usando **un solo** buffer temporaneo: dalla
seconda iterazione in poi `delta == delta_buf`, quindi scrivere `delta_buf[j]`
distruggeva il `delta[j]` ancora necessario alle iterazioni `j` successive.

**Su questo branch il bug era latente, non attivo.** La topologia discreta di
default e' `[4, 64, 2]`, cioe' **2 soli layer di pesi**: il delta di uscita e'
un array del chiamante e la passata si ferma prima di propagare una seconda
volta, quindi l'aliasing non si verifica mai.

Diventa reale appena si mette `USE_CONTINUOUS_ACTION 1`, che porta la topologia
a `[4, 64, 64, 1]` — 3 layer di pesi. Il test lo mostra su entrambe:

```
ok   discreta[4,64,2] L0 dW           err.rel.max 0
ok   discreta[4,64,2] L1 dW           err.rel.max 0
FAIL continua[4,64,64,1] L0 dW        err.rel.max 1 > 0.0001
ok   continua[4,64,64,1] L1 dW        err.rel.max 0
ok   continua[4,64,64,1] L2 dW        err.rel.max 0
```

Il gradiente del primo layer — la matrice piu' grande della rete — usciva
sbagliato del 100%, senza NaN, senza crash, senza alcun sintomo osservabile dal
training.

**Fix:** pool a **due meta' usate a ping-pong**, con la destinazione che non
coincide mai con la sorgente. Vale per qualsiasi topologia.

> **Conseguenza pratica:** le learning curve **discrete** raccolte finora
> restano valide (il bug non si attivava). Quelle eventualmente raccolte in
> modalita' continua no.

**Secondo difetto, corretto insieme:** la capacita' del buffer veniva ampliata
con `malloc` *dentro* la passata all'indietro, mentre un puntatore al buffer
vecchio era ancora vivo. Ora si dimensiona una volta sola, sul layer piu' largo,
prima di iniziare.

---

## BUG-8 — `uart_send_floats_action` senza bounds check *(latente)*

**File:** `Core/Src/uart.c`

Il buffer di TX e' dimensionato per 16 dimensioni d'azione, ma `n` non veniva
validato: con `n > 16` si scriveva oltre `pkt`, sullo stack. Con
`N_ACT_DIMS = 1` non poteva scattare, ma la funzione e' pubblica e il limite non
era scritto da nessuna parte. Ora `uart_send_action` ritorna 0 se `n` eccede
`UART_MAX_ACTION_DIMS`, verificato nel controllo di protocollo.

---

## BUG-9 — `episode_buffer_push` senza validazione dell'azione *(latente)*

**File:** `Core/Src/reinforce.c`

L'indice dell'azione e' ora memorizzato in `uint8_t` (§A9). Un valore fuori
range verrebbe troncato silenziosamente e finirebbe nella traiettoria come
un'azione **diversa da quella eseguita**, cioe' come una transizione falsa che
avvelena il gradiente Monte-Carlo. La push ora la scarta, ed e' verificato in
`test_update`.

---

## Lavoro inutile trovato (non bug, ma sprechi puri)

- **`policy_forward_continuous` calcolava sempre la log-probabilita'**, che
  costa `D` divisioni e `D` `logf` per campione, e poi la buttava via:
  `policy_sample_action` passa `NULL` come `log_prob_out`, e **nessuno in tutto
  il branch legge quel valore**. In PPO la log-prob al campionamento serve
  (finisce nel buffer per il ratio), in REINFORCE no: il gradiente la ricava da
  `(a - mu)`, non da `lp`. Ora e' dentro `if (log_prob_out)`.
- **`logf(probs[k])` ricalcolato due volte** in `policy_backward` — una per
  l'entropia, una per il gradiente. `2*n_acts` `logf` per step, tutte in
  software su M7.
- **`network_scale_grad`**: una passata di read-modify-write su tutti i
  parametri della rete dopo ogni episodio, per applicare `1/T`. Ripiegata nei
  due coefficienti scalari del delta (§B4), sparisce del tutto.

---

## Falsi positivi scartati

- **`rewards` e `returns` si potrebbero fondere in un array solo.**
  `compute_returns` legge `rewards[t]` e scrive `returns[t]` andando
  all'indietro, e dopo quel punto `rewards` non serve piu': si potrebbe scrivere
  in place e risparmiare 2 KB. Non fatto: 2 KB su 512 KB non giustificano di
  rendere `compute_returns` non ripetibile e di far sparire un nome che dice
  cosa contiene l'array. **Su F446 vale la pena rivalutarlo.**
- **Fondere il calcolo della media dentro `compute_returns`.** Risparmierebbe
  una passata su 500 float, ma cambierebbe l'ordine di somma (all'indietro
  invece che in avanti) e quindi il risultato in virgola mobile, violando il
  vincolo "solo trasformazioni esatte" per un guadagno trascurabile rispetto ai
  500 forward+backward che dominano l'update.
- **`log p` dai logit invece che con `logf(p)`.** Il trucco log-softmax
  (`log p_k = (z_k - max) - log(sum)`) costerebbe **1** `logf` invece di
  `n_acts`. Non e' applicabile qui: `softmax()` clampa `out[i]` a un minimo di
  `1e-7`, quindi `log` del valore clampato e la formula dai logit **non
  coincidono**. Sarebbe un cambio numerico, non una trasformazione esatta.
- **`uart_send_action_discrete` non e' un duplicato di `uart_send_action`.** Il
  payload e' **un byte**, non un float: e' il formato che il trainer lato PC si
  aspetta in modalita' discreta (`_ACTION_STRUCT = "<B"`, frame da 4 byte).
  Fonderle cambierebbe il protocollo sul filo. Restano due funzioni.

- **"L'episodio arriva a 501 step, ma CartPole si ferma a 500."** Controllato a
  fondo: **non e' un errore dell'algoritmo.** 501 e' il numero di scambi UART,
  500 e' il numero di transizioni di ambiente, ed e' anche esattamente quello che
  finisce nel buffer. Il frame in piu' e' l'handshake che porta `done = 1` al PC;
  l'azione che contiene e' campionata in uno stato terminale e viene scartata da
  entrambi i lati (il micro non la memorizza, il trainer fa `break` prima di
  contare reward e di steppare). Invariante `buffer.size == transizioni di
  ambiente` verificata su 400 episodi, con `buffer.size` mai sopra la capacita'.
  Era la metrica del banco di prova software-in-the-loop a essere etichettata
  male, non il conteggio del firmware.
