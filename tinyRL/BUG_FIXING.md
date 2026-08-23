# Bug trovati nel porting delle ottimizzazioni su DQN

## BUG-7 — aliasing del buffer del delta nel backward *(correttezza, silenzioso)*

**File:** `Core/Src/neural_net.c` → `dqn_backward`
**Trovato da:** `test/test_equiv.c`, sezione [2]. **Nessuna learning curve lo
avrebbe segnalato.**

La versione precedente propagava il delta all'indietro usando **un solo**
buffer temporaneo:

```c
static float *delta_buf = NULL;
...
for (int j = 0; j < in_dim; ++j) {
    float acc = 0.0f;
    for (int i = 0; i < out_dim; ++i)
        acc += delta[i] * ly->W[i][j];   /* legge delta[] */
    ...
    delta_buf[j] = acc;                  /* scrive delta_buf[] */
}
delta = delta_buf;                       /* dal prossimo giro sono lo stesso! */
```

Alla prima iterazione (layer 2) `delta` e' l'array del gradiente di uscita e
`delta_buf` e' un blocco distinto: nessun problema. Dalla **seconda** iterazione
in poi `delta == delta_buf`, quindi scrivere `delta_buf[j]` distrugge il
`delta[j]` che serve ancora alle iterazioni `j` successive.

Sulla topologia `[3, 32, 32, 5]` (3 layer di pesi) il risultato e':

| layer | gradiente |
|---|---|
| 2 (32→5)  | corretto |
| 1 (32→32) | corretto |
| **0 (3→32)** | **sbagliato** |

Niente NaN, niente crash, nessun sintomo osservabile: solo il gradiente del
primo layer sistematicamente errato. Il test lo riproduce esattamente:

```
FAIL relu layer 0 dW   err.rel.max 1 > 0.0001
FAIL relu layer 0 db   err.rel.max 1 > 0.0001
ok   relu layer 1 dW   err.rel.max 0
ok   relu layer 2 dW   err.rel.max 0
```

**Fix:** il buffer e' ora un pool a **due meta' usate a ping-pong**, con la
destinazione che non coincide mai con la sorgente.

**Secondo difetto, latente:** la capacita' veniva ampliata con `malloc` *dentro*
la passata all'indietro, quando un puntatore al buffer vecchio era ancora vivo.
Sulle topologie attuali non scatta (andando indietro `in_dim` non cresce mai),
ma su una rete tipo `[3, 64, 32, 5]` sarebbe un use-after-free. Ora la capacita'
si dimensiona **una volta sola, sul layer piu' largo, prima di iniziare**.

> ⚠️ **Impatto sulle misure:** tutte le learning curve DQN raccolte prima di
> questo fix sono state prodotte con il gradiente del primo layer sbagliato.
> Vanno riacquisite prima di qualsiasi confronto. I tempi di update non sono
> invece influenzati: il numero di operazioni era lo stesso.

---

## BUG-8 — `uart_send_action` senza bounds check *(latente)*

**File:** `Core/Src/uart.c`

La vecchia `uart_send_float_action` aveva `n` fissato a 1 e quindi era sicura,
ma unificandola nella `uart_send_action(huart, actions, n, ...)` di PPO il
parametro `n` diventa libero: senza validazione, `n > UART_MAX_ACTION_DIMS`
scriverebbe oltre il buffer di TX. Il controllo c'e' ora, e la funzione ritorna
0 invece di corrompere lo stack.

---

## BUG-9 — `replay_buffer_push` senza validazione dell'azione *(latente)*

**File:** `Core/Src/dqn.c`

L'indice dell'azione e' passato a `uint8_t` (§A9). Un valore fuori range verrebbe
troncato silenziosamente e finirebbe nel buffer come un'azione diversa da quella
eseguita — cioe' come una transizione falsa. La push ora scarta la transizione
invece di troncarla, ed e' verificato in `test_update`.

---

## Falsi positivi scartati

- **`head` e `size` del replay buffer non sono ridondanti.** In PPO (§A2) i due
  contatori erano uguali per costruzione e sono stati fusi. Qui no: il buffer e'
  **circolare**, `head` torna a 0 mentre `size` si ferma a `capacity`. Restano
  due campi.
- **`zero_grad_q` a inizio update non e' morto.** Adam azzera gia' `dW`/`db`
  mentre li consuma, quindi in regime la chiamata non ha nulla da ripulire (lo
  verifica `test_update`). E' pero' la rete di sicurezza che rende l'update
  indipendente dallo stato lasciato dal precedente, e costa due `memset` per
  layer contro le ~40k MAC dell'update. Resta.
