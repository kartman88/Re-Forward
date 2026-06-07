# Istruzioni: Calcolatore consumo di memoria per PPO

Questo documento descrive come realizzare uno script Python che calcola il
consumo di memoria heap dell'implementazione PPO su STM32, nello stesso stile
e con la stessa struttura di output usata per la versione DQN.

L'obiettivo è ottenere uno script interattivo che chiede all'utente la
topologia delle reti e i parametri rilevanti, e produce in output **tabelle
dettagliate per ogni sezione di memoria** e un **totale finale espresso in
kilobyte**.

---

## 1. Cosa deve fare lo script

1. **Analizzare il codice C corrente** (header `.h` e source `.c`) per
   identificare tutte le strutture allocate dinamicamente (`malloc`/`calloc`)
   o staticamente che contribuiscono al consumo di memoria significativo.
2. **Chiedere all'utente in modo interattivo**:
   - La topologia delle reti neurali (lista di dimensioni separate da spazio
     o virgola, es. `3 64 64 1` per la value net, `3 64 64 5` per la policy).
     In PPO ci sono tipicamente **due reti** (actor / policy e critic / value),
     quindi vanno chieste entrambe.
   - Le dimensioni dei buffer principali (rollout buffer, trajectory buffer,
     ecc. — vedi sezione 3).
   - Eventuali altri parametri necessari (batch size, n_epochs, obs_dim,
     n_actions).
   - Permettere valori di default e override (input vuoto = default).
3. **Calcolare il consumo** di ogni struttura applicando le regole della
   sezione 4 (dimensione tipi, puntatori, struct overhead).
4. **Stampare tabelle separate** per ogni macro-sezione, con subtotale.
5. **Stampare un riepilogo finale** con il totale in byte e in KB.

---

## 2. Strutture dati tipiche di PPO da analizzare

Prima di scrivere lo script **devi leggere il codice sorgente PPO** (Inc/Src)
ed elencare tutte le struct rilevanti. Le sezioni tipiche da aspettarsi sono:

### a) Policy network (actor)
- Per ogni layer denso, di solito: `W`, `b`, `out`, più momenti Adam
  (`mW`, `vW`, `mb`, `vb`) e gradienti (`dW`, `db`).
- Può avere logits, distribuzione, log-prob buffer.

### b) Value network (critic)
- Analogo alla policy, ma output scalare (V(s)).
- Anch'essa con momenti Adam e gradienti.

### c) Rollout / Trajectory buffer
PPO non usa un replay buffer; raccoglie traiettorie da N step per K episodi.
Strutture tipiche:
- `states[N][obs_dim]` (float)
- `actions[N]` (uint32 o float)
- `rewards[N]` (float)
- `values[N]` (float)
- `log_probs[N]` (float)
- `advantages[N]` (float)
- `returns[N]` (float)
- `dones[N]` (uint8)

### d) Stack / locali principali
- Buffer temporanei in `ppo_train`, `policy_forward`, `compute_gae`, ecc.
- Logits temporanei della policy (`logits[N_ACTIONS]`).

**IMPORTANTE**: la lista esatta va costruita leggendo il codice — non
assumere. Cerca tutti i `malloc`, `calloc`, e i campi nelle `struct` definite
negli header.

---

## 3. Domande da porre all'utente

Lo script deve chiedere almeno:

| Parametro | Esempio | Note |
|-----------|---------|------|
| Topologia policy network | `3 64 64 5` | input/hidden/output |
| Topologia value network | `3 64 64 1` | output scalare |
| `obs_dim` | `3` | dedotto da input layer, override possibile |
| `n_actions` | `5` | dedotto da output policy |
| Rollout buffer size (N step) | `2048` | quanti step per update |
| Batch size (mini-batch PPO) | `64` | |
| `n_epochs` PPO | `10` | non incide su RAM ma utile |
| Eventuale buffer di episodi | dipende | se il codice li separa |

Permetti default sensati (invio = default) e validazione (interi positivi).

---

## 4. Regole di calcolo (identiche a DQN)

Costanti da usare (target = STM32 Cortex-M, ARM 32-bit):

```python
SIZE_FLOAT   = 4    # byte
SIZE_UINT32  = 4
SIZE_UINT8   = 1
SIZE_PTR     = 4    # puntatori a 32 bit
```

### Matrice 2D allocata come array di puntatori a righe
Se nel codice C la matrice `W` è `float **` con `out_dim` righe di `in_dim`
elementi (come in `alloc_2d`), allora:
- Dati: `out_dim * in_dim * SIZE_FLOAT`
- Array di puntatori riga: `out_dim * SIZE_PTR`

### Vettore 1D
- `n * SIZE_FLOAT` (o tipo corrispondente)

### Struct overhead
Stima `sizeof(StructName)` sommando i campi con allineamento ARM standard:
- `int` = 4 byte
- `float *`, `void *`, qualsiasi puntatore = `SIZE_PTR` = 4 byte
- `enum` ≈ 4 byte
- `uint8_t` = 1 byte (può causare padding a 4)

Esempio per `DenseLayer` (DQN): 2 int + 9 ptr + 1 enum ≈ 48 byte.
**Riapplica la stessa logica per le struct PPO** che trovi nel codice.

### Layer della policy/value con Adam
Per ogni layer con ottimizzatore Adam allocato:
- 4 matrici `out×in` (W, dW, mW, vW) → 4 × (`out*in*4` + `out*4`) byte
- 5 vettori `out` (b, db, mb, vb, out) → 5 × `out*4` byte

Se PPO usa una struttura diversa (es. niente `dW` separato perché calcolato
on-the-fly, o momenti separati per layer), **adatta il calcolo a quello che
trovi nel codice**, non copiare ciecamente dal DQN.

---

## 5. Struttura dello script Python

```
memory_calculator_ppo.py
├── Costanti SIZE_FLOAT / SIZE_PTR / ...
├── ask_int(prompt, default)               # input intero validato
├── ask_topology(label)                    # input topologia validata
├── policy_layer_bytes(in, out)            # bytes di un layer policy
├── value_layer_bytes(in, out)             # bytes di un layer value
├── policy_breakdown(topology)             # lista (label, bytes)
├── value_breakdown(topology)              # lista (label, bytes)
├── rollout_breakdown(N, obs_dim, ...)     # lista (label, bytes)
├── stack_breakdown(...)                   # locali principali
├── print_table(title, items) -> subtotal
└── main()                                 # orchestra tutto
```

### Funzione `print_table` — formato tabella standard
Usa esattamente questo formato (è quello concordato per la versione DQN):

```
[N] TITOLO DELLA SEZIONE
------------------------------------------------------------------------------
Componente                                                       Bytes      KB
------------------------------------------------------------------------------
  voce 1                                                          XXXX   X.XX
  voce 2                                                          XXXX   X.XX
------------------------------------------------------------------------------
  SUBTOTALE                                                       XXXX   X.XX
```

Larghezze: label 60, bytes 10, KB 8. KB con 2 decimali.

### Riepilogo finale
Dopo le sezioni, stampa:

```
==============================================================================
 RIEPILOGO
==============================================================================
Sezione                                                          Bytes      KB
------------------------------------------------------------------------------
Policy network                                                    ...
Value network                                                     ...
Rollout buffer                                                    ...
Stack/locali                                                      ...
------------------------------------------------------------------------------
TOTALE                                                            ...
==============================================================================

>>> Consumo totale stimato: XXX.XX KB (XXXXX byte) <<<
```

### Informazioni aggiuntive utili
Alla fine, stampa anche:
- Numero totale di **parametri allenabili** (W + b) per policy e value.
- Memoria per **una sola copia dei pesi** (utile per capire l'overhead Adam,
  che richiede 4 copie per parametro: W + dW + mW + vW).

---

## 6. Direttive di stile e qualità

Queste sono le linee guida concordate per la versione DQN — applicare le
stesse:

1. **Lingua**: testo utente e commenti in italiano. Identificatori Python in
   inglese.
2. **Niente emoji** nel codice o nell'output (a meno che l'utente lo chieda).
3. **Niente docstring lunghe**; un docstring breve a inizio file basta.
4. **Niente commenti banali** che spiegano cosa fa il codice ovvio. Commenta
   solo dove serve chiarire una formula o un'assunzione non ovvia (es.
   "puntatore 4 byte perché ARM Cortex-M 32-bit").
5. **Validazione input**: rifiutare valori non interi / non positivi e
   richiedere di nuovo.
6. **Default sensati**: invio vuoto = default. I default devono riflettere
   tipici parametri PPO (es. rollout buffer 2048, batch 64, n_epochs 10).
7. **Tabelle allineate** con il formato della sezione 5.
8. **Output finale chiaro**: il totale in KB deve essere immediatamente
   leggibile e marcato con `>>> ... <<<`.
9. Lo script deve essere **eseguibile direttamente** (`python3 memory_calculator_ppo.py`).
10. **Test rapido finale**: dopo aver scritto lo script, esegui un dry-run con
    una topologia tipica e verifica che i numeri siano coerenti (es. per la
    versione DQN con `3-64-64-5` e replay 1000 il totale è ~136.54 KB).

---

## 7. Procedura consigliata

1. Esplora `Inc/` e `Src/` del progetto PPO. Elenca tutte le `struct` e tutte
   le allocazioni `malloc`/`calloc`.
2. Per ogni struttura, identifica i campi e calcola la formula di consumo
   parametrizzata sulla topologia / dimensione buffer.
3. Scrivi `memory_calculator_ppo.py` seguendo la struttura della sezione 5.
4. Testa con valori plausibili e verifica che le tabelle siano allineate e
   che il totale in KB sia ragionevole per un STM32H7 (tipicamente 1 MB RAM
   totale, ma DTCM/SRAM divisa in banchi più piccoli).
5. Salva lo script in una posizione accessibile del repo PPO (es. nella root
   del progetto, accanto al `Makefile` o al CMake).

---

## 8. Riferimento: cosa è stato fatto per la versione DQN

Per riferimento, nella versione DQN sono state identificate queste sezioni:

1. **QNetwork online** — per ogni `DenseLayer`: W + dW + mW + vW (matrici
   `out×in`) + b + db + mb + vb + out (vettori `out`) + array puntatori riga
   + overhead struct.
2. **TargetNetwork** — per ogni `TargetLayer`: solo W + b + out + puntatori
   riga.
3. **ReplayBuffer** — `state_pool`, `snext_pool` (`capacity * obs_dim *
   float`), array di puntatori `state[]`/`next_state[]`, `action[]` (uint32),
   `reward[]` (float), `done[]` (uint8).
4. **Stack/locali** — `q_online`, `q_tgt`, `q` (vettori `N_ACTIONS` float in
   `dqn_train` e `dqn_select_action`).

Risultato con topology `3-64-64-5`, replay `1000`: **~136.54 KB**.

Per PPO le sezioni saranno diverse (niente target net, niente replay buffer,
ma rollout buffer + value net) — usa questa lista solo come modello mentale.

