# Istruzioni: Aggiungere il logging CSV al trainer PPO (HIL + PC)

Queste istruzioni descrivono esattamente come aggiungere il logging CSV a un trainer PPO
hardware-in-the-loop (PC ↔ STM32 via seriale) e al suo corrispettivo PC-only.
Il codice di riferimento da cui sono state derivate è `hil_trainer_log.py` (lato HIL)
e `pc_dqn_trainer.py` (lato PC-only, usato come confronto).

---

## Contesto architetturale

Il progetto confronta learning curve ottenute:
1. **Sul micro (HIL)** — il PC manda l'osservazione via seriale, il micro esegue
   forward + training (PPO su STM32), il PC riceve azione e done flag.
2. **Sul PC** — implementazione Python/NumPy speculare all'algoritmo C sul micro,
   usata come baseline di confronto.

Entrambi i trainer salvano un CSV `episode, reward` nella stessa cartella, così da
poter sovrapporre le curve su un unico grafico.

---

## Regola chiave sulle performance

**Non scrivere su disco ad ogni episodio.** Il loop HIL è real-time e ogni I/O
blocca il thread. La strategia corretta è:

- Accumulare i dati in una lista Python durante il training.
- Scrivere tutto il CSV **una sola volta** nel blocco `finally`, quando il training
  viene interrotto (`KeyboardInterrupt`) o termina naturalmente.

Il trainer PC-only (script autonomo con numero fisso di episodi) può invece fare
`flush()` ad ogni episodio senza problemi perché non ha vincoli real-time.

---

## Modifiche lato HIL (trainer seriale PPO)

Analizza il file del trainer HIL esistente e applica le seguenti modifiche nell'ordine.

### 1. Import

Aggiungi `csv` agli import in cima al file:

```python
import csv
```

### 2. Variabili di configurazione

Nella sezione `CONFIGURAZIONE` (dove stanno `PORT`, `BAUDRATE`, ecc.),
aggiungi subito dopo i parametri grafici:

```python
# Logging CSV
LOG_FOLDER   = "learning_curve_comparison"   # cartella di destinazione
LOG_FILENAME = "training_mcu_1.csv"          # nome del file (incrementa il numero per ogni run)
```

Convezione nomi: `training_mcu_1.csv`, `training_mcu_2.csv`, … per il micro;
`training_pc_1.csv`, `training_pc_2.csv`, … per il PC.

### 3. Inizializzazione lista (dentro `main()`, prima del loop)

Dove vengono inizializzate le variabili di stato (prima di `env.reset()`), aggiungi:

```python
# Setup CSV — nessun file aperto qui, si scrive solo alla fine
episode_log = []
```

### 4. Fine episodio (dentro il loop, subito dopo `# ---- FINE EPISODIO ----`)

Sostituisci qualsiasi scrittura su file esistente con un semplice append:

```python
episode_log.append((episode_count, current_ep_reward))
```

Deve stare **prima** di `reward_history.append(...)`, così l'ordine delle operazioni
a fine episodio rimane leggibile.

### 5. Scrittura CSV nel `finally`

Nel blocco `finally` (dove si chiudono `env` e `ser`), aggiungi la scrittura del file
**dopo** `ser.close()` e **prima** di `plt.ioff()`:

```python
os.makedirs(LOG_FOLDER, exist_ok=True)
csv_path = os.path.join(LOG_FOLDER, LOG_FILENAME)
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["episode", "reward"])
    w.writerows(episode_log)
```

`os.makedirs(..., exist_ok=True)` garantisce che la cartella venga creata se non esiste,
senza errori se esiste già.

---

## Modifiche lato PC-only (trainer Python puro)

Il trainer PC-only ha un numero fisso di episodi e termina da solo, quindi può
permettersi di scrivere su disco più spesso. Lo schema usato in `pc_dqn_trainer.py` è:

### 1. Import e configurazione

Stesse due variabili `LOG_FOLDER` / `LOG_FILENAME` nella sezione configurazione.
Aggiungi anche `import csv` se non presente.

### 2. Argparse (opzionale ma consigliato)

Aggiungi un argomento `--csv` per poter sovrascrivere il percorso da riga di comando
senza modificare il codice:

```python
parser.add_argument("--csv", type=str,
                    default=os.path.join(LOG_FOLDER, LOG_FILENAME))
```

### 3. Apertura file prima del loop

```python
csv_dir = os.path.dirname(args.csv)
if csv_dir:
    os.makedirs(csv_dir, exist_ok=True)
csv_file = open(args.csv, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["episode", "reward"])
```

### 4. Fine episodio

```python
csv_writer.writerow([episode, total_reward])
csv_file.flush()   # OK qui: non è real-time
```

### 5. Chiusura file dopo il loop

```python
csv_file.close()
```

---

## Struttura CSV risultante

Entrambi i file producono un CSV con questa struttura:

```
episode,reward
1,-1234.56
2,-987.12
...
```

- `episode`: intero, numero progressivo dell'episodio (parte da 1).
- `reward`: float, reward totale accumulato nell'episodio.

---

## Checklist finale

Dopo aver applicato le modifiche, verifica:

- [ ] `import csv` presente in cima al file.
- [ ] `LOG_FOLDER` e `LOG_FILENAME` definiti nella sezione configurazione.
- [ ] HIL: `episode_log = []` inizializzato prima del loop, nessun file aperto durante il training.
- [ ] HIL: `episode_log.append(...)` dentro il loop a fine episodio, nessun `flush()`.
- [ ] HIL: scrittura CSV completa nel `finally` con `w.writerows(episode_log)`.
- [ ] PC: file aperto prima del loop, `flush()` ad ogni episodio, `close()` dopo il loop.
- [ ] Colonne: `["episode", "reward"]` — esattamente questi nomi, in questo ordine.
- [ ] La cartella `LOG_FOLDER` viene creata automaticamente con `os.makedirs(..., exist_ok=True)`.

