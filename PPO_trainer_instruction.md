# Istruzioni: trainer PPO per il confronto delle learning curve

Questo file serve a una **nuova chat** che lavorerà sulla versione del codice con **PPO**.

> ⚠️ **Non copiare la rete, le azioni o gli iperparametri da questa versione DQN.**
> Devi **analizzare il codice PPO che trovi in quella versione** (sia il sorgente che gira
> sul micro, sia eventuale codice host) e **adattarti a ciò che vedi lì**: la rete che usano,
> la task/ambiente che usano, la discretizzazione/parametrizzazione delle azioni di *quella*
> versione. L'unico vincolo è produrre **le stesse misure** descritte sotto, nello **stesso
> formato di output**, così che le curve siano confrontabili con quelle di questa versione.

---

## Cosa devi produrre (questo è il vincolo da rispettare)

Uno script di training PPO in Python che, **usando la rete e la task della versione PPO**,
salvi una learning curve in CSV con **esattamente lo stesso formato** usato qui:

1. **CSV con due colonne: `episode,reward`** (header incluso, in quest'ordine).
   - una riga per episodio;
   - `reward` = **somma dei reward dell'episodio** (reward totale, non normalizzato/mediato).
2. Salvataggio nella cartella **`learning_curve_comparison/`**.
3. Nome file **configurabile da una variabile in cima allo script**, p.es.:
   ```python
   LOG_FILENAME = "training_ppo_1.csv"      # cambia qui per le run successive
   LOG_FOLDER   = "learning_curve_comparison"
   ```
   - convenzione nomi già in uso nella cartella: `training_mcu_N.csv` (micro),
     `training_pc_N.csv` (PC). Usa `training_ppo_N.csv` per PPO (o conferma col tuo utente).
   - crea la cartella con `os.makedirs(..., exist_ok=True)` e fai `flush()` a ogni episodio.

Questo è l'unico requisito rigido: **stesse misure, stesso formato CSV, stessa cartella,
nome configurabile da variabile**.

---

## Cosa devi RICAVARE dal codice PPO (non assumere — analizzalo)

Prima di scrivere lo script, ispeziona la versione PPO e determina:

- **Ambiente / task**: quale environment usano (es. Pendulum-v1 o altro), quante dimensioni
  ha l'osservazione, quanti step per episodio, come viene definito `done` (termination vs
  truncation), e qual è la lunghezza dell'episodio. Usa **quella** task, non assumere.
- **Rete**: topologia, numero di layer, dimensioni, attivazioni, inizializzazione dei pesi.
  Replica **quella** rete (policy e, se presente, value/critic). Non forzare `{3,64,64,5}`.
- **Azioni**: come sono parametrizzate in *quella* versione — discrete o continue, quante,
  e con quali valori/mapping. Usa **quella** definizione.
- **Reward**: come viene calcolato il reward in *quella* versione. Usa **quello**.
- **Iperparametri PPO**: leggi quelli definiti lì (LR, gamma, GAE lambda, clip, epoche,
  rollout length, entropy/value coef, gradient clip, Adam betas/eps). Riusali quando esistono.

Il sorgente del micro di solito sta in una cartella tipo `Core/Src/` (es. `main.c`,
`*_net.c`, `*_layer.c`, e un file specifico dell'algoritmo, qui era `dqn.c` → lì sarà
qualcosa come `ppo.c` o `reinforce.c`). Cerca anche eventuale trainer host già presente.

---

## Dettagli di formato da rispettare (per coerenza con questa versione)

Per rendere le curve realmente confrontabili, allinea questi aspetti **di misura/output**
(non di algoritmo):

- **Unità sull'asse X = episodio** (intero crescente da 1).
- **Unità sull'asse Y = reward totale per episodio** (somma sugli step dell'episodio).
- Se la task della versione PPO ha episodi di lunghezza fissa, somma i reward su tutto
  l'episodio; se ha terminazione anticipata, somma fino al `done`.
- Header CSV **esatto**: `episode,reward`.
- Reward scritto come float (qui sono valori negativi, es. da ~-1700 verso 0, ma dipende
  dalla task: NON forzare un range, usa quello reale della task PPO).

---

## CLI consigliata (per uniformità con il trainer di questa versione)

Stessi argomenti usati qui, utili per le run ripetute e la riproducibilità:

```
--episodes   (default = numero episodi di confronto, es. 100)
--seed       (default None = casuale; se impostato, semina anche env.reset(seed=...))
--csv        (default = os.path.join(LOG_FOLDER, LOG_FILENAME))
--no-plot    (disabilita il grafico live)
```

Plot live opzionale: reward per episodio + media mobile (qui a 20 episodi), titolo
"Learning Curve (PPO)". Serve solo per controllo visivo, **il dato che conta è il CSV**.

---

## Plotting di confronto (già esistente in questa versione)

In `learning_curve_comparison/plot_complete.py` c'è già lo script che confronta le curve
dai vari `training_*.csv`. Assicurati che i tuoi `training_ppo_N.csv` abbiano lo stesso
header (`episode,reward`) così da essere caricati senza modifiche (o adatta il pattern dei
nomi se necessario).

---

## Checklist finale per la nuova chat

- [ ] Ho **analizzato** il codice PPO (micro + eventuale host) prima di scrivere.
- [ ] Uso la **rete** che trovo in quella versione (non quella del DQN).
- [ ] Uso la **task/ambiente, azioni e reward** di quella versione.
- [ ] Riuso gli **iperparametri PPO** definiti lì dove presenti.
- [ ] Output CSV `episode,reward`, una riga per episodio, reward = somma per-episodio.
- [ ] CSV salvato in `learning_curve_comparison/`, nome da variabile `LOG_FILENAME`.
- [ ] CLI `--episodes --seed --csv --no-plot` + plot live opzionale.
- [ ] Verificato che `plot_complete.py` carichi i nuovi file di confronto.

