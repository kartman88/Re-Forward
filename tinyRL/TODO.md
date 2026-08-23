# TODO

- [x] Unire `uart_send_float_action` e `uart_send_floats_action` in un'unica funzione
      generica che accetta un array di float e una dimensione, eliminando la versione
      scalare. → ora è `uart_send_action(huart, actions, n, done, timeout)`, con il
      bounds check su `n` che prima mancava.

- [ ] Rimisurare `ppo_update` sulla scheda con `TIME_LOG = 1` e aggiornare la tabella
      dei risultati in `OPTIMIZATIONS.md` (la riga tempo è ancora una stima).

- [ ] Girare una volta con `BENCH_KERNELS = 1` per sapere quanto pesano davvero le
      262.144 `tanhf` per update. È il numero che decide se valga la pena riaprire la
      questione dell'approssimazione polinomiale.

- [ ] Riacquisire le learning curve: BUG-7 (vedi `BUG_FIXING.md`) rendeva sbagliato il
      gradiente del primo layer, quindi tutte le curve precedenti sono da rifare.

- [ ] Riportare le stesse modifiche sul branch `PPO_F446` prima di pubblicare qualsiasi
      confronto fra le due schede.

- [ ] Decidere sul bonus di entropia in modalità continua: `PPO_C2` è definito ma non
      entra nel gradiente, l'esplorazione è governata solo dallo schedule di sigma.
      O si applica, o si toglie la costante.
