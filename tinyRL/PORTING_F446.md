# Port di tinyRL/PPO da NUCLEO-H743ZI2 a NUCLEO-F446RE

Questo branch (`PPO_F446`) porta il progetto PPO, nato per la NUCLEO-H743ZI2,
sulla NUCLEO-F446RE. La base di partenza per il livello piattaforma e' il
progetto F446 gia' presente sul branch `main` (versione REINFORCE).

## 1. Il vincolo che decide tutto: la RAM

| | NUCLEO-H743ZI2 | NUCLEO-F446RE |
|---|---|---|
| Core | Cortex-M7 @ 480 MHz | Cortex-M4 @ 84 MHz |
| FPU | doppia precisione | singola precisione (`fpv4-sp-d16`) |
| RAM usata dal linker | `RAM_D1` 512 KB | `RAM` 128 KB |
| Flash | 2 MB | 512 KB |
| Cache L1 / MPU | I+D cache, MPU | assenti |
| Virtual COM ST-LINK | USART3 (PD8/PD9) | USART2 (PA2/PA3) |
| LED utente | LD1 / LD2 / LD3 | solo LD2 (PA5) |

La configurazione H7 (topologia `11-64-64-*`, `ROLLOUT_STEPS 2048`) consuma
**325.75 KB** (verificato con `memory_calculator_ppo.py`), contro i 128 KB
totali della F446.

Il punto chiave: **non basta accorciare il rollout**. Con hidden 64 le sole due
reti pesano ~163 KB, perche' Adam tiene 4 copie di ogni peso (`W`, `dW`, `mW`,
`vW`). Anche con rollout azzerato la F446 non le contiene. La topologia *deve*
scendere.

Configurazioni valutate (KB stimati, budget utile ~110 KB):

| hidden | T=2048 | T=1024 | T=512 | T=256 |
|---|---|---|---|---|
| 64 | 325.8 | 244.8 | 204.3 | 184.0 |
| 48 | 260.9 | 179.9 | 139.4 | 119.2 |
| **32** | 212.1 | 131.1 | **90.6** | 70.4 |
| 24 | 193.7 | 112.7 | 72.2 | 52.0 |

Scelta: **hidden 32, `ROLLOUT_STEPS` 512** — il compromesso che mantiene il
rollout piu' lungo possibile (la qualita' della stima GAE dipende da T) restando
con margine dentro i 128 KB.

### Consumo misurato, non solo stimato

Lo stimatore non conta l'overhead per-chunk di `malloc`, e qui le allocazioni
sono tante e piccole (`alloc_2d` alloca una riga per volta). Misurato con un
test host che replica il setup di `main.c`:

```
allocazioni       = 593
byte richiesti    = 90.84 KB   <- coincide con la stima (90.63 KB)
stima newlib-nano = 96.52 KB   <- header 8 byte/chunk, allineamento 8
heap glibc reale  = 105.69 KB  <- solo host, glibc ha overhead maggiore
```

Sul target (newlib-nano, `--specs=nano.specs`) ci si aspettano **~97 KB di
heap**, piu' ~3 KB di `.bss`/`.data` e lo stack: **~101 KB su 128 KB**.

## 2. Modifiche al livello piattaforma

Sostituiti in blocco dal progetto F446 di `main`:

- `Drivers/STM32F4xx_HAL_Driver/` e `Drivers/CMSIS/Device/ST/STM32F4xx/`
  (rimossi gli equivalenti H7)
- `Core/Startup/startup_stm32f446retx.s`
- `Core/Src/system_stm32f4xx.c`, `stm32f4xx_it.c`, `stm32f4xx_hal_msp.c`
- `Core/Inc/stm32f4xx_hal_conf.h`, `stm32f4xx_it.h`
- `STM32F446RETX_FLASH.ld`, `STM32F446RETX_RAM.ld`
- `.cproject`, `.project`, `.mxproject`, `tinyRL.ioc`, `.settings/`,
  `tinyRL Debug.launch`

`Core/Inc/main.h` viene dal progetto F446 (pin map `LD2`/`B1`/`USART_TX`/
`USART_RX`), con `TIME_LOG` riportato dal branch H7.

### Include cambiati nei file portabili

`uart.h`, `dense_layer.c` → `stm32f4xx_hal.h`; `utils.h` → `stm32f4xx.h`.
Tutto il resto (`ppo.c/h`, `neural_net.c/h`, `rng.c`, `uart.c`, `utils.c`) e'
identico all'H7: nessuna dipendenza dalla famiglia.

`rng.c` usa PCG32 in software, non il peripherale RNG hardware (che la F446 non
ha): nessuna modifica necessaria.

`utils.c` (DWT) funziona invariato: il ciclo-contatore c'e' anche sul
Cortex-M4. Cambia solo il tempo di wrap del CYCCNT a 32 bit, ~51 s a 84 MHz
contro ~8.9 s a 480 MHz — gli accumulatori a 64 bit di `TrainTiming` restano
comunque necessari.

## 3. Modifiche a `main.c`

- **Rimosse** `SCB_EnableICache()` / `SCB_EnableDCache()` e tutta
  `MPU_Config()`: il Cortex-M4 non ha cache L1 ne' MPU da configurare qui.
- **Clock**: HSI 16 MHz → PLL M=16, N=336, P=4 → **84 MHz** SYSCLK,
  `FLASH_LATENCY_2`, `PWR_REGULATOR_VOLTAGE_SCALE3` (l'H7 usava LDO + VOS0 +
  `FLASH_LATENCY_4` per 480 MHz).
- **UART**: `USART3` → `USART2`. L'USART della serie F4 non ha FIFO ne'
  prescaler, quindi spariscono `ClockPrescaler`, `OneBitSampling` e le chiamate
  `HAL_UARTEx_Set{Tx,Rx}FifoThreshold` / `HAL_UARTEx_DisableFifoMode`.
- **Flush RX dopo il training**: `UART_RXDATA_FLUSH_REQUEST` non esiste sulla
  F4; si usa `__HAL_UART_FLUSH_DRREGISTER()` (leggere SR+DR azzera ORE e scarta
  il byte residuo del registro a singolo byte).
- **LED**: la F446 ha un solo LED utente. `LD1` (ready) e `LD3` (alloc fail)
  spariscono; il fallimento di allocazione ora si segnala con LD2 lampeggiante
  a 100 ms, distinguibile dal LED fisso durante gli update.
- **Topologia**: `HIDDEN_DIM 32` (era 64), applicata ad actor e critic.

## 4. Modifiche a `ppo.h`

- `ROLLOUT_STEPS` 2048 → 512.
- `PPO_SIGMA_N_STEPS` ora e' riscalato sul rollout:
  `500000 * PPO_SIGMA_REF_ROLLOUT / ROLLOUT_STEPS`.
  Motivo: `ppo_sigma_decay()` viene chiamata **una volta per update**, cioe'
  ogni `ROLLOUT_STEPS` step di ambiente. Accorciando il rollout di 4x gli update
  diventano 4x piu' frequenti e sigma collasserebbe 4x prima *a parita' di step
  di ambiente*. Con il riscalamento la traiettoria di sigma in funzione degli
  step di ambiente resta identica al build H7.

Invariati: `PPO_BATCH_SIZE 64`, `PPO_EPOCHS 8`, lr, gamma, lambda, clip.

## 5. Linker script

`_Min_Stack_Size` 0x400 → 0x1000. `_sbrk()` usa quel simbolo come guardia
(`heap_end + incr > _estack - _Min_Stack_Size` → `ENOMEM`); con solo ~27 KB fra
il tetto dell'heap e la fine della RAM conviene una riserva di stack piu'
generosa di 1 KB.

## 6. Lato PC

`Micro_RL/hil_trainer_continuous.py` non richiede modifiche: usa
`/dev/ttyACM0` a 115200 baud e il protocollo STX/checksum/ETX, identici. Anche
la NUCLEO-F446RE espone il virtual COM dello ST-LINK come `/dev/ttyACM0`.

Nota: le righe `<<<PROF>>>` ora arrivano ogni 512 step invece che ogni 2048.

## 7. Cosa NON e' stato verificato

Su questa macchina non e' installato `arm-none-eabi-gcc`, quindi **il firmware
non e' stato compilato per il target ne' provato su scheda**. Le verifiche
fatte sono:

- compilazione host (`gcc -Wall -Wextra`, pulita) di `ppo.c`, `neural_net.c`,
  `dense_layer.c`, `rng.c`, `utils.c`, `uart.c` con un HAL stub;
- smoke test funzionale host: init reti + agente, 532 step di `ppo_step` fino a
  innescare un `ppo_update` completo — nessuna azione non finita, un update
  eseguito, heap misurato come sopra.

Da fare alla prima build reale: aprire il progetto in STM32CubeIDE (o
`make -C Debug`) e controllare che il consuntivo del linker riporti un `.bss`
piccolo e RAM residua coerente con i ~101 KB previsti.
