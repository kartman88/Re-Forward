#ifndef UART_H
#define UART_H

#include "stm32h7xx_hal.h"
#include <stddef.h>

// Numero massimo di dimensioni d'azione per frame (capienza del buffer di TX).
#define UART_MAX_ACTION_DIMS 16

/* Frame RX: STX(0x02) + payload(dim*4) + checksum(XOR) + ETX(0x03).
 * Frame TX: STX(0x02) + payload(n*4) + done + ETX(0x03).
 * L'asimmetria (il TX non porta checksum) e' voluta: il formato deve restare
 * quello atteso da Micro_RL/hil_trainer_log.py.
 *
 * Queste due funzioni stavano in dqn.c ed erano l'unico punto in cui il file
 * dell'algoritmo toccava l'HAL. Spostate qui, dqn.c/neural_net.c/dense_layer.c/
 * rng.c compilano in nativo con due header stub vuoti: e' cio' che rende
 * possibili i test host in test/. */
int uart_recv_floats(UART_HandleTypeDef *huart, float *dst, size_t dim,
                     uint32_t timeout);

/* Invia n azioni float. Sostituisce le vecchie uart_send_float_action (n=1) e
 * uart_send_action_discrete (mai usata): passare &val, 1 per il caso scalare.
 * Ritorna 0 se n eccede UART_MAX_ACTION_DIMS o se la trasmissione fallisce. */
int uart_send_action(UART_HandleTypeDef *huart, const float *actions, size_t n,
                     uint8_t done, uint32_t timeout);

#endif
