#ifndef UART_H
#define UART_H

#include "stm32h7xx_hal.h"
#include <stddef.h>

// Numero massimo di dimensioni d'azione per frame (capienza del buffer di TX).
#define UART_MAX_ACTION_DIMS 16

/* Frame RX: STX(0x02) + payload(dim*4) + checksum(XOR) + ETX(0x03).
 * Deve corrispondere a send_state() di Micro_RL/hil_trainer_continuous.py. */
int uart_recv_floats(UART_HandleTypeDef *huart, float *dst, size_t dim,
                     uint32_t timeout);

/* Frame TX continuo: STX(0x02) + payload(n*4) + done + ETX(0x03).
 * Unifica le vecchie uart_send_float_action (n = 1, mai usata) e
 * uart_send_floats_action: per il caso scalare si passa &val, 1.
 * Ritorna 0 se n eccede UART_MAX_ACTION_DIMS o se la trasmissione fallisce —
 * il controllo su n prima mancava, e con n > 16 si scriveva oltre il buffer. */
int uart_send_action(UART_HandleTypeDef *huart, const float *actions, size_t n,
                     uint8_t done, uint32_t timeout);

/* Frame TX discreto: STX(0x02) + azione(1 byte) + done + ETX(0x03).
 * NON e' una specializzazione di uart_send_action: il payload e' un byte, non
 * un float, ed e' il formato che si aspetta il trainer lato PC in modalita'
 * discreta (_ACTION_STRUCT = "<B", frame da 4 byte). Va tenuta separata. */
int uart_send_action_discrete(UART_HandleTypeDef *huart, uint32_t action,
                              uint8_t done, uint32_t timeout);

#endif
