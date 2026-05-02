#ifndef UART_H
#define UART_H

#include "stm32h7xx_hal.h"
#include <stdlib.h>

int uart_recv_floats(UART_HandleTypeDef *huart, float *dst, size_t dim,
                     uint32_t timeout);
int uart_send_float_action(UART_HandleTypeDef *huart, float action_val,
                           uint8_t done, uint32_t timeout);
int uart_send_action_discrete(UART_HandleTypeDef *huart, uint32_t action,
                              uint8_t done, uint32_t timeout);

#endif
