#ifndef UART_H
#define UART_H

#include "stm32f4xx_hal.h"
#include <stdlib.h>

int uart_recv_floats(UART_HandleTypeDef *huart, float *dst, size_t dim,
                     uint32_t timeout);
int uart_send_float_action(UART_HandleTypeDef *huart, float action_val,
                           uint8_t done, uint32_t timeout);
int uart_send_floats_action(UART_HandleTypeDef *huart, const float *actions,
                            size_t n, uint8_t done, uint32_t timeout);
int uart_send_action_discrete(UART_HandleTypeDef *huart, uint32_t action,
                              uint8_t done, uint32_t timeout);

#endif
