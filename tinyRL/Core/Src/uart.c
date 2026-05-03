#include "uart.h"
#include <string.h>

int uart_recv_floats(UART_HandleTypeDef *huart, float *dst, size_t dim,
                     uint32_t timeout) {
    uint8_t stx;
    if (HAL_UART_Receive(huart, &stx, 1, timeout) != HAL_OK || stx != 0x02)
        return 0;
    if (HAL_UART_Receive(huart, (uint8_t *)dst, dim * sizeof(float), timeout) != HAL_OK)
        return 0;
    uint8_t etx;
    HAL_UART_Receive(huart, &etx, 1, timeout);
    return 1;
}

int uart_send_float_action(UART_HandleTypeDef *huart, float action_val,
                           uint8_t done, uint32_t timeout) {
    uint8_t pkt[7];
    pkt[0] = 0x02;
    memcpy(&pkt[1], &action_val, 4);
    pkt[5] = done;
    pkt[6] = 0x03;
    return HAL_UART_Transmit(huart, pkt, 7, timeout) == HAL_OK ? 1 : 0;
}

int uart_send_floats_action(UART_HandleTypeDef *huart, const float *actions,
                            size_t n, uint8_t done, uint32_t timeout) {
    // packet: [STX][float_0]...[float_{n-1}][done][ETX]
    uint8_t pkt[2 + 16 * sizeof(float) + 1]; // supports up to 16 action dims
    pkt[0] = 0x02;
    memcpy(&pkt[1], actions, n * sizeof(float));
    pkt[1 + n * sizeof(float)] = done;
    pkt[2 + n * sizeof(float)] = 0x03;
    return HAL_UART_Transmit(huart, pkt, 3 + n * sizeof(float), timeout) == HAL_OK ? 1 : 0;
}

int uart_send_action_discrete(UART_HandleTypeDef *huart, uint32_t action,
                              uint8_t done, uint32_t timeout) {
    uint8_t pkt[4] = {0x02, (uint8_t)(action & 0xFF), done, 0x03};
    return HAL_UART_Transmit(huart, pkt, 4, timeout) == HAL_OK ? 1 : 0;
}
