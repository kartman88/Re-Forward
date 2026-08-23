#include "uart.h"
#include <string.h>

int uart_recv_floats(UART_HandleTypeDef *huart, float *dst, size_t dim,
                     uint32_t timeout) {
    // Frame: STX(0x02) + payload(dim*4) + checksum(XOR) + ETX(0x03).
    // Checksum + validazione ETX scartano i frame disallineati post-overrun:
    // senza di essi un frame spazzatura passerebbe i controlli e finirebbe nel
    // replay buffer. Deve corrispondere a send_state() di hil_trainer_log.py.
    uint8_t stx;
    if (HAL_UART_Receive(huart, &stx, 1, timeout) != HAL_OK || stx != 0x02)
        return 0;

    const size_t nbytes = dim * sizeof(float);
    if (HAL_UART_Receive(huart, (uint8_t *)dst, nbytes, timeout) != HAL_OK)
        return 0;

    uint8_t chk;
    if (HAL_UART_Receive(huart, &chk, 1, timeout) != HAL_OK)
        return 0;

    uint8_t etx;
    if (HAL_UART_Receive(huart, &etx, 1, timeout) != HAL_OK || etx != 0x03)
        return 0;

    uint8_t calc = 0;
    const uint8_t *p = (const uint8_t *)dst;
    for (size_t i = 0; i < nbytes; i++)
        calc ^= p[i];
    if (calc != chk)
        return 0;   // frame corrotto/disallineato: il PC lo ritrasmette

    return 1;
}

int uart_send_action(UART_HandleTypeDef *huart, const float *actions, size_t n,
                     uint8_t done, uint32_t timeout) {
    // packet: [STX][float_0]...[float_{n-1}][done][ETX]
    uint8_t pkt[2 + UART_MAX_ACTION_DIMS * sizeof(float) + 1];

    // La versione precedente non validava n: con n > 16 scriveva oltre pkt.
    if (n == 0 || n > UART_MAX_ACTION_DIMS)
        return 0;

    const size_t payload = n * sizeof(float);
    pkt[0] = 0x02;
    memcpy(&pkt[1], actions, payload);
    pkt[1 + payload] = done;
    pkt[2 + payload] = 0x03;
    return HAL_UART_Transmit(huart, pkt, payload + 3, timeout) == HAL_OK ? 1 : 0;
}
