#ifndef DQN_H
#define DQN_H

#include "neural_net.h"
#include "stm32h7xx_hal.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    float    *state_pool;
    float    *snext_pool;
    float   **state;
    float   **next_state;
    uint32_t *action;
    float    *reward;
    uint8_t  *done;
    uint32_t  head;
    uint32_t  size;
    uint32_t  capacity;
    uint32_t  obs_dim;
} ReplayBuffer;

int  replay_buffer_init(ReplayBuffer *buf, uint32_t capacity, uint32_t obs_dim);
void replay_buffer_push(ReplayBuffer *buf, float *s, uint32_t action,
                        float reward, float *s_next, uint8_t done);

int  uart_recv_floats(UART_HandleTypeDef *huart, float *dst, size_t dim,
                      uint32_t timeout);
int  uart_send_float_action(UART_HandleTypeDef *huart, float action_val,
                            uint8_t done, uint32_t timeout);
int  uart_send_action_discrete(UART_HandleTypeDef *huart, uint32_t action,
                               uint8_t done, uint32_t timeout);

float    calc_epsilon(uint32_t step);
uint32_t dqn_select_action(QNetwork *net, float *obs, float epsilon,
                            uint32_t n_actions);

void dqn_train(QNetwork *online, TargetNetwork *target,
               ReplayBuffer *buf, uint32_t batch_size, uint32_t n_actions);

#endif
