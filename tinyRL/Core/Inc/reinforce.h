#ifndef REINFORCE_H
#define REINFORCE_H

#include "neural_net.h"
#include "stm32h7xx_hal.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define STARTING_ACTION_SIGMA 0.8f // For continuous action exploration

typedef struct {
  float **state_buffer;
#if USE_CONTINUOUS_ACTIONS
  action_t *
      *action_buffer; // Matrice [n_steps][action_dim] per N azioni continue
#else
  action_t *action_buffer; // Vettore [n_steps] per azione discreta scalare
#endif
  uint8_t *done_buffer;
  float *log_prob_old_buffer;
  float *advantage_buffer;
  float *critic_buffer;
  float *terminal_value_buffer;
  float *sigma_buffer; // Sigma usato al momento della raccolta (condiviso tra
                       // dimensioni)
  uint16_t indices[MAX_STEPS]; // Pre-allocato per evitare malloc in training
  uint32_t action_dim;         // Dimensione dello spazio di azione
} Buffer;

typedef enum {
  CART_POLE,
  ACROBOT
  // FUTURE IMPLEMENTATIONS NEW ENVS
} EnvType;

// buffer init
int buffer_init(Buffer *buf, uint32_t n_steps, uint32_t obs_dim,
                uint32_t action_dim);
// uart recieve
int uart_recv_floats(UART_HandleTypeDef *huart, float *dst, size_t dim,
                     uint32_t timeout);
// uart_send action
#if USE_CONTINUOUS_ACTIONS
int uart_send_action(UART_HandleTypeDef *huart, float *actions,
                     uint32_t action_dim, uint8_t done, uint32_t timeout);
#else
int uart_send_action(UART_HandleTypeDef *huart, action_t action, uint8_t done,
                     uint32_t timeout);
#endif
int uart_send_log(UART_HandleTypeDef *huart, uint32_t dt, uint32_t step,
                  uint32_t timeout);
// sample action
uint32_t sample_action(float *p, uint32_t action_dim);
// step function
int step(SharedBackbone *net, float *obs, float manual_reward,
         uint8_t manual_done, action_t *out_action, uint32_t *step_count,
         Buffer *buffer);
// finish episode
uint32_t finish_episode(Buffer *buf, SharedBackbone *net, uint32_t step_count,
                        uint8_t done);
// evaluate returns and advantages
void evaluate_advantages_and_returns(Buffer *buf, uint32_t step_count);

#endif
