#ifndef REINFORCE_H
#define REINFORCE_H

#include "stm32f4xx_hal.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "neural_net.h"

typedef struct{
	float **state_buffer;
	uint32_t *action_buffer;
	float *reward_buffer;
	float *advantage_buffer;
}Buffer;


//buffer init
int buffer_init(Buffer *buf, uint32_t n_steps, uint32_t obs_dim);
//uart recieve
int uart_recv_floats(UART_HandleTypeDef *huart, float *dst, size_t dim, uint32_t timeout);
//uart_send action
int uart_send_action(UART_HandleTypeDef *huart, uint8_t action, uint8_t done, uint32_t timeout);
//clip obs not mandatory
//sample action
uint32_t sample_action(float *p, uint32_t action_dim);
//done function
uint8_t done_check(float *state, uint32_t step);
//step function
int step(Buffer *buf, NeuralNet *net, float *obs, uint32_t step, uint8_t *action);
//store step
void store_step(Buffer *buf, float *state, uint32_t choosen_action, float reward, uint32_t step, uint32_t obs_dim);
//reward function
float evaluate_reward(float *obs);
//finish episode
void finish_episode(Buffer *buf, NeuralNet *net, uint32_t step_count);

#endif
