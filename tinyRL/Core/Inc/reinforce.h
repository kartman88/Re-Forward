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

typedef enum {
    CART_POLE,
	ACROBOT
	//FUTURE IMPLEMENTATIONS NEW ENVS
} EnvType;


//buffer init
int buffer_init(Buffer *buf, uint32_t n_steps, uint32_t obs_dim);
//uart recieve
int uart_recv_floats(UART_HandleTypeDef *huart, float *dst, size_t dim, uint32_t timeout);
//uart_send action
int uart_send_action(UART_HandleTypeDef *huart, uint8_t action, uint8_t done, uint32_t timeout);
int uart_send_log(UART_HandleTypeDef *huart, uint32_t dt, uint32_t step, uint32_t timeout);
//sample action
uint32_t sample_action(float *p, uint32_t action_dim);
//step function
int step(Buffer *buf, SharedBackbone *net, float *obs, uint32_t step, uint8_t *action);
//store step
void store_step(Buffer *buf, float *state, uint32_t choosen_action, float reward, uint32_t step, uint32_t obs_dim);
//finish episode
uint32_t finish_episode(Buffer *buf, SharedBackbone *net, uint32_t step_count);

//done function cartpole
uint8_t done_check(float *state, uint32_t step);
//reward function
float evaluate_reward(float *state);



#endif
