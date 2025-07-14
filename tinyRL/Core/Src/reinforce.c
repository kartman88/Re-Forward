#include "reinforce.h"

#define CART_LIMIT   2.4f                 /* ±2.4 m */
#define POLE_LIMIT   0.20943951f          /* ±12°   */
#define STEP_LIMIT   500

static inline float frand(void) { return (float)rand() / RAND_MAX;}


int buffer_init(Buffer *buf, uint32_t n_steps, uint32_t obs_dim){
    /* azzera i campi così, in caso di errore, i free sono sicuri */
    buf->state_buffer    = NULL;
    buf->action_buffer   = NULL;
    buf->reward_buffer   = NULL;
    buf->advantage_buffer= NULL;

    /* ---------- malloc principali ----------------------------- */
    buf->state_buffer   = malloc(n_steps * sizeof(float*));
    buf->action_buffer  = malloc(n_steps * sizeof(uint32_t));
    buf->reward_buffer  = malloc(n_steps * sizeof(float));
    buf->advantage_buffer = malloc(n_steps * sizeof(float));

    if (!buf->state_buffer || !buf->action_buffer ||
        !buf->reward_buffer || !buf->advantage_buffer)
        goto fail;

    /* righe per la matrice delle osservazioni */
    for (uint32_t i = 0; i < n_steps; ++i) {
        buf->state_buffer[i] = malloc(obs_dim * sizeof(float));
        if (!buf->state_buffer[i])
            goto fail;
    }
    return 1;                       /* tutto OK */

    fail:   /* qualsiasi malloc fallita → libera e segnala errore */
		if (buf->state_buffer) {
			for (uint32_t i = 0; i < n_steps; ++i)
				free(buf->state_buffer[i]);
			free(buf->state_buffer);
		}
		free(buf->action_buffer);
		free(buf->reward_buffer);
		free(buf->advantage_buffer);
		return 0;
}

int uart_recv_floats(UART_HandleTypeDef *huart,
                     float             *dst,
                     size_t             dim,
                     uint32_t           timeout)
{
    const size_t nbytes = dim * sizeof(float);
    uint8_t byte;
    uint8_t buf[nbytes];                 /* dim=4*4 */
    uint32_t t0 = HAL_GetTick();

    /* 1. Cerca lo STX ------------------------------------------------ */
    do {
        if (HAL_UART_Receive(huart, &byte, 1, 1) != HAL_OK) {
            if (HAL_GetTick() - t0 > timeout) return 0; /* timeout totale */
            continue;                                   /* riprova */
        }
    } while (byte != 0x02);

    /* 2. Legge esattamente nbytes + ETX ------------------------------ */
    if (HAL_UART_Receive(huart, buf, nbytes, timeout) != HAL_OK) return 0;
    if (HAL_UART_Receive(huart, &byte, 1, timeout) != HAL_OK) return 0;
    if (byte != 0x03)                                            return 0;

    memcpy(dst, buf, nbytes);             /* OK: frame completo */
    return 1;
}

int uart_send_action(UART_HandleTypeDef *huart, uint8_t action, uint8_t done, uint32_t timeout, uint16_t obs_dim){
	uint8_t frame[] = { 0x02, action, done, 0x03 };
	return (HAL_UART_Transmit(huart, frame, sizeof(frame), timeout) == HAL_OK);
}


uint32_t sample_action(float *p, uint32_t dim){
	const float EPSILON = 0.1f;               /* 1 % */

	/* ─── 1. esplorazione pura ogni tanto ─── */
	float r = (float)rand() / (float)RAND_MAX; /* uniform [0,1) */
	if (r < EPSILON)
		return (uint8_t)(rand() % dim);        /* azione random */

	/* ─── 2. campionamento “roulette-wheel” ─── */
	float c = (float)rand() / (float)RAND_MAX; /* [0,1) */
	for (uint32_t i = 0; i < dim; ++i) {
		if (c < p[i]) return (uint8_t)i;
		c -= p[i];
	}
	return (uint8_t)(dim - 1);                 /* fallback numerico */

}


void store_step(Buffer *buf, float *state, uint32_t choosen_action, float reward, uint32_t step, uint32_t obs_dim){

	memcpy(buf->state_buffer[step], state, obs_dim * sizeof(float));
	buf->action_buffer[step] = choosen_action;
	buf->reward_buffer[step] = reward;
}


int step(Buffer *buf, NeuralNet *net, float *obs, uint32_t step, uint8_t *action){
	uint32_t out_dim = net->layers[net->num_layers - 1].out_dim;
	uint32_t dim = net->layers[0].in_dim;

	//float *output_forward = malloc(out_dim * sizeof(float));
	float output_forward[out_dim];
	//if (!output_forward) return 0; //no RAM available

	if(!forward(net, obs, output_forward)) return 0;
	uint32_t a = sample_action(output_forward, out_dim);
	*action = a;
	float r = evaluate_reward(obs); //CONTROLLARE ORDINE REWARD AZIONE
	store_step(buf, obs, a, r, step, dim);

	//free(output_forward);
	return 1;
}

void finish_episode(Buffer *buf, NeuralNet *net, uint32_t step_count){

	if (step_count == 0) return; //no step in the buffer
	//zero grad
	zero_grad(net);
	float *adv_buf = buf->advantage_buffer;
	//returns & baseline
	float G = 0.0f;
	for (int t = step_count - 1; t >= 0; --t) {
		float r = buf->reward_buffer[t]; //POTREI SISTEMARE QUI PER RISOLVERE IL PROBLEMA DEGLI EPISODI CON T+1
		G = r + 0.99f * G;
		adv_buf[t] = G;
	}
	//adv normalization
	float mean = 0.0f;
	for (int t = 0; t < step_count; ++t) mean += adv_buf[t];
	mean /= step_count;
	for (int t = 0; t < step_count; ++t) adv_buf[t] -= mean;

	float var = 0.0f;
	for (int t = 0; t < step_count; ++t) var += adv_buf[t] * adv_buf[t];
	float std = sqrtf(var / step_count) + 1e-6f;
	for (int t = 0; t < step_count; ++t) adv_buf[t] /= std;



	for(int t = 0; t < step_count; ++t){
		float *state = buf->state_buffer[t];
		float r = buf->reward_buffer[t];
		forward(net, state, NULL);
		float adv = adv_buf[t];
		uint32_t a = buf->action_buffer[t];
		backward_pg(net, state, a, adv, r);
	}

	//Normalize gradient
	//gradient_norm_l2(net);
	adam_optimizer(net);
}

/*CODE FOR MOUNTAIN CAR
uint8_t done_check(float *state, uint32_t step){
	if (state[0] >= 0.5) return 1; //goal reached
	if (step >= 200) return 1; //timeout
	return 0;
}

float prev_pos = 0.f;
float evaluate_reward(float *obs){
	float r = 0.f;
	r-=1.0;
	if(obs[0]>prev_pos) r+=2.0;
	return r;
}
*/

//CODE FOR CARTPOLE
uint8_t done_check(float *state, uint32_t step){
	if (fabsf(state[0]) > CART_LIMIT) return 1;      //out of bound
	if (fabsf(state[2]) > POLE_LIMIT) return 1;      //±12°
	if (step >= STEP_LIMIT) return 1;   //timeout
	return 0;
}

float evaluate_reward(float *obs){
	return 1.f;
}



