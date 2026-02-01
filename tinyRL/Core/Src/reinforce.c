#include "reinforce.h"
#include "utils.h"

//CARTPOLE
#define CART_LIMIT 2.4f                 /* ±2.4 m */
#define POLE_LIMIT 0.20943951f          /* ±12°   */
#define STEP_LIMIT 500

//ACROBOT
#define HEIGHT_THRESHOLD  1.0f



static inline float frand(void) { return (float)rand() / RAND_MAX;}


int buffer_init(Buffer *buf, uint32_t n_steps, uint32_t obs_dim){
    /* azzera i campi così, in caso di errore, i free sono sicuri */
    buf->state_buffer = NULL;
    buf->action_buffer = NULL;
    buf->log_prob_old_buffer = NULL;
    buf->advantage_buffer = NULL;
    buf->critic_buffer = NULL;

    /* ---------- malloc principali ----------------------------- */
    buf->state_buffer = malloc(n_steps * sizeof(float*));
    buf->action_buffer = malloc(n_steps * sizeof(uint32_t));
    buf->log_prob_old_buffer = malloc(n_steps * sizeof(float));
    buf->advantage_buffer = malloc(n_steps * sizeof(float));
    buf->critic_buffer = malloc(n_steps * sizeof(float));

    if (!buf->state_buffer || !buf->action_buffer ||
        !buf->log_prob_old_buffer || !buf->advantage_buffer || !buf->critic_buffer)
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
		free(buf->log_prob_old_buffer);
		free(buf->advantage_buffer);
		free(buf->critic_buffer);
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

int uart_send_action(UART_HandleTypeDef *huart, uint8_t action, uint8_t done, uint32_t timeout){
	uint8_t frame[] = { 0x02, action, done, 0x03 };
	return (HAL_UART_Transmit(huart, frame, sizeof(frame), timeout) == HAL_OK);
}

int uart_send_log(UART_HandleTypeDef *huart, uint32_t dt, uint32_t step, uint32_t timeout){
	uint8_t frame[10];
	frame[0] = 0x01; //STX

	frame[1] = (uint8_t)(dt >> 24); //MSB
	frame[2] = (uint8_t)(dt >> 16);
	frame[3] = (uint8_t)(dt >>  8);
	frame[4] = (uint8_t)(dt >>  0); //LSB

	frame[5] = (uint8_t)(step >> 24);
	frame[6] = (uint8_t)(step >> 16);
	frame[7] = (uint8_t)(step >>  8);
	frame[8] = (uint8_t)(step >>  0);

	frame[9] = 0x04; //ETX
	return (HAL_UART_Transmit(huart, frame, sizeof(frame), timeout) == HAL_OK);
}



uint32_t sample_action(float *p, uint32_t dim){
	const float EPSILON = 0.0f;               /* 1 % */

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


void store_step(Buffer *buf, float *state, uint32_t choosen_action, float reward, float log_prob, float output_critic, uint32_t step_count, uint32_t obs_dim, uint8_t done){
	if(!done){
		memcpy(buf->state_buffer[step_count], state, obs_dim * sizeof(float));
		buf->action_buffer[step_count] = choosen_action;
		buf->log_prob_old_buffer[step_count] = log_prob;
		buf->critic_buffer[step_count] = output_critic;
		if(step_count != 0){
			buf->advantage_buffer[step_count-1] = reward; //reward is the consequence of the previous action
		}
	}
	else{ //if the episode is done the last step reward needs to be buffered
		buf->advantage_buffer[step_count-1] = reward;
	}
}


int step(SharedBackbone *net, float *obs, uint8_t *action, float *reward,
		uint8_t *done, uint32_t *step_count, Buffer *buffer){

	Head *actor = &net->actor;
	Head *critic = &net->critic;
	uint32_t out_dim_actor = actor->layers[actor->num_layers - 1].out_dim;
	uint32_t out_dim_critic = critic->layers[critic->num_layers - 1].out_dim;
	//uint32_t dim = net->layers[0].in_dim;

	//float *output_forward = malloc(out_dim * sizeof(float));
	float output_actor[out_dim_actor];
	float output_critic[out_dim_critic];
	float log_prob = 0.f;
	//if (!output_forward) return 0; //no RAM available

	if(*step_count != 0) *reward = evaluate_reward(obs); //reward will not be stored for the first step
	//CHECK DONE BEFORE WASTING TIME SELECTING AN ACTION
	*done = done_check(obs, *step_count);
	if(*done == 1){ //Truncated so we do the last forward to check the potential of that state
		if(!forward(net, obs, output_actor, output_critic)) return 0; //last forward just to save the last V(s) from critic
		buffer->advantage_buffer[*step_count-1] = *reward; //needed to store only the last step reward
		buffer->terminal_critic_value = output_critic[0];
		return 1;
	}
	if(*done == 2){ //Terminated it was a bad state we do not compute the forward
		buffer->advantage_buffer[*step_count-1] = *reward; //needed to store only the last step reward
		buffer->terminal_critic_value = 0.f;
		return 1;
	}

	if(!forward(net, obs, output_actor, output_critic)) return 0;
	uint32_t a = sample_action(output_actor, out_dim_actor);
	*action = a;

	//evaluate log_prob_old of the selected action useful for the PPO ratio
	float action_prob = output_actor[a];
	log_prob = logf(action_prob + 1e-8f); //Add epsilon 1e-8f to avoid log high values when the argument is near 0

	//-----STORE STEP IN BUFFER-----
	store_step(buffer, obs, *action, *reward, log_prob, output_critic[0], *step_count, net->layers[0].in_dim, *done);
	//free(output_forward);
	*step_count = *step_count + 1;
	return 1;
}

void evaluate_return(Buffer *buf, uint32_t step_count, uint8_t done){
	float *adv_buf = buf->advantage_buffer;
	//returns & baseline
	float G = buf->terminal_critic_value;
	for (int t = step_count - 1; t >= 0; t--) {
		float r = adv_buf[t]; //initially adv_buf has raw rewards
		G = r + GAMMA * G;
		adv_buf[t] = G;
	}
	//normalize returns
	if(step_count > 2){
		float mean = 0.0f;
		for (int t = 0; t < step_count; t++) mean += adv_buf[t];
			mean /= step_count;
		for (int t = 0; t < step_count; t++) adv_buf[t] -= mean;

		float var = 0.0f;
		for (int t = 0; t < step_count; t++) var += adv_buf[t] * adv_buf[t];
			float std = sqrtf(var / step_count) + 1e-6f;
		for (int t = 0; t < step_count; t++) adv_buf[t] /= std;
	}
}

float evaluate_advantages(float ret, float value, float mean, float std){
	float A = ret - value;
	A -= mean;
	A /= std;
	return A;
}

void evaluate_mean_std(Buffer *buf, uint32_t step_count, float *m, float *s){
	if(step_count > 2){
		float *adv_buf = buf->advantage_buffer;
		//adv normalization
		float mean = 0.0f;
		for (int t = 0; t < step_count; t++) mean += adv_buf[t];
			mean /= step_count;
		*m = mean;
		//for (int t = 0; t < step_count; t++) adv_buf[t] -= mean;

		float var = 0.0f;
		for (int t = 0; t < step_count; t++) var += adv_buf[t] * adv_buf[t];
			float std = sqrtf(var / step_count) + 1e-6f;
		*s = std;
		//for (int t = 0; t < step_count; t++) adv_buf[t] /= std;
	}
}

float evaluate_entropy(float *probs, int n_actions){
	float entropy = 0.0f;
	for (int i = 0; i < n_actions; i++){
		if (probs[i] > 1e-8f) { // Avoid log(0) that is -inf
			entropy -= probs[i] * logf(probs[i]);
	    }
	}
	return entropy;
}

uint32_t finish_episode(Buffer *buf, SharedBackbone *net, uint32_t step_count, uint8_t done){
	if (step_count == 0) return 1; //no step in the buffer
	float mean = 0;
	float std = 0;
	//Evaluate advantages and normalize
	evaluate_return(buf, step_count, done);
	evaluate_mean_std(buf, step_count, &mean, &std);

	//zero grad
	zero_grad(net); //TODO

	//Re-Forward + PPO
	Head *sub_net = &net->actor;
	uint8_t out_dim_actor = sub_net->layers[sub_net->num_layers].out_dim;
	float output_actor[out_dim_actor];
	sub_net = &net->critic;
	uint8_t out_dim_critic = sub_net->layers[sub_net->num_layers].out_dim;
	float output_critic[out_dim_critic];
	float log_prob_new = 0.f;
	for(int epoch =  0; epoch < N_EPOCHS; epoch++){
		for(int t = 0; t < step_count; t++){
			float *state = buf->state_buffer[t];
			forward(net, state, output_actor, output_critic);

			//-----ACTOR LOSS-----
			//evaluate log_prob_new of the selected action useful for the PPO ratio
			float action_prob = output_actor[buf->action_buffer[t]];
			log_prob_new = logf(action_prob + 1e-8f); //Add epsilon 1e-8f to avoid log high values when the argument is near 0
			//PPO ratio
			float ratio = expf((log_prob_new - buf->log_prob_old_buffer[t]));
			float surr1 = ratio * evaluate_advantages(buf->advantage_buffer[t], buf->critic_buffer[t], mean, std);
			float surr2 = clip(ratio, 1.0f - 0.2, 1.0f + 0.2);
			surr2 = surr2 * evaluate_advantages(buf->advantage_buffer[t], buf->critic_buffer[t], mean, std);
			float actor_loss_step = (surr1 < surr2) ? surr1 : surr2;
			actor_loss_step = -actor_loss_step; //minus because we have to maximize it

			//-----CRITIC LOSS-----
			float critic_loss_step = powf((output_critic[0] - buf->advantage_buffer[t]), 2);

			//TOTAL LOSS
			//float total_loss = actor_loss_step + (CRIT_LOSS * critic_loss_step) - (ENT_BETA * evaluate_entropy(output_actor, out_dim_actor));
		}
	}


	return 1;
}

/*uint32_t finish_episode(Buffer *buf, SharedBackbone *net, uint32_t step_count){
	if (step_count == 0) return 1; //no step in the buffer
	//zero grad
	zero_grad(net);
	float *adv_buf = buf->advantage_buffer;
	//returns & baseline
	float G = 0.0f;
	for (int t = step_count - 1; t >= 0; --t) {
		float r = adv_buf[t]; //TODO
		G = r + 0.99f * G;
		adv_buf[t] = G;
	}
	//adv normalization
	float mean = 0.0f;
	for (int t = 0; t < step_count; t++) mean += adv_buf[t];
	mean /= step_count;
	for (int t = 0; t < step_count; t++) adv_buf[t] -= mean;

	float var = 0.0f;
	for (int t = 0; t < step_count; t++) var += adv_buf[t] * adv_buf[t];
	float std = sqrtf(var / step_count) + 1e-6f;
	for (int t = 0; t < step_count; t++) adv_buf[t] /= std;


	//Re-Forward
	for(int t = 0; t < step_count; t++){
		float *state = buf->state_buffer[t];
		float r = buf->advantage_buffer[t]; //TODO

		forward(net, state, NULL, NULL);

		float adv = adv_buf[t];
		uint32_t a = buf->action_buffer[t];


		backward_pg(net, state, a, adv, r, step_count);

	}

	//Normalize gradient
	//gradient_norm_l2(net);


	adam_optimizer(net);


	return 0;
} */

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

/*CODE FOR CARTPOLE*/
uint8_t done_check(float *state, uint32_t step){
	if (fabsf(state[0]) > CART_LIMIT) return 2;      //out of bound
	if (fabsf(state[2]) > POLE_LIMIT) return 2;      //±12°
	if (step >= STEP_LIMIT){
		return 1;   //timeout
	}
	return 0;
}

float evaluate_reward(float *state){
	return 1.f;
}


/*CODE FOR ACROBOT
uint8_t done_check(float *state, uint32_t step){
	uint16_t a1 = 20;
	uint16_t a2 = 10;
	uint32_t goal1 = 0;
	uint32_t goal2 = 180;
	float cos1 = state[0];
	float sin1 = state[1];
	float cos2 = state[2];
	float sin2 = state[3];
	float theta1 = atan2f(sin1, cos1) * (180.0 / M_PI);
	float theta2 = atan2f(sin2, cos2) * (180.0 / M_PI);
	uint8_t angle1_reached = (theta1<goal1+a1)&&(theta1>goal1-a1);
	uint8_t angle2_reached = (theta2<goal2+a2)&&(theta2>goal2-a2);
	if((angle2_reached) || step>=500) return 1;
	else return 0;
}


float evaluate_reward(float *state, uint32_t step){
	int reward = -1;
	uint16_t a1 = 20;
	uint16_t a2 = 10;
	uint32_t goal1 = 0;
	uint32_t goal2 = 180;
	float cos1 = state[0];
	float sin1 = state[1];
	float cos2 = state[2];
	float sin2 = state[3];
	float theta1 = atan2f(sin1, cos1) * (180.0 / M_PI);
	float theta2 = atan2f(sin2, cos2) * (180.0 / M_PI);
	uint8_t angle1_reached = (theta1<goal1+a1)&&(theta1>goal1-a1);
	uint8_t angle2_reached = (theta2<goal2+a2)&&(theta2>goal2-a2);
	if((angle2_reached) && (angle1_reached)) reward = reward + 100;
	return reward;
}
*/

