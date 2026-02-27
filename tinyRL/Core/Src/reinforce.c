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
    buf->state_buffer    = NULL;
    buf->action_buffer   = NULL;
    buf->reward_buffer   = NULL;
    buf->advantage_buffer= NULL;

    /* ---------- malloc principali ----------------------------- */
    buf->state_buffer   = malloc(n_steps * sizeof(float*));
    buf->action_buffer  = malloc(n_steps * sizeof(action_t));
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

int uart_send_action(UART_HandleTypeDef *huart, action_t action, uint8_t done, uint32_t timeout){
	#if USE_CONTINUOUS_ACTIONS
		uint8_t frame[7];
		frame[0] = 0x02; // STX
		memcpy(&frame[1], &action, sizeof(float)); // Copia i 4 byte del float
		frame[5] = done;
		frame[6] = 0x03; // ETX
		return (HAL_UART_Transmit(huart, frame, sizeof(frame), timeout) == HAL_OK);
	#else
		uint8_t frame[] = { 0x02, action, done, 0x03 };
		return (HAL_UART_Transmit(huart, frame, sizeof(frame), timeout) == HAL_OK);
	#endif
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

// L'azione salvata nel buffer non sarà più un intero (uint8_t/uint32_t) ma un float!
float sample_continuous_action(float mu, float sigma){
    // Genera due numeri uniformi tra 0 e 1
    float u1 = fmaxf(frand(), 1e-7f); // Evita log(0)
    float u2 = frand();

    // Box-Muller transform per rumore Normale standard N(0,1)
    float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);

    // Scala con la nostra media e varianza
    float action = mu + sigma * z0;

    // (Opzionale) Clamp dell'azione se il tuo motore accetta ad es. solo [-1, 1]
    if (action > MAX_CONTINUOUS_ACTION) action = 1.0f;
    if (action < MIN_CONTINUOUS_ACTION) action = -1.0f;

    return action;
}


void store_step(Buffer *buf, float *state, uint32_t choosen_action, float reward, uint32_t step, uint32_t obs_dim){

	memcpy(buf->state_buffer[step], state, obs_dim * sizeof(float));
	buf->action_buffer[step] = choosen_action;
	buf->reward_buffer[step] = reward;
}


int step(Buffer *buf, NeuralNet *net, float *obs, uint32_t *step_count, action_t *action, uint8_t *done){
    uint32_t out_dim = net->layers[net->num_layers - 1].out_dim;
    uint32_t in_dim = net->layers[0].in_dim;

    //Immediately check if is a termination state
    *done = done_check(obs, *step_count);

    //Reward goes into the previous step
    //The current obs is the conseguence of taking the action at t-1
    if (*step_count > 0) {
        float r = evaluate_reward(obs);
        buf->reward_buffer[*step_count - 1] = r;
    }

    //Exit if episode is terminated
    if (*done) {
        return 1;
    }

    //Forward pass, we choose the action not for a terminal state
    float output_forward[out_dim];
    if(!forward(net, obs, output_forward)) return 0;

    action_t a;
    #if USE_CONTINUOUS_ACTIONS
        // Usiamo una sigma fissa di 0.5 per esplorare
        a = sample_continuous_action(output_forward[0], 0.5f);
    #else
        a = sample_action(output_forward, out_dim);
    #endif

        *action = a;

        memcpy(buf->state_buffer[*step_count], obs, in_dim * sizeof(float));
        buf->action_buffer[*step_count] = a;

        (*step_count)++;
        return 1;
}

uint32_t finish_episode(Buffer *buf, NeuralNet *net, uint32_t step_count){
	if (step_count == 0) return 1; //no step in the buffer
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
	//adv normalization, REINFORCE with costant baseline
	float mean = 0.0f;
	for (int t = 0; t < step_count; ++t) mean += adv_buf[t];
	mean /= step_count;
	for (int t = 0; t < step_count; ++t) adv_buf[t] -= mean;

	float var = 0.0f;
	for (int t = 0; t < step_count; ++t) var += adv_buf[t] * adv_buf[t];
	float std = sqrtf(var / step_count) + 1e-6f;
	for (int t = 0; t < step_count; ++t) adv_buf[t] /= std;


	//re-forward
	for(int t = 0; t < step_count; ++t){
		float *state = buf->state_buffer[t];
		float r = buf->reward_buffer[t];

		forward(net, state, NULL);

		float adv = adv_buf[t];
		action_t a = buf->action_buffer[t];


		backward_pg(net, state, a, adv, r);

	}

	//Normalize gradient
	gradient_norm(net, step_count);


	adam_optimizer(net);


	return 0;
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

/*CODE FOR CARTPOLE
uint8_t done_check(float *state, uint32_t step){
	if (fabsf(state[0]) > CART_LIMIT) return 1;      //out of bound
	if (fabsf(state[2]) > POLE_LIMIT) return 1;      //±12°
	if (step >= STEP_LIMIT){
		return 1;   //timeout
	}
	return 0;
}

float evaluate_reward(float *state){
	return 1.f;
}*/

/*CODE FOR PENDULUM*/
uint8_t done_check(float *state, uint32_t step) {
    // Il documento conferma che non ci sono condizioni di "out of bounds".
    // Si tronca solo al raggiungimento dei 200 step.
    if (step >= MAX_STEPS) {
        return 1;   // Timeout
    }
    return 0;
}

// --- Funzione di Reward ---
float evaluate_reward(float *state) {
    // In base alla documentazione (Observation Space), l'array state contiene:
    // state[0] = x = cos(theta)
    // state[1] = y = sin(theta)
    // state[2] = Angular Velocity (theta_dt)

    // 1. Ricaviamo l'angolo theta già normalizzato tra [-pi, pi]
    float theta = atan2f(state[1], state[0]);
    float theta_dt = state[2];

    // 2. La coppia (torque).
    // Come detto in precedenza, la firma della tua funzione non accetta l'azione.
    // Se non puoi modificare la firma o leggere l'azione, poniamo torque a 0.0f.
    // Il range valido della torque è tra -2.0 e 2.0.
    float torque = 0.0f;

    // 3. Applichiamo la formula ESATTA del documento:
    // r = -(theta^2 + 0.1 * theta_dt^2 + 0.001 * torque^2)
    float reward = -( (theta * theta) + 0.1f * (theta_dt * theta_dt) + 0.001f * (torque * torque) );

    return reward;
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

