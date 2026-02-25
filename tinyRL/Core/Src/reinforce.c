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
    n_steps++; //to prevent limit errors

    /* ---------- malloc principali ----------------------------- */
    buf->state_buffer = malloc(n_steps * sizeof(float*));
    buf->action_buffer = malloc(n_steps * sizeof(uint32_t));
    buf->log_prob_old_buffer = malloc(n_steps * sizeof(float));
    buf->advantage_buffer = malloc(n_steps * sizeof(float));
    buf->critic_buffer = malloc(n_steps * sizeof(float));
    buf->done_buffer = calloc((n_steps + 1), sizeof(uint8_t));
	buf->terminal_value_buffer = calloc((n_steps + 1), sizeof(float));

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


void store_step(Buffer *buf, float *state, uint32_t choosen_action, float reward, float log_prob, float output_critic,
		uint32_t step_count, uint32_t obs_dim, uint8_t done){
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

    // Variabili statiche per mantenere lo stato temporale tra le chiamate
    // in modo indipendente dal riempimento del buffer.
    static uint8_t new_episode = 1;
    static uint32_t ep_step = 0;

	// 1. Assegna il reward all'azione PRECEDENTE
    // Protezione: assegniamo il reward solo se non stiamo iniziando un nuovo episodio
	if(!new_episode && *step_count != 0) {
		*reward = evaluate_reward(obs);
		buffer->advantage_buffer[*step_count - 1] = *reward;
	}

	// 2. Controllo Terminazione per lo step ATTUALE
    // Usiamo ep_step per misurare la lunghezza dell'episodio, non il buffer
	*done = done_check(obs, ep_step);

	if(*done > 0) {
		// Assegniamo la morte all'azione PRECEDENTE (che l'ha causata)
		buffer->done_buffer[*step_count - 1] = *done;

		if(*done == 1) { // Truncated (Timeout) -> Serve il bootstrapping
			uint32_t out_dim_actor = net->actor.layers[net->actor.num_layers - 1].out_dim;
			uint32_t out_dim_critic = net->critic.layers[net->critic.num_layers - 1].out_dim;
			float output_actor[out_dim_actor];
			float output_critic[out_dim_critic];
			forward(net, obs, output_actor, output_critic);

            // Salviamo il valore terminale nel buffer dedicato
			buffer->terminal_value_buffer[*step_count - 1] = output_critic[0];
		} else { // Terminated (Morto) -> Nessun valore futuro
			buffer->terminal_value_buffer[*step_count - 1] = 0.0f;
		}

        // Prepariamoci per la prossima chiamata (il reset dell'ambiente)
        new_episode = 1;
        ep_step = 0;

		// RITORNIAMO SENZA INCREMENTARE STEP_COUNT.
		// L'indice attuale (*step_count) verrà sovrascritto dallo step 0
		// del NUOVO episodio, evitando buchi temporali nel buffer.
		return 1;
	}

	// 3. Se il gioco continua, azzeriamo i flag
    new_episode = 0;
	buffer->done_buffer[*step_count] = 0;
	buffer->terminal_value_buffer[*step_count] = 0.0f;

	// 4. Forward e Selezione Azione
	uint32_t out_dim_actor = net->actor.layers[net->actor.num_layers - 1].out_dim;
	uint32_t out_dim_critic = net->critic.layers[net->critic.num_layers - 1].out_dim;
	float output_actor[out_dim_actor];
	float output_critic[out_dim_critic];

	if(!forward(net, obs, output_actor, output_critic)) return 0;

    uint32_t a = sample_action(output_actor, out_dim_actor);
	*action = a;

	float action_prob = output_actor[a];
	float log_prob = logf(action_prob + 1e-8f);

	// 5. Salva i dati correnti (il reward lo mettiamo al ciclo dopo)
	memcpy(buffer->state_buffer[*step_count], obs, net->layers[0].in_dim * sizeof(float));
	buffer->action_buffer[*step_count] = *action;
	buffer->log_prob_old_buffer[*step_count] = log_prob;
	buffer->critic_buffer[*step_count] = output_critic[0];

    // 6. Incrementa i contatori
	*step_count = *step_count + 1; // Avanza nel buffer globale
    ep_step++;                     // Avanza nell'episodio corrente

	return 1;
}

void evaluate_return(Buffer *buf, uint32_t step_count){
    float *adv_buf = buf->advantage_buffer;
    float G = 0.0f;

    for (int t = step_count - 1; t >= 0; t--) {
        if (buf->done_buffer[t] == 2) {
            G = 0.0f; // Morto
        } else if (buf->done_buffer[t] == 1) {
            G = buf->terminal_value_buffer[t]; // Troncato, usa il bootstrap salvato
        }

        float r = adv_buf[t];
        G = r + GAMMA * G;
        adv_buf[t] = G;
    }
}

void evaluate_advantages(Buffer *buf, uint32_t step_count){
    float sum_adv = 0.0f;
    float sum_adv_sq = 0.0f;

    // 1. Calcola l'Advantage grezzo per TUTTO il buffer
    for(int t = 0; t < step_count; t++){
        float ret = buf->advantage_buffer[t]; // Contiene Return (R)
        float v_old = buf->critic_buffer[t];  // Contiene Valore Vecchio (V_old)

        float adv_raw = ret - v_old;          // A = R - V

        // Salviamo temporaneamente l'adv grezzo nel critic_buffer per non perderlo
        buf->critic_buffer[t] = adv_raw;

        sum_adv += adv_raw;
        sum_adv_sq += adv_raw * adv_raw;
    }

    // 2. Calcola Media e Varianza GLOBALI su tutto il batch
    float mean = sum_adv / (float)step_count;
    float variance = (sum_adv_sq / (float)step_count) - (mean * mean);

    // Evita radici di numeri negativi a causa di imprecisioni del float
    float std = sqrtf(variance > 0.0f ? variance : 0.0f) + 1e-8f;

    // 3. Normalizza e Salva
    for(int t = 0; t < step_count; t++){
        float adv_raw = buf->critic_buffer[t];

        // Sostituiamo il valore grezzo con quello normalizzato
        buf->critic_buffer[t] = (adv_raw - mean) / std;
    }
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

void backward_core_head(Head *net, float *dout_last, float *input, float *accumulate_out){
    // backprop of the gradient
    float *delta = dout_last;

    // Buffer dinamico per i calcoli INTERNI (tra i layer della head)
    static float *delta_buf = NULL;
    static uint32_t delta_cap = 0;

    float *prev = NULL;

    for(int l = net->num_layers - 1; l >= 0; --l){
    	DenseLayer *ly = &net->layers[l];

        // Se l=0 l'input è quello passato alla funzione, altrimenti è l'out del layer precedente
    	float *inp = (l == 0) ? input : net->layers[l-1].out; // BUCO ADT: ok logica puntatori

        // 1. Accumulate grad (Aggiornamento Pesi e Bias del layer corrente)
        // Questo passaggio è identico per tutti i layer
        for (int i = 0; i < ly->out_dim; ++i) {
            ly->db[i] += delta[i];
            for (int j = 0; j < ly->in_dim; ++j)
                ly->dW[i][j] += delta[i] * inp[j];
        }

        // 2. Prepare next delta (Propagazione all'indietro)
        // Dobbiamo decidere DOVE scrivere il risultato e SE applicare la derivata dell'attivazione

        float *target_buf = NULL;

        if (l == 0) {
            // SIAMO ALL'USCITA: Scriviamo nel buffer di output per i linked layers
            target_buf = accumulate_out;
        } else {
            // SIAMO DENTRO LA HEAD: Usiamo il buffer temporaneo
            // memory optimization (tua logica originale)
            if (ly->in_dim > delta_cap){
                free(delta_buf);
                delta_buf  = malloc(ly->in_dim * sizeof(float));
                delta_cap  = ly->in_dim;
                if (!delta_buf) return; // Gestione errore
            }
            target_buf = delta_buf;
        }

        // Se abbiamo un buffer valido dove scrivere (accumulate_out non deve essere NULL)
        if (target_buf != NULL) {
            prev = target_buf;

            for(int j = 0; j < ly->in_dim; ++j){
                float acc = 0.0f;

                // A. Calcolo parte lineare (Moltiplicazione matriciale Delta * W trasposta)
                for(int i = 0; i < ly->out_dim; ++i){
                    acc += delta[i] * ly->W[i][j];
                }

                // B. Gestione Derivata Attivazione
                if (l > 0) {
                    // Caso INTERNO: Dobbiamo derivare l'attivazione del layer precedente (l-1)
                    // che appartiene ancora a questa Head.
                    float h_prev = net->layers[l-1].out[j];
                    switch(net->layers[l-1].activation){
                        case ACT_RELU: acc = (h_prev > 0.f) ? acc : 0.f; break;
                        default: break; // TODO ACT_NONE etc...
                    }
                }
                else {
                    // Caso USCITA (l == 0): Stiamo uscendo verso lo Shared/Linked.
                    // NON applichiamo la derivata dell'attivazione qui.
                    // Passiamo il gradiente "lineare" puro. La derivata dell'attivazione
                    // che ha generato 'input' verrà fatta nella funzione dello shared.
                }

                prev[j] = acc;
            }
            // Aggiorniamo delta per il prossimo giro (solo se non siamo alla fine)
            delta = prev;
        }
    }
}

void backward_shared_backbone(SharedBackbone *net, float *grad_from_actor, float *grad_from_critic, float *sensor_input) {

    // --- MAPPING DEI LAYER ---
    // Assumiamo la struttura che abbiamo concordato:
    // layers[0]: Shared Trunk (Output usato da Link Actor e Link Critic)
    // layers[1]: Link Actor (Input: Trunk Out -> Output: Actor Head In)
    // layers[2]: Link Critic (Input: Trunk Out -> Output: Critic Head In)

    // Attenzione: verifica che net->num_layers sia gestito coerentemente.
    // Qui uso indici fissi per chiarezza, ma potresti volerlo parametrizzare.
    DenseLayer *trunk       = &net->layers[0];
    DenseLayer *link_actor  = &net->layers[1];
    DenseLayer *link_critic = &net->layers[2];

    // Buffer per l'accumulo che scende verso il tronco (dL/da del tronco)
    // Lo inizializziamo a 0 perché ci sommeremo dentro due contributi.
    // WARNING: Verifica che lo stack supporti questa dimensione (VLA)
    float trunk_grad_accum[trunk->out_dim];
    memset(trunk_grad_accum, 0, trunk->out_dim * sizeof(float));

    // ============================================================
    // FASE 1: BACKPROP SU LINK ACTOR (Layer 1)
    // ============================================================
    {
        float *delta = grad_from_actor; // Questo è dL/da (gradiente rispetto all'output)
        float *input = trunk->out;      // Input ricevuto nella forward pass

        for (int i = 0; i < link_actor->out_dim; i++) {
            // A. Derivata dell'attivazione (dL/da -> dL/dz)
            // Trasformiamo il gradiente "grezzo" in "delta" locale
            float out_val = link_actor->out[i]; // TODO: Verificare se l'attivazione serve pre o post

            switch (link_actor->activation) {
                case ACT_RELU:
                    delta[i] = (out_val > 0.0f) ? delta[i] : 0.0f;
                    break;
                /*case ACT_TANH: // Esempio
                    delta[i] = delta[i] * (1.0f - out_val * out_val);
                    break;*/
                case ACT_NONE:
                default:
                    // delta rimane invariato
                    break;
            }

            // B. Aggiornamento Pesi (dW) e Bias (db)
            link_actor->db[i] += delta[i];
            for (int j = 0; j < link_actor->in_dim; j++) {
                link_actor->dW[i][j] += delta[i] * input[j];
            }
        }

        // C. Calcolo Accumulo verso il Tronco (W^T * delta)
        for (int j = 0; j < link_actor->in_dim; j++) {
            float acc = 0.0f;
            for (int i = 0; i < link_actor->out_dim; i++) {
                acc += delta[i] * link_actor->W[i][j];
            }
            // Scriviamo nel buffer comune (PRIMA SCRITTURA o +=, qui è 0 all'inizio quindi += va bene)
            trunk_grad_accum[j] += acc;
        }
    }

    // ============================================================
    // FASE 2: BACKPROP SU LINK CRITIC (Layer 2)
    // ============================================================
    {
        float *delta = grad_from_critic;
        float *input = trunk->out;

        for (int i = 0; i < link_critic->out_dim; i++) {
            // A. Derivata Attivazione
            float out_val = link_critic->out[i];

            switch (link_critic->activation) {
                case ACT_RELU:
                    delta[i] = (out_val > 0.0f) ? delta[i] : 0.0f;
                    break;
                // ... altri casi ...
                default: break;
            }

            // B. Aggiornamento Pesi
            link_critic->db[i] += delta[i];
            for (int j = 0; j < link_critic->in_dim; j++) {
                link_critic->dW[i][j] += delta[i] * input[j];
            }
        }

        // C. Calcolo Accumulo verso il Tronco (SOMMA)
        for (int j = 0; j < link_critic->in_dim; j++) {
            float acc = 0.0f;
            for (int i = 0; i < link_critic->out_dim; i++) {
                acc += delta[i] * link_critic->W[i][j];
            }
            // SOMMA CRUCIALE: Aggiungiamo il contributo del Critic a quello dell'Actor
            trunk_grad_accum[j] += acc;
        }
    }

    // ============================================================
    // FASE 3: BACKPROP SUL TRONCO CONDIVISO (Loop all'indietro)
    // ============================================================

    // Inizia la discesa dal layer 0 verso l'input dei sensori.
    // Attualmente hai solo il layer 0, ma il ciclo lo rende robusto se ne aggiungi altri.
    // Usiamo 'num_layers' inteso come layer del tronco (nel tuo caso, immagino sia 1).
    // SE invece net->num_layers include anche i link (quindi è 3), il loop deve partire da 0.
    // Assumo qui che tu voglia iterare solo sul tronco (indice 0).

    float *next_layer_delta = trunk_grad_accum; // Questo è il delta che arriva dall'alto

    // NOTA: Se hai più layer nel tronco, gestisci malloc/free per buffer intermedi come fatto in core_head
    // Per ora assumiamo 1 solo layer (l=0), quindi niente buffer dinamici complessi.

    for (int l = 0; l >= 0; l--) { // Loop fatto per 1 solo layer (0)
        DenseLayer *ly = &net->layers[l];

        // Input: Se l=0 è il sensore, altrimenti output layer precedente (l-1)
        float *inp = (l == 0) ? sensor_input : net->layers[l-1].out;

        // 1. Derivata Attivazione Layer Corrente (Trunk)
        // Trasformiamo next_layer_delta (dL/da) in delta (dL/dz)
        for (int i = 0; i < ly->out_dim; i++) {
            float out_val = ly->out[i];
            switch (ly->activation) {
                case ACT_RELU:
                    next_layer_delta[i] = (out_val > 0.0f) ? next_layer_delta[i] : 0.0f;
                    break;
                // ...
                default: break;
            }
        }

        // 2. Aggiornamento Pesi
        for (int i = 0; i < ly->out_dim; i++) {
            ly->db[i] += next_layer_delta[i];
            for (int j = 0; j < ly->in_dim; j++) {
                ly->dW[i][j] += next_layer_delta[i] * inp[j];
            }
        }

        // 3. Accumulo per layer precedenti (SOLO SE l > 0)
        if (l > 0) {
            // Qui dovresti calcolare il gradiente per il layer l-1 e metterlo in un buffer
            // per il prossimo giro del loop.
            // Dato che per ora hai solo l=0, questa parte non viene eseguita e risparmiamo calcoli.
            // Se espandi il tronco, copia la logica di backward_core_head qui.
        }
    }
}

void backward_actor_critic(SharedBackbone *net, float *sensor_input, float *output_actor_new, uint8_t out_dim, float *output_critic_new, float ret_norm,
		uint8_t action_buf, float old_log_prob, float norm_adv, int current_batch_size){

	//-----ACTOR LOSS-----
	float log_prob_new = 0.f;
	float batch_scale = 1.0f / (float)current_batch_size;
	//evaluate log_prob_new of the selected action useful for the PPO ratio
	float action_prob = output_actor_new[action_buf];
	log_prob_new = logf(action_prob + 1e-8f); //Add epsilon 1e-8f to avoid log high values when the argument is near 0
	//PPO ratio
	float ratio = expf((log_prob_new - old_log_prob));
	float surr1 = ratio * norm_adv;
	float surr2 = clip(ratio, 1.0f - 0.2, 1.0f + 0.2);
	surr2 = surr2 * norm_adv;
	//float actor_loss_step = (surr1 < surr2) ? surr1 : surr2;
	//actor_loss_step = -actor_loss_step; //minus because we have to maximize it

	//-----ENTROPY LOSS-----
	float entropy_val = evaluate_entropy(output_actor_new, out_dim);

	//-----ACTOR GRADIENT-----
	float d_logits_actor[out_dim];
	for(int i = 0; i < out_dim; i++){
		float p = output_actor_new[i];
		float grad_ppo = 0.0f;
		float grad_ent = 0.0f;

		if(surr1 <= surr2){ //else the gradient is 0 because of the PPO safety clipping
			if(i == action_buf){
				//For the choosen action: -(1 - p) * A
				grad_ppo = -(1.0f - p) * norm_adv * ratio;
			}
			else{
				//For other actions: p * A
				grad_ppo = p * norm_adv * ratio;
			}
		}
		//-----ENTROPY GRADIENT-----
		// Formula: coeff * p * (log(p) + H_total)
		float log_p = logf(p + 1e-8f);
		grad_ent = ENT_BETA * p * (log_p + entropy_val);

		// --- C. Somma Finale ---
		// Questo è il valore che passerai indietro al layer precedente
		d_logits_actor[i] = (grad_ppo + grad_ent) * batch_scale;
	}

	//-----CRITIC LOSS-----
	//float critic_loss_step = powf((output_critic[0] - buf->advantage_buffer[t]), 2);
	float d_logits_critic[1];
	for(int i = 0; i < 1; i++) d_logits_critic[i] = (CRITIC_COEFF * 2.0f * (output_critic_new[0] - ret_norm)) * batch_scale;

	//POSSO CHIAMARE UN'UNICA FUNZIONE PER ENTRAMBE DOVE DENTRO IN MODO SEPARATO CALCOLO LE DERIVATE E L'ACCUMULO
	//POI SOMMO GLI ACCUMULI FINALI (ACTOR E CRITIC CON SHARED DEVONO AVERE STESSO LAYER INIZIALE) E LI PASSO ALLO SHARED PER LA PARTE FINALE
	float grad_from_actor[net->actor.layers[0].in_dim];
	float grad_from_critic[net->critic.layers[0].in_dim];
	backward_core_head(&net->actor, d_logits_actor, net->layers[net->num_layers].out, grad_from_actor);
	backward_core_head(&net->critic, d_logits_critic, net->layers[net->num_layers+1].out, grad_from_critic);
	backward_shared_backbone(net, grad_from_actor, grad_from_critic, sensor_input);
}



uint32_t finish_episode(Buffer *buf, SharedBackbone *net, uint32_t step_count, uint8_t done){
    if (step_count == 0) return 1;

    // --- FASE 1: PRE-CALCOLO (Una volta sola per episodio/buffer) ---

    // 1. Calcola i Returns (R) e li mette in advantage_buffer
    evaluate_return(buf, step_count);

    // 2. Evaluate Advantages that will be stored into critic buffer to save and reuse memory
    evaluate_advantages(buf, step_count);

    // --- FASE 2: TRAINING LOOP (PPO) ---

    // Buffer temporanei per le uscite (allocati nello stack)
    uint8_t out_dim_actor = net->actor.layers[net->actor.num_layers-1].out_dim;
    float output_actor[out_dim_actor];

    uint8_t out_dim_critic = net->critic.layers[net->critic.num_layers-1].out_dim;
    float output_critic[out_dim_critic];

    // --- NUOVA LOGICA: Impostazioni Mini-Batch ---

    // Alloca l'array degli indici (usiamo un VLA - Variable Length Array)
    int *indices = malloc(step_count * sizeof(int));
	if(indices == NULL) return 0; // Protezione sicurezza

	for (int i = 0; i < step_count; i++) {
		indices[i] = i;
	}

    for(int epoch = 0; epoch < N_EPOCHS; epoch++){

        // SHUFFLE: Mischia gli indici all'inizio di ogni epoca (Fisher-Yates)
        for (int i = step_count - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        // LOOP DEI MINI-BATCH
        for(int start = 0; start < step_count; start += BATCH_SIZE){

            int end = start + BATCH_SIZE;
            if (end > step_count) end = step_count;

            int current_batch_size = end - start;

            // ---> SPOSTATO: Reset dei gradienti per il mini-batch corrente
            zero_grad(net);

            // LOOP SUI SINGOLI CAMPIONI DEL MINI-BATCH
            for(int b = start; b < end; b++){

                // Prendi l'indice randomizzato
                int t = indices[b];

                // Recupera lo stato grezzo dal buffer usando l'indice 't'
                float *state = buf->state_buffer[t];

                // Forward Pass: Calcola probabilità e valori correnti
                forward(net, state, output_actor, output_critic);

                // Recupera i dati pre-calcolati
                float normalized_advantage = buf->critic_buffer[t];     // A_norm (per Actor)
                float return_target = buf->advantage_buffer[t];         // R (per Critic)
                float old_log_prob = buf->log_prob_old_buffer[t];
                uint8_t action = (uint8_t)buf->action_buffer[t];

                // Backward Pass
                backward_actor_critic(
                    net,
                    state,              // Input Sensori
                    output_actor,       // output_actor_new
                    out_dim_actor,      // out_dim
                    output_critic,      // output_critic_new
                    return_target,      // ret_norm (Target del Critic)
                    action,             // action_buf
                    old_log_prob,       // old_log_prob
                    normalized_advantage, // norm_adv (Advantage per Actor)
                    current_batch_size  // <--- NUOVO PARAMETRO PER LA SCALA
                );
            }

            // ---> SPOSTATO: Applica le modifiche ai pesi per QUESTO mini-batch
            adam_optimizer(net);
        }
    }

	free(indices);
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
	if (step >= MAX_STEPS){
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

