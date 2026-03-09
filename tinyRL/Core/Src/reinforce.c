#include "reinforce.h"
#include "utils.h"

// CARTPOLE
#define CART_LIMIT 2.4f        /* ±2.4 m */
#define POLE_LIMIT 0.20943951f /* ±12°   */
#define STEP_LIMIT 500

// ACROBOT
#define HEIGHT_THRESHOLD 1.0f

static inline float frand(void) { return (float)rand() / RAND_MAX; }

int buffer_init(Buffer *buf, uint32_t n_steps, uint32_t obs_dim) {
  /* azzera i campi così, in caso di errore, i free sono sicuri */
  buf->state_buffer = NULL;
  buf->action_buffer = NULL;
  buf->log_prob_old_buffer = NULL;
  buf->advantage_buffer = NULL;
  buf->critic_buffer = NULL;
  buf->sigma_buffer = NULL;
  n_steps++; // to prevent limit errors

  /* ---------- malloc principali ----------------------------- */
  buf->state_buffer = malloc(n_steps * sizeof(float *));
  buf->action_buffer = malloc(n_steps * sizeof(action_t));
  buf->log_prob_old_buffer = malloc(n_steps * sizeof(float));
  buf->advantage_buffer = malloc(n_steps * sizeof(float));
  buf->critic_buffer = malloc(n_steps * sizeof(float));
  buf->done_buffer = calloc((n_steps + 1), sizeof(uint8_t));
  buf->terminal_value_buffer = calloc((n_steps + 1), sizeof(float));
  buf->sigma_buffer = malloc(n_steps * sizeof(float));

  if (!buf->state_buffer || !buf->action_buffer || !buf->log_prob_old_buffer ||
      !buf->advantage_buffer || !buf->critic_buffer || !buf->sigma_buffer)
    goto fail;

  /* righe per la matrice delle osservazioni */
  for (uint32_t i = 0; i < n_steps; ++i) {
    buf->state_buffer[i] = malloc(obs_dim * sizeof(float));
    if (!buf->state_buffer[i])
      goto fail;
  }
  return 1; /* tutto OK */

fail: /* qualsiasi malloc fallita → libera e segnala errore */
  if (buf->state_buffer) {
    for (uint32_t i = 0; i < n_steps; ++i)
      free(buf->state_buffer[i]);
    free(buf->state_buffer);
  }
  free(buf->action_buffer);
  free(buf->log_prob_old_buffer);
  free(buf->advantage_buffer);
  free(buf->critic_buffer);
  free(buf->sigma_buffer);
  return 0;
}

int uart_recv_floats(UART_HandleTypeDef *huart, float *dst, size_t dim,
                     uint32_t timeout) {
  const size_t nbytes = dim * sizeof(float);
  uint8_t byte;
  uint8_t buf[nbytes]; /* dim=4*4 */
  uint32_t t0 = HAL_GetTick();

  /* 1. Cerca lo STX ------------------------------------------------ */
  do {
    if (HAL_UART_Receive(huart, &byte, 1, 1) != HAL_OK) {
      if (HAL_GetTick() - t0 > timeout)
        return 0; /* timeout totale */
      continue;   /* riprova */
    }
  } while (byte != 0x02);

  /* 2. Legge esattamente nbytes + ETX ------------------------------ */
  if (HAL_UART_Receive(huart, buf, nbytes, timeout) != HAL_OK)
    return 0;
  if (HAL_UART_Receive(huart, &byte, 1, timeout) != HAL_OK)
    return 0;
  if (byte != 0x03)
    return 0;

  memcpy(dst, buf, nbytes); /* OK: frame completo */
  return 1;
}

int uart_send_action(UART_HandleTypeDef *huart, action_t action, uint8_t done,
                     uint32_t timeout) {
#if USE_CONTINUOUS_ACTIONS
  uint8_t frame[7];
  frame[0] = 0x02; // STX
  memcpy(&frame[1], &action, sizeof(float));
  frame[5] = done;
  frame[6] = 0x03; // ETX
  return (HAL_UART_Transmit(huart, frame, sizeof(frame), timeout) == HAL_OK);
#else
  uint8_t frame[] = {0x02, action, done, 0x03};
  return (HAL_UART_Transmit(huart, frame, sizeof(frame), timeout) == HAL_OK);
#endif
}

int uart_send_log(UART_HandleTypeDef *huart, uint32_t dt, uint32_t step,
                  uint32_t timeout) {
  uint8_t frame[10];
  frame[0] = 0x01; // STX

  frame[1] = (uint8_t)(dt >> 24); // MSB
  frame[2] = (uint8_t)(dt >> 16);
  frame[3] = (uint8_t)(dt >> 8);
  frame[4] = (uint8_t)(dt >> 0); // LSB

  frame[5] = (uint8_t)(step >> 24);
  frame[6] = (uint8_t)(step >> 16);
  frame[7] = (uint8_t)(step >> 8);
  frame[8] = (uint8_t)(step >> 0);

  frame[9] = 0x04; // ETX
  return (HAL_UART_Transmit(huart, frame, sizeof(frame), timeout) == HAL_OK);
}

uint32_t sample_action(float *p, uint32_t dim) {
  const float EPSILON = 0.0f; /* 1 % */

  /* ─── 1. esplorazione pura ogni tanto ─── */
  float r = (float)rand() / (float)RAND_MAX; /* uniform [0,1) */
  if (r < EPSILON)
    return (uint8_t)(rand() % dim); /* azione random */

  /* ─── 2. campionamento “roulette-wheel” ─── */
  float c = (float)rand() / (float)RAND_MAX; /* [0,1) */
  for (uint32_t i = 0; i < dim; ++i) {
    if (c < p[i])
      return (uint8_t)i;
    c -= p[i];
  }
  return (uint8_t)(dim - 1); /* fallback numerico */
}

#if USE_CONTINUOUS_ACTIONS
float sample_continuous_action(float mu, float sigma) {
  float u1 = fmaxf((float)rand() / RAND_MAX, 1e-7f);
  float u2 = (float)rand() / RAND_MAX;
  float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
  float action = mu + sigma * z0;

  return action;
}

float gaussian_log_prob(float action, float mu, float sigma) {
  float var = sigma * sigma;
  // Formula: -0.5 * log(2 * PI * var) - (x - mu)^2 / (2 * var)
  return -0.5f * logf(2.0f * (float)M_PI * var) -
         ((action - mu) * (action - mu)) / (2.0f * var);
}
#endif

int step(SharedBackbone *net, float *obs, float manual_reward,
         uint8_t manual_done, action_t *out_action, uint32_t *step_count,
         Buffer *buffer) {

  static uint8_t new_episode = 1;
  static uint32_t ep_step = 0;

  // =========================================================================
  // FASE 1: CHIUSURA DELLO STEP PRECEDENTE (t-1)
  // Qui possediamo l'osservazione futura s_t (risultante da a_t-1)
  // =========================================================================
  if (!new_episode && *step_count > 0) {
    uint32_t prev_t = *step_count - 1;

    // 1. IL MAGIC TRICK DEL TIMING (Data-Driven Step):
    // La libreria NON calcola più il reward o il done. Prende 'manual_reward'
    // e 'manual_done' forniti ciecamente dall'esterno (es. main.c) al tempo 't'
    // e li assegna come CAUSA di ciò che l'Actor ha deciso al tempo 't-1'.
    buffer->advantage_buffer[prev_t] = manual_reward;
    buffer->done_buffer[prev_t] = manual_done;

    if (manual_done > 0) {

      // BOOTSTRAP: Se Troncato (timeout), serve il V(s_t) futuro
      if (manual_done == 1) {
        uint32_t out_dim_actor =
            net->actor.layers[net->actor.num_layers - 1].out_dim;
        uint32_t out_dim_critic =
            net->critic.layers[net->critic.num_layers - 1].out_dim;
        float output_actor[out_dim_actor];
        float output_critic[out_dim_critic];

        forward(net, obs, output_actor, output_critic);
        buffer->terminal_value_buffer[prev_t] = output_critic[0];
      } else {
        // Terminated: valore futuro è intrinsecamente 0
        buffer->terminal_value_buffer[prev_t] = 0.0f;
      }

      // Prepara le variabili fisiche e logiche per il reset sim-to-real
      new_episode = 1;
      ep_step = 0;

      // Ritorniamo *senza* incrementare step_count.
      // Il prossimo passo ricomincerà scrivendo allo stesso indice pulito.
      return 1;
    }
  }

  // =========================================================================
  // FASE 2: GESTIONE RIEMPIMENTO BUFFER
  // =========================================================================
  if (*step_count >= MAX_STEPS) {
    // Abbiamo raccolto tutto il possibile, informiamo il main in modo da
    // avviare l'aggiornamento e resettare l'ambiente all'episodio successivo
    return 2; // Trigger training code
  }

  // =========================================================================
  // FASE 3: ELABORAZIONE DELLO STEP CORRENTE (t)
  // =========================================================================
  new_episode = 0; // Se eravamo in un nuovo episodio, ora non lo siamo più

  // Inizializza i flag futuri per sicurezza e ripulisce il ritorno per il main
  // Non modifichiamo output return done, lo gestisce il main
  buffer->done_buffer[*step_count] = 0;
  buffer->terminal_value_buffer[*step_count] = 0.0f;

  uint32_t out_dim_actor = net->actor.layers[net->actor.num_layers - 1].out_dim;
  uint32_t out_dim_critic =
      net->critic.layers[net->critic.num_layers - 1].out_dim;
  float output_actor[out_dim_actor];
  float output_critic[out_dim_critic];

  if (!forward(net, obs, output_actor, output_critic))
    return 0;

  action_t a;
  float log_prob;

#if USE_CONTINUOUS_ACTIONS
  float progress = (float)net->adam_t / (float)(TOTAL_ADAM_STEPS + 1);
  if (progress > 1.0f)
    progress = 1.0f;

  float current_sigma = STARTING_ACTION_SIGMA * (1.0f - progress) + 0.15f;
  if (current_sigma < 0.15f)
    current_sigma = 0.15f;

  float mu = output_actor[0];
  float a_raw = sample_continuous_action(mu, current_sigma);

  // SOFT GAUSSIAN CLIPPING (Fix for Gradient Explosion & Bias):
  // Limit a_raw mathematically to mu +/- 3*sigma.
  // This preserves the valid Gaussian PDF shape for 99.7% of samples,
  // while actively blocking the 0.3% of extreme numerical outliers (a_raw
  // = 4.0) that cause massive destructive gradient explosions ((a_raw -
  // mu)/sigma^2).
  float bound = 3.0f * current_sigma;
  if (a_raw > mu + bound)
    a_raw = mu + bound;
  if (a_raw < mu - bound)
    a_raw = mu - bound;

  // L'azione fisica mandata ai motori è SEMPRE rigidamente tagliata a [-1, 1]
  a = fmaxf(fminf(a_raw, 1.0f), -1.0f);
  *out_action = a;

  // IMPORTANTE: Calcoliamo la log_prob e riempiamo il buffer usando l'azione
  // a_raw "Soft Clipped". Questo garantisce la validità matematica della PDF
  // evitando che la policy venga "assorbita" e bloccata in modo irreversibile
  // ai bordi +/- 2.0 (Action Saturation).
  log_prob = gaussian_log_prob(a_raw, mu, current_sigma);

  buffer->action_buffer[*step_count] = a_raw;
  buffer->sigma_buffer[*step_count] = current_sigma;
#else
  a = (action_t)sample_action(output_actor, out_dim_actor);
  float action_prob = output_actor[(uint8_t)a];
  log_prob = logf(action_prob + 1e-8f);
  *out_action = a;

  buffer->action_buffer[*step_count] = a;
#endif

  // Salva i dati correnti nel buffer allo slot `t`
  memcpy(buffer->state_buffer[*step_count], obs,
         net->layers[0].in_dim * sizeof(float));
  buffer->log_prob_old_buffer[*step_count] = log_prob;
  buffer->critic_buffer[*step_count] = output_critic[0];

  // Incrementa contatori per prepararsi al passo futuro `t+1`
  *step_count = *step_count + 1;
  ep_step++;

  return 1;
}

void evaluate_advantages_and_returns(Buffer *buf, uint32_t step_count) {
  float sum_adv = 0.0f;
  float sum_adv_sq = 0.0f;
  float gae = 0.0f;
  float next_v_curr = 0.0f;

  // GAE Lambda parameter (typically 0.95 for PPO)
  const float lambda = 0.95f;

  // 1. Calculate GAE and Returns for the whole buffer
  for (int t = step_count - 1; t >= 0; t--) {
    float r = buf->advantage_buffer[t];   // Contains immediate Reward (R)
    float v_curr = buf->critic_buffer[t]; // Current Value V(s_t)

    // Next value V(s_{t+1})
    float v_next = 0.0f;
    if (t == step_count - 1) {
      v_next = buf->terminal_value_buffer[t];
    } else {
      // If the next step is a new episode or terminated, v_next is 0
      if (buf->done_buffer[t] == 2)
        v_next = 0.0f;
      else if (buf->done_buffer[t] == 1)
        v_next = buf->terminal_value_buffer[t];
      else
        v_next = next_v_curr;
    }

    // TD Error: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    float delta = r + GAMMA * v_next - v_curr;

    // Reset GAE if episode ends at this step
    if (buf->done_buffer[t] > 0) {
      gae =
          0.0f; // No future advantages flow backwards across episode boundaries
    }

    // GAE: A_t = delta_t + gamma * lambda * A_{t+1}
    gae = delta + GAMMA * lambda * gae;

    // Store Return into advantage_buffer (Return = GAE + V_curr)
    buf->advantage_buffer[t] = gae + v_curr;

    // Temporarily store raw GAE in critic_buffer
    buf->critic_buffer[t] = gae;

    // Save current value for the next iteration (t-1 evaluates t as t+1)
    next_v_curr = v_curr;

    sum_adv += gae;
    sum_adv_sq += gae * gae;
  }

  // 2. Calculate Global Mean and Variance
  float mean = sum_adv / (float)step_count;
  float variance = (sum_adv_sq / (float)step_count) - (mean * mean);

  // Prevent sqrt of negative number
  float std = sqrtf(variance > 0.0f ? variance : 0.0f) + 1e-8f;

  // 3. Normalize and Save
  for (int t = 0; t < step_count; t++) {
    float adv_raw = buf->critic_buffer[t];
    // Replace raw with normalized advantage
    buf->critic_buffer[t] = (adv_raw - mean) / std;
  }
}

void evaluate_mean_std(Buffer *buf, uint32_t step_count, float *m, float *s) {
  if (step_count > 2) {
    float *adv_buf = buf->advantage_buffer;
    // adv normalization
    float mean = 0.0f;
    for (int t = 0; t < step_count; t++)
      mean += adv_buf[t];
    mean /= step_count;
    *m = mean;
    // for (int t = 0; t < step_count; t++) adv_buf[t] -= mean;

    float var = 0.0f;
    for (int t = 0; t < step_count; t++)
      var += adv_buf[t] * adv_buf[t];
    float std = sqrtf(var / step_count) + 1e-6f;
    *s = std;
    // for (int t = 0; t < step_count; t++) adv_buf[t] /= std;
  }
}

float evaluate_entropy(float *probs, int n_actions) {
  float entropy = 0.0f;
  for (int i = 0; i < n_actions; i++) {
    if (probs[i] > 1e-8f) { // Avoid log(0) that is -inf
      entropy -= probs[i] * logf(probs[i]);
    }
  }
  return entropy;
}

void backward_core_head(Head *net, float *dout_last, float *input,
                        float *accumulate_out) {
  // backprop of the gradient
  float *delta = dout_last;

  // Buffer dinamico per i calcoli INTERNI (tra i layer della head)
  static float *delta_buf = NULL;
  static uint32_t delta_cap = 0;

  float *prev = NULL;

  for (int l = net->num_layers - 1; l >= 0; --l) {
    DenseLayer *ly = &net->layers[l];

    // Se l=0 l'input è quello passato alla funzione, altrimenti è l'out del
    // layer precedente
    float *inp = (l == 0)
                     ? input
                     : net->layers[l - 1].out; // BUCO ADT: ok logica puntatori

    const int out_dim = ly->out_dim;
    const int in_dim = ly->in_dim;
    const float *restrict inp_vec = inp;
    const float *restrict d = delta;

    // 1. Accumulate grad (Aggiornamento Pesi e Bias del layer corrente)
    float *restrict db = ly->db;
    for (int i = 0; i < out_dim; ++i) {
      db[i] += d[i];
      float *restrict dw_row = ly->dW[i];
      float di = d[i];
      for (int j = 0; j < in_dim; ++j) {
        dw_row[j] += di * inp_vec[j];
      }
    }

    // 2. Prepare next delta (Propagazione all'indietro)
    // Dobbiamo decidere DOVE scrivere il risultato e SE applicare la derivata
    // dell'attivazione

    float *target_buf = NULL;

    if (l == 0) {
      // SIAMO ALL'USCITA: Scriviamo nel buffer di output per i linked layers
      target_buf = accumulate_out;
    } else {
      // SIAMO DENTRO LA HEAD: Usiamo il buffer temporaneo
      // memory optimization (tua logica originale)
      if (ly->in_dim > delta_cap) {
        free(delta_buf);
        delta_buf = malloc(ly->in_dim * sizeof(float));
        delta_cap = ly->in_dim;
        if (!delta_buf)
          return; // Gestione errore
      }
      target_buf = delta_buf;
    }

    // Se abbiamo un buffer valido dove scrivere (accumulate_out non deve essere
    // NULL)
    if (target_buf != NULL) {
      prev = target_buf;

      float *restrict p_buf = prev;
      for (int j = 0; j < in_dim; ++j) {
        float acc = 0.0f;

        // A. Calcolo parte lineare (Moltiplicazione matriciale Delta * W
        // trasposta)
        for (int i = 0; i < out_dim; ++i) {
          acc += d[i] * ly->W[i][j];
        }

        // B. Gestione Derivata Attivazione
        if (l > 0) {
          // Caso INTERNO: Dobbiamo derivare l'attivazione del layer precedente
          // (l-1) che appartiene ancora a questa Head.
          float h_prev = net->layers[l - 1].out[j];
          switch (net->layers[l - 1].activation) {
          case ACT_RELU:
            acc = (h_prev > 0.f) ? acc : 0.f;
            break;
          case ACT_TANH:
            acc = acc * (1.f - h_prev * h_prev);
            break;
          default:
            break; // TODO ACT_NONE etc...
          }
        } else {
          // Caso USCITA (l == 0): Stiamo uscendo verso lo Shared/Linked.
          // NON applichiamo la derivata dell'attivazione qui.
          // Passiamo il gradiente "lineare" puro. La derivata dell'attivazione
          // che ha generato 'input' verrà fatta nella funzione dello shared.
        }

        p_buf[j] = acc;
      }
      // Aggiorniamo delta per il prossimo giro (solo se non siamo alla fine)
      delta = prev;
    }
  }
}

void backward_shared_backbone(SharedBackbone *net, float *grad_from_actor,
                              float *grad_from_critic, float *sensor_input) {

  // --- MAPPING DEI LAYER ---
  // Assumiamo la struttura che abbiamo concordato:
  // layers[0]: Shared Trunk (Output usato da Link Actor e Link Critic)
  // layers[1]: Link Actor (Input: Trunk Out -> Output: Actor Head In)
  // layers[2]: Link Critic (Input: Trunk Out -> Output: Critic Head In)

  DenseLayer *trunk =
      &net->layers[net->num_layers - 1]; // L'ultimo layer del Trunk
  DenseLayer *link_actor =
      &net->layers[net->num_layers]; // Subito dopo il Trunk
  DenseLayer *link_critic =
      &net->layers[net->num_layers + 1]; // Subito dopo il link Actor

  float trunk_grad_accum[trunk->out_dim];
  memset(trunk_grad_accum, 0, trunk->out_dim * sizeof(float));

  // ============================================================
  // FASE 1: BACKPROP SU LINK ACTOR (Layer 1)
  // ============================================================
  {
    float *delta =
        grad_from_actor;       // Questo è dL/da (gradiente rispetto all'output)
    float *input = trunk->out; // Input ricevuto nella forward pass

    for (int i = 0; i < link_actor->out_dim; i++) {
      // A. Derivata dell'attivazione (dL/da -> dL/dz)
      // Trasformiamo il gradiente "grezzo" in "delta" locale
      float out_val =
          link_actor
              ->out[i]; // TODO: Verificare se l'attivazione serve pre o post

      switch (link_actor->activation) {
      case ACT_RELU:
        delta[i] = (out_val > 0.0f) ? delta[i] : 0.0f;
        break;
      case ACT_TANH:
        delta[i] = delta[i] * (1.0f - out_val * out_val);
        break;
      case ACT_NONE:
      default:
        // delta rimane invariato
        break;
      }

      // B. Aggiornamento Pesi (dW) e Bias (db)
      link_actor->db[i] += delta[i];
      float *restrict dw_row = link_actor->dW[i];
      float di = delta[i];
      for (int j = 0; j < link_actor->in_dim; j++) {
        dw_row[j] += di * input[j];
      }
    }

    // C. Calcolo Accumulo verso il Tronco (W^T * delta)
    const int act_in_dim = link_actor->in_dim;
    const int act_out_dim = link_actor->out_dim;
    for (int j = 0; j < act_in_dim; j++) {
      float acc = 0.0f;
      for (int i = 0; i < act_out_dim; i++) {
        acc += delta[i] * link_actor->W[i][j];
      }
      // Scriviamo nel buffer comune (PRIMA SCRITTURA o +=, qui è 0 all'inizio
      // quindi += va bene)
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
      case ACT_TANH:
        delta[i] = delta[i] * (1.0f - out_val * out_val);
        break;
      // ... altri casi ...
      default:
        break;
      }

      // B. Aggiornamento Pesi
      link_critic->db[i] += delta[i];
      float *restrict dw_row = link_critic->dW[i];
      float di = delta[i];
      for (int j = 0; j < link_critic->in_dim; j++) {
        dw_row[j] += di * input[j];
      }
    }

    // C. Calcolo Accumulo verso il Tronco (SOMMA)
    const int crit_in_dim = link_critic->in_dim;
    const int crit_out_dim = link_critic->out_dim;
    for (int j = 0; j < crit_in_dim; j++) {
      float acc = 0.0f;
      for (int i = 0; i < crit_out_dim; i++) {
        acc += delta[i] * link_critic->W[i][j];
      }
      // SOMMA CRUCIALE: Aggiungiamo il contributo del Critic a quello
      // dell'Actor
      trunk_grad_accum[j] += acc;
    }
  }

  // ============================================================
  // FASE 3: BACKPROP SUL TRONCO CONDIVISO (Loop all'indietro)
  // ============================================================

  // Inizia la discesa dal layer 0 verso l'input dei sensori.
  // Usiamo 'num_layers' inteso come layer del tronco.

  float *next_layer_delta =
      trunk_grad_accum;    // Questo è il delta che arriva dall'alto
  float *delta_buf = NULL; // Buffer dinamico temporaneo per prop.
  uint32_t delta_cap = 0;

  for (int l = net->num_layers - 1; l >= 0; l--) {
    DenseLayer *ly = &net->layers[l];

    // Input: Se l=0 è il sensore, altrimenti output layer precedente (l-1)
    float *inp = (l == 0) ? sensor_input : net->layers[l - 1].out;

    // 1. Derivata Attivazione Layer Corrente (Trunk)
    // Trasformiamo next_layer_delta (dL/da) in delta (dL/dz)
    for (int i = 0; i < ly->out_dim; i++) {
      float out_val = ly->out[i];
      switch (ly->activation) {
      case ACT_RELU:
        next_layer_delta[i] = (out_val > 0.0f) ? next_layer_delta[i] : 0.0f;
        break;
      case ACT_TANH:
        next_layer_delta[i] = next_layer_delta[i] * (1.0f - out_val * out_val);
        break;
      // ...
      default:
        break;
      }
    }

    // 2. Aggiornamento Pesi e Bias
    for (int i = 0; i < ly->out_dim; i++) {
      ly->db[i] += next_layer_delta[i];
      for (int j = 0; j < ly->in_dim; j++) {
        ly->dW[i][j] += next_layer_delta[i] * inp[j];
      }
    }

    // 3. Accumulo per layer precedenti
    if (l > 0) {
      // Allocate memory optimization
      if (ly->in_dim > delta_cap) {
        free(delta_buf);
        delta_buf = malloc(ly->in_dim * sizeof(float));
        delta_cap = ly->in_dim;
        if (!delta_buf)
          return; // Error handling
      }

      float *prev = delta_buf;

      for (int j = 0; j < ly->in_dim; ++j) {
        float acc = 0.0f;
        // Accumulo gradiente (Delta * W trasposta)
        for (int i = 0; i < ly->out_dim; ++i) {
          acc += next_layer_delta[i] * ly->W[i][j];
        }
        prev[j] = acc;
      }

      // Il delta calcolato ora diventerà il next_layer_delta per l-1
      // Dobbiamo re-indirizzare o copiare (copia per sicurezza dei puntatori)
      // Selezioniamo il delta_buf come puntatore per il prossimo giro
      next_layer_delta = prev;
    }
  }

  if (delta_buf)
    free(delta_buf);
}

void backward_actor_critic(SharedBackbone *net, float *sensor_input,
                           float *output_actor_new, uint8_t out_dim,
                           float *output_critic_new, float ret_norm,
                           action_t action_buf, float old_log_prob,
                           float norm_adv, int current_batch_size,
                           float old_sigma) {

  float batch_scale = 1.0f / (float)current_batch_size;
  float log_prob_new;
  float d_logits_actor[out_dim];

#if USE_CONTINUOUS_ACTIONS
  // --- CASO CONTINUO ---
  // Usa old_sigma: il sigma che era attivo al momento della RACCOLTA del
  // rollout, calcolato una volta sola in finish_episode() prima di qualsiasi
  // adam_t++. Questo garantisce che log_prob_old e log_prob_new usino la stessa
  // distribuzione di riferimento, rendendo il ratio PPO matematicamente valido.
  float mu = output_actor_new[0];
  log_prob_new = gaussian_log_prob(action_buf, mu, old_sigma);

  // Clamp log-ratio BEFORE expf() to prevent inf/nan when policies diverge.
  // Without this, expf(very_large_number) = INF -> grad_ppo = INF -> mu
  // saturates to ±1.
  // clamp to [-4, 4]
  float log_ratio = log_prob_new - old_log_prob;
  log_ratio = clip(log_ratio, -4.0f,
                   4.0f); // exp(±4) keeps ratio in safe range [0.018, 54]
  float ratio = expf(log_ratio);

  // Calcolo del PPO Clip Surrogate Gradient Corretto
  // Se la policy si sposta troppo (fuori dal trust region 1-eps .. 1+eps)
  // il gradiente DEVE essere matematicamente 0.0f, bloccando l'ottimizzatore
  // Adam.
  float grad_ppo = 0.0f;
  float d_log_prob = (action_buf - mu) / (old_sigma * old_sigma);

  // STABILITY FIX: Hard limit the analytical gradient of the log_prob.
  // When an outlier action is sampled (e.g. a - mu = 3.0) and sigma is small
  // (e.g. 0.2), d_log_prob blows up to 75.0! Multiplied by an Advantage, this
  // instantly destroys the Actor weights and permanently saturates Tanh to
  // +/- 1.0. Standard PPO frameworks safely clamp this derivative.
  d_log_prob = clip(d_log_prob, -5.0f, 5.0f);

  if (norm_adv > 0.0f) {
    if (ratio < 1.0f + EPS_CLIPPING) {
      grad_ppo = -(ratio * norm_adv * d_log_prob);
    } // altrimenti grad_ppo resta 0.0 (Capped)
  } else {
    if (ratio > 1.0f - EPS_CLIPPING) {
      grad_ppo = -(ratio * norm_adv * d_log_prob);
    } // altrimenti grad_ppo resta 0.0 (Capped)
  }

  // Penalità L2 su mu: con sigma fisso, l'entropia gaussiana H =
  // 0.5*log(2πe*σ²) non dipende da mu (∂H/∂mu = 0). Al suo posto usiamo una
  // penalità L2 che spinge mu verso 0, prevenendo la saturazione del tanh
  // verso ±1. ENT_BETA = 0.0 disabilita questo termine.
  float grad_ent = ENT_BETA * mu;

  float d_mu = (grad_ppo + grad_ent) * batch_scale;

  // FIX CRITICO: Derivata dell'attivazione TANH dell'ultimo layer dell'Actor!
  // La funzione backward_core_head tratta dout_last come dL/dz (derivata
  // pre-attivazione). Quindi dobbiamo moltiplicare qui il nostro dL/dmu per
  // (1 - mu^2). Se non lo facessimo, Tanh non saturerebbe i gradienti e i
  // pesi esploderebbero a -inf o +inf, bloccando l'agente fisso a un'azione
  // estrema (-2.0 o +2.0).
  float tanh_deriv = 1.0f - mu * mu;

  // Floor di sicurezza contro lo Strict Vanishing Gradient:
  // Se mu è già accidentalmente a 1.0/-1.0, tanh_deriv sarebbe 0.0 e la rete
  // non si sbloccherebbe MAI. Assicuriamo una deviazione minima per
  // permettere all'adam di recuperare.
  if (tanh_deriv < 0.05f) {
    tanh_deriv = 0.05f;
  }

  d_logits_actor[0] = d_mu * tanh_deriv;
#else
  // --- CASO DISCRETO ---
  float action_prob = output_actor_new[(uint8_t)action_buf];
  log_prob_new = logf(action_prob + 1e-8f);
  float ratio = expf((log_prob_new - old_log_prob));
  float surr1 = ratio * norm_adv;
  float surr2 = clip(ratio, 1.0f - 0.2f, 1.0f + 0.2f) * norm_adv;

  float entropy_val = evaluate_entropy(output_actor_new, out_dim);

  for (int i = 0; i < out_dim; i++) {
    float p = output_actor_new[i];
    float grad_ppo = 0.0f;
    float grad_ent = 0.0f;

    if (surr1 <= surr2) {
      if (i == (uint8_t)action_buf)
        grad_ppo = -(1.0f - p) * norm_adv * ratio;
      else
        grad_ppo = p * norm_adv * ratio;
    }
    float log_p = logf(p + 1e-8f);
    grad_ent = ENT_BETA * p * (log_p + entropy_val);
    d_logits_actor[i] = (grad_ppo + grad_ent) * batch_scale;
  }
#endif

  //-----CRITIC LOSS E BACKPROP -----
  // Added Value Clipping to avoid Gradient Interference when computing loss
  // on shared backbones V(s) current evaluation
  float v_curr = output_critic_new[0];

  // Temporal Difference error: v_curr - R_target
  // P4 FIX: rimosso il clip ±1 su v_err. Il critic deve imparare return
  // nel range [-16, 0] per il Pendulum; clippare a ±1 rallenta enormemente
  // l'apprendimento della scala. Il global gradient_norm_l2 (clip 0.5)
  // protegge già da gradienti esplosivi a livello di rete intera.
  float v_err = v_curr - ret_norm;

  float d_logits_critic[1];
  d_logits_critic[0] = (CRITIC_COEFF * v_err) * batch_scale;

  float grad_from_actor[net->actor.layers[0].in_dim];
  float grad_from_critic[net->critic.layers[0].in_dim];
  backward_core_head(&net->actor, d_logits_actor,
                     net->layers[net->num_layers].out, grad_from_actor);
  backward_core_head(&net->critic, d_logits_critic,
                     net->layers[net->num_layers + 1].out, grad_from_critic);
  backward_shared_backbone(net, grad_from_actor, grad_from_critic,
                           sensor_input);
}

uint32_t finish_episode(Buffer *buf, SharedBackbone *net, uint32_t step_count,
                        uint8_t done) {
  if (step_count == 0)
    return 1;

  // --- FASE 1: PRE-CALCOLO (Una volta sola per episodio/buffer) ---

  // FIX: Calcola sigma_at_collection PRIMA di qualsiasi adam_t++ per
  // garantire che tutti i mini-batch usino lo stesso sigma con cui è stato
  // raccolto il rollout. Questo rende il ratio PPO = exp(log_new - log_old)
  // matematicamente valido (stessa distribuzione di riferimento per entrambi
  // i termini).
#if USE_CONTINUOUS_ACTIONS
  // sigma_at_collection is now stored per-sample in sigma_buffer during
  // collection in step(). No need to recompute it here from adam_t.
  // This prevents mismatch when other episodes ran training between
  // collection and this call, which would have incremented adam_t and changed
  // the sigma.
  (void)0; // placeholder to keep preprocessor block valid
#else
  (void)0;
#endif

  // Calculate GAE Advantages (saved to critic_buffer) and Returns (saved to
  // advantage_buffer)
  evaluate_advantages_and_returns(buf, step_count);

  // --- FASE 2: TRAINING LOOP (PPO) ---

  // Buffer temporanei per le uscite (allocati nello stack)
  uint8_t out_dim_actor = net->actor.layers[net->actor.num_layers - 1].out_dim;
  float output_actor[out_dim_actor];

  uint8_t out_dim_critic =
      net->critic.layers[net->critic.num_layers - 1].out_dim;
  float output_critic[out_dim_critic];

  // --- NUOVA LOGICA: Impostazioni Mini-Batch ---

  // Inizializza l'array degli indici preallocato nel Buffer
  for (int i = 0; i < step_count; i++) {
    buf->indices[i] = i;
  }

  for (int epoch = 0; epoch < N_EPOCHS; epoch++) {

    // SHUFFLE: Mischia gli indici all'inizio di ogni epoca (Fisher-Yates)
    for (int i = step_count - 1; i > 0; i--) {
      int j = rand() % (i + 1);
      uint16_t temp = buf->indices[i];
      buf->indices[i] = buf->indices[j];
      buf->indices[j] = temp;
    }

    // LOOP DEI MINI-BATCH
    for (int start = 0; start < step_count; start += BATCH_SIZE) {

      int end = start + BATCH_SIZE;
      if (end > step_count)
        end = step_count;

      int current_batch_size = end - start;

      // ---> SPOSTATO: Reset dei gradienti per il mini-batch corrente
      zero_grad(net);

      // LOOP SUI SINGOLI CAMPIONI DEL MINI-BATCH
      for (int b = start; b < end; b++) {

        // Prendi l'indice randomizzato
        int t = buf->indices[b];

        // Recupera lo stato grezzo dal buffer usando l'indice 't'
        float *state = buf->state_buffer[t];

        // Forward Pass: Calcola probabilità e valori correnti
        forward(net, state, output_actor, output_critic);

        // Recupera i dati pre-calcolati
        float normalized_advantage =
            buf->critic_buffer[t];                      // A_norm (per Actor)
        float return_target = buf->advantage_buffer[t]; // R (per Critic)
        float old_log_prob = buf->log_prob_old_buffer[t];
        action_t action = buf->action_buffer[t];
#if USE_CONTINUOUS_ACTIONS
        // Per-sample sigma: il sigma esatto usato per campionare questa
        // azione. È più preciso di sigma_at_collection (che usava adam_t
        // attuale).
        float sample_sigma = buf->sigma_buffer[t];
#else
        float sample_sigma = 0.0f;
#endif

        // Backward Pass
        backward_actor_critic(
            net,
            state,                // Input Sensori
            output_actor,         // output_actor_new
            out_dim_actor,        // out_dim
            output_critic,        // output_critic_new
            return_target,        // ret_norm (Target del Critic)
            action,               // action_buf
            old_log_prob,         // old_log_prob
            normalized_advantage, // norm_adv (Advantage per Actor)
            current_batch_size,   // scala per mini-batch
            sample_sigma          // sigma esatto al momento della raccolta
        );
      }

      // normalize gradient and optimize
      gradient_norm_l2(net);
      adam_optimizer(net);
    }
  }

  return 1;
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
        if (fabsf(state[0]) > CART_LIMIT) return 2;      //out of bound
        if (fabsf(state[2]) > POLE_LIMIT) return 2;      //±12°
        if (step >= MAX_STEPS){
                return 1;   //timeout
        }
        return 0;
}
*/

