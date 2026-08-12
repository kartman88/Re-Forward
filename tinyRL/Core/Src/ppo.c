#include "ppo.h"
#include "rng.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#if TIME_LOG
#include "utils.h"   /* dwt_ticks / dwt_delta per il profiling */

TrainTiming g_train_timing = {0};
#endif

// ─── Rollout Buffer ───────────────────────────────────────────────────────────

int rollout_buffer_init(RolloutBuffer *buf, uint32_t T, uint32_t obs_dim) {
    buf->obs_dim   = obs_dim;
    buf->head      = 0;
    buf->size      = 0;
    buf->capacity  = T;

    buf->states        = malloc(T * obs_dim * sizeof(float));
    buf->log_probs_old = malloc(T * sizeof(float));
    buf->values        = malloc(T * sizeof(float));
    buf->rewards       = malloc(T * sizeof(float));
    buf->advantages    = malloc(T * sizeof(float));
    buf->returns       = malloc(T * sizeof(float));
    buf->dones         = malloc(T * sizeof(uint8_t));
#if USE_CONTINUOUS_ACTION
    buf->actions = malloc(T * N_ACT_DIMS * sizeof(float));
#else
    buf->actions = malloc(T * sizeof(uint32_t));
#endif

    if (!buf->states || !buf->log_probs_old || !buf->values  ||
        !buf->rewards || !buf->advantages   || !buf->returns  ||
        !buf->actions || !buf->dones)
        return 0;

    return 1;
}

#if USE_CONTINUOUS_ACTION
void rollout_buffer_push(RolloutBuffer *buf, float *obs, float *action,
                         float reward, uint8_t done, float log_prob, float value) {
    if (buf->head >= buf->capacity) return;
    uint32_t idx = buf->head;
    memcpy(&buf->states[idx * buf->obs_dim], obs, buf->obs_dim * sizeof(float));
    memcpy(&buf->actions[idx * N_ACT_DIMS], action, N_ACT_DIMS * sizeof(float));
    buf->rewards[idx]       = reward;
    buf->dones[idx]         = done;
    buf->log_probs_old[idx] = log_prob;
    buf->values[idx]        = value;
    buf->head++;
    buf->size++;
}
#else
void rollout_buffer_push(RolloutBuffer *buf, float *obs, uint32_t action,
                         float reward, uint8_t done, float log_prob, float value) {
    if (buf->head >= buf->capacity) return;
    uint32_t idx = buf->head;
    memcpy(&buf->states[idx * buf->obs_dim], obs, buf->obs_dim * sizeof(float));
    buf->actions[idx]       = action;
    buf->rewards[idx]       = reward;
    buf->dones[idx]         = done;
    buf->log_probs_old[idx] = log_prob;
    buf->values[idx]        = value;
    buf->head++;
    buf->size++;
}
#endif

// ─── GAE and Advantage Normalization ─────────────────────────────────────────

void compute_gae(RolloutBuffer *buf, float last_value, float gamma, float lambda) {
    float gae = 0.f;
    for (int t = (int)buf->size - 1; t >= 0; --t) {
        float not_done = buf->dones[t] ? 0.f : 1.f;
        float v_next = (t == (int)buf->size - 1)
                       ? last_value * not_done
                       : buf->values[t + 1] * not_done;
        float delta = buf->rewards[t] + gamma * v_next - buf->values[t];
        gae = delta + gamma * lambda * not_done * gae;
        buf->advantages[t] = gae;
        buf->returns[t]    = gae + buf->values[t];
    }
}

void normalize_advantages(RolloutBuffer *buf) {
    uint32_t n = buf->size;
    float mean = 0.f;
    for (uint32_t i = 0; i < n; i++)
        mean += buf->advantages[i];
    mean /= (float)n;

    float var = 0.f;
    for (uint32_t i = 0; i < n; i++) {
        float d = buf->advantages[i] - mean;
        var += d * d;
    }
    float inv_std = 1.f / (sqrtf(var / (float)n) + 1e-8f);

    for (uint32_t i = 0; i < n; i++)
        buf->advantages[i] = (buf->advantages[i] - mean) * inv_std;
}

// ─── actor_sample_action ─────────────────────────────────────────────────────

#if USE_CONTINUOUS_ACTION

#define LOG_2PI_PPO  1.8378770664093453f

float g_ppo_log_sigma[N_ACT_DIMS];

void ppo_sigma_init(void) {
    float init = logf(PPO_SIGMA_INIT);
    for (int i = 0; i < N_ACT_DIMS; i++)
        g_ppo_log_sigma[i] = init;
}

void ppo_sigma_decay(void) {
    float min_ls = logf(PPO_SIGMA_MIN);
    for (int i = 0; i < N_ACT_DIMS; i++) {
        g_ppo_log_sigma[i] -= PPO_SIGMA_DECAY;
        if (g_ppo_log_sigma[i] < min_ls)
            g_ppo_log_sigma[i] = min_ls;
    }
}

static float randn(void) {
    return rng_normal();
}

void actor_sample_action(Network *actor, float *obs, float *action_out,
                         float *log_prob_out, float *value_out,
                         Network *critic) {
    network_forward(actor, obs, NULL);
    float *mu = actor->layers[actor->num_layers - 1].out;
    for (int i = 0; i < N_ACT_DIMS; i++)
        mu[i] = fmaxf(fminf(mu[i], MU_CLAMP), -MU_CLAMP);

    float log_prob = 0.f;
    for (int i = 0; i < N_ACT_DIMS; i++) {
        float ls  = g_ppo_log_sigma[i];
        float eps = randn();
        float z   = mu[i] + expf(ls) * eps;
#if PPO_USE_TANH_SQUASH
        float a       = tanhf(z);
        action_out[i] = a;
        log_prob += -0.5f * (eps * eps + 2.f * ls + LOG_2PI_PPO)
                    - logf(1.f - a * a + 1e-6f);
#else
        action_out[i] = z;
        log_prob += -0.5f * (eps * eps + 2.f * ls + LOG_2PI_PPO);
#endif
    }
    *log_prob_out = log_prob;

    if (value_out)
        *value_out = critic_forward(critic, obs);
}

#else  /* discrete */

uint32_t actor_sample_action(Network *actor, float *obs,
                              float *log_prob_out, float *value_out,
                              Network *critic) {
    static float probs[PPO_N_ACTIONS];
    network_forward(actor, obs, probs);

    float r = rng_uniform();
    uint32_t action = (uint32_t)(PPO_N_ACTIONS - 1);
    float cumsum = 0.f;
    for (uint32_t i = 0; i < (uint32_t)PPO_N_ACTIONS; i++) {
        cumsum += probs[i];
        if (r < cumsum) { action = i; break; }
    }

    *log_prob_out = logf(probs[action]);

    if (value_out)
        *value_out = critic_forward(critic, obs);

    return action;
}

#endif  /* USE_CONTINUOUS_ACTION */

// ─── ppo_update ──────────────────────────────────────────────────────────────

void ppo_update(Network *actor, Network *critic, RolloutBuffer *buf) {
    uint32_t N = buf->size;

    static uint32_t idx[ROLLOUT_STEPS];
#if !USE_CONTINUOUS_ACTION
    static float probs[PPO_N_ACTIONS];
#endif

#if TIME_LOG
    /* --- profiling: azzera e avvia i contatori DWT ---
     * Il totale NON si misura con un unico delta start/stop: il CYCCNT e' a
     * 32 bit (wrap ogni ~8.9 s a 480 MHz) mentre un update dura di piu'. Lo
     * accumuliamo a pezzi, uno per minibatch (pochi ms l'uno), su un uint64. */
    g_train_timing = (TrainTiming){0};
    uint64_t _fwd_cyc = 0, _bwd_cyc = 0, _adam_cyc = 0, _tot_cyc = 0;
    uint32_t _t_chunk = dwt_ticks();
#endif

    for (uint32_t i = 0; i < N; i++) idx[i] = i;

    network_zero_grad(actor);
    network_zero_grad(critic);

    for (int epoch = 0; epoch < PPO_EPOCHS; epoch++) {

        for (uint32_t i = N - 1; i > 0; i--) {
            uint32_t j   = rng_u32() % (i + 1);
            uint32_t tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        }

        for (uint32_t start = 0; start < N; start += PPO_BATCH_SIZE) {
            uint32_t end     = start + PPO_BATCH_SIZE;
            if (end > N) end = N;
            float    inv_bsz = 1.f / (float)(end - start);

            float log_prob_new, entropy;

            for (uint32_t bi = start; bi < end; bi++) {
                uint32_t  t            = idx[bi];
                float    *obs_t        = &buf->states[t * buf->obs_dim];
                float     adv_t        = buf->advantages[t];
                float     ret_t        = buf->returns[t];
                float     log_prob_old = buf->log_probs_old[t];

#if TIME_LOG
                uint32_t _tf, _tb;
#endif

#if USE_CONTINUOUS_ACTION
                float *a_t = &buf->actions[t * N_ACT_DIMS];
#if TIME_LOG
                _tf = dwt_ticks();
#endif
                actor_forward_continuous(actor, obs_t, a_t, g_ppo_log_sigma,
                                         &log_prob_new, &entropy);
#if TIME_LOG
                _fwd_cyc += dwt_delta(_tf, dwt_ticks());
#endif
                float ratio = expf(fmaxf(fminf(log_prob_new - log_prob_old, 10.f), -10.f));
#if TIME_LOG
                _tb = dwt_ticks();
#endif
                actor_backward_continuous(actor, obs_t, a_t, g_ppo_log_sigma,
                                          adv_t, ratio, PPO_CLIP_EPS);
#if TIME_LOG
                _bwd_cyc += dwt_delta(_tb, dwt_ticks());
#endif
#else
                uint32_t a_t = buf->actions[t];
#if TIME_LOG
                _tf = dwt_ticks();
#endif
                actor_forward(actor, obs_t, probs, &log_prob_new, a_t, &entropy);
#if TIME_LOG
                _fwd_cyc += dwt_delta(_tf, dwt_ticks());
#endif
                float ratio = expf(fmaxf(fminf(log_prob_new - log_prob_old, 10.f), -10.f));
#if TIME_LOG
                _tb = dwt_ticks();
#endif
                actor_backward(actor, obs_t, a_t, adv_t, ratio,
                               PPO_CLIP_EPS, PPO_C2);
#if TIME_LOG
                _bwd_cyc += dwt_delta(_tb, dwt_ticks());
#endif
#endif

#if TIME_LOG
                _tf = dwt_ticks();
#endif
                critic_forward(critic, obs_t);
#if TIME_LOG
                _fwd_cyc += dwt_delta(_tf, dwt_ticks());
                _tb = dwt_ticks();
#endif
                critic_backward(critic, obs_t, ret_t, PPO_C1);
#if TIME_LOG
                _bwd_cyc += dwt_delta(_tb, dwt_ticks());
#endif
            }

#if TIME_LOG
            uint32_t _ta = dwt_ticks();
#endif
            for (int l = 0; l < (int)actor->num_layers; l++) {
                DenseLayer *la = &actor->layers[l];
                for (int i = 0; i < la->out_dim; i++) {
                    la->db[i] *= inv_bsz;
                    for (int j = 0; j < la->in_dim; j++)
                        la->dW[i][j] *= inv_bsz;
                }
            }
            for (int l = 0; l < (int)critic->num_layers; l++) {
                DenseLayer *lc = &critic->layers[l];
                for (int i = 0; i < lc->out_dim; i++) {
                    lc->db[i] *= inv_bsz;
                    for (int j = 0; j < lc->in_dim; j++)
                        lc->dW[i][j] *= inv_bsz;
                }
            }

            network_clip_grad(actor);
            network_clip_grad(critic);
            network_adam_update(actor, PPO_LR_ACTOR);
            network_adam_update(critic, PPO_LR_CRITIC);
#if TIME_LOG
            uint32_t _now = dwt_ticks();
            _adam_cyc += dwt_delta(_ta, _now);
            _tot_cyc  += dwt_delta(_t_chunk, _now);   /* chiude il pezzo di totale */
            _t_chunk   = _now;
#endif
        }
    }
#if USE_CONTINUOUS_ACTION
    ppo_sigma_decay();
#endif

#if TIME_LOG
    _tot_cyc += dwt_delta(_t_chunk, dwt_ticks());   /* coda dopo l'ultimo minibatch */

    g_train_timing.adam_cycles     = _adam_cyc;
    g_train_timing.forward_cycles  = _fwd_cyc;
    g_train_timing.backward_cycles = _bwd_cyc;
    g_train_timing.total_cycles    = _tot_cyc;
#endif
}

// ─── PPOAgent ─────────────────────────────────────────────────────────────────

int ppo_agent_init(PPOAgent *agent, Network *actor, Network *critic,
                   ppo_reward_fn reward_fn, ppo_done_fn done_fn,
                   ppo_event_fn on_train_begin, ppo_event_fn on_train_end)
{
    agent->actor           = actor;
    agent->critic          = critic;
    agent->reward_fn       = reward_fn;
    agent->done_fn         = done_fn;
    agent->on_train_begin  = on_train_begin;
    agent->on_train_end    = on_train_end;

    agent->rollout_step_count = 0;
    agent->step_in_ep         = 0;
    agent->first_step         = 1;
    agent->prev_log_prob      = 0.f;
    agent->prev_value         = 0.f;
    agent->prev_done          = 0;
    agent->done               = 0;

    memset(agent->prev_obs, 0, sizeof(agent->prev_obs));
#if USE_CONTINUOUS_ACTION
    memset(agent->prev_action, 0, sizeof(agent->prev_action));
#else
    agent->prev_action = 0;
#endif

    if (!rollout_buffer_init(&agent->buf, ROLLOUT_STEPS, OBS_DIM))
        return 0;

#if USE_CONTINUOUS_ACTION
    ppo_sigma_init();
#endif

    return 1;
}

#if USE_CONTINUOUS_ACTION

void ppo_step(PPOAgent *agent, const float *obs, float *action_out)
{
    // 1. done dello stato CORRENTE (inviato via UART; comanda il reset lato PC).
    //    Convenzione allineata a pc_ppo_trainer.py: done = is_done(s_t).
    agent->done = agent->done_fn(agent->step_in_ep);

    // 2. Push della transizione precedente (s_{t-1}, a_{t-1}). Il suo done e'
    //    prev_done = is_done(s_{t-1}), NON il done corrente. La transizione dello
    //    stato terminale NON viene scartata: verra' spinta alla chiamata
    //    successiva, esattamente come fa la baseline PC.
    if (!agent->first_step) {
        float reward = agent->reward_fn(agent->prev_obs, agent->prev_action);
        rollout_buffer_push(&agent->buf,
                            agent->prev_obs,
                            agent->prev_action,
                            reward,
                            agent->prev_done,
                            agent->prev_log_prob,
                            agent->prev_value);
    }

    // 3. Sample action from current observation
    float log_prob, value_critic;
    float action[N_ACT_DIMS];
    actor_sample_action(agent->actor, (float *)obs, action,
                        &log_prob, &value_critic, agent->critic);

    // 4. Store unclipped action and obs (log_prob consistency requires unclipped)
    memcpy(agent->prev_obs,    obs,    OBS_DIM    * sizeof(float));
    memcpy(agent->prev_action, action, N_ACT_DIMS * sizeof(float));
    agent->prev_log_prob = log_prob;
    agent->prev_value    = value_critic;
    agent->prev_done     = agent->done;   // done di s_t, usato al prossimo push

    // 5. Output raw action — user clips before sending if desired
    memcpy(action_out, action, N_ACT_DIMS * sizeof(float));

    // 6. Update counters — su done resettiamo solo il contatore di episodio.
    //    NON impostiamo first_step (la transizione terminale va comunque spinta).
    agent->first_step = 0;
    if (agent->done) {
        agent->step_in_ep = 0;
    } else {
        agent->step_in_ep++;
    }
    agent->rollout_step_count++;

    // 7. Trigger training when rollout is full
    if (agent->rollout_step_count >= ROLLOUT_STEPS) {
        // Bootstrap con V(obs): obs e' sempre il successore dell'ultima
        // transizione spinta. Se quella era terminale, GAE lo ignora (not_done=0).
        float last_val = critic_forward(agent->critic, (float *)obs);
        compute_gae(&agent->buf, last_val, PPO_GAMMA, PPO_LAMBDA);
        normalize_advantages(&agent->buf);
        if (agent->buf.size >= PPO_BATCH_SIZE) {
            if (agent->on_train_begin) agent->on_train_begin();
            ppo_update(agent->actor, agent->critic, &agent->buf);
            if (agent->on_train_end)   agent->on_train_end();
        }
        agent->buf.head           = 0;
        agent->buf.size           = 0;
        agent->rollout_step_count = 0;
    }
}

#else  /* discrete */

uint32_t ppo_step_discrete(PPOAgent *agent, const float *obs)
{
    // 1. done dello stato corrente (convenzione allineata a pc)
    agent->done = agent->done_fn(agent->step_in_ep);

    // 2. Push della transizione precedente con il suo prev_done (transizione
    //    terminale mantenuta, spinta al passo successivo come fa pc).
    if (!agent->first_step) {
        float reward = agent->reward_fn(agent->prev_obs, agent->prev_action);
        rollout_buffer_push(&agent->buf,
                            agent->prev_obs,
                            agent->prev_action,
                            reward,
                            agent->prev_done,
                            agent->prev_log_prob,
                            agent->prev_value);
    }

    // 3. Sample action
    float    log_prob, value_critic;
    uint32_t action = actor_sample_action(agent->actor, (float *)obs,
                                          &log_prob, &value_critic, agent->critic);

    // 4. Store prev state
    memcpy(agent->prev_obs, obs, OBS_DIM * sizeof(float));
    agent->prev_action   = action;
    agent->prev_log_prob = log_prob;
    agent->prev_value    = value_critic;
    agent->prev_done     = agent->done;

    // 5. Update counters — su done reset solo del contatore episodio.
    agent->first_step = 0;
    if (agent->done) {
        agent->step_in_ep = 0;
    } else {
        agent->step_in_ep++;
    }
    agent->rollout_step_count++;

    // 6. Trigger training when rollout is full
    if (agent->rollout_step_count >= ROLLOUT_STEPS) {
        float last_val = critic_forward(agent->critic, (float *)obs);
        compute_gae(&agent->buf, last_val, PPO_GAMMA, PPO_LAMBDA);
        normalize_advantages(&agent->buf);
        if (agent->buf.size >= PPO_BATCH_SIZE) {
            if (agent->on_train_begin) agent->on_train_begin();
            ppo_update(agent->actor, agent->critic, &agent->buf);
            if (agent->on_train_end)   agent->on_train_end();
        }
        agent->buf.head           = 0;
        agent->buf.size           = 0;
        agent->rollout_step_count = 0;
    }

    return action;
}

#endif  /* USE_CONTINUOUS_ACTION */
