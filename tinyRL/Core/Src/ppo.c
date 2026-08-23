#include "ppo.h"
#include "rng.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* Profiling: quando TIME_LOG e' 0 le macro spariscono e non lasciano
 * riferimenti a variabili inesistenti, cosi' il loop caldo di ppo_update resta
 * leggibile invece di essere spezzato da una dozzina di #if. */
#if TIME_LOG
#include "utils.h"   /* dwt_ticks / dwt_delta per il profiling */

TrainTiming g_train_timing = {0};

#define PROF_T(name)       uint32_t name = dwt_ticks()
#define PROF_ADD(acc, t0)  ((acc) += dwt_delta((t0), dwt_ticks()))
#else
#define PROF_T(name)       ((void)0)
#define PROF_ADD(acc, t0)  ((void)0)
#endif

// ─── Rollout Buffer ───────────────────────────────────────────────────────────

/* Un'unica arena per l'intero buffer invece di 8 malloc separate: meno header
 * di heap e nessun percorso di fallimento parziale (prima, se la settima
 * malloc falliva, le prime sei restavano allocate e la init tornava 0). */
int rollout_buffer_init(RolloutBuffer *buf, uint32_t T, uint32_t obs_dim) {
    buf->obs_dim  = obs_dim;
    buf->size     = 0;
    buf->capacity = T;

#if USE_CONTINUOUS_ACTION
    const uint32_t n_act = T * N_ACT_DIMS;   /* z pre-squash, un float per dim */
#else
    const uint32_t n_act = T;                /* indice azione, uint32 = 4 byte */
#endif
    /* states + 5 vettori scalari + azioni, tutti a 4 byte; dones a 1 byte. */
    const size_t n_word = (size_t)T * obs_dim + 5u * T + n_act;
    const size_t bytes  = n_word * sizeof(float) + (size_t)T;

    float *arena = malloc(bytes);
    if (!arena) return 0;
    buf->arena = arena;

    float *p = arena;
    buf->states        = p; p += (size_t)T * obs_dim;
    buf->log_probs_old = p; p += T;
    buf->values        = p; p += T;
    buf->rewards       = p; p += T;
    buf->advantages    = p; p += T;
    buf->returns       = p; p += T;
#if USE_CONTINUOUS_ACTION
    buf->actions = p; p += n_act;
#else
    buf->actions = (uint32_t *)p; p += n_act;
#endif
    buf->dones = (uint8_t *)p;

    return 1;
}

void rollout_buffer_free(RolloutBuffer *buf) {
    free(buf->arena);
    buf->arena = NULL;
    buf->size  = 0;
}

#if USE_CONTINUOUS_ACTION
/* `z` e' l'azione PRE-squash: e' quella che serve per rivalutare la gaussiana
 * durante l'update. L'azione squashata a = tanh(z) serve solo all'ambiente e
 * al reward, e non entra nel buffer. */
void rollout_buffer_push(RolloutBuffer *buf, const float *obs, const float *z,
                         float reward, uint8_t done, float log_prob, float value) {
    if (buf->size >= buf->capacity) return;
    uint32_t idx = buf->size;
    memcpy(&buf->states[idx * buf->obs_dim], obs, buf->obs_dim * sizeof(float));
    memcpy(&buf->actions[idx * N_ACT_DIMS], z, N_ACT_DIMS * sizeof(float));
    buf->rewards[idx]       = reward;
    buf->dones[idx]         = done;
    buf->log_probs_old[idx] = log_prob;
    buf->values[idx]        = value;
    buf->size++;
}
#else
void rollout_buffer_push(RolloutBuffer *buf, const float *obs, uint32_t action,
                         float reward, uint8_t done, float log_prob, float value) {
    if (buf->size >= buf->capacity) return;
    uint32_t idx = buf->size;
    memcpy(&buf->states[idx * buf->obs_dim], obs, buf->obs_dim * sizeof(float));
    buf->actions[idx]       = action;
    buf->rewards[idx]       = reward;
    buf->dones[idx]         = done;
    buf->log_probs_old[idx] = log_prob;
    buf->values[idx]        = value;
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

/* Restituisce sia l'azione squashata `a` (per l'ambiente e per il reward) sia
 * la pre-squash `z` (per il buffer). Prima solo `a` veniva salvata e l'update
 * ricostruiva z = atanhf(a) ad ogni epoca: 6 atanhf per campione per epoca per
 * riottenere un valore che avevamo gia' qui, per giunta degradato dal clamp a
 * |a| <= 1-1e-6 che satura z a +-7.25.
 *
 * Il termine Jacobiano -log(1 - a^2) non entra in log_prob: dipende solo
 * dall'azione, fissa nel buffer, quindi compare identico in log_prob_old e
 * log_prob_new e si cancella nel ratio. */
void actor_sample_action(Network *actor, float *obs, float *action_out,
                         float *z_out, float *log_prob_out, float *value_out,
                         Network *critic) {
    network_forward(actor, obs, NULL);
    float *mu = actor->layers[actor->num_layers - 1].out;

    float log_prob = 0.f;
    for (int i = 0; i < N_ACT_DIMS; i++) {
        float m   = fmaxf(fminf(mu[i], MU_CLAMP), -MU_CLAMP);
        mu[i]     = m;
        float ls  = g_ppo_log_sigma[i];
        float eps = rng_normal();
        float z   = m + expf(ls) * eps;
        z_out[i]  = z;
#if PPO_USE_TANH_SQUASH
        action_out[i] = tanhf(z);
#else
        action_out[i] = z;
#endif
        log_prob += -0.5f * (eps * eps + 2.f * ls + LOG_2PI);
    }
    *log_prob_out = log_prob;

    if (value_out)
        *value_out = critic_forward(critic, obs);
}

#else  /* discrete */

uint32_t actor_sample_action(Network *actor, float *obs,
                              float *log_prob_out, float *value_out,
                              Network *critic) {
    float probs[PPO_N_ACTIONS];
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

    /* ROLLOUT_STEPS entra in 16 bit: 1 KB di .bss invece di 2. */
    static uint16_t idx[ROLLOUT_STEPS];

#if USE_CONTINUOUS_ACTION
    /* sigma e' costante per tutta la durata dell'update (ppo_sigma_decay gira
     * una sola volta, in fondo): var e log_var si calcolano qui una volta sola
     * invece di 6 expf per campione dentro i kernel. */
    float var[N_ACT_DIMS], log_var[N_ACT_DIMS];
    for (int i = 0; i < N_ACT_DIMS; i++) {
        log_var[i] = 2.f * g_ppo_log_sigma[i];
        var[i]     = expf(log_var[i]);
    }
#else
    float probs[PPO_N_ACTIONS];
#endif

#if TIME_LOG
    /* Il totale NON si misura con un unico delta start/stop: il CYCCNT e' a
     * 32 bit (wrap ogni ~8.9 s a 480 MHz) mentre un update dura di piu'. Lo
     * accumuliamo a pezzi, uno per minibatch (pochi ms l'uno), su un uint64. */
    g_train_timing = (TrainTiming){0};
    uint64_t _fwd_cyc = 0, _bwd_cyc = 0, _adam_cyc = 0, _tot_cyc = 0;
    uint32_t _t_chunk = dwt_ticks();
#endif

    for (uint32_t i = 0; i < N; i++) idx[i] = (uint16_t)i;

    network_zero_grad(actor);
    network_zero_grad(critic);

    for (int epoch = 0; epoch < PPO_EPOCHS; epoch++) {

        for (uint32_t i = N - 1; i > 0; i--) {
            uint32_t j   = rng_u32() % (i + 1);
            uint16_t tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
        }

        for (uint32_t start = 0; start < N; start += PPO_BATCH_SIZE) {
            uint32_t end     = start + PPO_BATCH_SIZE;
            if (end > N) end = N;
            /* inv_bsz viaggia dentro il delta invece di essere applicato dopo
             * con una passata su tutti i parametri di actor+critic (64 volte
             * per update). Il ramo di clipping dipende da ratio, non da
             * advantage, quindi scalare l'advantage non lo sposta. */
            float inv_bsz = 1.f / (float)(end - start);

            float log_prob_new;

            for (uint32_t bi = start; bi < end; bi++) {
                uint32_t  t            = idx[bi];
                float    *obs_t        = &buf->states[t * buf->obs_dim];
                float     adv_t        = buf->advantages[t];
                float     ret_t        = buf->returns[t];
                float     log_prob_old = buf->log_probs_old[t];

#if USE_CONTINUOUS_ACTION
                const float *z_t = &buf->actions[t * N_ACT_DIMS];

                PROF_T(_t_af);
                actor_forward_continuous(actor, obs_t, z_t, var, log_var,
                                         &log_prob_new);
                PROF_ADD(_fwd_cyc, _t_af);

                float ratio = expf(fmaxf(fminf(log_prob_new - log_prob_old, 10.f), -10.f));

                PROF_T(_t_ab);
                actor_backward_continuous(actor, obs_t, z_t, var,
                                          adv_t * inv_bsz, ratio, PPO_CLIP_EPS);
                PROF_ADD(_bwd_cyc, _t_ab);
#else
                uint32_t a_t = buf->actions[t];
                float    entropy;

                PROF_T(_t_af);
                actor_forward(actor, obs_t, probs, &log_prob_new, a_t, &entropy);
                PROF_ADD(_fwd_cyc, _t_af);

                float ratio = expf(fmaxf(fminf(log_prob_new - log_prob_old, 10.f), -10.f));

                PROF_T(_t_ab);
                actor_backward(actor, obs_t, a_t, adv_t * inv_bsz, ratio,
                               PPO_CLIP_EPS, PPO_C2 * inv_bsz);
                PROF_ADD(_bwd_cyc, _t_ab);
#endif

                PROF_T(_t_cf);
                critic_forward(critic, obs_t);
                PROF_ADD(_fwd_cyc, _t_cf);

                PROF_T(_t_cb);
                critic_backward(critic, obs_t, ret_t, PPO_C1 * inv_bsz);
                PROF_ADD(_bwd_cyc, _t_cb);
            }

            PROF_T(_t_adam);
            float sa = network_clip_grad(actor,  PPO_GRAD_CLIP);
            float sc = network_clip_grad(critic, PPO_GRAD_CLIP);
            network_adam_update(actor,  PPO_LR_ACTOR,  sa);
            network_adam_update(critic, PPO_LR_CRITIC, sc);
#if TIME_LOG
            uint32_t _now = dwt_ticks();
            _adam_cyc += dwt_delta(_t_adam, _now);
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
    memset(agent->prev_z,      0, sizeof(agent->prev_z));
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
    //    Nel buffer va prev_z (pre-squash); il reward usa prev_action (squashata).
    if (!agent->first_step) {
        float reward = agent->reward_fn(agent->prev_obs, agent->prev_action);
        rollout_buffer_push(&agent->buf,
                            agent->prev_obs,
                            agent->prev_z,
                            reward,
                            agent->prev_done,
                            agent->prev_log_prob,
                            agent->prev_value);
    }

    // 3. Sample action from current observation
    float log_prob, value_critic;
    float action[N_ACT_DIMS], z[N_ACT_DIMS];
    actor_sample_action(agent->actor, (float *)obs, action, z,
                        &log_prob, &value_critic, agent->critic);

    // 4. Store obs, azione squashata (per il reward) e pre-squash (per il buffer)
    memcpy(agent->prev_obs,    obs,    OBS_DIM    * sizeof(float));
    memcpy(agent->prev_action, action, N_ACT_DIMS * sizeof(float));
    memcpy(agent->prev_z,      z,      N_ACT_DIMS * sizeof(float));
    agent->prev_log_prob = log_prob;
    agent->prev_value    = value_critic;
    agent->prev_done     = agent->done;   // done di s_t, usato al prossimo push

    // 5. Output action — con tanh squash e' gia' in (-1, 1)
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
        agent->buf.size           = 0;
        agent->rollout_step_count = 0;
    }

    return action;
}

#endif  /* USE_CONTINUOUS_ACTION */
