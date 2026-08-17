#include "dqn.h"
#include "rng.h"
#if TIME_LOG
#include "utils.h"   /* dwt_ticks / dwt_delta per il profiling */

TrainTiming g_train_timing = {0};
#endif

int replay_buffer_init(ReplayBuffer *buf, uint32_t capacity, uint32_t obs_dim) {
    buf->capacity = capacity;
    buf->obs_dim  = obs_dim;
    buf->head     = 0;
    buf->size     = 0;

    buf->state_pool = malloc(capacity * obs_dim * sizeof(float));
    buf->snext_pool = malloc(capacity * obs_dim * sizeof(float));
    buf->state      = malloc(capacity * sizeof(float *));
    buf->next_state = malloc(capacity * sizeof(float *));
    buf->action     = malloc(capacity * sizeof(uint32_t));
    buf->reward     = malloc(capacity * sizeof(float));
    buf->done       = malloc(capacity * sizeof(uint8_t));

    if (!buf->state_pool || !buf->snext_pool || !buf->state ||
        !buf->next_state || !buf->action || !buf->reward || !buf->done)
        return 0;

    for (uint32_t i = 0; i < capacity; i++) {
        buf->state[i]      = buf->state_pool + i * obs_dim;
        buf->next_state[i] = buf->snext_pool + i * obs_dim;
    }
    return 1;
}

void replay_buffer_push(ReplayBuffer *buf, float *s, uint32_t action,
                        float reward, float *s_next, uint8_t done) {
    uint32_t idx = buf->head;
    memcpy(buf->state[idx],      s,      buf->obs_dim * sizeof(float));
    memcpy(buf->next_state[idx], s_next, buf->obs_dim * sizeof(float));
    buf->action[idx] = action;
    buf->reward[idx] = reward;
    buf->done[idx]   = done;
    buf->head = (idx + 1) % buf->capacity;
    if (buf->size < buf->capacity)
        buf->size++;
}

float calc_epsilon(uint32_t step) {
    float t = (float)step / EPSILON_DECAY;
    if (t > 1.0f) t = 1.0f;
    return EPSILON_START + t * (EPSILON_END - EPSILON_START);
}

uint32_t dqn_select_action(QNetwork *net, float *obs, float epsilon,
                            uint32_t n_actions) {
    float r = rng_uniform();
    if (r < epsilon)
        return rng_u32() % n_actions;

    float q[N_ACTIONS];
    forward_q(net, obs, q);
    uint32_t best = 0;
    for (uint32_t i = 1; i < n_actions; i++)
        if (q[i] > q[best]) best = i;
    return best;
}

void dqn_train(QNetwork *online, TargetNetwork *target,
               ReplayBuffer *buf, uint32_t batch_size, uint32_t n_actions) {
    float q_online[N_ACTIONS];
    float q_tgt[N_ACTIONS];
    float inv_batch = 1.0f / (float)batch_size;

#if TIME_LOG
    /* --- profiling: azzera e avvia i contatori DWT --- */
    g_train_timing = (TrainTiming){0};
    uint32_t _t_total = dwt_ticks();
    uint32_t _fwd_cyc = 0, _bwd_cyc = 0;
#endif

    zero_grad_q(online);

    for (uint32_t b = 0; b < batch_size; b++) {
        uint32_t idx    = rng_u32() % buf->size;
        float   *s      = buf->state[idx];
        float   *s_next = buf->next_state[idx];
        uint32_t act    = buf->action[idx];
        float    rew    = buf->reward[idx];
        uint8_t  dn     = buf->done[idx];

#if TIME_LOG
        uint32_t _tf = dwt_ticks();
#endif
        forward_q(online, s, q_online);
        forward_target(target, s_next, q_tgt);
#if TIME_LOG
        _fwd_cyc += dwt_delta(_tf, dwt_ticks());
#endif

        float max_qt = q_tgt[0];
        for (uint32_t i = 1; i < n_actions; i++)
            if (q_tgt[i] > max_qt) max_qt = q_tgt[i];

        float target_val = dn ? rew : rew + GAMMA * max_qt;
        float td_error   = q_online[act] - target_val;

#if TIME_LOG
        uint32_t _tb = dwt_ticks();
#endif
        dqn_backward(online, s, act, td_error * inv_batch);
#if TIME_LOG
        _bwd_cyc += dwt_delta(_tb, dwt_ticks());
#endif
    }

#if TIME_LOG
    uint32_t _ta = dwt_ticks();
#endif
    gradient_norm_q(online);
    adam_optimizer_q(online);
#if TIME_LOG
    g_train_timing.adam_cycles     = dwt_delta(_ta, dwt_ticks());

    g_train_timing.forward_cycles  = _fwd_cyc;
    g_train_timing.backward_cycles = _bwd_cyc;
    g_train_timing.total_cycles    = dwt_delta(_t_total, dwt_ticks());
#endif
}
