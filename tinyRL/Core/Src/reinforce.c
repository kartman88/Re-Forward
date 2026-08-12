#include "reinforce.h"
#include "rng.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#if TIME_LOG
#include "utils.h"   /* dwt_ticks / dwt_delta per il profiling */

TrainTiming g_train_timing = {0};
#endif

#if USE_CONTINUOUS_ACTION
float g_reinforce_sigma[N_ACT_DIMS];

void reinforce_sigma_init(void) {
    for (int i = 0; i < N_ACT_DIMS; i++)
        g_reinforce_sigma[i] = REINFORCE_SIGMA;
}
#endif

// ─── Episode Buffer ───────────────────────────────────────────────────────────

int episode_buffer_init(EpisodeBuffer *buf, uint32_t T, uint32_t obs_dim) {
    buf->obs_dim  = obs_dim;
    buf->size     = 0;
    buf->capacity = T;

    buf->states  = malloc(T * obs_dim * sizeof(float));
    buf->rewards = malloc(T * sizeof(float));
    buf->returns = malloc(T * sizeof(float));
#if USE_CONTINUOUS_ACTION
    buf->actions = malloc(T * N_ACT_DIMS * sizeof(float));
#else
    buf->actions = malloc(T * sizeof(uint32_t));
#endif

    if (!buf->states || !buf->rewards || !buf->returns || !buf->actions)
        return 0;

    return 1;
}

void episode_buffer_reset(EpisodeBuffer *buf) {
    buf->size = 0;
}

#if USE_CONTINUOUS_ACTION
void episode_buffer_push(EpisodeBuffer *buf, const float *obs,
                         const float *action, float reward) {
    if (buf->size >= buf->capacity) return;
    uint32_t idx = buf->size;
    memcpy(&buf->states[idx * buf->obs_dim], obs, buf->obs_dim * sizeof(float));
    memcpy(&buf->actions[idx * N_ACT_DIMS], action, N_ACT_DIMS * sizeof(float));
    buf->rewards[idx] = reward;
    buf->size++;
}
#else
void episode_buffer_push(EpisodeBuffer *buf, const float *obs,
                         uint32_t action, float reward) {
    if (buf->size >= buf->capacity) return;
    uint32_t idx = buf->size;
    memcpy(&buf->states[idx * buf->obs_dim], obs, buf->obs_dim * sizeof(float));
    buf->actions[idx] = action;
    buf->rewards[idx] = reward;
    buf->size++;
}
#endif

// ─── Returns e baseline ──────────────────────────────────────────────────────

// Ritorni Monte-Carlo scontati: G_t = r_t + gamma * G_{t+1}.
void compute_returns(EpisodeBuffer *buf, float gamma) {
    float G = 0.f;
    for (int t = (int)buf->size - 1; t >= 0; --t) {
        G = buf->rewards[t] + gamma * G;
        buf->returns[t] = G;
    }
}

// REINFORCE con baseline costante: sottrae la media e divide per la deviazione
// standard dei ritorni dell'episodio. Riduce la varianza del gradiente senza
// introdurre bias (la baseline non dipende dall'azione).
void normalize_returns(EpisodeBuffer *buf) {
    uint32_t n = buf->size;
    if (n == 0) return;

    float mean = 0.f;
    for (uint32_t i = 0; i < n; i++)
        mean += buf->returns[i];
    mean /= (float)n;

    float var = 0.f;
    for (uint32_t i = 0; i < n; i++) {
        float d = buf->returns[i] - mean;
        var += d * d;
    }
    float inv_std = 1.f / (sqrtf(var / (float)n) + 1e-6f);

    for (uint32_t i = 0; i < n; i++)
        buf->returns[i] = (buf->returns[i] - mean) * inv_std;
}

// ─── policy_sample_action ────────────────────────────────────────────────────

#if USE_CONTINUOUS_ACTION

void policy_sample_action(Network *policy, float *obs, float *action_out) {
    policy_forward_continuous(policy, obs, action_out, g_reinforce_sigma, NULL);
}

#else

// Campionamento "roulette-wheel" dalla categorica prodotta dal softmax.
uint32_t policy_sample_action(Network *policy, float *obs) {
    float probs[N_ACTIONS];
    network_forward(policy, obs, probs);

    int   n = policy->layers[policy->num_layers - 1].out_dim;
    float c = rng_uniform();
    for (int i = 0; i < n; ++i) {
        if (c < probs[i]) return (uint32_t)i;
        c -= probs[i];
    }
    return (uint32_t)(n - 1);   // fallback numerico
}

#endif

// ─── reinforce_update ────────────────────────────────────────────────────────

void reinforce_update(Network *policy, EpisodeBuffer *buf) {
    if (buf->size == 0) return;

#if TIME_LOG
    /* Il DWT e' a 32 bit e a 84 MHz wrappa ogni ~51 s: il totale viene
     * accumulato a pezzi (uno per step di traiettoria) in un uint64, cosi'
     * regge anche update molto lunghi. */
    uint64_t _fwd_cyc = 0, _bwd_cyc = 0, _tot_cyc = 0;
    uint32_t _t_chunk = dwt_ticks();
#endif

    compute_returns(buf, REINFORCE_GAMMA);
    normalize_returns(buf);

    network_zero_grad(policy);

    for (uint32_t t = 0; t < buf->size; t++) {
        float *state = &buf->states[t * buf->obs_dim];
        float  ret   = buf->returns[t];

#if TIME_LOG
        uint32_t _tf = dwt_ticks();
#endif
        // Re-forward: il backward legge le attivazioni dell'ultimo forward.
        network_forward(policy, state, NULL);
#if TIME_LOG
        _fwd_cyc += dwt_delta(_tf, dwt_ticks());
        uint32_t _tb = dwt_ticks();
#endif
#if USE_CONTINUOUS_ACTION
        policy_backward_continuous(policy, state,
                                   &buf->actions[t * N_ACT_DIMS],
                                   g_reinforce_sigma, ret);
#else
        policy_backward(policy, state, buf->actions[t], ret,
                        REINFORCE_ENT_COEF);
#endif
#if TIME_LOG
        uint32_t _now = dwt_ticks();
        _bwd_cyc += dwt_delta(_tb, _now);
        _tot_cyc += dwt_delta(_t_chunk, _now);   /* chiude il pezzo di totale */
        _t_chunk  = _now;
#endif
    }

#if TIME_LOG
    uint32_t _ta = dwt_ticks();
#endif
    network_scale_grad(policy, 1.f / (float)buf->size);
    network_clip_grad(policy);
    network_adam_update(policy, REINFORCE_LR);
#if TIME_LOG
    uint32_t _end = dwt_ticks();
    _tot_cyc += dwt_delta(_t_chunk, _end);   /* coda: scale/clip/adam */

    g_train_timing.adam_cycles     = dwt_delta(_ta, _end);
    g_train_timing.forward_cycles  = _fwd_cyc;
    g_train_timing.backward_cycles = _bwd_cyc;
    g_train_timing.total_cycles    = _tot_cyc;
#endif
}

// ─── ReinforceAgent ──────────────────────────────────────────────────────────

int reinforce_agent_init(ReinforceAgent *agent, Network *policy,
                         reinforce_reward_fn reward_fn,
                         reinforce_done_fn done_fn,
                         reinforce_event_fn on_train_begin,
                         reinforce_event_fn on_train_end)
{
    agent->policy         = policy;
    agent->reward_fn      = reward_fn;
    agent->done_fn        = done_fn;
    agent->on_train_begin = on_train_begin;
    agent->on_train_end   = on_train_end;

    agent->step_in_ep = 0;
    agent->first_step = 1;
    agent->done       = 0;

    memset(agent->prev_obs, 0, sizeof(agent->prev_obs));
#if USE_CONTINUOUS_ACTION
    memset(agent->prev_action, 0, sizeof(agent->prev_action));
    reinforce_sigma_init();
#else
    agent->prev_action = 0;
#endif

    if (!episode_buffer_init(&agent->buf, MAX_STEPS_PER_EP, OBS_DIM))
        return 0;

    return 1;
}

#if USE_CONTINUOUS_ACTION

void reinforce_step(ReinforceAgent *agent, const float *obs, float *action_out)
{
    // 1. done dello stato CORRENTE (inviato via UART; comanda il reset lato PC).
    //    Convenzione allineata a PPO: done = is_done(s_t).
    agent->done = agent->done_fn(agent->step_in_ep);

    // 2. Push della transizione precedente (s_{t-1}, a_{t-1}): la sua reward
    //    dipende dall'azione presa in s_{t-1}, quindi il push e' ritardato di
    //    un passo. Quando arriva s_T terminale il buffer contiene esattamente
    //    le T transizioni 0..T-1 dell'episodio.
    if (!agent->first_step) {
        float reward = agent->reward_fn(agent->prev_obs, agent->prev_action);
        episode_buffer_push(&agent->buf, agent->prev_obs,
                            agent->prev_action, reward);
    }

    // 3. Campiona l'azione dallo stato corrente.
    float action[N_ACT_DIMS];
    policy_sample_action(agent->policy, (float *)obs, action);

    // 4. Memorizza (s_t, a_t) per il push del prossimo giro. L'azione salvata e'
    //    quella NON clippata: il gradiente deve essere coerente con il campione
    //    effettivamente estratto dalla gaussiana.
    memcpy(agent->prev_obs,    obs,    OBS_DIM    * sizeof(float));
    memcpy(agent->prev_action, action, N_ACT_DIMS * sizeof(float));

    // 5. Azione grezza in uscita — chi chiama la clippa prima di inviarla.
    memcpy(action_out, action, N_ACT_DIMS * sizeof(float));

    agent->first_step = 0;
    agent->step_in_ep++;

    // 6. Fine episodio: update Monte-Carlo sulla traiettoria completa.
    //    first_step torna a 1 cosi' la coppia (s_T, a_T) — azione presa in uno
    //    stato terminale, che il PC scarta al reset — non finisce nell'episodio
    //    successivo.
    if (agent->done) {
        if (agent->on_train_begin) agent->on_train_begin();
        reinforce_update(agent->policy, &agent->buf);
        if (agent->on_train_end)   agent->on_train_end();
        episode_buffer_reset(&agent->buf);
        agent->step_in_ep = 0;
        agent->first_step = 1;
    }
}

#else  /* discrete */

uint32_t reinforce_step_discrete(ReinforceAgent *agent, const float *obs)
{
    // 1. done dello stato CORRENTE (convenzione allineata a PPO).
    agent->done = agent->done_fn(agent->step_in_ep);

    // 2. Push ritardato della transizione precedente (s_{t-1}, a_{t-1}).
    if (!agent->first_step) {
        float reward = agent->reward_fn(agent->prev_obs, agent->prev_action);
        episode_buffer_push(&agent->buf, agent->prev_obs,
                            agent->prev_action, reward);
    }

    // 3. Campiona l'azione dallo stato corrente.
    uint32_t action = policy_sample_action(agent->policy, (float *)obs);

    // 4. Memorizza (s_t, a_t) per il push del prossimo giro.
    memcpy(agent->prev_obs, obs, OBS_DIM * sizeof(float));
    agent->prev_action = action;

    agent->first_step = 0;
    agent->step_in_ep++;

    // 5. Fine episodio: update Monte-Carlo sulla traiettoria completa.
    if (agent->done) {
        if (agent->on_train_begin) agent->on_train_begin();
        reinforce_update(agent->policy, &agent->buf);
        if (agent->on_train_end)   agent->on_train_end();
        episode_buffer_reset(&agent->buf);
        agent->step_in_ep = 0;
        agent->first_step = 1;
    }

    return action;
}

#endif  /* USE_CONTINUOUS_ACTION */
