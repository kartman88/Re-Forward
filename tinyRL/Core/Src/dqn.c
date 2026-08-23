#include "dqn.h"
#include "rng.h"

/* Profiling: quando TIME_LOG e' 0 le macro spariscono e non lasciano
 * riferimenti a variabili inesistenti, cosi' il loop caldo di dqn_train resta
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

// ─── Replay Buffer ────────────────────────────────────────────────────────────

/* Un'unica arena per l'intero buffer invece di 7 malloc separate: meno header
 * di heap, nessun percorso di fallimento parziale (prima, se la sesta malloc
 * falliva, le prime cinque restavano allocate e la init tornava 0 — leak
 * silenzioso senza OS) e via gli 8 KB di array di puntatori di riga. */
int replay_buffer_init(ReplayBuffer *buf, uint32_t capacity, uint32_t obs_dim) {
    if (capacity == 0 || obs_dim == 0) return 0;

    buf->capacity = capacity;
    buf->obs_dim  = obs_dim;
    buf->head     = 0;
    buf->size     = 0;
    buf->arena    = NULL;

    /* states + next_states + reward a 4 byte; action e done a 1 byte, in coda
     * cosi' l'aritmetica dei puntatori resta allineata senza padding. */
    const size_t n_word = 2u * (size_t)capacity * obs_dim + (size_t)capacity;
    const size_t bytes  = n_word * sizeof(float) + 2u * (size_t)capacity;

    float *arena = malloc(bytes);
    if (!arena) return 0;
    buf->arena = arena;

    float *p = arena;
    buf->states      = p; p += (size_t)capacity * obs_dim;
    buf->next_states = p; p += (size_t)capacity * obs_dim;
    buf->reward      = p; p += capacity;
    buf->action      = (uint8_t *)p;
    buf->done        = buf->action + capacity;

    return 1;
}

void replay_buffer_free(ReplayBuffer *buf) {
    if (!buf || !buf->arena) return;
    free(buf->arena);
    buf->arena  = NULL;
    buf->states = buf->next_states = buf->reward = NULL;
    buf->action = buf->done = NULL;
    buf->head   = 0;
    buf->size   = 0;
}

void replay_buffer_push(ReplayBuffer *buf, const float *s, uint32_t action,
                        float reward, const float *s_next, uint8_t done) {
    /* Bounds check: head e' sempre < capacity per costruzione, ma un buffer
     * non inizializzato (capacity 0) farebbe una divisione per zero qui sotto
     * e una scrittura su puntatore nullo. */
    if (!buf->arena || buf->head >= buf->capacity || action >= N_ACTIONS)
        return;

    const uint32_t idx = buf->head;
    const size_t   n   = (size_t)buf->obs_dim * sizeof(float);
    memcpy(replay_state(buf, idx),      s,      n);
    memcpy(replay_next_state(buf, idx), s_next, n);
    buf->action[idx] = (uint8_t)action;
    buf->reward[idx] = reward;
    buf->done[idx]   = done;
    buf->head = (idx + 1) % buf->capacity;
    if (buf->size < buf->capacity)
        buf->size++;
}

// ─── Policy ───────────────────────────────────────────────────────────────────

float calc_epsilon(uint32_t step) {
    float t = (float)step / EPSILON_DECAY;
    if (t > 1.0f) t = 1.0f;
    return EPSILON_START + t * (EPSILON_END - EPSILON_START);
}

uint32_t dqn_select_action(QNetwork *net, const float *obs, float epsilon,
                            uint32_t n_actions) {
    float r = rng_uniform();
    if (r < epsilon)
        return rng_u32() % n_actions;

    /* q_out = NULL: l'argmax si legge direttamente dal buffer di uscita
     * dell'ultimo layer, senza copiarlo prima in un array locale. */
    forward_q(net, obs, NULL);
    const float *q = net->layers[net->num_layers - 1].out;
    uint32_t best = 0;
    for (uint32_t i = 1; i < n_actions; i++)
        if (q[i] > q[best]) best = i;
    return best;
}

// ─── Update ───────────────────────────────────────────────────────────────────

void dqn_train(QNetwork *online, TargetNetwork *target,
               ReplayBuffer *buf, uint32_t batch_size, uint32_t n_actions) {
    /* Costante per l'intero update: il fattore 1/batch_size si ripiega nel
     * delta di uscita (un solo float per campione) invece di scalare a valle
     * tutti i gradienti di tutti i layer. */
    const float inv_batch = 1.0f / (float)batch_size;

    const float *q_online = online->layers[online->num_layers - 1].out;
    const float *q_tgt    = target->layers[target->num_layers - 1].out;

#if TIME_LOG
    g_train_timing = (TrainTiming){0};
    uint32_t _fwd_cyc = 0, _bwd_cyc = 0;
#endif
    PROF_T(_t_total);

    zero_grad_q(online);

    for (uint32_t b = 0; b < batch_size; b++) {
        uint32_t idx    = rng_u32() % buf->size;
        const float *s      = replay_state(buf, idx);
        const float *s_next = replay_next_state(buf, idx);
        uint32_t     act    = buf->action[idx];
        float        rew    = buf->reward[idx];
        uint8_t      dn     = buf->done[idx];

        PROF_T(_tf);
        forward_q(online, s, NULL);
        /* La rete target serve solo per max_a' Q(s',a'), che entra nel target
         * unicamente quando la transizione NON e' terminale: su una transizione
         * done il suo risultato veniva calcolato e poi scartato. Saltarla e'
         * esatto, ed e' un forward intero risparmiato. */
        float target_val = rew;
        if (!dn) {
            forward_target(target, s_next, NULL);
            float max_qt = q_tgt[0];
            for (uint32_t i = 1; i < n_actions; i++)
                if (q_tgt[i] > max_qt) max_qt = q_tgt[i];
            target_val = rew + GAMMA * max_qt;
        }
        PROF_ADD(_fwd_cyc, _tf);

        float td_error = q_online[act] - target_val;

        PROF_T(_tb);
        dqn_backward(online, s, act, td_error * inv_batch);
        PROF_ADD(_bwd_cyc, _tb);
    }

    PROF_T(_t_adam);
    /* Il clipping non riscrive piu' i gradienti: ritorna il fattore di scala,
     * che Adam applica al volo mentre carica dW/db. Una passata completa su
     * tutti i parametri in meno ogni volta che il clip scatta. */
    float gscale = gradient_norm_q(online, GRAD_CLIP);
    adam_optimizer_q(online, LR, gscale);

#if TIME_LOG
    g_train_timing.adam_cycles     = dwt_delta(_t_adam, dwt_ticks());
    g_train_timing.forward_cycles  = _fwd_cyc;
    g_train_timing.backward_cycles = _bwd_cyc;
    g_train_timing.total_cycles    = dwt_delta(_t_total, dwt_ticks());
#endif
}
