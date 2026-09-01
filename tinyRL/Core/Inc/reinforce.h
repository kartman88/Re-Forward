#ifndef REINFORCE_H
#define REINFORCE_H

#include "main.h"   /* HAL della famiglia target + TIME_LOG */
#include "neural_net.h"
#include <stdint.h>
#include <stdlib.h>

// REINFORCE Hyperparameters (Monte-Carlo policy gradient, baseline costante)
#define REINFORCE_LR        0.01f   // 0.01 / 0.02 vanno bene per CartPole
#define REINFORCE_GAMMA     0.99f
#define REINFORCE_ENT_COEF  0.001f  // 0.001 va bene per CartPole
#define MAX_STEPS_PER_EP    500     // CartPole-v1: troncamento a 500 step
#define MAX_EPISODE         1000

// Policy output size: usare REINFORCE_POLICY_OUT_DIM per l'ultimo layer.
#if USE_CONTINUOUS_ACTION
#define REINFORCE_POLICY_OUT_DIM  N_ACT_DIMS
// Bound simmetrico: l'utente puo' clippare in [-ACTION_SCALE, +ACTION_SCALE].
#define ACTION_SCALE              2.0f
// Deviazione standard fissa usata sia nel campionamento sia nel gradiente.
#define REINFORCE_SIGMA           0.5f
extern float g_reinforce_sigma[N_ACT_DIMS];
void reinforce_sigma_init(void);
#else
#define REINFORCE_POLICY_OUT_DIM  N_ACTIONS
#endif

// ── Episode Buffer ────────────────────────────────────────────────────────────
// REINFORCE e' Monte-Carlo: serve la traiettoria completa di UN episodio, che
// viene consumata e svuotata a ogni update.

/* L'azione discreta sta in un byte: con N_ACTIONS = 2 un uint32_t per step
 * sprecava 1,5 KB su MAX_STEPS_PER_EP = 500. Lo _Static_assert lega il tipo
 * alla costante che lo giustifica. */
#if !USE_CONTINUOUS_ACTION
_Static_assert(N_ACTIONS <= 255, "action index deve stare in uint8_t");
#endif
_Static_assert(MAX_STEPS_PER_EP > 0, "MAX_STEPS_PER_EP deve essere positivo");

/* Buffer di episodio in UNA sola allocazione, con i vettori come viste dentro
 * l'arena, invece di 4 malloc separate. Oltre all'overhead sparisce il
 * percorso di fallimento parziale: prima, se la terza malloc falliva, le
 * prime due restavano allocate e la init tornava 0 — leak silenzioso in un
 * contesto senza OS.
 *
 * Layout con i campi a 4 byte prima di quelli a 1 byte, cosi' l'aritmetica dei
 * puntatori resta allineata senza padding esplicito:
 *     [ states | rewards | returns | actions ]
 *
 * `size` e `capacity` non sono ridondanti: il buffer si riempie fino a
 * capacity e viene svuotato a ogni update. */
typedef struct {
    float    *arena;      // base dell'allocazione, la libera episode_buffer_free()
    float    *states;     // [capacity * obs_dim]
    float    *rewards;    // [capacity]
    float    *returns;    // ritorni scontati, poi normalizzati in place
#if USE_CONTINUOUS_ACTION
    float    *actions;    // [capacity * N_ACT_DIMS]
#else
    uint8_t  *actions;    // [capacity]
#endif
    uint32_t  size;
    uint32_t  capacity;
    uint32_t  obs_dim;
} EpisodeBuffer;

#if TIME_LOG
/* Profiling: cicli DWT misurati dentro reinforce_update (una chiamata = un
 * update, cioe' un episodio). total = intero update; forward = re-forward della
 * traiettoria; backward = policy_backward; adam = media/clip dei gradienti +
 * network_adam_update.
 * Accumulatori a 64 bit: il DWT e' a 32 bit e a 480 MHz wrappa ogni ~8.9 s,
 * potenzialmente meno di un update su un episodio lungo. */
typedef struct {
    uint64_t total_cycles;
    uint64_t forward_cycles;
    uint64_t backward_cycles;
    uint64_t adam_cycles;
} TrainTiming;

extern TrainTiming g_train_timing;
#endif /* TIME_LOG */

int  episode_buffer_init(EpisodeBuffer *buf, uint32_t T, uint32_t obs_dim);
void episode_buffer_free(EpisodeBuffer *buf);
void episode_buffer_reset(EpisodeBuffer *buf);
#if USE_CONTINUOUS_ACTION
void episode_buffer_push(EpisodeBuffer *buf, const float *obs,
                         const float *action, float reward);
#else
void episode_buffer_push(EpisodeBuffer *buf, const float *obs,
                         uint32_t action, float reward);
#endif

void compute_returns(EpisodeBuffer *buf, float gamma);
void normalize_returns(EpisodeBuffer *buf);

// ── Action sampling ───────────────────────────────────────────────────────────

#if USE_CONTINUOUS_ACTION
void     policy_sample_action(Network *policy, float *obs, float *action_out);
#else
uint32_t policy_sample_action(Network *policy, float *obs);
#endif

void reinforce_update(Network *policy, EpisodeBuffer *buf);

// ── ReinforceAgent — high-level API ───────────────────────────────────────────

#if USE_CONTINUOUS_ACTION
typedef float (*reinforce_reward_fn)(const float *obs, const float *action);
#else
typedef float (*reinforce_reward_fn)(const float *obs, uint32_t action);
#endif
typedef uint8_t (*reinforce_done_fn)(uint32_t step_in_ep);
typedef void    (*reinforce_event_fn)(void);

typedef struct {
    Network       *policy;
    EpisodeBuffer  buf;

    uint32_t       step_in_ep;
    uint8_t        first_step;

    float          prev_obs[OBS_DIM];
#if USE_CONTINUOUS_ACTION
    float          prev_action[N_ACT_DIMS];
#else
    uint32_t       prev_action;
#endif

    uint8_t        done;           // set inside reinforce_step, readable by caller

    reinforce_reward_fn reward_fn;
    reinforce_done_fn   done_fn;
    reinforce_event_fn  on_train_begin; // optional (NULL = no-op)
    reinforce_event_fn  on_train_end;   // optional (NULL = no-op)
} ReinforceAgent;

int reinforce_agent_init(ReinforceAgent *agent, Network *policy,
                         reinforce_reward_fn reward_fn,
                         reinforce_done_fn done_fn,
                         reinforce_event_fn on_train_begin,
                         reinforce_event_fn on_train_end);

#if USE_CONTINUOUS_ACTION
void     reinforce_step(ReinforceAgent *agent, const float *obs, float *action_out);
#else
uint32_t reinforce_step_discrete(ReinforceAgent *agent, const float *obs);
#endif

#endif
