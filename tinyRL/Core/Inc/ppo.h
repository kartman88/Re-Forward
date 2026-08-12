#ifndef PPO_H
#define PPO_H

#include "main.h"   /* HAL della famiglia target + TIME_LOG */
#include "neural_net.h"
#include <stdint.h>
#include <stdlib.h>

// PPO Hyperparameters
#define PPO_LR_ACTOR      5e-4f
#define PPO_LR_CRITIC     1e-3f
#define PPO_GAMMA         0.99f
#define PPO_LAMBDA        0.95f
#define PPO_CLIP_EPS      0.2f
#define PPO_EPOCHS        8
#define PPO_BATCH_SIZE    64
/* NUCLEO-F446RE: 128 KB di SRAM in tutto, contro i 512 KB di RAM_D1 usati
 * sull'H7. Con la topologia 64-64 dell'H7 le sole due reti (pesi + dW + mW + vW)
 * occupano ~163 KB, quindi non entrano a prescindere dal rollout: la topologia
 * scende a 32-32 (vedi main.c) e il rollout da 2048 a 512 step.
 * Budget risultante ~90.6 KB su 128 KB (memory_calculator_ppo.py). */
#define ROLLOUT_STEPS     512
#define PPO_C1            0.5f
#define PPO_C2            0.01f
#define PPO_GRAD_CLIP     0.5f
#define PPO_OBS_DIM       OBS_DIM

// Actor output size: use PPO_ACTOR_OUT_DIM to set the last layer of the actor network.
#if USE_CONTINUOUS_ACTION
// Actor outputs only means (mu); std is state-independent and decayed externally.
#define PPO_ACTOR_OUT_DIM  N_ACT_DIMS
// Symmetric bound: action values can be clipped to [-ACTION_SCALE, +ACTION_SCALE] by the user.
#define ACTION_SCALE       1.0f
// Sigma schedule: starts at PPO_SIGMA_INIT, decreases by PPO_SIGMA_DECAY each ppo_update,
// never drops below PPO_SIGMA_MIN.
#define PPO_SIGMA_INIT     1.5f
#define PPO_SIGMA_MIN      0.1f
/* ppo_sigma_decay() viene chiamata una volta per ppo_update, cioe' ogni
 * ROLLOUT_STEPS step di ambiente. Accorciando il rollout (2048 -> 512) gli
 * update diventano 4x piu' frequenti: a parita' di PPO_SIGMA_N_STEPS sigma
 * collasserebbe 4x prima in termini di step di ambiente. Il conteggio viene
 * quindi riscalato sul rollout di riferimento dell'H7, cosi' la traiettoria di
 * sigma in funzione degli step di ambiente resta identica al build H7. */
#define PPO_SIGMA_REF_ROLLOUT  2048
#define PPO_SIGMA_N_STEPS  (500000 * PPO_SIGMA_REF_ROLLOUT / ROLLOUT_STEPS)
#define PPO_SIGMA_DECAY    (logf(PPO_SIGMA_INIT / PPO_SIGMA_MIN) / (float)PPO_SIGMA_N_STEPS)

extern float g_ppo_log_sigma[N_ACT_DIMS];
void ppo_sigma_init(void);
void ppo_sigma_decay(void);
#else
#define PPO_ACTOR_OUT_DIM  N_ACTIONS
#define PPO_N_ACTIONS      N_ACTIONS
#endif

// ── Rollout Buffer ─────────────────────────────────────────────────────────────

typedef struct {
    float    *states;
    float    *log_probs_old;
    float    *values;
    float    *rewards;
    float    *advantages;
    float    *returns;
#if USE_CONTINUOUS_ACTION
    float    *actions;
#else
    uint32_t *actions;
#endif
    uint8_t  *dones;
    uint32_t  head;
    uint32_t  size;
    uint32_t  capacity;
    uint32_t  obs_dim;
} RolloutBuffer;

#if TIME_LOG
/* Profiling: cicli DWT misurati dentro ppo_update (una chiamata = un update).
 * total = intero update; forward = actor_forward+critic_forward sui minibatch;
 * backward = actor_backward+critic_backward; adam = normalizzazione gradienti +
 * network_clip_grad + network_adam_update (actor e critic).
 * Accumulatori a 64 bit: il DWT e' a 32 bit e a 480 MHz wrappa ogni ~8.9 s,
 * meno di quanto dura un update intero (8 epoche x ROLLOUT_STEPS campioni). */
typedef struct {
    uint64_t total_cycles;
    uint64_t forward_cycles;
    uint64_t backward_cycles;
    uint64_t adam_cycles;
} TrainTiming;

extern TrainTiming g_train_timing;
#endif /* TIME_LOG */

int  rollout_buffer_init(RolloutBuffer *buf, uint32_t T, uint32_t obs_dim);
#if USE_CONTINUOUS_ACTION
void rollout_buffer_push(RolloutBuffer *buf, float *obs, float *action,
                         float reward, uint8_t done, float log_prob, float value);
#else
void rollout_buffer_push(RolloutBuffer *buf, float *obs, uint32_t action,
                         float reward, uint8_t done, float log_prob, float value);
#endif

void compute_gae(RolloutBuffer *buf, float last_value, float gamma, float lambda);
void normalize_advantages(RolloutBuffer *buf);

// ── Action sampling ───────────────────────────────────────────────────────────

#if USE_CONTINUOUS_ACTION
void     actor_sample_action(Network *actor, float *obs, float *action_out,
                             float *log_prob_out, float *value_out,
                             Network *critic);
#else
uint32_t actor_sample_action(Network *actor, float *obs,
                              float *log_prob_out, float *value_out,
                              Network *critic);
#endif

void ppo_update(Network *actor, Network *critic, RolloutBuffer *buf);

// ── PPOAgent — high-level API ─────────────────────────────────────────────────

#if USE_CONTINUOUS_ACTION
typedef float (*ppo_reward_fn)(const float *obs, const float *action);
#else
typedef float (*ppo_reward_fn)(const float *obs, uint32_t action);
#endif
typedef uint8_t (*ppo_done_fn)(uint32_t step_in_ep);
typedef void    (*ppo_event_fn)(void);

typedef struct {
    Network       *actor;
    Network       *critic;
    RolloutBuffer  buf;

    uint32_t       rollout_step_count;
    uint32_t       step_in_ep;
    uint8_t        first_step;

    float          prev_obs[OBS_DIM];
#if USE_CONTINUOUS_ACTION
    float          prev_action[N_ACT_DIMS];
#else
    uint32_t       prev_action;
#endif
    float          prev_log_prob;
    float          prev_value;

    uint8_t        prev_done;      // done dello stato precedente (per push allineato a pc)
    uint8_t        done;           // set inside ppo_step, readable by caller

    ppo_reward_fn  reward_fn;
    ppo_done_fn    done_fn;
    ppo_event_fn   on_train_begin; // optional (NULL = no-op)
    ppo_event_fn   on_train_end;
} PPOAgent;

int ppo_agent_init(PPOAgent *agent, Network *actor, Network *critic,
                   ppo_reward_fn reward_fn, ppo_done_fn done_fn,
                   ppo_event_fn on_train_begin, ppo_event_fn on_train_end);

#if USE_CONTINUOUS_ACTION
void     ppo_step(PPOAgent *agent, const float *obs, float *action_out);
#else
uint32_t ppo_step_discrete(PPOAgent *agent, const float *obs);
#endif

#endif
