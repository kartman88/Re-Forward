#ifndef NEURAL_NET_H
#define NEURAL_NET_H
#include "dense_layer.h"
#include <math.h>
#include <stdint.h>

// ── Action mode ────────────────────────────────────────────────────────────────
// 0 = discrete (categorical softmax)
// 1 = continuous (diagonal Gaussian, state-independent std); actor outputs only mu_0..mu_{D-1}
#define USE_CONTINUOUS_ACTION   1

// ── Tanh squashing (continuous mode only) ──────────────────────────────────────
// 0 = standard Gaussian (clip externally if needed) — original behaviour
// 1 = squashed Gaussian: a = tanh(z), log_prob corrected with Jacobian
//     Actions are bounded in (-1, 1) without any external clipping.
//     Fixes the boundary-saturation issue at the cost of atanhf() calls in training.
#define PPO_USE_TANH_SQUASH     1   //1 to enable Tanh squashing of continuous actions
#define N_ACT_DIMS              3     // continuous: number of independent action dims (Hopper: hip, knee, ankle)

// ── Network / optimizer constants ──────────────────────────────────────────────
#define BETA1           0.9f
#define BETA2           0.999f
#define EPS_ADAM        1e-8f
#define N_ACTIONS       5       // discrete mode only (unused for Hopper)
#define OBS_DIM         11      // Hopper-v4: [z, torso_angle, thigh, leg, foot, vx, vz, v_torso, v_thigh, v_leg, v_foot]

// Limite sull'uscita dell'actor (mu, pre-tanh). tanh(8)=0.99997: nessuna perdita
// di espressivita', ma spezza il feedback che farebbe esplodere i pesi in float32.
// Usando fminf/fmaxf, un eventuale mu NaN viene sanificato a +-MU_CLAMP.
#define MU_CLAMP        8.0f

// log(2*pi) — usata dalla log-likelihood gaussiana in neural_net.c e ppo.c.
#define LOG_2PI         1.8378770664093453f

typedef struct {
    DenseLayer *layers;
    uint8_t     num_layers;
    uint32_t    adam_t;
} Network;

void softmax(float *in, float *out, int n);

int  network_init(Network *net, int num_layers, int *topology,
                  ActivationType *activations);
int  network_forward(Network *net, float *input, float *out);
void  network_zero_grad(Network *net);
// Ritorna il fattore di scala da passare a network_adam_update (1.0 = nessun
// clipping, 0.0 = gradiente non finito, update da scartare).
float network_clip_grad(Network *net, float max_norm);
void  network_adam_update(Network *net, float lr, float gscale);

// PPO discrete
void  actor_forward(Network *actor, float *obs, float *probs_out,
                    float *log_prob_out, uint32_t action, float *entropy_out);
float critic_forward(Network *critic, float *obs);
void  actor_backward(Network *actor, float *obs, uint32_t action,
                     float advantage, float ratio, float clip_eps,
                     float entropy_coef);
void  critic_backward(Network *critic, float *obs, float value_target, float coeff);

#if USE_CONTINUOUS_ACTION
// PPO continuous diagonal Gaussian, state-independent std.
// Actor outputs N_ACT_DIMS means (μ); σ lives in g_ppo_log_sigma (ppo.h).
// `z` e' l'azione pre-squash presa dal rollout buffer; `var`/`log_var` sono
// precalcolati una volta per update (sigma e' costante durante l'update).
// Il bonus di entropia (PPO_C2) NON e' applicato in modalita' continua:
// l'esplorazione e' governata dallo schedule di sigma (ppo_sigma_decay), non
// da un termine nella loss. Vedi il commento su PPO_C2 in ppo.h.
void  actor_forward_continuous(Network *actor, float *obs, const float *z,
                               const float *var, const float *log_var,
                               float *log_prob_out);
void  actor_backward_continuous(Network *actor, float *obs, const float *z,
                                const float *var,
                                float advantage, float ratio, float clip_eps);
#endif

#endif
