#ifndef NEURAL_NET_H
#define NEURAL_NET_H
#include "dense_layer.h"
#include <math.h>
#include <stdint.h>

// ── Action mode ────────────────────────────────────────────────────────────────
// 0 = discrete (categorical softmax)
// 1 = continuous (diagonal Gaussian, state-independent std); actor outputs only mu_0..mu_{D-1}
#define USE_CONTINUOUS_ACTION   1
#define N_ACT_DIMS              1     // continuous: number of independent action dims

// ── Network / optimizer constants ──────────────────────────────────────────────
#define BETA1           0.9f
#define BETA2           0.999f
#define EPS_ADAM        1e-8f
#define N_ACTIONS       5       // {-2, -1, 0, +1, +2} Nm  (discrete mode only)
#define OBS_DIM         3       // [cos_theta, sin_theta, theta_dot]

typedef struct {
    DenseLayer *layers;
    uint8_t     num_layers;
    uint32_t    adam_t;
} Network;

void softmax(float *in, float *out, int n);

int  network_init(Network *net, int num_layers, int *topology,
                  ActivationType *activations);
int  network_forward(Network *net, float *input, float *out);
void network_zero_grad(Network *net);
void network_clip_grad(Network *net);
void network_adam_update(Network *net, float lr);

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
void  actor_forward_continuous(Network *actor, float *obs, float *action,
                               float *log_sigma,
                               float *log_prob_out, float *entropy_out);
void  actor_backward_continuous(Network *actor, float *obs, float *action,
                                float *log_sigma,
                                float advantage, float ratio, float clip_eps);
#endif

#endif
