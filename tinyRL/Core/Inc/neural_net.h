#ifndef NEURAL_NET_H
#define NEURAL_NET_H
#include "dense_layer.h"
#include <math.h>
#include <stdint.h>

// ── Action mode ────────────────────────────────────────────────────────────────
// 0 = discrete (categorical softmax)
// 1 = continuous (diagonal Gaussian with fixed std); the policy outputs only
//     mu_0..mu_{D-1}
#define USE_CONTINUOUS_ACTION   0

#define N_ACT_DIMS              1     // continuous: numero di dim di azione indipendenti

// ── Network / optimizer constants ──────────────────────────────────────────────
#define BETA1           0.9f
#define BETA2           0.999f
#define EPS_ADAM        1e-8f
#define N_ACTIONS       2       // discrete mode: CartPole-v1 -> {sinistra, destra}
#define OBS_DIM         4       // CartPole-v1: [x, x_dot, theta, theta_dot]

// Limite sull'uscita della policy continua (mu). Non toglie espressivita' ma
// spezza il feedback che farebbe esplodere i pesi in float32; usando fminf/fmaxf
// un eventuale mu NaN viene sanificato a +-MU_CLAMP.
#define MU_CLAMP        8.0f

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
void network_scale_grad(Network *net, float scale);
void network_clip_grad(Network *net);
void network_adam_update(Network *net, float lr);

// ── REINFORCE policy ──────────────────────────────────────────────────────────
// Il gradiente accumulato e' sempre quello della LOSS (da minimizzare):
//   L = -log pi(a|s) * G  -  ent_coef * H[pi]
// cosi' network_adam_update puo' fare discesa come in PPO/DQN, invece della
// risalita con il segno invertito della vecchia versione F446.
void  policy_forward(Network *policy, float *obs, float *probs_out,
                     float *log_prob_out, uint32_t action, float *entropy_out);
void  policy_backward(Network *policy, float *obs, uint32_t action,
                      float ret, float entropy_coef);

#if USE_CONTINUOUS_ACTION
// Gaussiana diagonale a sigma fissa: la policy emette N_ACT_DIMS medie (mu),
// sigma vive in g_reinforce_sigma (reinforce.h).
void  policy_forward_continuous(Network *policy, float *obs, float *action_out,
                                const float *sigma, float *log_prob_out);
void  policy_backward_continuous(Network *policy, float *obs,
                                 const float *action, const float *sigma,
                                 float ret);
#endif

#endif
