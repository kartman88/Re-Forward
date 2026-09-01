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

// Norma massima del gradiente globale (era una costante locale dentro
// network_clip_grad, invisibile da fuori).
#define GRAD_CLIP       0.5f

typedef struct {
    DenseLayer *layers;
    uint8_t     num_layers;
    uint32_t    adam_t;
} Network;

void softmax(float *in, float *out, int n);

int  network_init(Network *net, int num_layers, int *topology,
                  ActivationType *activations);
void network_free(Network *net);
/* Se `out` e' NULL l'uscita resta solo nel buffer `out` dell'ultimo layer,
 * evitando una memcpy quando al chiamante basta leggerla da li'. */
int  network_forward(Network *net, const float *input, float *out);
void network_zero_grad(Network *net);

/* Ritorna il fattore di scala da applicare ai gradienti (1.0 se il clip non
 * scatta, 0.0 se la norma non e' finita): non riscrive piu' i gradienti, li
 * scala network_adam_update al volo mentre li carica. */
float network_clip_grad(Network *net, float max_norm);
void  network_adam_update(Network *net, float lr, float gscale);

/* Backprop generico da un gradiente denso sull'uscita: e' la via di riferimento
 * del test di equivalenza, e la usano i due policy_backward. */
void  network_backward_from_delta(Network *net, const float *input,
                                  const float *delta_out);

// ── REINFORCE policy ──────────────────────────────────────────────────────────
// Il gradiente accumulato e' sempre quello della LOSS (da minimizzare):
//   L = -log pi(a|s) * G  -  ent_coef * H[pi]
// cosi' network_adam_update puo' fare discesa come in PPO/DQN, invece della
// risalita con il segno invertito della vecchia versione F446.
//
// `ret` ed `entropy_coef` arrivano gia' moltiplicati per 1/T (media
// sull'episodio): il fattore e' ripiegato nel delta di uscita invece di essere
// applicato a valle su tutti i parametri.
void  policy_forward(Network *policy, float *obs, float *probs_out,
                     float *log_prob_out, uint32_t action, float *entropy_out);
void  policy_backward(Network *policy, const float *obs, uint32_t action,
                      float ret, float entropy_coef);

#if USE_CONTINUOUS_ACTION
// Gaussiana diagonale a sigma fissa: la policy emette N_ACT_DIMS medie (mu),
// sigma vive in g_reinforce_sigma (reinforce.h).
// `log_prob_out` NULL salta del tutto il calcolo della log-probabilita': in
// REINFORCE nessuno la usa al campionamento (a differenza di PPO, che la
// conserva per il ratio), ed e' D divisioni + D logf per campione.
void  policy_forward_continuous(Network *policy, float *obs, float *action_out,
                                const float *sigma, float *log_prob_out);
// `inv_var[i] = 1/sigma_i^2`, precalcolato una volta per update: sigma e'
// costante per tutta la durata del training.
void  policy_backward_continuous(Network *policy, const float *obs,
                                 const float *action, const float *inv_var,
                                 float ret);
#endif

#endif
