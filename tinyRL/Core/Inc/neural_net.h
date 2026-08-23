#ifndef NEURAL_NET_H
#define NEURAL_NET_H
#include "dense_layer.h"
#include <math.h>
#include <stdint.h>

// DQN Hyperparameters — tuned for Pendulum-v1 (5 discrete torques)
#define LR              0.001f
#define BETA1           0.9f
#define BETA2           0.999f
#define EPS_ADAM        1e-8f
#define GAMMA           0.99f
#define EPSILON_START   1.0f
#define EPSILON_END     0.05f
#define EPSILON_DECAY   10000
#define TARGET_UPDATE   100
#define REPLAY_MIN      300
#define REPLAY_SIZE     1000
#define BATCH_SIZE      32
#define MAX_EPISODE     1000
#define N_ACTIONS       5       // {-2, -1, 0, +1, +2} Nm
#define OBS_DIM         3       // [cos_theta, sin_theta, theta_dot]

/* Norma massima del gradiente globale (era una costante locale dentro
 * gradient_norm_q, invisibile da fuori). */
#define GRAD_CLIP       0.5f

typedef struct {
    DenseLayer *layers;
    uint8_t     num_layers;
    uint32_t    adam_t;
} QNetwork;

/* La rete target e' di sola lettura: le servono W, b e out, non gradienti ne'
 * momenti. Stesso trattamento del DenseLayer: una sola allocazione contigua
 *     arena = [ W (out_dim*in_dim) | b (out_dim) | out (out_dim) ]
 * con W row-major, invece di un array di puntatori di riga piu' una calloc per
 * riga. `W` e' la base dell'arena, la libera target_free(). */
typedef struct {
    float          *W;      // [out_dim * in_dim] — base dell'arena
    float          *b;      // [out_dim]
    float          *out;    // [out_dim]
    int             in_dim;
    int             out_dim;
    ActivationType  activation;
} TargetLayer;

typedef struct {
    TargetLayer *layers;
    uint8_t      num_layers;
} TargetNetwork;

void softmax(float *in, float *out, int n);

int  init_qnetwork(QNetwork *net, int num_layers, int *topology,
                   ActivationType *activations);
int  init_target_network(TargetNetwork *tgt, int num_layers, int *topology);
void qnetwork_free(QNetwork *net);
void target_free(TargetNetwork *tgt);

/* forward_*: se q_out e' NULL l'uscita resta solo nel buffer `out` dell'ultimo
 * layer, evitando una memcpy quando al chiamante basta leggerla da li'. */
int  forward_q(QNetwork *net, const float *input, float *q_out);
int  forward_target(TargetNetwork *tgt, const float *input, float *q_out);
void copy_weights_to_target(QNetwork *src, TargetNetwork *dst);

/* Backprop generico da un gradiente denso sull'uscita: e' la via di riferimento
   del test di equivalenza. */
void dqn_backward_delta(QNetwork *net, const float *input,
                        const float *delta_out);

/* Gradiente sul solo Q(s,a) eseguito: il vettore di uscita e' nullo ovunque
   tranne in `action`, e il motore salta le righe moltiplicate per zero. */
void dqn_backward(QNetwork *net, const float *input, uint32_t action,
                  float td_error);

/* Ritorna il fattore di scala da applicare ai gradienti (1.0 se il clip non
 * scatta, 0.0 se la norma non e' finita): non riscrive piu' i gradienti, li
 * scala adam_optimizer_q al volo mentre li carica. */
float gradient_norm_q(QNetwork *net, float max_norm);
void  adam_optimizer_q(QNetwork *net, float lr, float gscale);
void  zero_grad_q(QNetwork *net);

#endif
