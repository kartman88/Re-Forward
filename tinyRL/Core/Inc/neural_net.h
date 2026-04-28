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

typedef struct {
    DenseLayer *layers;
    uint8_t     num_layers;
    uint32_t    adam_t;
} QNetwork;

typedef struct {
    float         **W;
    float          *b;
    float          *out;
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
int  forward_q(QNetwork *net, float *input, float *q_out);
int  forward_target(TargetNetwork *tgt, float *input, float *q_out);
void copy_weights_to_target(QNetwork *src, TargetNetwork *dst);
void dqn_backward(QNetwork *net, float *input, uint32_t action,
                  float td_error);
void adam_optimizer_q(QNetwork *net);
void gradient_norm_q(QNetwork *net);
void zero_grad_q(QNetwork *net);

#endif
