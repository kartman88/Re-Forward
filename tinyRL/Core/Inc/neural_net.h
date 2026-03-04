#ifndef NEURAL_NET_H
#define NEURAL_NET_H
#include "dense_layer.h"
#include <math.h>
#include <stdint.h>

#define LR 0.0005f // 0.003 good for CartPole
#define BETA1 0.9f
#define BETA2 0.999f
#define EPS_ADAM 1e-8f
#define GAMMA 0.99f
#define ENT_BETA                                                               \
  0.005f // L2 penalty on mu to prevent tanh saturation (exploration)
#define STARTING_ACTION_SIGMA 0.8f // Initial std dev for continuous actions
#define CRIT_LOSS 0.5
#define MAX_EPISODE 2000
#define MAX_STEPS 2048
#define BATCH_SIZE 64
#define ROLLOUT 200

#define N_EPOCHS 10
// Total expected adam_optimizer() calls over the full training run.
// adam_t increments once per mini-batch (not per env step), so the correct
// denominator for sigma decay is: episodes × (buffer / batch) × epochs.
#define TOTAL_ADAM_STEPS (MAX_EPISODE * (MAX_STEPS / BATCH_SIZE) * N_EPOCHS)
#define EPS_CLIPPING 0.2f
#define CRITIC_COEFF 0.5F
// #define ENTROPY_W 0.001 //0.001 good for CartPole
#define PPO_EPSILON 0.2 // 0.2

#define USE_CONTINUOUS_ACTIONS                                                 \
  1 // 1=Continuo (es. Pendulum), 0=Discreto (es. CartPole)

#if USE_CONTINUOUS_ACTIONS
typedef float action_t;
#else
typedef uint8_t action_t;
#endif

typedef struct {
  DenseLayer *layers;
  uint8_t num_layers;
} Head;

typedef struct {
  DenseLayer *layers;
  uint8_t num_layers;
  uint32_t adam_t; // adam steps counter
  Head actor;
  Head critic;
} SharedBackbone;

int init_network(SharedBackbone *net, int num_layers, int num_layers_actor,
                 int num_layers_critic, int *net_topology,
                 int *net_topology_actor, int *net_topology_critic,
                 ActivationType *activations, ActivationType *activations_actor,
                 ActivationType *activations_critic);
void softmax(float *in, float *out, int n);
int forward(SharedBackbone *net, float *input, float *output_actor,
            float *output_critic);
void backward_core(SharedBackbone *net, float *dout_last, float *input);
void backward_pg(SharedBackbone *net, float *input, action_t action,
                 float advantage, float reward, uint32_t step_count);
void adam_optimizer(SharedBackbone *net);
void gradient_norm_l2(SharedBackbone *net);
void zero_grad(SharedBackbone *net);

#endif
