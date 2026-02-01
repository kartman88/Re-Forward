#ifndef NEURAL_NET_H
#define NEURAL_NET_H
#include "dense_layer.h"

#define LR 0.02f //0.01 and 0.02 good for CartPole
#define BETA1 0.9f
#define BETA2 0.999f
#define EPS_ADAM 1e-8f
#define GAMMA 0.99f
#define ENT_BETA 0.01f //0.001 good for cartpole
#define CRIT_LOSS 0.5
#define MAX_EPISODE 100
#define MAX_STEPS 500

#define N_EPOCHS 3
#define EPS_CLIPPING 0.2f
#define VALUE_W 0.5F
#define ENTROPY_W 0.01

typedef struct{
	DenseLayer *layers;
	uint8_t num_layers;
} Head;

typedef struct{
    DenseLayer *layers;
    uint8_t num_layers;
    uint32_t adam_t; //adam steps counter
    Head actor;
    Head critic;
} SharedBackbone;

int init_network(SharedBackbone *net, int num_layers, int num_layers_actor, int num_layers_critic,
		int *net_topology, int *net_topology_actor, int *net_topology_critic,
		ActivationType *activations, ActivationType *activations_actor, ActivationType *activations_critic);
void softmax(float *in, float *out, int n);
int forward(SharedBackbone *net, float *input, float *output_actor, float *output_critic);
void backward_core(SharedBackbone *net, float *dout_last, float *input);
void backward_pg(SharedBackbone *net, float *input, uint8_t action, float advantage, float reward, uint32_t step_count);
void adam_optimizer(SharedBackbone *net);
void gradient_norm_l2(SharedBackbone *net);
void zero_grad(SharedBackbone *net);

#endif
