#ifndef NEURAL_NET_H
#define NEURAL_NET_H
#include "dense_layer.h"

#define LR 0.02f //0.01 and 0.02 good for CartPole
#define BETA1 0.9f
#define BETA2 0.999f
#define EPS_ADAM 1e-8f
#define GAMMA 0.99f
#define ENT_BETA 0.01f //0.001 good for cartpole
#define MAX_EPISODE 100
#define MAX_STEPS 500

typedef struct {
    DenseLayer *layers;
    int num_layers;
    uint32_t adam_t; //adam steps counter
} NeuralNet;

int init_network(NeuralNet *net, int num_layers, int *net_topology, ActivationType *activations);
void softmax(float *in, float *out, int n);
int forward(NeuralNet *net, float *input, float *output_final);
void backward_core(NeuralNet *net, float *dout_last, float *input);
void backward_pg(NeuralNet *net, float *input, uint8_t action, float advantage, float reward, uint32_t step_count);
void adam_optimizer(NeuralNet *net);
void gradient_norm_l2(NeuralNet *net);
void zero_grad(NeuralNet *net);

#endif
