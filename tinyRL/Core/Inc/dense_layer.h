#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H
#include "main.h"

typedef enum {
  ACT_NONE,
  ACT_RELU,
  ACT_SOFTMAX,
  ACT_TANH
  // future implementation, ACT_SIGMOID...
} ActivationType;

// Network datas and optimizer
typedef struct {
  // layer params
  int in_dim;
  int out_dim;
  float **W;
  float *b;
  float *out;
  ActivationType activation;
  // Adam moments
  float **mW;
  float **vW;
  float **dW;
  float *db;
  float *mb;
  float *vb;
} DenseLayer;

void init_layer_params(DenseLayer *layer);
int dense_init(DenseLayer *layer, int in_dim, int out_dim,
               ActivationType activation);
int alloc_2d(float ***matrix, int rows, int cols);
int free_2d(float ***matrix, int rows);

#endif
