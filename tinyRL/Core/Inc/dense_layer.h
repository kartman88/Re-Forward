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

/* Un layer denso vive in UNA sola allocazione contigua ("arena"), non in
 * 4*out_dim+9 blocchi separati come nella versione a float**.
 *
 * Layout dell'arena, tutto float, in quest'ordine:
 *     W  [out_dim*in_dim]  dW [out_dim*in_dim]  mW [out_dim*in_dim]  vW [out_dim*in_dim]
 *     b  [out_dim]  db [out_dim]  mb [out_dim]  vb [out_dim]  out [out_dim]
 *
 * Le matrici sono row-major: l'elemento (i,j) e' W[i*in_dim + j] e la riga i
 * inizia a W + i*in_dim. Righe contigue e nessun array di puntatori di riga:
 * il loop interno fa una load in meno per elemento e l'accesso e' sequenziale.
 *
 * Su questa topologia il guadagno e' particolarmente alto: il primo layer ha
 * in_dim = 4, cioe' righe da 16 byte che newlib-nano arrotonda a 24 con
 * l'header — il 50% di spreco sulla matrice piu' grande della rete.
 *
 * `W` e' anche il puntatore base dell'arena: dense_free() libera quello. */
typedef struct {
  // layer params
  int in_dim;
  int out_dim;
  float *W;    // [out_dim * in_dim] — base dell'arena
  float *b;    // [out_dim]
  float *out;  // [out_dim]
  ActivationType activation;
  // Adam moments
  float *mW;   // [out_dim * in_dim]
  float *vW;   // [out_dim * in_dim]
  float *dW;   // [out_dim * in_dim]
  float *db;   // [out_dim]
  float *mb;   // [out_dim]
  float *vb;   // [out_dim]
} DenseLayer;

void init_layer_params(DenseLayer *layer);
int  dense_init(DenseLayer *layer, int in_dim, int out_dim,
                ActivationType activation);
void dense_free(DenseLayer *layer);

#endif
