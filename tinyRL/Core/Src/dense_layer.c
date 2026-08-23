#include "dense_layer.h"
#include "rng.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void init_layer_params(DenseLayer *layer) {
    const int in_dim  = layer->in_dim;
    const int out_dim = layer->out_dim;
    const int n_w     = out_dim * in_dim;

    // Limite di Xavier (Glorot): sqrt(6 / (in_dim + out_dim)).
    // Mantiene stabile la varianza del segnale attraverso i layer.
    const float limit = sqrtf(6.0f / (float)(in_dim + out_dim));

    // rng_uniform() -> [0,1);  (u - 0.5) * 2 * limit -> [-limit, +limit)
    for (int k = 0; k < n_w; ++k)
        layer->W[k] = (rng_uniform() - 0.5f) * 2.0f * limit;

    // Gradienti e momenti Adam a zero: dW, mW, vW sono contigui subito dopo W.
    memset(layer->dW, 0, 3 * (size_t)n_w * sizeof(float));

    // Bias a zero, insieme ai loro gradienti/momenti (b, db, mb, vb contigui).
    memset(layer->b, 0, 4 * (size_t)out_dim * sizeof(float));
}

int dense_init(DenseLayer *layer, int in_dim, int out_dim,
               ActivationType activation) {
    if (in_dim <= 0 || out_dim <= 0) return 0;

    layer->in_dim     = in_dim;
    layer->out_dim    = out_dim;
    layer->activation = activation;

    // Una sola allocazione per l'intero layer: 4 matrici + 5 vettori.
    const size_t n_w     = (size_t)out_dim * (size_t)in_dim;
    const size_t n_total = 4u * n_w + 5u * (size_t)out_dim;

    float *arena = malloc(n_total * sizeof(float));
    if (!arena) return 0;

    layer->W   = arena;              // base dell'arena, la libera dense_free()
    layer->dW  = layer->W  + n_w;
    layer->mW  = layer->dW + n_w;
    layer->vW  = layer->mW + n_w;

    float *v   = arena + 4u * n_w;
    layer->b   = v;                  // init_layer_params azzera b..vb in blocco,
    layer->db  = v +      out_dim;   // quindi quest'ordine non va cambiato senza
    layer->mb  = v + 2u * out_dim;   // aggiornare anche la memset la' dentro
    layer->vb  = v + 3u * out_dim;
    layer->out = v + 4u * out_dim;

    init_layer_params(layer);
    return 1;
}

void dense_free(DenseLayer *layer) {
    if (!layer || !layer->W) return;
    free(layer->W);                  // libera l'intera arena in un colpo
    layer->W = layer->dW = layer->mW = layer->vW = NULL;
    layer->b = layer->db = layer->mb = layer->vb = layer->out = NULL;
}
