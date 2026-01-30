//#include "backprop.h"
#include "stm32h7xx_hal.h"
#include "main.h"
#include "dense_layer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline float frand(void) { return (float)rand() / RAND_MAX;}

int free_2d(float ***matrix, int rows) {
    if (matrix == NULL || *matrix == NULL || rows <= 0)
        return 0;

    for (int i = 0; i < rows; ++i) {
        if ((*matrix)[i] != NULL) {
            free((*matrix)[i]);
            (*matrix)[i] = NULL;
        }
    }

    free(*matrix);
    *matrix = NULL;
    return 1;
}

int alloc_2d(float ***matrix, int rows, int cols){
    if (rows <= 0 || cols <= 0 || matrix == NULL) return 0;

    *matrix = malloc(rows * sizeof(float *));
    if (*matrix == NULL) return 0;

    for (int i = 0; i < rows; ++i) {
        (*matrix)[i] = calloc(cols, sizeof(float));
        if ((*matrix)[i] == NULL) {
            //deallocate
            for (int j = 0; j < i; ++j)
                free((*matrix)[j]);
            free(*matrix);
            *matrix = NULL;
            return 0;
        }
    }

    return 1;
}

void init_layer_params(DenseLayer *layer){
    for (int i = 0; i < layer->out_dim; ++i) {
        for (int j = 0; j < layer->in_dim; ++j){
            layer->W[i][j] = 0.05f * (frand() - 0.5f);
        	layer->dW[i][j] = 0.0f;
            layer->mW[i][j] = 0.0f;
            layer->vW[i][j] = 0.0f;
        }
        layer->b[i] = 0.0f;
        layer->db[i] = 0.0f;
        layer->mb[i] = 0.0f;
        layer->vb[i] = 0.0f;
    }
}


int dense_init(DenseLayer *layer, int in_dim, int out_dim, ActivationType activation){
    layer->in_dim = in_dim;
    layer->out_dim = out_dim;
    layer->activation = activation;

    //allocations
    if (!alloc_2d(&layer->W, out_dim, in_dim)) return 0;
    if (!alloc_2d(&layer->dW, out_dim, in_dim)) return 0;
    if (!alloc_2d(&layer->mW, out_dim, in_dim)) return 0;
    if (!alloc_2d(&layer->vW, out_dim, in_dim)) return 0;

    layer->b  = malloc(out_dim * sizeof(float));
    layer->out = malloc(out_dim * sizeof(float));
    layer->db = malloc(out_dim * sizeof(float));
    layer->mb = malloc(out_dim * sizeof(float));
    layer->vb = malloc(out_dim * sizeof(float));
    if (!layer->b || !layer->db || !layer->mb || !layer->vb || !layer->out) return 0;

    init_layer_params(layer);
    return 1;
}
