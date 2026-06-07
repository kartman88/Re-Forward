#include "neural_net.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void softmax(float *in, float *out, int n) {
    float max = in[0];
    for (int i = 1; i < n; ++i)
        if (in[i] > max)
            max = in[i];
    float sum = 0.f;
    for (int i = 0; i < n; ++i) {
        out[i] = expf(in[i] - max);
        sum += out[i];
    }
    float inv = 1.f / sum;
    for (int i = 0; i < n; ++i) {
        out[i] *= inv;
        if (out[i] < 1e-7f)
            out[i] = 1e-7f;
    }
}

static inline void forward_dense_layer(const float *restrict in_vec,
                                        float *restrict out_vec,
                                        const DenseLayer *restrict layer) {
    const int in_dim  = layer->in_dim;
    const int out_dim = layer->out_dim;
    for (int i = 0; i < out_dim; ++i) {
        float acc = layer->b[i];
        const float *restrict w_row = layer->W[i];
        for (int j = 0; j < in_dim; ++j)
            acc += w_row[j] * in_vec[j];
        switch (layer->activation) {
        case ACT_RELU: out_vec[i] = (acc > 0.f) ? acc : 0.f; break;
        case ACT_TANH: out_vec[i] = tanhf(acc);               break;
        default:       out_vec[i] = acc;                       break;
        }
    }
    if (layer->activation == ACT_SOFTMAX)
        softmax(out_vec, out_vec, out_dim);
}

static inline void forward_target_layer(const float *restrict in_vec,
                                         float *restrict out_vec,
                                         const TargetLayer *restrict layer) {
    const int in_dim  = layer->in_dim;
    const int out_dim = layer->out_dim;
    for (int i = 0; i < out_dim; ++i) {
        float acc = layer->b[i];
        const float *restrict w_row = layer->W[i];
        for (int j = 0; j < in_dim; ++j)
            acc += w_row[j] * in_vec[j];
        switch (layer->activation) {
        case ACT_RELU: out_vec[i] = (acc > 0.f) ? acc : 0.f; break;
        case ACT_TANH: out_vec[i] = tanhf(acc);               break;
        default:       out_vec[i] = acc;                       break;
        }
    }
    if (layer->activation == ACT_SOFTMAX)
        softmax(out_vec, out_vec, out_dim);
}

static void adam_update_single_layer(DenseLayer *restrict ly,
                                      float b1t, float b2t) {
    const int out_dim = ly->out_dim;
    const int in_dim  = ly->in_dim;

    for (int i = 0; i < out_dim; ++i) {
        float db = ly->db[i];
        float mb = BETA1 * ly->mb[i] + (1.f - BETA1) * db;
        float vb = BETA2 * ly->vb[i] + (1.f - BETA2) * db * db;
        ly->mb[i] = mb;
        ly->vb[i] = vb;
        float m_hat = mb / b1t;
        float v_hat = vb / b2t;
        ly->b[i]  -= LR * m_hat / (sqrtf(v_hat) + EPS_ADAM);
        ly->db[i]  = 0.0f;
    }

    for (int i = 0; i < out_dim; ++i) {
        float *restrict mw_row = ly->mW[i];
        float *restrict vw_row = ly->vW[i];
        float *restrict  w_row = ly->W[i];
        float *restrict dw_row = ly->dW[i];
        for (int j = 0; j < in_dim; ++j) {
            float dw   = dw_row[j];
            float mw   = BETA1 * mw_row[j] + (1.f - BETA1) * dw;
            float vw   = BETA2 * vw_row[j] + (1.f - BETA2) * dw * dw;
            mw_row[j]  = mw;
            vw_row[j]  = vw;
            float m_hat = mw / b1t;
            float v_hat = vw / b2t;
            w_row[j]  -= LR * m_hat / (sqrtf(v_hat) + EPS_ADAM);
            dw_row[j]  = 0.0f;
        }
    }
}

int init_qnetwork(QNetwork *net, int num_layers, int *topology,
                  ActivationType *activations) {
    net->adam_t     = 0;
    int n_weights   = num_layers - 1;
    net->num_layers = (uint8_t)n_weights;
    net->layers     = malloc(n_weights * sizeof(DenseLayer));
    if (!net->layers)
        return 0;
    for (int i = 0; i < n_weights; i++) {
        if (!dense_init(&net->layers[i], topology[i], topology[i + 1],
                        activations[i]))
            return 0;
        init_layer_params(&net->layers[i]);
    }
    return 1;
}

int init_target_network(TargetNetwork *tgt, int num_layers, int *topology) {
    int n_weights   = num_layers - 1;
    tgt->num_layers = (uint8_t)n_weights;
    tgt->layers     = malloc(n_weights * sizeof(TargetLayer));
    if (!tgt->layers)
        return 0;
    for (int i = 0; i < n_weights; i++) {
        int in_d  = topology[i];
        int out_d = topology[i + 1];
        tgt->layers[i].in_dim     = in_d;
        tgt->layers[i].out_dim    = out_d;
        tgt->layers[i].activation = ACT_NONE;
        if (!alloc_2d(&tgt->layers[i].W, out_d, in_d))
            return 0;
        tgt->layers[i].b   = calloc(out_d, sizeof(float));
        tgt->layers[i].out = malloc(out_d * sizeof(float));
        if (!tgt->layers[i].b || !tgt->layers[i].out)
            return 0;
    }
    return 1;
}

int forward_q(QNetwork *net, float *input, float *q_out) {
    const float *curr_in = input;
    for (int l = 0; l < net->num_layers; l++) {
        forward_dense_layer(curr_in, net->layers[l].out, &net->layers[l]);
        curr_in = net->layers[l].out;
    }
    if (q_out)
        memcpy(q_out, curr_in,
               net->layers[net->num_layers - 1].out_dim * sizeof(float));
    return 1;
}

int forward_target(TargetNetwork *tgt, float *input, float *q_out) {
    const float *curr_in = input;
    for (int l = 0; l < tgt->num_layers; l++) {
        forward_target_layer(curr_in, tgt->layers[l].out, &tgt->layers[l]);
        curr_in = tgt->layers[l].out;
    }
    if (q_out)
        memcpy(q_out, curr_in,
               tgt->layers[tgt->num_layers - 1].out_dim * sizeof(float));
    return 1;
}

void dqn_backward(QNetwork *net, float *input, uint32_t action,
                  float td_error) {
    // Sparse output gradient: only position 'action' is non-zero
    static float delta_out[N_ACTIONS];
    static float *delta_buf = NULL;
    static uint32_t delta_cap = 0;

    int out_dim_last = net->layers[net->num_layers - 1].out_dim;
    for (int i = 0; i < out_dim_last; i++)
        delta_out[i] = 0.0f;
    delta_out[action] = td_error;

    float *delta = delta_out;

    for (int l = net->num_layers - 1; l >= 0; --l) {
        DenseLayer *ly       = &net->layers[l];
        const int   in_dim   = ly->in_dim;
        const int   out_dim  = ly->out_dim;
        const float *inp     = (l == 0) ? input : net->layers[l - 1].out;

        // Accumulate gradients for this layer
        float *restrict db = ly->db;
        for (int i = 0; i < out_dim; ++i) {
            float di = delta[i];
            db[i] += di;
            float *restrict dw_row = ly->dW[i];
            for (int j = 0; j < in_dim; ++j)
                dw_row[j] += di * inp[j];
        }

        // Propagate delta to the layer below (not needed at input layer)
        if (l > 0) {
            if ((uint32_t)in_dim > delta_cap) {
                free(delta_buf);
                delta_buf = malloc(in_dim * sizeof(float));
                delta_cap = (uint32_t)in_dim;
                if (!delta_buf) return;
            }

            const float    *h_prev = net->layers[l - 1].out;
            ActivationType  act    = net->layers[l - 1].activation;

            for (int j = 0; j < in_dim; ++j) {
                float acc = 0.0f;
                for (int i = 0; i < out_dim; ++i)
                    acc += delta[i] * ly->W[i][j];
                float hp = h_prev[j];
                switch (act) {
                case ACT_RELU: acc = (hp > 0.f) ? acc : 0.f; break;
                case ACT_TANH: acc = acc * (1.f - hp * hp);  break;
                default: break;
                }
                delta_buf[j] = acc;
            }
            delta = delta_buf;
        }
    }
}

void copy_weights_to_target(QNetwork *src, TargetNetwork *dst) {
    for (int l = 0; l < src->num_layers; l++) {
        DenseLayer  *sl = &src->layers[l];
        TargetLayer *tl = &dst->layers[l];
        tl->activation  = sl->activation;
        for (int i = 0; i < sl->out_dim; i++)
            memcpy(tl->W[i], sl->W[i], sl->in_dim * sizeof(float));
        memcpy(tl->b, sl->b, sl->out_dim * sizeof(float));
    }
}

void zero_grad_q(QNetwork *net) {
    for (int l = 0; l < net->num_layers; l++) {
        DenseLayer *ly = &net->layers[l];
        for (int i = 0; i < ly->out_dim; i++)
            memset(ly->dW[i], 0, ly->in_dim * sizeof(float));
        memset(ly->db, 0, ly->out_dim * sizeof(float));
    }
}

void gradient_norm_q(QNetwork *net) {
    float gnorm_sq = 0.f;
    for (int l = 0; l < net->num_layers; l++) {
        DenseLayer *ly = &net->layers[l];
        for (int i = 0; i < ly->out_dim; i++) {
            gnorm_sq += ly->db[i] * ly->db[i];
            for (int j = 0; j < ly->in_dim; j++)
                gnorm_sq += ly->dW[i][j] * ly->dW[i][j];
        }
    }
    const float CLIP  = 0.5f;
    float gnorm = sqrtf(gnorm_sq);
    if (!isfinite(gnorm)) {
        // gradiente non finito: scarta l'update azzerando i gradienti
        for (int l = 0; l < net->num_layers; l++) {
            DenseLayer *ly = &net->layers[l];
            memset(ly->db, 0, ly->out_dim * sizeof(float));
            for (int i = 0; i < ly->out_dim; i++)
                memset(ly->dW[i], 0, ly->in_dim * sizeof(float));
        }
        return;
    }
    if (gnorm > CLIP) {
        float s = CLIP / gnorm;
        for (int l = 0; l < net->num_layers; l++) {
            DenseLayer *ly = &net->layers[l];
            for (int i = 0; i < ly->out_dim; i++) {
                ly->db[i] *= s;
                for (int j = 0; j < ly->in_dim; j++)
                    ly->dW[i][j] *= s;
            }
        }
    }
}

void adam_optimizer_q(QNetwork *net) {
    net->adam_t++;
    float b1t = 1.f - powf(BETA1, (float)net->adam_t);
    float b2t = 1.f - powf(BETA2, (float)net->adam_t);
    for (int l = 0; l < net->num_layers; l++)
        adam_update_single_layer(&net->layers[l], b1t, b2t);
}
