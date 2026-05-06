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

static void adam_update_single_layer(DenseLayer *restrict ly,
                                      float b1t, float b2t, float lr) {
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
        ly->b[i]  -= lr * m_hat / (sqrtf(v_hat) + EPS_ADAM);
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
            w_row[j]  -= lr * m_hat / (sqrtf(v_hat) + EPS_ADAM);
            dw_row[j]  = 0.0f;
        }
    }
}

int network_init(Network *net, int num_layers, int *topology,
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

int network_forward(Network *net, float *input, float *out) {
    const float *curr_in = input;
    for (int l = 0; l < net->num_layers; l++) {
        forward_dense_layer(curr_in, net->layers[l].out, &net->layers[l]);
        curr_in = net->layers[l].out;
    }
    if (out)
        memcpy(out, curr_in,
               net->layers[net->num_layers - 1].out_dim * sizeof(float));
    return 1;
}

// Shared backprop engine: accumulates dW/db and propagates delta backward.
static void backward_from_delta(Network *net, float *input, float *delta_out) {
    static float   *delta_buf = NULL;
    static uint32_t delta_cap = 0;

    float *delta = delta_out;

    for (int l = net->num_layers - 1; l >= 0; --l) {
        DenseLayer *ly      = &net->layers[l];
        const int   in_dim  = ly->in_dim;
        const int   out_dim = ly->out_dim;
        const float *inp    = (l == 0) ? input : net->layers[l - 1].out;

        float *restrict db = ly->db;
        for (int i = 0; i < out_dim; ++i) {
            float di = delta[i];
            db[i] += di;
            float *restrict dw_row = ly->dW[i];
            for (int j = 0; j < in_dim; ++j)
                dw_row[j] += di * inp[j];
        }

        if (l > 0) {
            if ((uint32_t)in_dim > delta_cap) {
                free(delta_buf);
                delta_buf = malloc(in_dim * sizeof(float));
                delta_cap = (uint32_t)in_dim;
                if (!delta_buf) return;
            }

            const float   *h_prev = net->layers[l - 1].out;
            ActivationType act    = net->layers[l - 1].activation;

            for (int j = 0; j < in_dim; ++j) {
                float acc = 0.f;
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

// Actor backward — clipped surrogate + entropy gradient w.r.t. logits.
void actor_backward(Network *actor, float *obs, uint32_t action,
                    float advantage, float ratio, float clip_eps,
                    float entropy_coef) {
    int    last   = actor->num_layers - 1;
    int    n_acts = actor->layers[last].out_dim;
    float *probs  = actor->layers[last].out;

    int clipped = (advantage >= 0.f && ratio > 1.f + clip_eps) ||
                  (advantage <  0.f && ratio < 1.f - clip_eps);
    float w = clipped ? 0.f : ratio * advantage;

    float H = 0.f;
    for (int i = 0; i < n_acts; i++)
        H -= probs[i] * logf(probs[i]);

    static float delta_out[N_ACTIONS];
    for (int k = 0; k < n_acts; k++) {
        float ind          = (k == (int)action) ? 1.f : 0.f;
        float policy_grad  = -w * (ind - probs[k]);
        float entropy_grad = entropy_coef * probs[k] * (logf(probs[k]) + H);
        delta_out[k]       = policy_grad + entropy_grad;
    }

    backward_from_delta(actor, obs, delta_out);
}

// Critic backward — MSE value loss.
void critic_backward(Network *critic, float *obs, float value_target, float coeff) {
    float delta = coeff * (critic->layers[critic->num_layers - 1].out[0] - value_target);
    backward_from_delta(critic, obs, &delta);
}

void network_zero_grad(Network *net) {
    for (int l = 0; l < net->num_layers; l++) {
        DenseLayer *ly = &net->layers[l];
        for (int i = 0; i < ly->out_dim; i++)
            memset(ly->dW[i], 0, ly->in_dim * sizeof(float));
        memset(ly->db, 0, ly->out_dim * sizeof(float));
    }
}

void network_clip_grad(Network *net) {
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

void network_adam_update(Network *net, float lr) {
    net->adam_t++;
    float b1t = 1.f - powf(BETA1, (float)net->adam_t);
    float b2t = 1.f - powf(BETA2, (float)net->adam_t);
    for (int l = 0; l < net->num_layers; l++)
        adam_update_single_layer(&net->layers[l], b1t, b2t, lr);
}

#if USE_CONTINUOUS_ACTION

#define LOG_2PI_C  1.8378770664093453f   // log(2*pi)

void actor_forward_continuous(Network *actor, float *obs, float *action,
                              float *log_sigma,
                              float *log_prob_out, float *entropy_out) {
    int D = N_ACT_DIMS;
    network_forward(actor, obs, NULL);
    float *mu = actor->layers[actor->num_layers - 1].out;

    float log_prob = 0.f;
    float H        = 0.5f * (float)D * (1.f + LOG_2PI_C);
    for (int i = 0; i < D; i++) {
        float ls = log_sigma[i];
        H += ls;
#if PPO_USE_TANH_SQUASH
        // action[i] = tanh(z); recover z = atanh(action[i]) to evaluate Gaussian log_prob,
        // then subtract the log-abs-Jacobian: log(1 - tanh²(z)) = log(1 - a²).
        float a    = fmaxf(fminf(action[i], 1.f - 1e-6f), -1.f + 1e-6f);
        float z    = atanhf(a);
        float diff = z - mu[i];
        log_prob += -0.5f * (diff * diff / expf(2.f * ls) + 2.f * ls + LOG_2PI_C)
                    - logf(1.f - a * a + 1e-6f);
#else
        float diff  = action[i] - mu[i];
        log_prob += -0.5f * (diff * diff / expf(2.f * ls) + 2.f * ls + LOG_2PI_C);
#endif
    }
    *log_prob_out = log_prob;
    *entropy_out  = H;
}

void actor_backward_continuous(Network *actor, float *obs, float *action,
                               float *log_sigma,
                               float advantage, float ratio, float clip_eps) {
    int    D  = N_ACT_DIMS;
    float *mu = actor->layers[actor->num_layers - 1].out;

    int clipped = (advantage >= 0.f && ratio > 1.f + clip_eps) ||
                  (advantage <  0.f && ratio < 1.f - clip_eps);
    float w = clipped ? 0.f : ratio * advantage;

    static float delta_out[N_ACT_DIMS];
    for (int i = 0; i < D; i++) {
#if PPO_USE_TANH_SQUASH
        // action[i] = tanh(z); gradient of log π w.r.t. mu_i is (z - mu_i)/sigma_i²,
        // same form as the Gaussian case but with z = atanh(a) in place of the raw action.
        float a = fmaxf(fminf(action[i], 1.f - 1e-6f), -1.f + 1e-6f);
        float z = atanhf(a);
        delta_out[i] = -w * (z - mu[i]) / expf(2.f * log_sigma[i]);
#else
        delta_out[i] = -w * (action[i] - mu[i]) / expf(2.f * log_sigma[i]);
#endif
    }
    backward_from_delta(actor, obs, delta_out);
}

#endif  /* USE_CONTINUOUS_ACTION */

// Actor forward — computes softmax probs, log π(a|s), and entropy H[π]
void actor_forward(Network *actor, float *obs, float *probs_out,
                   float *log_prob_out, uint32_t action, float *entropy_out) {
    network_forward(actor, obs, probs_out);

    *log_prob_out = logf(probs_out[action]);

    float H = 0.f;
    int n = actor->layers[actor->num_layers - 1].out_dim;
    for (int i = 0; i < n; i++)
        H -= probs_out[i] * logf(probs_out[i]);
    *entropy_out = H;
}

// Critic forward — returns scalar V(s)
float critic_forward(Network *critic, float *obs) {
    network_forward(critic, obs, NULL);
    return critic->layers[critic->num_layers - 1].out[0];
}
