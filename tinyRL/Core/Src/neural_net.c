#include "neural_net.h"
#include "rng.h"
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
        case ACT_TANH: out_vec[i] = tanhf(acc);              break;
        default:       out_vec[i] = acc;                     break;
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

void network_zero_grad(Network *net) {
    for (int l = 0; l < net->num_layers; l++) {
        DenseLayer *ly = &net->layers[l];
        for (int i = 0; i < ly->out_dim; i++)
            memset(ly->dW[i], 0, ly->in_dim * sizeof(float));
        memset(ly->db, 0, ly->out_dim * sizeof(float));
    }
}

/* Media del gradiente sull'episodio: REINFORCE accumula un contributo per ogni
 * step, quindi l'ampiezza dipenderebbe dalla lunghezza dell'episodio (per
 * CartPole varia da ~10 a 500 step). scale = 1/step_count rende l'update
 * indipendente dalla durata. */
void network_scale_grad(Network *net, float scale) {
    for (int l = 0; l < net->num_layers; l++) {
        DenseLayer *ly = &net->layers[l];
        for (int i = 0; i < ly->out_dim; i++) {
            ly->db[i] *= scale;
            for (int j = 0; j < ly->in_dim; j++)
                ly->dW[i][j] *= scale;
        }
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
    if (!isfinite(gnorm)) {
        // Gradiente non finito (Inf/NaN): scarta l'update azzerando i gradienti
        // per non scrivere NaN nei pesi. Rete di sicurezza, non dovrebbe scattare.
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

void network_adam_update(Network *net, float lr) {
    net->adam_t++;
    float b1t = 1.f - powf(BETA1, (float)net->adam_t);
    float b2t = 1.f - powf(BETA2, (float)net->adam_t);
    for (int l = 0; l < net->num_layers; l++)
        adam_update_single_layer(&net->layers[l], b1t, b2t, lr);
}

// ── Policy discreta (categorical softmax) ────────────────────────────────────

// Forward — softmax probs, log pi(a|s) ed entropia H[pi]
void policy_forward(Network *policy, float *obs, float *probs_out,
                    float *log_prob_out, uint32_t action, float *entropy_out) {
    network_forward(policy, obs, probs_out);

    if (log_prob_out)
        *log_prob_out = logf(probs_out[action]);

    if (entropy_out) {
        float H = 0.f;
        int n = policy->layers[policy->num_layers - 1].out_dim;
        for (int i = 0; i < n; i++)
            H -= probs_out[i] * logf(probs_out[i]);
        *entropy_out = H;
    }
}

// Backward — gradiente della loss REINFORCE rispetto ai logit:
//   L      = -log pi(a|s) * G  -  ent_coef * H[pi]
//   dL/dz_k = -G * (1{k=a} - p_k)  +  ent_coef * p_k * (log p_k + H)
// Richiede che l'ultimo forward sulla rete sia stato fatto con questa obs.
void policy_backward(Network *policy, float *obs, uint32_t action,
                     float ret, float entropy_coef) {
    int    last   = policy->num_layers - 1;
    int    n_acts = policy->layers[last].out_dim;
    float *probs  = policy->layers[last].out;

    float H = 0.f;
    for (int i = 0; i < n_acts; i++)
        H -= probs[i] * logf(probs[i]);

    static float delta_out[N_ACTIONS];
    for (int k = 0; k < n_acts; k++) {
        float ind          = (k == (int)action) ? 1.f : 0.f;
        float policy_grad  = -ret * (ind - probs[k]);
        float entropy_grad = entropy_coef * probs[k] * (logf(probs[k]) + H);
        delta_out[k]       = policy_grad + entropy_grad;
    }

    backward_from_delta(policy, obs, delta_out);
}

#if USE_CONTINUOUS_ACTION

#define LOG_2PI_C  1.8378770664093453f   // log(2*pi)

// Forward + campionamento: a_i ~ N(mu_i, sigma_i^2) con sigma fissa.
void policy_forward_continuous(Network *policy, float *obs, float *action_out,
                               const float *sigma, float *log_prob_out) {
    int D = N_ACT_DIMS;
    network_forward(policy, obs, NULL);
    float *mu = policy->layers[policy->num_layers - 1].out;
    for (int i = 0; i < D; i++)
        mu[i] = fmaxf(fminf(mu[i], MU_CLAMP), -MU_CLAMP);

    float log_prob = 0.f;
    for (int i = 0; i < D; i++) {
        float s = sigma[i];
        float a = mu[i] + s * rng_normal();
        action_out[i] = a;
        float diff = a - mu[i];
        log_prob += -0.5f * (diff * diff / (s * s) + 2.f * logf(s) + LOG_2PI_C);
    }
    if (log_prob_out)
        *log_prob_out = log_prob;
}

// Backward — dL/dmu_i = -G * (a_i - mu_i) / sigma_i^2
// (entropia costante a sigma fissa: nessun contributo al gradiente).
void policy_backward_continuous(Network *policy, float *obs,
                                const float *action, const float *sigma,
                                float ret) {
    float *mu = policy->layers[policy->num_layers - 1].out;

    static float delta_out[N_ACT_DIMS];
    for (int i = 0; i < N_ACT_DIMS; i++)
        delta_out[i] = -ret * (action[i] - mu[i]) / (sigma[i] * sigma[i]);

    backward_from_delta(policy, obs, delta_out);
}

#endif  /* USE_CONTINUOUS_ACTION */
