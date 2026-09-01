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

/* Prodotto matrice-vettore del forward, con blocking 2x1 sulle righe di W.
 *
 * La versione a una riga per volta e' una catena FMA seriale: ogni
 * `acc += w[j]*in[j]` dipende dal precedente, e la VFMA.F32 del Cortex-M7 ha
 * latenza ~3 cicli con throughput 1/ciclo -> ~3 cicli per MAC di sola latenza.
 * In piu' servono 2 load per MAC e la M7 ne ritira una per ciclo.
 *
 * Elaborando due righe di output per iterazione si ottengono due catene di
 * accumulo indipendenti e in_vec[j] viene caricato una volta per due MAC.
 * GCC non lo fa da solo: la M7 non ha FPU vettoriale, quindi lo splitting
 * della riduzione non arriva dalla vettorizzazione e -Ofast da solo non
 * riscrive la catena di dipendenza. */
static inline void forward_dense_layer(const float *restrict in_vec,
                                        float *restrict out_vec,
                                        const DenseLayer *restrict layer) {
    const int in_dim  = layer->in_dim;
    const int out_dim = layer->out_dim;
    const float *restrict W = layer->W;
    const float *restrict b = layer->b;

    int i = 0;
    for (; i + 1 < out_dim; i += 2) {
        const float *restrict w0 = W + (size_t)i * in_dim;
        const float *restrict w1 = w0 + in_dim;
        float acc0 = b[i];
        float acc1 = b[i + 1];
        for (int j = 0; j < in_dim; ++j) {
            float x = in_vec[j];
            acc0 += w0[j] * x;
            acc1 += w1[j] * x;
        }
        out_vec[i]     = acc0;
        out_vec[i + 1] = acc1;
    }
    if (i < out_dim) {                      /* coda per out_dim dispari */
        const float *restrict w0 = W + (size_t)i * in_dim;
        float acc0 = b[i];
        for (int j = 0; j < in_dim; ++j)
            acc0 += w0[j] * in_vec[j];
        out_vec[i] = acc0;
    }

    /* Attivazione in una passata separata: lo switch esce dal loop di output
     * (era una branch per neurone) e i due loop restano stretti. */
    switch (layer->activation) {
    case ACT_RELU:
        for (int k = 0; k < out_dim; ++k)
            out_vec[k] = (out_vec[k] > 0.f) ? out_vec[k] : 0.f;
        break;
    case ACT_TANH:
        for (int k = 0; k < out_dim; ++k)
            out_vec[k] = tanhf(out_vec[k]);
        break;
    case ACT_SOFTMAX:
        softmax(out_vec, out_vec, out_dim);
        break;
    default:
        break;
    }
}

/* Un passo di Adam su un layer.
 *
 * `lr_t` ed `eps_t` incorporano gia' la bias correction (vedi
 * network_adam_update) e `gscale` il fattore di gradient clipping: cosi' per
 * ogni peso restano una sola divisione e una sola sqrt, contro le tre
 * divisioni + sqrt della forma testuale, che su M7 non sono pipelined.
 *
 * Il layout piatto collassa il vecchio doppio loop i/j su un unico loop
 * lineare sugli out_dim*in_dim pesi. */
static void adam_update_single_layer(DenseLayer *restrict ly,
                                      float lr_t, float eps_t, float gscale) {
    const int    out_dim = ly->out_dim;
    const size_t n_w     = (size_t)out_dim * (size_t)ly->in_dim;

    float *restrict b  = ly->b;
    float *restrict db = ly->db;
    float *restrict mb = ly->mb;
    float *restrict vb = ly->vb;
    for (int i = 0; i < out_dim; ++i) {
        float g = db[i] * gscale;
        float m = BETA1 * mb[i] + (1.f - BETA1) * g;
        float v = BETA2 * vb[i] + (1.f - BETA2) * g * g;
        mb[i] = m;
        vb[i] = v;
        b[i] -= lr_t * m / (sqrtf(v) + eps_t);
        db[i] = 0.0f;
    }

    float *restrict w  = ly->W;
    float *restrict dw = ly->dW;
    float *restrict mw = ly->mW;
    float *restrict vw = ly->vW;
    for (size_t k = 0; k < n_w; ++k) {
        float g = dw[k] * gscale;
        float m = BETA1 * mw[k] + (1.f - BETA1) * g;
        float v = BETA2 * vw[k] + (1.f - BETA2) * g * g;
        mw[k] = m;
        vw[k] = v;
        w[k] -= lr_t * m / (sqrtf(v) + eps_t);
        dw[k] = 0.0f;
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
        /* dense_init chiama gia' init_layer_params: chiamarla di nuovo qui
         * rifarebbe il lavoro e consumerebbe il doppio di estrazioni RNG,
         * cioe' cambierebbe i pesi iniziali a parita' di seed. */
        if (!dense_init(&net->layers[i], topology[i], topology[i + 1],
                        activations[i]))
            return 0;
    }
    return 1;
}

void network_free(Network *net) {
    if (!net || !net->layers) return;
    for (int l = 0; l < net->num_layers; l++)
        dense_free(&net->layers[l]);
    free(net->layers);
    net->layers = NULL;
}

int network_forward(Network *net, const float *input, float *out) {
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

/* Motore di backprop condiviso: accumula dW/db e propaga il delta all'indietro.
 *
 * Il delta vive in un pool a due meta' usate a ping-pong. Serve davvero: il
 * delta del layer precedente si scrive mentre quello corrente e' ancora in
 * lettura, e dal secondo layer in poi i due starebbero nello stesso buffer.
 * Con un buffer solo, scrivere l'elemento j corrompe il delta[j] che serve
 * ancora alle iterazioni j successive.
 *
 * Sulla topologia discreta [4,64,2] (2 layer di pesi) l'aliasing non si
 * manifesta: il delta di uscita e' un array del chiamante e il secondo giro si
 * ferma prima di propagare. Ma con USE_CONTINUOUS_ACTION la topologia diventa
 * [4,64,64,1] — 3 layer di pesi — e il gradiente del layer 0 uscirebbe
 * sbagliato senza NaN ne' crash. Il ping-pong e' la forma corretta in generale.
 *
 * La capacita' si dimensiona una volta sul layer piu' largo PRIMA di iniziare:
 * una realloc a meta' passata invaliderebbe il delta in uso. */
void network_backward_from_delta(Network *net, const float *restrict input,
                                 const float *restrict delta_out) {
    static float   *delta_pool = NULL;
    static uint32_t delta_cap  = 0;

    uint32_t need = 0;
    for (int l = 1; l < net->num_layers; ++l)
        if ((uint32_t)net->layers[l].in_dim > need)
            need = (uint32_t)net->layers[l].in_dim;
    if (need > delta_cap) {
        float *nb = realloc(delta_pool, 2u * (size_t)need * sizeof(float));
        if (!nb) return;                 /* delta_cap resta coerente col pool */
        delta_pool = nb;
        delta_cap  = need;
    }

    const float *delta = delta_out;
    int half = 0;

    for (int l = net->num_layers - 1; l >= 0; --l) {
        DenseLayer *ly      = &net->layers[l];
        const int   in_dim  = ly->in_dim;
        const int   out_dim = ly->out_dim;
        const float *restrict inp = (l == 0) ? input : net->layers[l - 1].out;

        /* Accumulo gradienti: db += delta, dW[i][:] += delta[i]*inp[:].
         * Due righe per volta cosi' inp[j] serve una load per due MAC. */
        float *restrict db = ly->db;
        int i = 0;
        for (; i + 1 < out_dim; i += 2) {
            float d0 = delta[i];
            float d1 = delta[i + 1];
            db[i]     += d0;
            db[i + 1] += d1;
            float *restrict r0 = ly->dW + (size_t)i * in_dim;
            float *restrict r1 = r0 + in_dim;
            for (int j = 0; j < in_dim; ++j) {
                float x = inp[j];
                r0[j] += d0 * x;
                r1[j] += d1 * x;
            }
        }
        if (i < out_dim) {
            float d0 = delta[i];
            db[i] += d0;
            float *restrict r0 = ly->dW + (size_t)i * in_dim;
            for (int j = 0; j < in_dim; ++j)
                r0[j] += d0 * inp[j];
        }

        if (l == 0)
            break;

        /* W^T * delta con la riga di W nel loop interno.
         * Prima il loop era j esterno / i interno, cioe' accesso per colonna:
         * stride in_dim fra elementi consecutivi e accumulo seriale su un solo
         * registro. Ora l'accesso e' sequenziale sulla riga e ogni j ha il
         * proprio accumulatore in dst, senza catena di dipendenza. */
        float *restrict dst = delta_pool + (size_t)half * delta_cap;
        half ^= 1;                       /* dst non e' mai l'attuale delta */
        memset(dst, 0, (size_t)in_dim * sizeof(float));
        i = 0;
        for (; i + 1 < out_dim; i += 2) {
            float d0 = delta[i];
            float d1 = delta[i + 1];
            const float *restrict w0 = ly->W + (size_t)i * in_dim;
            const float *restrict w1 = w0 + in_dim;
            for (int j = 0; j < in_dim; ++j)
                dst[j] += d0 * w0[j] + d1 * w1[j];
        }
        if (i < out_dim) {
            float d0 = delta[i];
            const float *restrict w0 = ly->W + (size_t)i * in_dim;
            for (int j = 0; j < in_dim; ++j)
                dst[j] += d0 * w0[j];
        }

        /* Derivata dell'attivazione in una passata separata dallo swap. */
        const float *restrict h_prev = net->layers[l - 1].out;
        switch (net->layers[l - 1].activation) {
        case ACT_RELU:
            for (int j = 0; j < in_dim; ++j)
                dst[j] = (h_prev[j] > 0.f) ? dst[j] : 0.f;
            break;
        case ACT_TANH:
            for (int j = 0; j < in_dim; ++j)
                dst[j] *= (1.f - h_prev[j] * h_prev[j]);
            break;
        default:
            break;
        }
        delta = dst;
    }
}

void network_zero_grad(Network *net) {
    for (int l = 0; l < net->num_layers; l++) {
        DenseLayer *ly = &net->layers[l];
        memset(ly->dW, 0, (size_t)ly->out_dim * ly->in_dim * sizeof(float));
        memset(ly->db, 0, (size_t)ly->out_dim * sizeof(float));
    }
}

/* Norma globale dei gradienti -> fattore di scala da applicare all'update.
 * Non riscrive piu' i gradienti: il fattore viaggia fino ad Adam, che lo
 * applica al volo. Prima, quando il clip scattava, tutti i gradienti venivano
 * riletti e riscritti in una passata dedicata, prima che Adam li rileggesse
 * ancora. */
float network_clip_grad(Network *net, float max_norm) {
    float gnorm_sq = 0.f;
    for (int l = 0; l < net->num_layers; l++) {
        DenseLayer  *ly  = &net->layers[l];
        const size_t n_w = (size_t)ly->out_dim * (size_t)ly->in_dim;
        const float *restrict dw = ly->dW;
        const float *restrict db = ly->db;
        for (int i = 0; i < ly->out_dim; i++)
            gnorm_sq += db[i] * db[i];
        for (size_t k = 0; k < n_w; k++)
            gnorm_sq += dw[k] * dw[k];
    }

    float gnorm = sqrtf(gnorm_sq);
    if (!isfinite(gnorm)) {
        /* Gradiente non finito (Inf/NaN): scarta l'update. gscale = 0 azzera
         * il contributo di questo episodio e Adam ripulisce comunque dW/db,
         * quindi l'effetto e' identico all'azzeramento esplicito di prima.
         * Rete di sicurezza, non dovrebbe scattare. */
        return 0.0f;
    }
    return (gnorm > max_norm) ? (max_norm / gnorm) : 1.0f;
}

void network_adam_update(Network *net, float lr, float gscale) {
    net->adam_t++;
    float b1t = 1.f - powf(BETA1, (float)net->adam_t);
    float b2t = 1.f - powf(BETA2, (float)net->adam_t);

    /* Forma "efficiente" dell'Algoritmo 2 di Kingma & Ba:
     *   lr*(m/b1t) / (sqrt(v/b2t) + eps)  ==  lr_t*m / (sqrt(v) + eps_t)
     * con lr_t = lr*sqrt(b2t)/b1t e eps_t = eps*sqrt(b2t).
     * Algebricamente identica, ma due divisioni per peso in meno. */
    float sb2t  = sqrtf(b2t);
    float lr_t  = lr * sb2t / b1t;
    float eps_t = EPS_ADAM * sb2t;

    for (int l = 0; l < net->num_layers; l++)
        adam_update_single_layer(&net->layers[l], lr_t, eps_t, gscale);
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
//
// `ret` ed `entropy_coef` arrivano gia' scalati per 1/T: entrambi i termini
// della loss sono lineari nel rispettivo coefficiente, quindi la media
// sull'episodio si ottiene qui invece che con una passata su tutti i parametri.
void policy_backward(Network *policy, const float *obs, uint32_t action,
                     float ret, float entropy_coef) {
    int    last   = policy->num_layers - 1;
    int    n_acts = policy->layers[last].out_dim;
    const float *restrict probs = policy->layers[last].out;

    /* log p_k serviva due volte — una per l'entropia, una per il gradiente —
     * ed era ricalcolato: 2*n_acts logf per step, tutte in software su M7.
     * Calcolato una volta e riusato. */
    float logp[N_ACTIONS];
    float H = 0.f;
    for (int i = 0; i < n_acts; i++) {
        float lp = logf(probs[i]);
        logp[i]  = lp;
        H       -= probs[i] * lp;
    }

    float delta_out[N_ACTIONS];
    for (int k = 0; k < n_acts; k++) {
        float ind          = (k == (int)action) ? 1.f : 0.f;
        float policy_grad  = -ret * (ind - probs[k]);
        float entropy_grad = entropy_coef * probs[k] * (logp[k] + H);
        delta_out[k]       = policy_grad + entropy_grad;
    }

    network_backward_from_delta(policy, obs, delta_out);
}

#if USE_CONTINUOUS_ACTION

#define LOG_2PI_C  1.8378770664093453f   // log(2*pi)

// Forward + campionamento: a_i ~ N(mu_i, sigma_i^2) con sigma fissa.
void policy_forward_continuous(Network *policy, float *obs, float *action_out,
                               const float *sigma, float *log_prob_out) {
    int D = N_ACT_DIMS;
    network_forward(policy, obs, NULL);
    float *mu = policy->layers[policy->num_layers - 1].out;

    for (int i = 0; i < D; i++) {
        float m = fmaxf(fminf(mu[i], MU_CLAMP), -MU_CLAMP);
        mu[i] = m;                       /* clamp in place: lo rilegge il backward */
        action_out[i] = m + sigma[i] * rng_normal();
    }

    /* La log-probabilita' costa D divisioni e D logf per campione, e in
     * REINFORCE nessuno la usa: il gradiente la ricava da (a - mu), non da
     * lp. Con log_prob_out == NULL non viene proprio calcolata. */
    if (log_prob_out) {
        float log_prob = 0.f;
        for (int i = 0; i < D; i++) {
            float s    = sigma[i];
            float diff = action_out[i] - mu[i];
            log_prob += -0.5f * (diff * diff / (s * s) + 2.f * logf(s) + LOG_2PI_C);
        }
        *log_prob_out = log_prob;
    }
}

// Backward — dL/dmu_i = -G * (a_i - mu_i) / sigma_i^2
// (entropia costante a sigma fissa: nessun contributo al gradiente).
// `inv_var` = 1/sigma^2 precalcolato: sigma non cambia mai, ricalcolare
// s*s e dividere per ogni step della traiettoria era lavoro ripetuto.
void policy_backward_continuous(Network *policy, const float *obs,
                                const float *action, const float *inv_var,
                                float ret) {
    const float *restrict mu = policy->layers[policy->num_layers - 1].out;

    float delta_out[N_ACT_DIMS];
    for (int i = 0; i < N_ACT_DIMS; i++)
        delta_out[i] = -ret * (action[i] - mu[i]) * inv_var[i];

    network_backward_from_delta(policy, obs, delta_out);
}

#endif  /* USE_CONTINUOUS_ACTION */
