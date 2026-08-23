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
 * riscrive la catena di dipendenza.
 *
 * Prende W/b sciolti invece del layer: rete online e rete target hanno struct
 * diverse ma lo stesso kernel, che prima era duplicato in due copie da tenere
 * allineate a mano. */
static inline void dense_matvec(const float *restrict W, const float *restrict b,
                                const float *restrict in_vec,
                                float *restrict out_vec,
                                int in_dim, int out_dim, ActivationType act) {
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
    switch (act) {
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
 * adam_optimizer_q) e `gscale` il fattore di gradient clipping: cosi' per ogni
 * peso restano una sola divisione e una sola sqrt, contro le tre divisioni +
 * sqrt della forma testuale, che su M7 non sono pipelined.
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

int init_qnetwork(QNetwork *net, int num_layers, int *topology,
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

int init_target_network(TargetNetwork *tgt, int num_layers, int *topology) {
    int n_weights   = num_layers - 1;
    tgt->num_layers = (uint8_t)n_weights;
    tgt->layers     = malloc(n_weights * sizeof(TargetLayer));
    if (!tgt->layers)
        return 0;
    for (int i = 0; i < n_weights; i++) {
        int in_d  = topology[i];
        int out_d = topology[i + 1];
        TargetLayer *tl = &tgt->layers[i];
        tl->in_dim     = in_d;
        tl->out_dim    = out_d;
        tl->activation = ACT_NONE;   /* la fissa copy_weights_to_target */

        /* Una sola allocazione: [ W | b | out ]. */
        const size_t n_w = (size_t)out_d * (size_t)in_d;
        float *arena = calloc(n_w + 2u * (size_t)out_d, sizeof(float));
        if (!arena)
            return 0;
        tl->W   = arena;             // base dell'arena, la libera target_free()
        tl->b   = arena + n_w;
        tl->out = arena + n_w + out_d;
    }
    return 1;
}

void qnetwork_free(QNetwork *net) {
    if (!net || !net->layers) return;
    for (int l = 0; l < net->num_layers; l++)
        dense_free(&net->layers[l]);
    free(net->layers);
    net->layers = NULL;
}

void target_free(TargetNetwork *tgt) {
    if (!tgt || !tgt->layers) return;
    for (int l = 0; l < tgt->num_layers; l++) {
        free(tgt->layers[l].W);      // libera l'intera arena in un colpo
        tgt->layers[l].W = tgt->layers[l].b = tgt->layers[l].out = NULL;
    }
    free(tgt->layers);
    tgt->layers = NULL;
}

int forward_q(QNetwork *net, const float *input, float *q_out) {
    const float *curr_in = input;
    for (int l = 0; l < net->num_layers; l++) {
        DenseLayer *ly = &net->layers[l];
        dense_matvec(ly->W, ly->b, curr_in, ly->out,
                     ly->in_dim, ly->out_dim, ly->activation);
        curr_in = ly->out;
    }
    if (q_out)
        memcpy(q_out, curr_in,
               net->layers[net->num_layers - 1].out_dim * sizeof(float));
    return 1;
}

int forward_target(TargetNetwork *tgt, const float *input, float *q_out) {
    const float *curr_in = input;
    for (int l = 0; l < tgt->num_layers; l++) {
        TargetLayer *tl = &tgt->layers[l];
        dense_matvec(tl->W, tl->b, curr_in, tl->out,
                     tl->in_dim, tl->out_dim, tl->activation);
        curr_in = tl->out;
    }
    if (q_out)
        memcpy(q_out, curr_in,
               tgt->layers[tgt->num_layers - 1].out_dim * sizeof(float));
    return 1;
}

/* Motore di backprop condiviso: accumula dW/db e propaga il delta all'indietro.
 *
 * Il delta vive in un pool a due meta' usate a ping-pong. Serve davvero: il
 * delta del layer precedente si scrive mentre quello corrente e' ancora in
 * lettura, e dal secondo layer in poi i due starebbero nello stesso buffer.
 * Con un buffer solo, scrivere l'elemento j corrompe il delta[j] che serve
 * ancora alle iterazioni j successive — sulla topologia [3,32,32,5] il
 * gradiente del layer 0 usciva sistematicamente sbagliato, senza NaN ne' crash.
 *
 * La capacita' si dimensiona una volta sul layer piu' largo PRIMA di iniziare:
 * una realloc a meta' passata invaliderebbe il delta in uso.
 *
 * `sparse_idx >= 0` e' il caso DQN: il gradiente sull'uscita ha un solo
 * elemento non nullo (l'azione eseguita), e `delta_out` punta a quel singolo
 * float. Saltare i 4/5 di righe moltiplicate per zero e' esatto — sommare
 * 0.0f non cambia un accumulatore inizializzato a zero — e toglie l'80% del
 * lavoro sull'ultimo layer. Vale solo alla prima iterazione: da li' in giu' il
 * delta e' denso. */
static void backward_engine(QNetwork *net, const float *restrict input,
                            const float *restrict delta_out, int sparse_idx) {
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
    const int    top   = net->num_layers - 1;
    int half = 0;

    for (int l = top; l >= 0; --l) {
        DenseLayer *ly      = &net->layers[l];
        const int   in_dim  = ly->in_dim;
        const int   out_dim = ly->out_dim;
        const float *restrict inp = (l == 0) ? input : net->layers[l - 1].out;
        const int   sparse  = (l == top && sparse_idx >= 0);

        /* Accumulo gradienti: db += delta, dW[i][:] += delta[i]*inp[:].
         * Due righe per volta cosi' inp[j] serve una load per due MAC. */
        float *restrict db = ly->db;
        if (sparse) {
            float d0 = delta[0];
            db[sparse_idx] += d0;
            float *restrict r0 = ly->dW + (size_t)sparse_idx * in_dim;
            for (int j = 0; j < in_dim; ++j)
                r0[j] += d0 * inp[j];
        } else {
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
        if (sparse) {
            /* Una sola riga contribuisce: assegnazione diretta, niente memset
             * e niente somma di out_dim-1 termini nulli. */
            float d0 = delta[0];
            const float *restrict w0 = ly->W + (size_t)sparse_idx * in_dim;
            for (int j = 0; j < in_dim; ++j)
                dst[j] = d0 * w0[j];
        } else {
            memset(dst, 0, (size_t)in_dim * sizeof(float));
            int i = 0;
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

/* Backprop generico da un gradiente denso sull'uscita: e' la via che il test
 * di equivalenza confronta con la versione sparsa qui sotto. */
void dqn_backward_delta(QNetwork *net, const float *input,
                        const float *delta_out) {
    backward_engine(net, input, delta_out, -1);
}

/* Gradiente della loss di Huber/MSE sul solo Q(s,a) eseguito: il vettore di
 * uscita e' nullo ovunque tranne in `action`. */
void dqn_backward(QNetwork *net, const float *input, uint32_t action,
                  float td_error) {
    backward_engine(net, input, &td_error, (int)action);
}

void copy_weights_to_target(QNetwork *src, TargetNetwork *dst) {
    for (int l = 0; l < src->num_layers; l++) {
        DenseLayer  *sl = &src->layers[l];
        TargetLayer *tl = &dst->layers[l];
        tl->activation  = sl->activation;
        /* Arene piatte da entrambi i lati: due memcpy per layer invece di
         * out_dim+1 (una per riga di W). */
        memcpy(tl->W, sl->W, (size_t)sl->out_dim * sl->in_dim * sizeof(float));
        memcpy(tl->b, sl->b, (size_t)sl->out_dim * sizeof(float));
    }
}

void zero_grad_q(QNetwork *net) {
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
float gradient_norm_q(QNetwork *net, float max_norm) {
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
         * il contributo di questo batch e Adam ripulisce comunque dW/db,
         * quindi l'effetto e' identico all'azzeramento esplicito di prima.
         * Rete di sicurezza, non dovrebbe scattare. */
        return 0.0f;
    }
    return (gnorm > max_norm) ? (max_norm / gnorm) : 1.0f;
}

void adam_optimizer_q(QNetwork *net, float lr, float gscale) {
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
