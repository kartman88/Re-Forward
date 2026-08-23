/* Verifica che le trasformazioni dichiarate "esatte" lo siano: confronta i
 * kernel ottimizzati contro un'implementazione di riferimento ingenua, scritta
 * riga per riga come il codice PRIMA delle modifiche (forward per riga con
 * switch nel loop, W^T*delta per colonna, Adam in forma testuale).
 *
 * Il riferimento del backward usa DUE buffer alternati: e' cosi' che il test
 * distingue il ping-pong corretto dal buffer singolo, che sulla topologia a 3
 * layer di pesi corrompeva il gradiente del layer 0 senza produrre NaN. */
#include "neural_net.h"
#include "dqn.h"
#include "rng.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* stub del profiling: dqn.c e' compilato con TIME_LOG=1 */
uint32_t dwt_ticks(void) { return 0; }
uint32_t dwt_delta(uint32_t a, uint32_t b) { (void)a; (void)b; return 0; }

static int    g_fail = 0;
static double g_worst = 0.0;

static void check(const char *what, const float *a, const float *b, int n, double tol) {
    double worst = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        double s = fabs((double)b[i]);
        double rel = (s > 1e-6) ? d / s : d;
        if (rel > worst) worst = rel;
    }
    if (worst > g_worst) g_worst = worst;
    if (worst > tol) { g_fail++; printf("  FAIL %-30s err.rel.max %.3g > %.g\n", what, worst, tol); }
    else             { printf("  ok   %-30s err.rel.max %.3g\n", what, worst); }
}

/* ── riferimento: forward come nel codice originale ─────────────────────── */
static void ref_forward(QNetwork *net, const float *in, float **outs) {
    const float *cur = in;
    for (int l = 0; l < net->num_layers; l++) {
        DenseLayer *ly = &net->layers[l];
        for (int i = 0; i < ly->out_dim; i++) {
            float acc = ly->b[i];
            for (int j = 0; j < ly->in_dim; j++)
                acc += ly->W[i * ly->in_dim + j] * cur[j];   /* era W[i][j] */
            switch (ly->activation) {                        /* switch DENTRO il loop */
            case ACT_RELU: outs[l][i] = (acc > 0.f) ? acc : 0.f; break;
            case ACT_TANH: outs[l][i] = tanhf(acc);             break;
            default:       outs[l][i] = acc;                    break;
            }
        }
        cur = outs[l];
    }
}

/* ── riferimento: backward come nel codice originale (delta per colonna) ── */
static void ref_backward(QNetwork *net, const float *input, const float *delta_in,
                         float **outs, float **dW, float **db_ref) {
    float dbuf_a[256], dbuf_b[256];
    float *delta = dbuf_a;
    memcpy(delta, delta_in, net->layers[net->num_layers - 1].out_dim * sizeof(float));

    for (int l = net->num_layers - 1; l >= 0; --l) {
        DenseLayer *ly = &net->layers[l];
        const int in_dim = ly->in_dim, out_dim = ly->out_dim;
        const float *inp = (l == 0) ? input : outs[l - 1];

        for (int i = 0; i < out_dim; ++i) {
            db_ref[l][i] += delta[i];
            for (int j = 0; j < in_dim; ++j)
                dW[l][i * in_dim + j] += delta[i] * inp[j];
        }
        if (l == 0) break;

        float *nd = (delta == dbuf_a) ? dbuf_b : dbuf_a;
        for (int j = 0; j < in_dim; ++j) {                    /* j esterno, i interno */
            float acc = 0.f;
            for (int i = 0; i < out_dim; ++i)
                acc += delta[i] * ly->W[i * in_dim + j];
            float hp = outs[l - 1][j];
            switch (net->layers[l - 1].activation) {
            case ACT_RELU: acc = (hp > 0.f) ? acc : 0.f; break;
            case ACT_TANH: acc = acc * (1.f - hp * hp);  break;
            default: break;
            }
            nd[j] = acc;
        }
        delta = nd;
    }
}

#define HID 32
#define NL  3    /* layer di pesi */

static void make_net(QNetwork *n, ActivationType h) {
    int topo[4] = {OBS_DIM, HID, HID, N_ACTIONS};
    ActivationType act[3] = {h, h, ACT_NONE};
    if (!init_qnetwork(n, 4, topo, act)) { printf("init_qnetwork fallita\n"); exit(1); }
}

int main(void) {
    rng_seed(12345);
    QNetwork relu_net, tanh_net;                 /* le attivazioni di DQN e una tanh */
    make_net(&relu_net, ACT_RELU);
    make_net(&tanh_net, ACT_TANH);
    QNetwork *nets[2] = {&relu_net, &tanh_net};
    const char *name[2] = {"relu", "tanh"};

    float obs[OBS_DIM];
    for (int i = 0; i < OBS_DIM; i++) obs[i] = 2.f * rng_uniform() - 1.f;

    float *outs[2][NL], *dWr[2][NL], *dbr[2][NL];
    for (int n = 0; n < 2; n++)
        for (int l = 0; l < NL; l++) {
            DenseLayer *ly = &nets[n]->layers[l];
            outs[n][l] = calloc(ly->out_dim, sizeof(float));
            dWr[n][l]  = calloc((size_t)ly->out_dim * ly->in_dim, sizeof(float));
            dbr[n][l]  = calloc(ly->out_dim, sizeof(float));
        }

    printf("\n[1] forward: blocking 2x1 + switch hoistato  vs  riga per riga\n");
    for (int n = 0; n < 2; n++) {
        forward_q(nets[n], obs, NULL);
        ref_forward(nets[n], obs, outs[n]);
        for (int l = 0; l < NL; l++) {
            char nm[64]; snprintf(nm, sizeof nm, "%s layer %d out", name[n], l);
            check(nm, nets[n]->layers[l].out, outs[n][l], nets[n]->layers[l].out_dim, 1e-5);
        }
    }

    printf("\n[2] backward: loop swap W^T*delta + ping-pong  vs  colonna e due buffer\n");
    /* Delta denso su tutte le uscite: e' il caso che espone l'aliasing del
     * buffer del delta fra layer 2 e layer 1 (con un buffer solo, il layer 0
     * esce sbagliato e gli altri due combaciano). */
    float ddense[N_ACTIONS];
    for (int i = 0; i < N_ACTIONS; i++) ddense[i] = 0.3f * (float)(i + 1) - 0.7f;
    for (int n = 0; n < 2; n++) {
        zero_grad_q(nets[n]);
        dqn_backward_delta(nets[n], obs, ddense);
        ref_backward(nets[n], obs, ddense, outs[n], dWr[n], dbr[n]);
        for (int l = 0; l < NL; l++) {
            DenseLayer *ly = &nets[n]->layers[l];
            char nm[64];
            snprintf(nm, sizeof nm, "%s layer %d dW", name[n], l);
            check(nm, ly->dW, dWr[n][l], ly->out_dim * ly->in_dim, 1e-4);
            snprintf(nm, sizeof nm, "%s layer %d db", name[n], l);
            check(nm, ly->db, dbr[n][l], ly->out_dim, 1e-4);
        }
    }

    printf("\n[3] dqn_backward sparso (una riga)  vs  gradiente denso con 4 zeri su 5\n");
    /* Il gradiente della loss DQN e' non nullo solo su Q(s,a): saltare le righe
     * moltiplicate per zero deve dare lo stesso identico risultato. */
    {
        QNetwork *net = &relu_net;
        const uint32_t act = 3;
        const float    td  = -0.4217f;
        float dsparse[N_ACTIONS] = {0};
        dsparse[act] = td;

        forward_q(net, obs, NULL);
        zero_grad_q(net);
        dqn_backward_delta(net, obs, dsparse);
        float *dW_dense[NL], *db_dense[NL];
        for (int l = 0; l < NL; l++) {
            DenseLayer *ly = &net->layers[l];
            size_t nw = (size_t)ly->out_dim * ly->in_dim;
            dW_dense[l] = malloc(nw * sizeof(float));
            db_dense[l] = malloc((size_t)ly->out_dim * sizeof(float));
            memcpy(dW_dense[l], ly->dW, nw * sizeof(float));
            memcpy(db_dense[l], ly->db, (size_t)ly->out_dim * sizeof(float));
        }

        zero_grad_q(net);
        dqn_backward(net, obs, act, td);
        for (int l = 0; l < NL; l++) {
            DenseLayer *ly = &net->layers[l];
            char nm[64];
            snprintf(nm, sizeof nm, "sparso vs denso layer %d dW", l);
            check(nm, ly->dW, dW_dense[l], ly->out_dim * ly->in_dim, 0.0);
            snprintf(nm, sizeof nm, "sparso vs denso layer %d db", l);
            check(nm, ly->db, db_dense[l], ly->out_dim, 0.0);
            free(dW_dense[l]); free(db_dense[l]);
        }
    }

    printf("\n[4] Adam: forma efficiente (1 div + 1 sqrt)  vs  forma testuale (3 div + 1 sqrt)\n");
    {
        QNetwork *net = &relu_net;
        const float gscale = 0.73f;
        int l = 2;
        DenseLayer *ly = &net->layers[l];
        int nw = ly->out_dim * ly->in_dim;
        float *W0 = malloc(nw * 4), *m0 = malloc(nw * 4), *v0 = malloc(nw * 4), *g0 = malloc(nw * 4);
        memcpy(W0, ly->W, nw * 4); memcpy(m0, ly->mW, nw * 4);
        memcpy(v0, ly->vW, nw * 4); memcpy(g0, ly->dW, nw * 4);

        uint32_t t = net->adam_t + 1;
        float b1t = 1.f - powf(BETA1, (float)t), b2t = 1.f - powf(BETA2, (float)t);
        adam_optimizer_q(net, LR, gscale);

        float *ref = malloc(nw * 4);
        for (int k = 0; k < nw; k++) {              /* forma testuale originale */
            float g = g0[k] * gscale;
            float m = BETA1 * m0[k] + (1.f - BETA1) * g;
            float v = BETA2 * v0[k] + (1.f - BETA2) * g * g;
            ref[k] = W0[k] - LR * (m / b1t) / (sqrtf(v / b2t) + EPS_ADAM);
        }
        check("layer 2 W dopo Adam", ly->W, ref, nw, 1e-5);
        free(ref); free(W0); free(m0); free(v0); free(g0);
    }

    printf("\n[5] gradient_norm_q: fattore di scala  vs  riscrittura dei gradienti\n");
    {
        QNetwork *net = &relu_net;
        zero_grad_q(net);
        dqn_backward_delta(net, obs, ddense);      /* gradienti di norma > GRAD_CLIP */
        double sq = 0;
        for (int l = 0; l < NL; l++) {
            DenseLayer *ly = &net->layers[l];
            for (int i = 0; i < ly->out_dim; i++) sq += (double)ly->db[i] * ly->db[i];
            for (int k = 0; k < ly->out_dim * ly->in_dim; k++) sq += (double)ly->dW[k] * ly->dW[k];
        }
        float gn  = (float)sqrt(sq);
        float exp_s = (gn > GRAD_CLIP) ? GRAD_CLIP / gn : 1.0f;
        float got_s = gradient_norm_q(net, GRAD_CLIP);
        printf("  norma %.6f, clip %.2f -> scala attesa %.6f, ottenuta %.6f\n",
               gn, (double)GRAD_CLIP, (double)exp_s, (double)got_s);
        check("fattore di scala", &got_s, &exp_s, 1, 1e-5);
    }

    printf("\n[6] rete target: stesso kernel della rete online\n");
    {
        TargetNetwork tgt;
        int topo[4] = {OBS_DIM, HID, HID, N_ACTIONS};
        if (!init_target_network(&tgt, 4, topo)) { printf("init target KO\n"); return 1; }
        copy_weights_to_target(&relu_net, &tgt);
        forward_q(&relu_net, obs, NULL);
        forward_target(&tgt, obs, NULL);
        check("target out == online out",
              tgt.layers[NL - 1].out, relu_net.layers[NL - 1].out, N_ACTIONS, 0.0);
        target_free(&tgt);
    }

    printf("\n%s  (scostamento relativo massimo osservato: %.3g)\n",
           g_fail ? ">>> FALLITO" : ">>> TUTTI I CONTROLLI PASSATI", g_worst);
    return g_fail ? 1 : 0;
}
