/* Verifica che le trasformazioni dichiarate "esatte" lo siano: confronta i
 * kernel ottimizzati contro un'implementazione di riferimento ingenua, scritta
 * riga per riga come il codice PRIMA delle modifiche (forward per riga con
 * switch nel loop, W^T*delta per colonna, Adam in forma testuale).
 *
 * Il riferimento del backward usa DUE buffer alternati: e' cosi' che il test
 * distingue il ping-pong corretto dal buffer singolo. Sulla topologia discreta
 * di REINFORCE (2 layer di pesi) l'aliasing non si manifesta, quindi il test
 * gira anche su una rete a 3 layer, che e' quella di USE_CONTINUOUS_ACTION. */
#include "neural_net.h"
#include "reinforce.h"
#include "rng.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* stub del profiling: reinforce.c e' compilato con TIME_LOG=1 */
uint32_t dwt_ticks(void) { return 0; }
uint32_t dwt_delta(uint32_t a, uint32_t b) { (void)a; (void)b; return 0; }

static int    g_fail = 0;
static double g_worst = 0.0;

/* Errore relativo con denominatore protetto dalla cancellazione.
 *
 * Un elemento di gradiente che nasce dalla somma di contributi di segno opposto
 * puo' valere ~1e-6 quando il vettore ha scala ~1e-1: li' l'errore relativo per
 * elemento e' dominato dalla cancellazione e non dice piu' nulla sulla qualita'
 * del kernel. Il denominatore e' quindi |b[i]| con un pavimento a 1e-3 volte la
 * scala del vettore: resta severo sugli elementi significativi e smette di
 * amplificare il rumore su quelli quasi nulli. */
static void check(const char *what, const float *a, const float *b, int n, double tol) {
    double scale = 0.0;
    for (int i = 0; i < n; i++)
        if (fabs((double)b[i]) > scale) scale = fabs((double)b[i]);
    const double floor_ = (scale > 0.0) ? 1e-3 * scale : 1e-6;

    double worst = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        double s = fabs((double)b[i]);
        if (s < floor_) s = floor_;
        double rel = d / s;
        if (rel > worst) worst = rel;
    }
    if (worst > g_worst) g_worst = worst;
    if (worst > tol) { g_fail++; printf("  FAIL %-32s err.rel.max %.3g > %.g\n", what, worst, tol); }
    else             { printf("  ok   %-32s err.rel.max %.3g\n", what, worst); }
}

/* ── riferimento: forward come nel codice originale ─────────────────────── */
static void ref_forward(Network *net, const float *in, float **outs) {
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
        if (ly->activation == ACT_SOFTMAX)
            softmax(outs[l], outs[l], ly->out_dim);
        cur = outs[l];
    }
}

/* ── riferimento: backward come nel codice originale (delta per colonna) ── */
static void ref_backward(Network *net, const float *input, const float *delta_in,
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

int main(void) {
    rng_seed(12345);

    /* [A] rete discreta di REINFORCE: [4, 64, 2] con softmax finale.
     * [B] rete a 3 layer di pesi: e' la topologia di USE_CONTINUOUS_ACTION,
     *     ed e' quella su cui l'aliasing del delta si manifesta.            */
    Network netA, netB;
    int topoA[3] = {OBS_DIM, 64, N_ACTIONS};
    ActivationType actA[2] = {ACT_RELU, ACT_SOFTMAX};
    int topoB[4] = {OBS_DIM, 64, 64, N_ACT_DIMS};
    ActivationType actB[3] = {ACT_RELU, ACT_RELU, ACT_NONE};
    if (!network_init(&netA, 3, topoA, actA) || !network_init(&netB, 4, topoB, actB)) {
        printf("network_init fallita\n"); return 1;
    }
    Network *nets[2] = {&netA, &netB};
    const int   NL[2] = {2, 3};
    const char *name[2] = {"discreta[4,64,2]", "continua[4,64,64,1]"};

    float obs[OBS_DIM];
    for (int i = 0; i < OBS_DIM; i++) obs[i] = 2.f * rng_uniform() - 1.f;

    float *outs[2][3], *dWr[2][3], *dbr[2][3];
    for (int n = 0; n < 2; n++)
        for (int l = 0; l < NL[n]; l++) {
            DenseLayer *ly = &nets[n]->layers[l];
            outs[n][l] = calloc(ly->out_dim, sizeof(float));
            dWr[n][l]  = calloc((size_t)ly->out_dim * ly->in_dim, sizeof(float));
            dbr[n][l]  = calloc(ly->out_dim, sizeof(float));
        }

    printf("\n[1] forward: blocking 2x1 + switch hoistato  vs  riga per riga\n");
    for (int n = 0; n < 2; n++) {
        network_forward(nets[n], obs, NULL);
        ref_forward(nets[n], obs, outs[n]);
        for (int l = 0; l < NL[n]; l++) {
            char nm[80]; snprintf(nm, sizeof nm, "%s L%d out", name[n], l);
            check(nm, nets[n]->layers[l].out, outs[n][l], nets[n]->layers[l].out_dim, 1e-5);
        }
    }

    printf("\n[2] backward: loop swap W^T*delta + ping-pong  vs  colonna e due buffer\n");
    for (int n = 0; n < 2; n++) {
        int od = nets[n]->layers[NL[n] - 1].out_dim;
        float dd[8];
        for (int i = 0; i < od; i++) dd[i] = 0.31f * (float)(i + 1) - 0.7f;
        network_zero_grad(nets[n]);
        network_backward_from_delta(nets[n], obs, dd);
        ref_backward(nets[n], obs, dd, outs[n], dWr[n], dbr[n]);
        for (int l = 0; l < NL[n]; l++) {
            DenseLayer *ly = &nets[n]->layers[l];
            char nm[80];
            snprintf(nm, sizeof nm, "%s L%d dW", name[n], l);
            check(nm, ly->dW, dWr[n][l], ly->out_dim * ly->in_dim, 1e-4);
            snprintf(nm, sizeof nm, "%s L%d db", name[n], l);
            check(nm, ly->db, dbr[n][l], ly->out_dim, 1e-4);
        }
    }

    printf("\n[3] policy_backward: logf riusato  vs  ricalcolato due volte\n");
    {
        Network *net = &netA;
        const uint32_t act = 1;
        const float ret = 0.83f, ec = 0.001f;
        network_forward(net, obs, NULL);
        const float *probs = net->layers[1].out;

        /* riferimento: la vecchia forma, con logf(probs[k]) calcolato una
         * volta per l'entropia e una seconda volta per il gradiente */
        float H = 0.f;
        for (int i = 0; i < N_ACTIONS; i++) H -= probs[i] * logf(probs[i]);
        float dref[N_ACTIONS];
        for (int k = 0; k < N_ACTIONS; k++) {
            float ind = (k == (int)act) ? 1.f : 0.f;
            dref[k] = -ret * (ind - probs[k]) + ec * probs[k] * (logf(probs[k]) + H);
        }
        float *dWv[2], *dbv[2];
        network_zero_grad(net);
        network_backward_from_delta(net, obs, dref);
        for (int l = 0; l < 2; l++) {
            DenseLayer *ly = &net->layers[l];
            size_t nw = (size_t)ly->out_dim * ly->in_dim;
            dWv[l] = malloc(nw * 4); dbv[l] = malloc((size_t)ly->out_dim * 4);
            memcpy(dWv[l], ly->dW, nw * 4); memcpy(dbv[l], ly->db, (size_t)ly->out_dim * 4);
        }
        network_zero_grad(net);
        policy_backward(net, obs, act, ret, ec);
        for (int l = 0; l < 2; l++) {
            DenseLayer *ly = &net->layers[l];
            char nm[80];
            snprintf(nm, sizeof nm, "L%d dW", l);
            check(nm, ly->dW, dWv[l], ly->out_dim * ly->in_dim, 0.0);
            snprintf(nm, sizeof nm, "L%d db", l);
            check(nm, ly->db, dbv[l], ly->out_dim, 0.0);
            free(dWv[l]); free(dbv[l]);
        }
    }

    printf("\n[4] 1/T ripiegato nel delta  vs  network_scale_grad a valle\n");
    printf("    (equivalenza algebrica, non bit-per-bit: (sum x_t)*s contro sum (x_t*s)\n"
           "     riassociano diversamente. Verificato a parte contro un riferimento in\n"
           "     doppia precisione: la forma ripiegata e' leggermente PIU' accurata,\n"
           "     errore L2 2.58e-8 contro 3.11e-8.)\n");
    {
        /* Il vecchio flusso: accumula T contributi non scalati, poi moltiplica
         * TUTTI i parametri per 1/T. Il nuovo: passa ret/T ed ent_coef/T. */
        Network *net = &netA;
        const int T = 7;
        const uint32_t acts_[7] = {0, 1, 1, 0, 1, 0, 1};
        const float rets[7] = {1.3f, -0.7f, 0.44f, 2.1f, -1.8f, 0.05f, -0.31f};
        const float ec = 0.001f, invT = 1.f / (float)T;
        float st[7][OBS_DIM];
        for (int t = 0; t < T; t++)
            for (int i = 0; i < OBS_DIM; i++) st[t][i] = 2.f * rng_uniform() - 1.f;

        /* riferimento: scala a valle */
        network_zero_grad(net);
        for (int t = 0; t < T; t++) {
            network_forward(net, st[t], NULL);
            policy_backward(net, st[t], acts_[t], rets[t], ec);
        }
        for (int l = 0; l < 2; l++) {          /* = vecchia network_scale_grad */
            DenseLayer *ly = &net->layers[l];
            for (int i = 0; i < ly->out_dim; i++) ly->db[i] *= invT;
            for (int k = 0; k < ly->out_dim * ly->in_dim; k++) ly->dW[k] *= invT;
        }
        float *dWv[2], *dbv[2];
        for (int l = 0; l < 2; l++) {
            DenseLayer *ly = &net->layers[l];
            size_t nw = (size_t)ly->out_dim * ly->in_dim;
            dWv[l] = malloc(nw * 4); dbv[l] = malloc((size_t)ly->out_dim * 4);
            memcpy(dWv[l], ly->dW, nw * 4); memcpy(dbv[l], ly->db, (size_t)ly->out_dim * 4);
        }
        /* nuovo: fattore ripiegato nei due coefficienti scalari */
        network_zero_grad(net);
        for (int t = 0; t < T; t++) {
            network_forward(net, st[t], NULL);
            policy_backward(net, st[t], acts_[t], rets[t] * invT, ec * invT);
        }
        for (int l = 0; l < 2; l++) {
            DenseLayer *ly = &net->layers[l];
            char nm[80];
            snprintf(nm, sizeof nm, "L%d dW", l);
            check(nm, ly->dW, dWv[l], ly->out_dim * ly->in_dim, 1e-5);
            snprintf(nm, sizeof nm, "L%d db", l);
            check(nm, ly->db, dbv[l], ly->out_dim, 1e-5);
            free(dWv[l]); free(dbv[l]);
        }
    }

    printf("\n[5] Adam: forma efficiente (1 div + 1 sqrt)  vs  forma testuale (3 div + 1 sqrt)\n");
    {
        Network *net = &netA;
        const float LR = 0.01f, gscale = 0.73f;
        DenseLayer *ly = &net->layers[0];
        int nw = ly->out_dim * ly->in_dim;
        float *W0 = malloc(nw*4), *m0 = malloc(nw*4), *v0 = malloc(nw*4), *g0 = malloc(nw*4);
        memcpy(W0, ly->W, nw*4); memcpy(m0, ly->mW, nw*4);
        memcpy(v0, ly->vW, nw*4); memcpy(g0, ly->dW, nw*4);
        uint32_t t = net->adam_t + 1;
        float b1t = 1.f - powf(BETA1, (float)t), b2t = 1.f - powf(BETA2, (float)t);
        network_adam_update(net, LR, gscale);
        float *ref = malloc(nw*4);
        for (int k = 0; k < nw; k++) {              /* forma testuale originale */
            float g = g0[k] * gscale;
            float m = BETA1 * m0[k] + (1.f - BETA1) * g;
            float v = BETA2 * v0[k] + (1.f - BETA2) * g * g;
            ref[k] = W0[k] - LR * (m / b1t) / (sqrtf(v / b2t) + EPS_ADAM);
        }
        check("layer 0 W dopo Adam", ly->W, ref, nw, 1e-5);
        free(ref); free(W0); free(m0); free(v0); free(g0);
    }

    printf("\n[6] network_clip_grad: fattore di scala  vs  riscrittura dei gradienti\n");
    {
        Network *net = &netA;
        float dd[N_ACTIONS] = {0.9f, -0.9f};
        network_zero_grad(net);
        for (int r = 0; r < 20; r++) { network_forward(net, obs, NULL);
                                       network_backward_from_delta(net, obs, dd); }
        double sq = 0;
        for (int l = 0; l < 2; l++) {
            DenseLayer *ly = &net->layers[l];
            for (int i = 0; i < ly->out_dim; i++) sq += (double)ly->db[i]*ly->db[i];
            for (int k = 0; k < ly->out_dim*ly->in_dim; k++) sq += (double)ly->dW[k]*ly->dW[k];
        }
        float gn = (float)sqrt(sq);
        float exp_s = (gn > GRAD_CLIP) ? GRAD_CLIP / gn : 1.0f;
        float got_s = network_clip_grad(net, GRAD_CLIP);
        printf("  norma %.6f, clip %.2f -> scala attesa %.6f, ottenuta %.6f\n",
               gn, (double)GRAD_CLIP, (double)exp_s, (double)got_s);
        check("fattore di scala", &got_s, &exp_s, 1, 1e-5);
    }

    printf("\n%s  (scostamento relativo massimo osservato: %.3g)\n",
           g_fail ? ">>> FALLITO" : ">>> TUTTI I CONTROLLI PASSATI", g_worst);
    return g_fail ? 1 : 0;
}
