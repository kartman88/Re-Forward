/* Verifica che le trasformazioni dichiarate "esatte" lo siano: confronta i
 * kernel ottimizzati contro un'implementazione di riferimento ingenua, scritta
 * riga per riga come il codice PRIMA delle modifiche (forward per riga con
 * switch nel loop, W^T*delta per colonna, Adam in forma testuale). */
#include "neural_net.h"
#include "ppo.h"
#include "rng.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* stub del profiling: ppo.c e' compilato con TIME_LOG=1 */
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
    if (worst > tol) { g_fail++; printf("  FAIL %-28s err.rel.max %.3g > %.g\n", what, worst, tol); }
    else             { printf("  ok   %-28s err.rel.max %.3g\n", what, worst); }
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
        cur = outs[l];
    }
}

/* ── riferimento: backward come nel codice originale (delta per colonna) ── */
static void ref_backward(Network *net, const float *input, const float *delta_in,
                         float **outs, float **dW, float *db_ref[]) {
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

static Network make_net(int a, int b, int c, int d, ActivationType h, ActivationType o) {
    Network n;
    int topo[4] = {a, b, c, d};
    ActivationType act[3] = {h, h, o};
    if (!network_init(&n, 4, topo, act)) { printf("network_init fallita\n"); exit(1); }
    return n;
}

int main(void) {
    rng_seed(12345);
    Network actor  = make_net(OBS_DIM, 32, 32, N_ACT_DIMS, ACT_TANH, ACT_NONE);
    Network critic = make_net(OBS_DIM, 32, 32, 1,          ACT_RELU, ACT_NONE);

    float obs[OBS_DIM];
    for (int i = 0; i < OBS_DIM; i++) obs[i] = 2.f * rng_uniform() - 1.f;

    /* buffer di riferimento */
    float *outs[2][3], *dWr[2][3], *dbr[2][3];
    Network *nets[2] = {&actor, &critic};
    for (int n = 0; n < 2; n++)
        for (int l = 0; l < 3; l++) {
            DenseLayer *ly = &nets[n]->layers[l];
            outs[n][l] = calloc(ly->out_dim, sizeof(float));
            dWr[n][l]  = calloc((size_t)ly->out_dim * ly->in_dim, sizeof(float));
            dbr[n][l]  = calloc(ly->out_dim, sizeof(float));
        }

    printf("\n[1] forward: blocking 2x1 + switch hoistato  vs  riga per riga\n");
    for (int n = 0; n < 2; n++) {
        network_forward(nets[n], obs, NULL);
        ref_forward(nets[n], obs, outs[n]);
        for (int l = 0; l < 3; l++) {
            char nm[64]; snprintf(nm, sizeof nm, "%s layer %d out", n ? "critic" : "actor", l);
            check(nm, nets[n]->layers[l].out, outs[n][l], nets[n]->layers[l].out_dim, 1e-5);
        }
    }

    printf("\n[2] backward: loop swap W^T*delta  vs  accesso per colonna\n");
    /* critic: delta scalare, come critic_backward */
    network_zero_grad(&critic);
    float target = 0.42f, coeff = 0.5f;
    float dc = coeff * (critic.layers[2].out[0] - target);
    critic_backward(&critic, obs, target, coeff);
    ref_backward(&critic, obs, &dc, outs[1], dWr[1], dbr[1]);
    for (int l = 0; l < 3; l++) {
        DenseLayer *ly = &critic.layers[l];
        char nm[64];
        snprintf(nm, sizeof nm, "critic layer %d dW", l);
        check(nm, ly->dW, dWr[1][l], ly->out_dim * ly->in_dim, 1e-4);
        snprintf(nm, sizeof nm, "critic layer %d db", l);
        check(nm, ly->db, dbr[1][l], ly->out_dim, 1e-4);
    }

    /* actor: delta vettoriale gaussiano */
    network_zero_grad(&actor);
    float z[N_ACT_DIMS], var[N_ACT_DIMS], log_var[N_ACT_DIMS];
    for (int i = 0; i < N_ACT_DIMS; i++) {
        z[i] = 0.7f * (2.f * rng_uniform() - 1.f);
        log_var[i] = 2.f * logf(1.5f);
        var[i] = expf(log_var[i]);
    }
    float lp; actor_forward_continuous(&actor, obs, z, var, log_var, &lp);
    ref_forward(&actor, obs, outs[0]);   /* rigenera outs con mu gia' clampato */
    float adv = 0.83f, ratio = 1.05f, w = ratio * adv;
    float da[N_ACT_DIMS];
    for (int i = 0; i < N_ACT_DIMS; i++)
        da[i] = -w * (z[i] - actor.layers[2].out[i]) / var[i];
    actor_backward_continuous(&actor, obs, z, var, adv, ratio, PPO_CLIP_EPS);
    ref_backward(&actor, obs, da, outs[0], dWr[0], dbr[0]);
    for (int l = 0; l < 3; l++) {
        DenseLayer *ly = &actor.layers[l];
        char nm[64];
        snprintf(nm, sizeof nm, "actor layer %d dW", l);
        check(nm, ly->dW, dWr[0][l], ly->out_dim * ly->in_dim, 1e-4);
        snprintf(nm, sizeof nm, "actor layer %d db", l);
        check(nm, ly->db, dbr[0][l], ly->out_dim, 1e-4);
    }

    printf("\n[3] Adam: forma efficiente (1 div + 1 sqrt)  vs  forma testuale (3 div + 1 sqrt)\n");
    const float LR = 5e-4f, gscale = 0.73f;
    for (int l = 0; l < 3; l++) {
        DenseLayer *ly = &critic.layers[l];
        int nw = ly->out_dim * ly->in_dim;
        float *W0 = malloc(nw * 4), *m0 = malloc(nw * 4), *v0 = malloc(nw * 4), *g0 = malloc(nw * 4);
        memcpy(W0, ly->W, nw * 4); memcpy(m0, ly->mW, nw * 4);
        memcpy(v0, ly->vW, nw * 4); memcpy(g0, ly->dW, nw * 4);
        if (l == 2) {
            uint32_t t = critic.adam_t + 1;
            float b1t = 1.f - powf(BETA1, (float)t), b2t = 1.f - powf(BETA2, (float)t);
            network_adam_update(&critic, LR, gscale);
            float *ref = malloc(nw * 4);
            for (int k = 0; k < nw; k++) {              /* forma testuale originale */
                float g = g0[k] * gscale;
                float m = BETA1 * m0[k] + (1.f - BETA1) * g;
                float v = BETA2 * v0[k] + (1.f - BETA2) * g * g;
                ref[k] = W0[k] - LR * (m / b1t) / (sqrtf(v / b2t) + EPS_ADAM);
            }
            check("critic layer 2 W dopo Adam", ly->W, ref, nw, 1e-5);
            free(ref);
        }
        free(W0); free(m0); free(v0); free(g0);
    }

    printf("\n[4] ratio invariante alla rimozione del termine Jacobiano\n");
    {
        float a[N_ACT_DIMS], lp_no = 0.f, lp_with = 0.f, lpo_no = 0.f, lpo_with = 0.f;
        float mu_old[N_ACT_DIMS] = {0.1f, -0.3f, 0.25f};
        for (int i = 0; i < N_ACT_DIMS; i++) a[i] = tanhf(z[i]);
        actor_forward_continuous(&actor, obs, z, var, log_var, &lp_no);
        for (int i = 0; i < N_ACT_DIMS; i++) {
            float jac = -logf(1.f - a[i] * a[i] + 1e-6f);
            lp_with += jac;                                   /* nuovo, con Jacobiano */
            float diff = z[i] - mu_old[i];
            float base = -0.5f * (diff * diff / var[i] + log_var[i] + LOG_2PI);
            lpo_no   += base;                                  /* vecchio, senza */
            lpo_with += base + jac;                            /* vecchio, con */
        }
        lp_with += lp_no;
        float r_no   = lp_no   - lpo_no;
        float r_with = lp_with - lpo_with;
        check("log_prob_new - log_prob_old", &r_no, &r_with, 1, 1e-4); /* cancellazione esatta; il residuo e' rumore float del test */
    }

    printf("\n%s  (scostamento relativo massimo osservato: %.3g)\n",
           g_fail ? ">>> FALLITO" : ">>> TUTTI I CONTROLLI PASSATI", g_worst);
    return g_fail ? 1 : 0;
}
