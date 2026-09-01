/* Smoke test end-to-end: riempie un episodio sintetico, gira reinforce_update
 * e verifica che l'arena del buffer sia coerente, che i pesi si muovano e che
 * non compaiano NaN/Inf. */
#include "neural_net.h"
#include "reinforce.h"
#include "rng.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

uint32_t dwt_ticks(void) { return 0; }
uint32_t dwt_delta(uint32_t a, uint32_t b) { (void)a; (void)b; return 0; }

static int nonfinite(Network *n) {
    int bad = 0;
    for (int l = 0; l < n->num_layers; l++) {
        DenseLayer *ly = &n->layers[l];
        for (int k = 0; k < ly->out_dim * ly->in_dim; k++) if (!isfinite(ly->W[k])) bad++;
        for (int i = 0; i < ly->out_dim; i++)              if (!isfinite(ly->b[i])) bad++;
    }
    return bad;
}
static double l2(Network *n) {
    double s = 0;
    for (int l = 0; l < n->num_layers; l++) {
        DenseLayer *ly = &n->layers[l];
        for (int k = 0; k < ly->out_dim * ly->in_dim; k++) s += (double)ly->W[k] * ly->W[k];
    }
    return sqrt(s);
}

int main(void) {
    rng_seed(7);
    Network policy;
    int topo[3] = {OBS_DIM, 64, N_ACTIONS};
    ActivationType acts[2] = {ACT_RELU, ACT_SOFTMAX};
    if (!network_init(&policy, 3, topo, acts)) { printf("init policy KO\n"); return 1; }

    EpisodeBuffer buf;
    if (!episode_buffer_init(&buf, MAX_STEPS_PER_EP, OBS_DIM)) { printf("init buffer KO\n"); return 1; }

    /* Traiettoria sintetica: la policy campiona davvero, cosi' il test copre
     * anche policy_sample_action. */
    float obs[OBS_DIM];
    const uint32_t T = MAX_STEPS_PER_EP;
    for (uint32_t t = 0; t < T; t++) {
        for (int i = 0; i < OBS_DIM; i++) obs[i] = 2.f * rng_uniform() - 1.f;
        uint32_t a = policy_sample_action(&policy, obs);
        if (a >= N_ACTIONS) { printf("azione fuori range dal campionamento!\n"); return 1; }
        episode_buffer_push(&buf, obs, a, 1.0f);   /* CartPole: +1 per step */
    }
    printf("buffer: size=%u capacity=%u  (attesi %u/%u)\n",
           buf.size, buf.capacity, (unsigned)T, (unsigned)T);
    printf("actions[ultimo]=%u  (l'arena arriva fino in fondo; che non la superi\n"
           "  lo verifica ASan)\n", buf.actions[T - 1]);

    uint32_t before = buf.size;
    episode_buffer_push(&buf, obs, 0, 1.0f);
    printf("push oltre capienza: size %u -> %u %s\n", before, buf.size,
           buf.size == before ? "(ignorato, ok)" : "(SCRITTURA FUORI BUFFER!)");

    episode_buffer_reset(&buf);
    episode_buffer_push(&buf, obs, N_ACTIONS + 7, 1.0f);
    printf("push con azione fuori range: size %u %s\n", buf.size,
           buf.size == 0 ? "(ignorato, ok)" : "(ACCETTATO!)");

    for (uint32_t t = 0; t < T; t++) {
        for (int i = 0; i < OBS_DIM; i++) obs[i] = 2.f * rng_uniform() - 1.f;
        episode_buffer_push(&buf, obs, policy_sample_action(&policy, obs), 1.0f);
    }

    /* returns: con reward costante 1, G_t = (1-gamma^(T-t))/(1-gamma) */
    compute_returns(&buf, REINFORCE_GAMMA);
    printf("returns: G[T-1]=%.4f (atteso 1.0), G[0]=%.4f (atteso %.4f)\n",
           buf.returns[T - 1], buf.returns[0],
           (1.0 - pow(REINFORCE_GAMMA, T)) / (1.0 - REINFORCE_GAMMA));

    normalize_returns(&buf);
    double mean = 0, var = 0;
    for (uint32_t i = 0; i < buf.size; i++) mean += buf.returns[i];
    mean /= buf.size;
    for (uint32_t i = 0; i < buf.size; i++) var += (buf.returns[i]-mean)*(buf.returns[i]-mean);
    printf("returns normalizzati: media %.2e, std %.6f (attesi ~0 e ~1)\n",
           mean, sqrt(var/buf.size));

    double a0 = l2(&policy);
    uint32_t adam_t0 = policy.adam_t;
    reinforce_update(&policy, &buf);
    double a1 = l2(&policy);

    printf("||W|| policy %.6f -> %.6f  (delta %.3e)\n", a0, a1, a1 - a0);
    printf("adam_t %u -> %u  (un solo passo per episodio)\n", adam_t0, policy.adam_t);

    double gsum = 0;
    for (int l = 0; l < policy.num_layers; l++) {
        DenseLayer *ly = &policy.layers[l];
        for (int k = 0; k < ly->out_dim * ly->in_dim; k++) gsum += fabs(ly->dW[k]);
        for (int i = 0; i < ly->out_dim; i++)              gsum += fabs(ly->db[i]);
    }
    printf("somma |gradienti| dopo l'update: %.3e (atteso 0)\n", gsum);

    int bad = nonfinite(&policy);
    printf("pesi non finiti dopo l'update: %d\n", bad);

    episode_buffer_reset(&buf);
    uint32_t at = policy.adam_t; double w = l2(&policy);
    reinforce_update(&policy, &buf);
    int noop = (policy.adam_t == at && l2(&policy) == w);
    printf("update su episodio vuoto: %s\n", noop ? "no-op (ok)" : "HA MODIFICATO LA RETE!");

    int ok = (bad == 0 && gsum == 0.0 && a1 != a0 &&
              policy.adam_t == adam_t0 + 1 && noop);

    episode_buffer_free(&buf);
    network_free(&policy);
    printf("\n%s\n", ok ? ">>> SMOKE TEST OK" : ">>> SMOKE TEST FALLITO");
    return ok ? 0 : 1;
}
