/* Smoke test end-to-end: riempie un rollout sintetico, gira ppo_update e
 * verifica che l'arena del buffer sia coerente, che i pesi si muovano e che
 * non compaiano NaN/Inf. */
#include "neural_net.h"
#include "ppo.h"
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
    Network actor, critic;
    int ta[4] = {OBS_DIM, 32, 32, N_ACT_DIMS}, tc[4] = {OBS_DIM, 32, 32, 1};
    ActivationType aa[3] = {ACT_TANH, ACT_TANH, ACT_NONE};
    ActivationType ac[3] = {ACT_RELU, ACT_RELU, ACT_NONE};
    if (!network_init(&actor, 4, ta, aa) || !network_init(&critic, 4, tc, ac)) return 1;
    ppo_sigma_init();

    RolloutBuffer buf;
    if (!rollout_buffer_init(&buf, ROLLOUT_STEPS, OBS_DIM)) { printf("init buffer KO\n"); return 1; }

    float obs[OBS_DIM], a[N_ACT_DIMS], z[N_ACT_DIMS], lp, v;
    for (uint32_t t = 0; t < ROLLOUT_STEPS; t++) {
        for (int i = 0; i < OBS_DIM; i++) obs[i] = 2.f * rng_uniform() - 1.f;
        actor_sample_action(&actor, obs, a, z, &lp, &v, &critic);
        rollout_buffer_push(&buf, obs, z, 2.f * rng_uniform() - 1.f,
                            (t % 200 == 199), lp, v);
    }
    printf("buffer: size=%u capacity=%u  (attesi %u/%u)\n",
           buf.size, buf.capacity, (unsigned)ROLLOUT_STEPS, (unsigned)ROLLOUT_STEPS);
    printf("dones[ultimo] = %u  (l'arena arriva fino in fondo; i limiti li verifica ASan)\n",
           buf.dones[ROLLOUT_STEPS - 1]);

    /* push oltre capienza: deve essere un no-op */
    uint32_t before = buf.size;
    rollout_buffer_push(&buf, obs, z, 0.f, 0, 0.f, 0.f);
    printf("push oltre capienza: size %u -> %u %s\n", before, buf.size,
           buf.size == before ? "(ignorato, ok)" : "(SCRITTURA FUORI BUFFER!)");

    compute_gae(&buf, 0.1f, PPO_GAMMA, PPO_LAMBDA);
    normalize_advantages(&buf);

    double mean = 0, var = 0;
    for (uint32_t i = 0; i < buf.size; i++) mean += buf.advantages[i];
    mean /= buf.size;
    for (uint32_t i = 0; i < buf.size; i++) var += (buf.advantages[i]-mean)*(buf.advantages[i]-mean);
    printf("advantages normalizzati: media %.2e, std %.6f (attesi ~0 e ~1)\n",
           mean, sqrt(var/buf.size));

    double a0 = l2(&actor), c0 = l2(&critic);
    float sig0 = g_ppo_log_sigma[0];
    ppo_update(&actor, &critic, &buf);
    double a1 = l2(&actor), c1 = l2(&critic);

    printf("||W|| actor  %.6f -> %.6f  (delta %.3e)\n", a0, a1, a1-a0);
    printf("||W|| critic %.6f -> %.6f  (delta %.3e)\n", c0, c1, c1-c0);
    printf("log_sigma[0] %.6f -> %.6f  (decay applicato una volta)\n", sig0, g_ppo_log_sigma[0]);

    int bad = nonfinite(&actor) + nonfinite(&critic);
    printf("pesi non finiti dopo l'update: %d\n", bad);

    rollout_buffer_free(&buf);
    printf("\n%s\n", (bad == 0 && buf.size == 0 && a1 != a0 && c1 != c0)
                     ? ">>> SMOKE TEST OK" : ">>> SMOKE TEST FALLITO");
    return (bad == 0) ? 0 : 1;
}
