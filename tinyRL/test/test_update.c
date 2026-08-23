/* Smoke test end-to-end: riempie un replay buffer sintetico, gira dqn_train e
 * verifica che l'arena del buffer sia coerente, che i pesi si muovano, che la
 * copia verso la rete target sia esatta e che non compaiano NaN/Inf. */
#include "neural_net.h"
#include "dqn.h"
#include "rng.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

uint32_t dwt_ticks(void) { return 0; }
uint32_t dwt_delta(uint32_t a, uint32_t b) { (void)a; (void)b; return 0; }

static int nonfinite(QNetwork *n) {
    int bad = 0;
    for (int l = 0; l < n->num_layers; l++) {
        DenseLayer *ly = &n->layers[l];
        for (int k = 0; k < ly->out_dim * ly->in_dim; k++) if (!isfinite(ly->W[k])) bad++;
        for (int i = 0; i < ly->out_dim; i++)              if (!isfinite(ly->b[i])) bad++;
    }
    return bad;
}
static double l2(QNetwork *n) {
    double s = 0;
    for (int l = 0; l < n->num_layers; l++) {
        DenseLayer *ly = &n->layers[l];
        for (int k = 0; k < ly->out_dim * ly->in_dim; k++) s += (double)ly->W[k] * ly->W[k];
    }
    return sqrt(s);
}

int main(void) {
    rng_seed(7);
    QNetwork online;
    TargetNetwork target;
    int topo[4] = {OBS_DIM, 32, 32, N_ACTIONS};
    ActivationType acts[3] = {ACT_RELU, ACT_RELU, ACT_NONE};
    if (!init_qnetwork(&online, 4, topo, acts))   { printf("init online KO\n"); return 1; }
    if (!init_target_network(&target, 4, topo))   { printf("init target KO\n"); return 1; }
    copy_weights_to_target(&online, &target);

    ReplayBuffer buf;
    if (!replay_buffer_init(&buf, REPLAY_SIZE, OBS_DIM)) { printf("init buffer KO\n"); return 1; }

    float s[OBS_DIM], s_next[OBS_DIM];
    for (uint32_t t = 0; t < REPLAY_SIZE; t++) {
        for (int i = 0; i < OBS_DIM; i++) {
            s[i]      = 2.f * rng_uniform() - 1.f;
            s_next[i] = 2.f * rng_uniform() - 1.f;
        }
        replay_buffer_push(&buf, s, rng_u32() % N_ACTIONS,
                           2.f * rng_uniform() - 1.f, s_next, (t % 200 == 199));
    }
    printf("buffer: size=%u capacity=%u head=%u  (attesi %u/%u/0)\n",
           buf.size, buf.capacity, buf.head,
           (unsigned)REPLAY_SIZE, (unsigned)REPLAY_SIZE);
    printf("done[ultimo]=%u action[ultimo]=%u  (l'arena arriva fino in fondo;\n"
           "  che non la superi lo verifica ASan)\n",
           buf.done[REPLAY_SIZE - 1], buf.action[REPLAY_SIZE - 1]);

    /* Il buffer e' circolare: oltre capienza sovrascrive, non cresce. */
    uint32_t before = buf.size;
    replay_buffer_push(&buf, s, 0, 0.f, s_next, 0);
    printf("push a buffer pieno: size %u -> %u, head %u %s\n", before, buf.size, buf.head,
           (buf.size == before && buf.head == 1) ? "(sovrascrive, ok)" : "(ANOMALIA)");

    /* Azione fuori range: deve essere scartata, non troncata a uint8_t. */
    before = buf.size;
    uint32_t head_before = buf.head;
    replay_buffer_push(&buf, s, N_ACTIONS + 7, 0.f, s_next, 0);
    printf("push con azione fuori range: head %u -> %u %s\n", head_before, buf.head,
           buf.head == head_before ? "(ignorato, ok)" : "(ACCETTATO!)");

    double a0 = l2(&online);
    uint32_t adam_t0 = online.adam_t;
    dqn_train(&online, &target, &buf, BATCH_SIZE, N_ACTIONS);
    double a1 = l2(&online);

    printf("||W|| online %.6f -> %.6f  (delta %.3e)\n", a0, a1, a1 - a0);
    printf("adam_t %u -> %u  (un solo passo per update)\n", adam_t0, online.adam_t);

    /* Dopo Adam i gradienti devono essere gia' azzerati: zero_grad_q a inizio
     * update non deve avere niente da ripulire. */
    double gsum = 0;
    for (int l = 0; l < online.num_layers; l++) {
        DenseLayer *ly = &online.layers[l];
        for (int k = 0; k < ly->out_dim * ly->in_dim; k++) gsum += fabs(ly->dW[k]);
        for (int i = 0; i < ly->out_dim; i++)              gsum += fabs(ly->db[i]);
    }
    printf("somma |gradienti| dopo l'update: %.3e (atteso 0)\n", gsum);

    /* copy_weights_to_target su arene piatte: deve essere bit-per-bit. */
    copy_weights_to_target(&online, &target);
    int diff = 0;
    for (int l = 0; l < online.num_layers; l++) {
        DenseLayer  *sl = &online.layers[l];
        TargetLayer *tl = &target.layers[l];
        if (memcmp(tl->W, sl->W, (size_t)sl->out_dim * sl->in_dim * sizeof(float))) diff++;
        if (memcmp(tl->b, sl->b, (size_t)sl->out_dim * sizeof(float)))              diff++;
        if (tl->activation != sl->activation)                                       diff++;
    }
    printf("copy_weights_to_target: %d blocchi diversi (atteso 0)\n", diff);

    int bad = nonfinite(&online);
    printf("pesi non finiti dopo l'update: %d\n", bad);

    int ok = (bad == 0 && diff == 0 && gsum == 0.0 && a1 != a0 &&
              buf.size == REPLAY_SIZE && online.adam_t == adam_t0 + 1);

    replay_buffer_free(&buf);
    qnetwork_free(&online);
    target_free(&target);
    printf("\n%s\n", ok ? ">>> SMOKE TEST OK" : ">>> SMOKE TEST FALLITO");
    return ok ? 0 : 1;
}
