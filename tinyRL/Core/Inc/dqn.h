#ifndef DQN_H
#define DQN_H

#include "neural_net.h"
#include "main.h"   /* HAL della famiglia target (stm32h7xx_hal.h) + TIME_LOG */
#include <stdlib.h>
#include <string.h>

/* L'azione discreta sta in un byte: con N_ACTIONS = 5 un uint32_t per
 * transizione sprecava 3 KB su REPLAY_SIZE = 1000. Lo _Static_assert lega il
 * tipo alla costante che lo giustifica: alzare N_ACTIONS oltre 255 non passa
 * piu' inosservato. */
_Static_assert(N_ACTIONS <= 255, "action index deve stare in uint8_t");
_Static_assert(REPLAY_SIZE > 0,  "REPLAY_SIZE deve essere positivo");

/* Buffer di replay in UNA sola allocazione, con i vettori come viste dentro
 * l'arena. Prima erano 7 malloc, due delle quali (state[] e next_state[]) solo
 * array di puntatori di riga sopra i pool contigui: 8 KB di pura indirezione,
 * piu' una load in ogni accesso. Ora l'indice si calcola:
 *     stato i      = states      + i*obs_dim
 *     stato succ i = next_states + i*obs_dim
 *
 * Layout, campi a 4 byte prima di quelli a 1 byte cosi' l'aritmetica dei
 * puntatori resta allineata senza padding esplicito:
 *     [ states | next_states | rewards | actions | dones ]
 *
 * `head` e `size` NON sono ridondanti qui: e' un buffer circolare, head torna
 * a 0 mentre size si ferma a capacity. */
typedef struct {
    float    *arena;        // base dell'allocazione, la libera replay_buffer_free()
    float    *states;       // [capacity * obs_dim]
    float    *next_states;  // [capacity * obs_dim]
    float    *reward;       // [capacity]
    uint8_t  *action;       // [capacity]
    uint8_t  *done;         // [capacity]
    uint32_t  head;         // prossima posizione di scrittura (circolare)
    uint32_t  size;         // transizioni valide, satura a capacity
    uint32_t  capacity;
    uint32_t  obs_dim;
} ReplayBuffer;

#if TIME_LOG
/* Profiling: cicli DWT misurati dentro dqn_train (una chiamata = un update).
 * total = intero update; forward = forward_q+forward_target sul batch;
 * backward = dqn_backward sul batch; adam = gradient_norm_q+adam_optimizer_q.
 * Contatori a 32 bit: il DWT wrappa ogni ~8.9 s a 480 MHz, mentre un singolo
 * update DQN (batch di 32 transizioni) sta nell'ordine dei millisecondi, quindi
 * un update non arriva mai a wrappare. */
typedef struct {
    uint32_t total_cycles;
    uint32_t forward_cycles;
    uint32_t backward_cycles;
    uint32_t adam_cycles;
} TrainTiming;

extern TrainTiming g_train_timing;
#endif /* TIME_LOG */

int  replay_buffer_init(ReplayBuffer *buf, uint32_t capacity, uint32_t obs_dim);
void replay_buffer_free(ReplayBuffer *buf);
void replay_buffer_push(ReplayBuffer *buf, const float *s, uint32_t action,
                        float reward, const float *s_next, uint8_t done);

/* Viste sulla i-esima transizione, senza array di puntatori di riga. */
static inline float *replay_state(const ReplayBuffer *buf, uint32_t i) {
    return buf->states + (size_t)i * buf->obs_dim;
}
static inline float *replay_next_state(const ReplayBuffer *buf, uint32_t i) {
    return buf->next_states + (size_t)i * buf->obs_dim;
}

float    calc_epsilon(uint32_t step);
uint32_t dqn_select_action(QNetwork *net, const float *obs, float epsilon,
                            uint32_t n_actions);

void dqn_train(QNetwork *online, TargetNetwork *target,
               ReplayBuffer *buf, uint32_t batch_size, uint32_t n_actions);

#endif
