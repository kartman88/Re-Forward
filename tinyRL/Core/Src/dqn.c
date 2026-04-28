#include "dqn.h"

int replay_buffer_init(ReplayBuffer *buf, uint32_t capacity, uint32_t obs_dim) {
    buf->capacity = capacity;
    buf->obs_dim  = obs_dim;
    buf->head     = 0;
    buf->size     = 0;

    buf->state_pool = malloc(capacity * obs_dim * sizeof(float));
    buf->snext_pool = malloc(capacity * obs_dim * sizeof(float));
    buf->state      = malloc(capacity * sizeof(float *));
    buf->next_state = malloc(capacity * sizeof(float *));
    buf->action     = malloc(capacity * sizeof(uint32_t));
    buf->reward     = malloc(capacity * sizeof(float));
    buf->done       = malloc(capacity * sizeof(uint8_t));

    if (!buf->state_pool || !buf->snext_pool || !buf->state ||
        !buf->next_state || !buf->action || !buf->reward || !buf->done)
        return 0;

    for (uint32_t i = 0; i < capacity; i++) {
        buf->state[i]      = buf->state_pool + i * obs_dim;
        buf->next_state[i] = buf->snext_pool + i * obs_dim;
    }
    return 1;
}

void replay_buffer_push(ReplayBuffer *buf, float *s, uint32_t action,
                        float reward, float *s_next, uint8_t done) {
    uint32_t idx = buf->head;
    memcpy(buf->state[idx],      s,      buf->obs_dim * sizeof(float));
    memcpy(buf->next_state[idx], s_next, buf->obs_dim * sizeof(float));
    buf->action[idx] = action;
    buf->reward[idx] = reward;
    buf->done[idx]   = done;
    buf->head = (idx + 1) % buf->capacity;
    if (buf->size < buf->capacity)
        buf->size++;
}

int uart_recv_floats(UART_HandleTypeDef *huart, float *dst, size_t dim,
                     uint32_t timeout) {
    uint8_t stx;
    if (HAL_UART_Receive(huart, &stx, 1, timeout) != HAL_OK || stx != 0x02)
        return 0;
    if (HAL_UART_Receive(huart, (uint8_t *)dst, dim * sizeof(float), timeout) != HAL_OK)
        return 0;
    uint8_t etx;
    HAL_UART_Receive(huart, &etx, 1, timeout);
    return 1;
}

int uart_send_float_action(UART_HandleTypeDef *huart, float action_val,
                           uint8_t done, uint32_t timeout) {
    uint8_t pkt[7];
    pkt[0] = 0x02;
    memcpy(&pkt[1], &action_val, 4);
    pkt[5] = done;
    pkt[6] = 0x03;
    return HAL_UART_Transmit(huart, pkt, 7, timeout) == HAL_OK ? 1 : 0;
}

int uart_send_action_discrete(UART_HandleTypeDef *huart, uint32_t action,
                              uint8_t done, uint32_t timeout) {
    uint8_t pkt[4] = {0x02, (uint8_t)(action & 0xFF), done, 0x03};
    return HAL_UART_Transmit(huart, pkt, 4, timeout) == HAL_OK ? 1 : 0;
}

float calc_epsilon(uint32_t step) {
    float t = (float)step / EPSILON_DECAY;
    if (t > 1.0f) t = 1.0f;
    return EPSILON_START + t * (EPSILON_END - EPSILON_START);
}

uint32_t dqn_select_action(QNetwork *net, float *obs, float epsilon,
                            uint32_t n_actions) {
    float r = (float)rand() / ((float)RAND_MAX + 1.0f);
    if (r < epsilon)
        return (uint32_t)((uint32_t)rand() % n_actions);

    float q[N_ACTIONS];
    forward_q(net, obs, q);
    uint32_t best = 0;
    for (uint32_t i = 1; i < n_actions; i++)
        if (q[i] > q[best]) best = i;
    return best;
}

void dqn_train(QNetwork *online, TargetNetwork *target,
               ReplayBuffer *buf, uint32_t batch_size, uint32_t n_actions) {
    float q_online[N_ACTIONS];
    float q_tgt[N_ACTIONS];
    float inv_batch = 1.0f / (float)batch_size;

    zero_grad_q(online);

    for (uint32_t b = 0; b < batch_size; b++) {
        uint32_t idx    = (uint32_t)rand() % buf->size;
        float   *s      = buf->state[idx];
        float   *s_next = buf->next_state[idx];
        uint32_t act    = buf->action[idx];
        float    rew    = buf->reward[idx];
        uint8_t  dn     = buf->done[idx];

        forward_q(online, s, q_online);
        forward_target(target, s_next, q_tgt);

        float max_qt = q_tgt[0];
        for (uint32_t i = 1; i < n_actions; i++)
            if (q_tgt[i] > max_qt) max_qt = q_tgt[i];

        float target_val = dn ? rew : rew + GAMMA * max_qt;
        float td_error   = q_online[act] - target_val;

        dqn_backward(online, s, act, td_error * inv_batch);
    }

    gradient_norm_q(online);
    adam_optimizer_q(online);
}
