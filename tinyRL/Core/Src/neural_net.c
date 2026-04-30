#include "neural_net.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

int init_network(SharedBackbone *net, int num_layers, int num_layers_actor,
                 int num_layers_critic, int *net_topology,
                 int *net_topology_actor, int *net_topology_critic,
                 ActivationType *activations, ActivationType *activations_actor,
                 ActivationType *activations_critic) {
  net->adam_t = 0;
  net->episode_count = 0;
  net->layers =
      // +2: one slot for link_actor, one for link_critic (both live in
      // net->layers)
      malloc((num_layers + 2) *
             sizeof(DenseLayer)); // should be num_layers +1 because we have
                                  // separate actor critic layers
  num_layers--; // there are num_layers-1 set of weight MAYBE I DON'T NEED THIS
                // BECAUSE I WILL HAVE 2 SET OF WEIGHTS AS OUT
  net->num_layers = num_layers;

  Head *actor = &net->actor;
  num_layers_actor--; // there are num_layers-1 set of weight
  actor->num_layers = num_layers_actor;
  actor->layers = malloc(num_layers_actor * sizeof(DenseLayer));
  Head *critic = &net->critic;
  num_layers_critic--; // there are num_layers-1 set of weight
  critic->num_layers = num_layers_critic;
  critic->layers = malloc(num_layers_critic * sizeof(DenseLayer));

  if (net->layers == NULL)
    return 0;

  // We do num_layer -1 inside the loop and 1 outside so the total set of
  // weights will be num_layer
  for (int i = 0; i < net->num_layers; i++) {
    if (!dense_init(&net->layers[i], net_topology[i], net_topology[i + 1],
                    activations[i]))
      return 0;
    init_layer_params(&net->layers[i]);
  }
  // This one is out of the loop because connect the shared backbone to the
  // actor/critic
  if (!dense_init(&net->layers[num_layers], net_topology[num_layers],
                  net_topology_actor[0], activations[num_layers]))
    return 0;
  init_layer_params(&net->layers[num_layers]);

  // Critic connection init
  if (!dense_init(&net->layers[num_layers + 1], net_topology[num_layers],
                  net_topology_critic[0], activations[num_layers]))
    return 0;
  init_layer_params(&net->layers[num_layers + 1]);

  for (int i = 0; i < actor->num_layers; i++) {
    if (!dense_init(&actor->layers[i], net_topology_actor[i],
                    net_topology_actor[i + 1], activations_actor[i]))
      return 0;
    init_layer_params(&actor->layers[i]);

    // PPO CONTINUOUS FIX - Final Layer Initialization Scaling
    // Standard practice is to scale the final policy weights by 0.01
    // to start training with actions perfectly centered at ~0.0 (mu=0).
    // Otherwise, Glorot limit pushes Tanh to +/- 1.0 immediately!
    if (i == actor->num_layers - 1) {
      for (int r = 0; r < actor->layers[i].out_dim; r++) {
        for (int c = 0; c < actor->layers[i].in_dim; c++) {
          actor->layers[i].W[r][c] *= 0.01f;
        }
      }
    }
  }

  for (int i = 0; i < critic->num_layers; i++) {
    if (!dense_init(&critic->layers[i], net_topology_critic[i],
                    net_topology_critic[i + 1], activations_critic[i]))
      return 0;
    init_layer_params(&critic->layers[i]);
  }
  return 1;
}

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

static inline void forward_dense_layer(const float *restrict in_vec,
                                       float *restrict out_vec,
                                       const DenseLayer *restrict layer) {
  const int in_dim = layer->in_dim;
  const int out_dim = layer->out_dim;

  for (int i = 0; i < out_dim; ++i) {
    float acc = layer->b[i];
    const float *restrict w_row = layer->W[i];
    for (int j = 0; j < in_dim; ++j) {
      acc += w_row[j] * in_vec[j];
    }

    switch (layer->activation) {
    case ACT_RELU:
      out_vec[i] = (acc > 0.f) ? acc : 0.f;
      break;
    case ACT_TANH:
      out_vec[i] = tanhf(acc);
      break;
    default:
      out_vec[i] = acc;
      break;
    }
  }

  if (layer->activation == ACT_SOFTMAX) {
    softmax(out_vec, out_vec, out_dim);
  }
}

int forward(SharedBackbone *net, float *input, float *output_actor,
            float *output_critic) {
  const float *curr_in = input;
  float *curr_out = NULL;

  // SHARED BACKBONE FORWARD
  for (int l = 0; l < net->num_layers; ++l) {
    DenseLayer *layer = &net->layers[l];
    curr_out = layer->out;
    forward_dense_layer(curr_in, curr_out, layer);
    curr_in = curr_out;
  }
  const float *backbone_out = curr_out;

  // Evaluate output for actor link
  DenseLayer *layer_actor = &net->layers[net->num_layers];
  curr_out = layer_actor->out;
  forward_dense_layer(backbone_out, curr_out, layer_actor);
  curr_in = curr_out;

  // ACTOR FORWARD
  for (int l = 0; l < net->actor.num_layers; ++l) {
    DenseLayer *ly = &net->actor.layers[l];
    curr_out = ly->out;
    forward_dense_layer(curr_in, curr_out, ly);
    curr_in = curr_out;
  }

  if (output_actor)
    memcpy(output_actor, curr_out,
           net->actor.layers[net->actor.num_layers - 1].out_dim *
               sizeof(float));

  // Evaluate output for critic link
  DenseLayer *layer_critic = &net->layers[net->num_layers + 1];
  curr_out = layer_critic->out;
  forward_dense_layer(backbone_out, curr_out, layer_critic);
  curr_in = curr_out;

  // CRITIC FORWARD
  for (int l = 0; l < net->critic.num_layers; ++l) {
    DenseLayer *ly = &net->critic.layers[l];
    curr_out = ly->out;
    forward_dense_layer(curr_in, curr_out, ly);
    curr_in = curr_out;
  }

  if (output_critic)
    memcpy(output_critic, curr_out,
           net->critic.layers[net->critic.num_layers - 1].out_dim *
               sizeof(float));

  return 1;
}

// Funzione helper per azzerare i gradienti di un singolo layer in sicurezza
static void zero_layer_grad(DenseLayer *ly) {
  for (int i = 0; i < ly->out_dim; i++) {
    memset(ly->dW[i], 0, ly->in_dim * sizeof(float));
  }
  memset(ly->db, 0, ly->out_dim * sizeof(float));
}

void zero_grad(SharedBackbone *net) {
  // 1. Zero grad Shared Backbone + Link Layers
  for (int l = 0; l < net->num_layers + 2; l++) {
    zero_layer_grad(&net->layers[l]);
  }

  // 2. Zero grad Actor Head
  for (int l = 0; l < net->actor.num_layers; l++) {
    zero_layer_grad(&net->actor.layers[l]);
  }

  // 3. Zero grad Critic Head
  for (int l = 0; l < net->critic.num_layers; l++) {
    zero_layer_grad(&net->critic.layers[l]);
  }
}

void backward_core(SharedBackbone *net, float *dout_last, float *input) {
  // backprop of the gradient
  float *delta = dout_last;
  float *prev = NULL;
  static float *delta_buf = NULL;
  static uint32_t delta_cap = 0;

  for (int l = net->num_layers - 1; l >= 0; --l) {
    DenseLayer *ly = &net->layers[l];
    const int in_dim = ly->in_dim;
    const int out_dim = ly->out_dim;

    const float *restrict inp = (l == 0) ? input : net->layers[l - 1].out;
    const float *restrict d = delta;

    // accumulate grad
    float *restrict db = ly->db;
    for (int i = 0; i < out_dim; ++i) {
      db[i] += d[i];
      float *restrict dw_row = ly->dW[i];
      float di = d[i];
      for (int j = 0; j < in_dim; ++j) {
        dw_row[j] += di * inp[j];
      }
    }

    // prepare next delta if not the last layer
    if (l > 0) {
      if (in_dim > delta_cap) {
        free(delta_buf);
        delta_buf = malloc(in_dim * sizeof(float));
        delta_cap = in_dim;
        if (!delta_buf)
          return;
      }
      prev = delta_buf;

      const float *restrict h_prev = net->layers[l - 1].out;
      ActivationType act = net->layers[l - 1].activation;

      for (int j = 0; j < in_dim; ++j) {
        float acc = 0.0f;
        for (int i = 0; i < out_dim; ++i) {
          acc += d[i] * ly->W[i][j];
        }
        float hp = h_prev[j];
        switch (act) {
        case ACT_RELU:
          acc = (hp > 0.f) ? acc : 0.f;
          break;
        case ACT_TANH:
          acc = acc * (1.f - hp * hp);
          break;
        default:
          break;
        }
        prev[j] = acc;
      }
      delta = prev;
    }
  }
}

void backward_pg(SharedBackbone *net, float *input, action_t action,
                 float advantage, float reward, uint32_t step_count) {
  DenseLayer *last = &net->layers[net->num_layers - 1];
  float *p = last->out; // softmax prob.0

  // float *dlogit = malloc(last->out_dim * sizeof(float));
  float dlogit[last->out_dim];
  for (int i = 0; i < last->out_dim; ++i) {
    float pi = fmaxf(p[i], 1e-6f); // clamp to avoid NaN values
    float pg = ((i == action) ? (1.f - pi) : -pi) * (advantage * reward);
    float ent = ENT_BETA * (-logf(pi) - 1.f);
    dlogit[i] = pg + ent;
  }
  backward_core(net, dlogit, input); // do the rest of the backprop
  // free(dlogit);
}

// Funzione helper che aggiorna un singolo layer (usabile per Shared, Actor e
// Critic)
void adam_update_single_layer(DenseLayer *restrict ly, float b1t, float b2t) {
  const int out_dim = ly->out_dim;
  const int in_dim = ly->in_dim;

  // Bias
  for (int i = 0; i < out_dim; ++i) {
    float db = ly->db[i];
    float mb = BETA1 * ly->mb[i] + (1.f - BETA1) * db;
    float vb = BETA2 * ly->vb[i] + (1.f - BETA2) * db * db;
    ly->mb[i] = mb;
    ly->vb[i] = vb;

    float m_hat = mb / b1t;
    float v_hat = vb / b2t;

    ly->b[i] -= LR * m_hat / (sqrtf(v_hat) + EPS_ADAM);
    ly->db[i] = 0.0f;
  }

  // Weights
  for (int i = 0; i < out_dim; ++i) {
    float *restrict mw_row = ly->mW[i];
    float *restrict vw_row = ly->vW[i];
    float *restrict w_row = ly->W[i];
    float *restrict dw_row = ly->dW[i];

    for (int j = 0; j < in_dim; ++j) {
      float dw = dw_row[j];
      float mw = BETA1 * mw_row[j] + (1.f - BETA1) * dw;
      float vw = BETA2 * vw_row[j] + (1.f - BETA2) * dw * dw;
      mw_row[j] = mw;
      vw_row[j] = vw;

      float m_hat = mw / b1t;
      float v_hat = vw / b2t;

      w_row[j] -= LR * m_hat / (sqrtf(v_hat) + EPS_ADAM);
      dw_row[j] = 0.0f;
    }
  }
}

void adam_optimizer(SharedBackbone *net) {
  // 1. Aggiornamento Time Step Globale
  net->adam_t++;

  // Calcolo coefficienti validi per tutti i layer di questa epoch
  // Limit bias correction to first 1500 steps to save intense MCU clock cycles
  // from power float operations when b1t/b2t practically approaches 1.0f
  // Bias correction terms: always active for mathematically correct Adam.
  // Old code disabled this after 1500 steps causing a ~12% LR discontinuity
  // (1 - 0.999^1500 ≈ 0.777, not 1.0). On STM32H7 with FPU, powf() is fast.
  float b1t = 1.f - powf(BETA1, (float)net->adam_t);
  float b2t = 1.f - powf(BETA2, (float)net->adam_t);

  // ----------------------------------------------------
  // A. Aggiornamento Shared Backbone + Link Layers
  // ----------------------------------------------------
  // Il loop va da 0 a num_layers + 2 (Trunk + LinkActor + LinkCritic)
  // Assicurati che l'allocazione di net->layers sia di dimensione 3!
  for (int l = 0; l < net->num_layers + 2; ++l) {
    adam_update_single_layer(&net->layers[l], b1t, b2t);
  }

  // ----------------------------------------------------
  // B. Aggiornamento Actor Head
  // ----------------------------------------------------
  for (int l = 0; l < net->actor.num_layers; ++l) {
    adam_update_single_layer(&net->actor.layers[l], b1t, b2t);
  }

  // ----------------------------------------------------
  // C. Aggiornamento Critic Head
  // ----------------------------------------------------
  for (int l = 0; l < net->critic.num_layers; ++l) {
    adam_update_single_layer(&net->critic.layers[l], b1t, b2t);
  }
}

void gradient_norm_l2(SharedBackbone *net) {
  float gnorm_sq = 0.f;

  // 1. Calcola norma Trunk + Links
  for (int l = 0; l < net->num_layers + 2; ++l) {
    DenseLayer *ly = &net->layers[l];
    for (int i = 0; i < ly->out_dim; ++i) {
      gnorm_sq += ly->db[i] * ly->db[i];
      for (int j = 0; j < ly->in_dim; ++j)
        gnorm_sq += ly->dW[i][j] * ly->dW[i][j];
    }
  }
  // 2. Calcola norma Actor
  for (int l = 0; l < net->actor.num_layers; ++l) {
    DenseLayer *ly = &net->actor.layers[l];
    for (int i = 0; i < ly->out_dim; ++i) {
      gnorm_sq += ly->db[i] * ly->db[i];
      for (int j = 0; j < ly->in_dim; ++j)
        gnorm_sq += ly->dW[i][j] * ly->dW[i][j];
    }
  }
  // 3. Calcola norma Critic
  for (int l = 0; l < net->critic.num_layers; ++l) {
    DenseLayer *ly = &net->critic.layers[l];
    for (int i = 0; i < ly->out_dim; ++i) {
      gnorm_sq += ly->db[i] * ly->db[i];
      for (int j = 0; j < ly->in_dim; ++j)
        gnorm_sq += ly->dW[i][j] * ly->dW[i][j];
    }
  }

  float gnorm = sqrtf(gnorm_sq);
  const float CLIP = 0.5f;

  // Applica il Clipping a tutta la rete
  if (gnorm > CLIP) {
    float s = CLIP / gnorm;

    for (int l = 0; l < net->num_layers + 2; ++l) {
      DenseLayer *ly = &net->layers[l];
      for (int i = 0; i < ly->out_dim; ++i) {
        ly->db[i] *= s;
        for (int j = 0; j < ly->in_dim; ++j)
          ly->dW[i][j] *= s;
      }
    }
    for (int l = 0; l < net->actor.num_layers; ++l) {
      DenseLayer *ly = &net->actor.layers[l];
      for (int i = 0; i < ly->out_dim; ++i) {
        ly->db[i] *= s;
        for (int j = 0; j < ly->in_dim; ++j)
          ly->dW[i][j] *= s;
      }
    }
    for (int l = 0; l < net->critic.num_layers; ++l) {
      DenseLayer *ly = &net->critic.layers[l];
      for (int i = 0; i < ly->out_dim; ++i) {
        ly->db[i] *= s;
        for (int j = 0; j < ly->in_dim; ++j)
          ly->dW[i][j] *= s;
      }
    }
  }
}
