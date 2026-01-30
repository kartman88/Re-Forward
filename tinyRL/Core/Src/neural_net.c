#include "neural_net.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

int init_network(SharedBackbone *net, int num_layers, int num_layers_actor, int num_layers_critic,
		int *net_topology, int *net_topology_actor, int *net_topology_critic,
		ActivationType *activations, ActivationType *activations_actor, ActivationType *activations_critic){
	net->adam_t = 0;
    net->layers  = malloc(num_layers * sizeof(DenseLayer));
    num_layers--; //there are num_layers-1 set of weight
    net->num_layers = num_layers;

    Head *actor = &net->actor;
    num_layers_actor--;
    actor->num_layers = num_layers_actor;
    actor->layers = malloc(num_layers_actor * sizeof(DenseLayer));
    Head *critic = &net->critic;
    num_layers_critic--;
    critic->num_layers = num_layers_critic;
    critic->layers = malloc(num_layers_critic * sizeof(DenseLayer));

    if(net->layers == NULL) return 0;

    for(int i = 0; i < net->num_layers; i++){
        if(!dense_init(&net->layers[i], net_topology[i], net_topology[i+1], activations[i])) return 0;
        init_layer_params(&net->layers[i]);
    }

    if(!dense_init(&net->layers[num_layers], net_topology[num_layers], net_topology_actor[0], activations[num_layers])) return 0;
    init_layer_params(&net->layers[num_layers]);

    for(int i = 0; i < actor->num_layers; i++){
    	if(!dense_init(&actor->layers[i], net_topology_actor[i], net_topology_actor[i+1], activations_actor[i])) return 0;
		init_layer_params(&actor->layers[i]);
	}

    for(int i = 0; i < critic->num_layers; i++){
    	if(!dense_init(&critic->layers[i], net_topology_critic[i], net_topology_critic[i+1], activations_critic[i])) return 0;
		init_layer_params(&critic->layers[i]);
	}
    return 1;
}

void softmax(float *in, float *out, int n)
{
    float max = in[0];
    for (int i = 1; i < n; ++i) if (in[i] > max) max = in[i];

    float sum = 0.f;
    for (int i = 0; i < n; ++i) {
        out[i] = expf(in[i] - max);
        sum   += out[i];
    }
    float inv = 1.f / sum;
    for (int i = 0; i < n; ++i){
    	out[i] *= inv;
    	if(out[i] < 1e-7f) out[i] = 1e-7f;
	}
}

int forward(SharedBackbone *net, float *input, float *output_final)
{
    const float *curr_in  = input;
    float *curr_out = NULL;

    for (int l = 0; l < net->num_layers; ++l) {

        DenseLayer *layer = &net->layers[l];
        curr_out = layer->out; //pointer to layer's output

        // fully connected product W·x + b
        for (int i = 0; i < layer->out_dim; ++i) {
            float acc = layer->b[i];
            for (int j = 0; j < layer->in_dim; ++j)
                acc += layer->W[i][j] * curr_in[j];
            curr_out[i] = acc; //logit
        }

        //activation
        switch (layer->activation) {
            case ACT_RELU:
                for (int i = 0; i < layer->out_dim; ++i)
                    if (curr_out[i] < 0.f) curr_out[i] = 0.f;
                break;
            case ACT_SOFTMAX:
                softmax(curr_out, curr_out, layer->out_dim);
                break;
            default: break;
        }

        //l'uscita diventa input per il prossimo layer
        curr_in = curr_out;
    }

    //copy the final output
    if(output_final) memcpy(output_final, curr_out, net->layers[net->num_layers-1].out_dim * sizeof(float));

    return 1;
}

void zero_grad(SharedBackbone *net){
	//zero grad
	for (int l = 0; l < net->num_layers; ++l) {
		DenseLayer *ly = &net->layers[l];
		memset(ly->dW[0], 0, ly->out_dim * ly->in_dim * sizeof(float));
		memset(ly->db   , 0, ly->out_dim * sizeof(float));
	}
}

void backward_core(SharedBackbone *net, float *dout_last, float *input){
	//backprop of the gradient
	float *delta = dout_last;
	float *prev = NULL;
	static float *delta_buf = NULL;
	static uint32_t delta_cap = 0;

	for(int l = net->num_layers - 1; l >= 0; --l){
		DenseLayer *ly = &net->layers[l];

		float *inp = (l == 0) ? input : net->layers[l-1].out; //BUCO ADT IMPLEMENTARE UNA GET

		//accumulate grad
		for (int i = 0; i < ly->out_dim; ++i) {
		    ly->db[i] += delta[i];
		    for (int j = 0; j < ly->in_dim; ++j)
		        ly->dW[i][j] += delta[i] * inp[j];
		}

		//prepare next delta if not the last layer
		if(l > 0){
			//memory optimization, allocate only if first time or dim is different
			if (ly->in_dim > delta_cap){
				free(delta_buf);
				delta_buf  = malloc(ly->in_dim * sizeof(float));
				delta_cap  = ly->in_dim;
				if (!delta_buf) return;
			}
			prev = delta_buf;

			//accumulate grad
			for(int j = 0; j < ly->in_dim; ++j){
				float acc = 0.0f;
				for(int i = 0; i< ly->out_dim; ++i){
					acc += delta[i] * ly->W[i][j];
				}
				float h_prev = net->layers[l-1].out[j]; //BUCO ADT CREARE UNA GET
				switch(net->layers[l-1].activation){ //BUCO ADT CREARE UNA GET
					case ACT_RELU: acc = (h_prev > 0.f) ? acc : 0.f; break;
					default: break; //TODO ACT_NONE etc...
				}
				prev[j] = acc;
			}
			delta = prev;
		}
	}
}

void backward_pg(SharedBackbone *net, float *input, uint8_t action, float advantage, float reward, uint32_t step_count){
	DenseLayer *last = &net->layers[net->num_layers - 1];
	float *p = last->out; //softmax prob.0

	//float *dlogit = malloc(last->out_dim * sizeof(float));
	float dlogit[last->out_dim];
	for (int i = 0; i < last->out_dim; ++i){
		float pi = fmaxf(p[i], 1e-6f); //clamp to avoid NaN values
		float pg  = ((i == action) ? (1.f - pi) : - pi) * (advantage * reward);
		float ent = ENT_BETA * (-logf(pi) - 1.f);
		dlogit[i] = pg + ent;
}
	backward_core(net, dlogit, input); //do the rest of the backprop
	//free(dlogit);
}

void adam_optimizer(SharedBackbone *net){
	//Adam update (ascent)
	net->adam_t++;
	const float b1t = 1.f - powf(BETA1, (float)net->adam_t);
	const float b2t = 1.f - powf(BETA2, (float)net->adam_t);

	for (int l = 0; l < net->num_layers; ++l) {
		DenseLayer *ly = &net->layers[l];

		//bias
		for (int i = 0; i < ly->out_dim; ++i) {
			ly->mb[i] = BETA1*ly->mb[i] + (1.f-BETA1)*ly->db[i];
			ly->vb[i] = BETA2*ly->vb[i] + (1.f-BETA2)*ly->db[i]*ly->db[i];
			ly->b [i]+= LR * (ly->mb[i]/b1t) / (sqrtf(ly->vb[i]/b2t) + EPS_ADAM);
		}
		//weight
		for (int i = 0; i < ly->out_dim; ++i)
			for (int j = 0; j < ly->in_dim; ++j) {
				ly->mW[i][j] = BETA1*ly->mW[i][j] + (1.f-BETA1)*ly->dW[i][j];
				ly->vW[i][j] = BETA2*ly->vW[i][j] + (1.f-BETA2)*ly->dW[i][j]*ly->dW[i][j];
				ly->W[i][j]+= LR * (ly->mW[i][j]/b1t) / (sqrtf(ly->vW[i][j]/b2t) + EPS_ADAM);
			}
	}
}

void gradient_norm_l2(SharedBackbone *net){
	/* ---- gradient-norm (L2) ----------------------------------- */
	float gnorm_sq = 0.f;

	for (int l = 0; l < net->num_layers; ++l) {
	    DenseLayer *ly = &net->layers[l];

	    for (int i = 0; i < ly->out_dim; ++i) {
	        gnorm_sq += ly->db[i] * ly->db[i];
	        for (int j = 0; j < ly->in_dim; ++j)
	            gnorm_sq += ly->dW[i][j] * ly->dW[i][j];
	    }
	}
	float gnorm = sqrtf(gnorm_sq);

	/* ---- clipping --------------------------------------------- */
	const float CLIP = 5.0f;               /* soglia consigliata */

	if (gnorm > CLIP) {
	    float s = CLIP / gnorm;            /* fattore di scala */

	    for (int l = 0; l < net->num_layers; ++l) {
	        DenseLayer *ly = &net->layers[l];
	        for (int i = 0; i < ly->out_dim; ++i) {
	            ly->db[i] *= s;
	            for (int j = 0; j < ly->in_dim; ++j)
	                ly->dW[i][j] *= s;
	        }
	    }
	}
}

