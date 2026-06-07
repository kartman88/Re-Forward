#ifndef RNG_H
#define RNG_H

#include <stdint.h>

// PRNG di qualità (PCG32) per il training PPO sul micro.
// Sostituisce rand()/RAND_MAX di newlib, il cui LCG debole produce coppie
// (u1, u2) correlate nel Box-Muller -> rumore di esplorazione non gaussiano.
// La baseline PC (pc_ppo_trainer.py) usa PCG64 di NumPy; qui usiamo PCG32 per
// avvicinare la qualità del random (esplorazione, shuffle, init pesi).

void     rng_seed(uint64_t seed);   // inizializza lo stato del generatore
uint32_t rng_u32(void);             // intero a 32 bit uniforme
float    rng_uniform(void);         // float uniforme in [0, 1)
float    rng_normal(void);          // normale standard N(0,1) via Box-Muller

#endif
