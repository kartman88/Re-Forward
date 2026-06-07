#include "rng.h"
#include <math.h>

// ─── PCG32 (Minimal C implementation, O'Neill 2014) ──────────────────────────
// Stato a 64 bit + incremento (stream) dispari. Output a 32 bit di alta qualità,
// senza le correlazioni tra estrazioni consecutive tipiche dell'LCG di newlib.

static uint64_t s_state = 0x853c49e6748fea9bULL;
static uint64_t s_inc   = 0xda3e39cb94b95bdbULL;   // deve essere dispari

uint32_t rng_u32(void) {
    uint64_t old = s_state;
    s_state = old * 6364136223846793005ULL + s_inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot        = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

void rng_seed(uint64_t seed) {
    s_state = 0u;
    s_inc   = (seed << 1u) | 1u;   // qualsiasi stream dispari va bene
    rng_u32();
    s_state += 0x853c49e6748fea9bULL ^ seed;
    rng_u32();
}

// float in [0, 1) con 24 bit di mantissa effettivi.
float rng_uniform(void) {
    return (rng_u32() >> 8) * (1.0f / 16777216.0f);   // 2^24
}

// Box-Muller: ora u1, u2 provengono da estrazioni PCG32 indipendenti.
float rng_normal(void) {
    float u1, u2;
    do { u1 = rng_uniform(); } while (u1 == 0.0f);
    u2 = rng_uniform();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}
