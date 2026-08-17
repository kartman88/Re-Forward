#include "stm32f4xx.h"
#include <stdint.h>

#ifndef INC_UTILS_H_
#define INC_UTILS_H_

float clip(float value, float range_min, float range_max);
int dwt_init(void);
uint32_t dwt_ticks(void);
uint32_t dwt_delta(uint32_t start, uint32_t stop);
uint32_t cycles_to_ms(uint32_t cycles);
uint32_t cycles_to_us(uint32_t cycles);
uint32_t cycles_to_ns(uint32_t cycles);
uint32_t cycles64_to_us(uint64_t cycles);

#endif /* INC_UTILS_H_ */
