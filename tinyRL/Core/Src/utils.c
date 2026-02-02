#include "utils.h"

float clip(float value, float range_min, float range_max, uint8_t *is_clipped){
	*is_clipped = 0;
	if(value > range_max){
		value = range_max;
		*is_clipped = 1;
	}
	if(value < range_min){
		value = range_min;
		*is_clipped = 1;
	}
	return value;
}

int dwt_init(void){
    /* Abilita il blocco di trace */
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;

    /* Sblocco DWT su alcune MCU (es. M7) */
#if defined(DWT_LAR)
    DWT->LAR = 0xC5ACCE55;  // se presente, sblocca i registri DWT
#endif

    /* Azzera e abilita il contatore di cicli */
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

    /* Verifica che stia contando: scrivo 0, leggo e vedo se cambia */
    uint32_t c1 = DWT->CYCCNT;
    __NOP(); __NOP(); __NOP();  // piccola attesa
    uint32_t c2 = DWT->CYCCNT;

    return (c2 != c1);  // 1=ok, 0=non disponibile
}

uint32_t dwt_ticks(void){
    return DWT->CYCCNT;  // 32 bit, overflow automatico
}

/* Delta tra due letture, gestendo l'overflow a 32 bit */
uint32_t dwt_delta(uint32_t start, uint32_t stop){
    return (uint32_t)(stop - start);  // aritmetica modulo 2^32
}

uint32_t cycles_to_ms(uint32_t cycles){
    return (uint32_t)(( (uint64_t)cycles * 1000ULL + (SystemCoreClock/2) ) / SystemCoreClock);
}

uint32_t cycles_to_us(uint32_t cycles){
    return (uint32_t)(( (uint64_t)cycles * 1000000ULL + (SystemCoreClock/2) ) / SystemCoreClock);
}

uint32_t cycles_to_ns(uint32_t cycles){
    return (uint32_t)(( (uint64_t)cycles * 1000000000ULL + (SystemCoreClock/2) ) / SystemCoreClock);
}
