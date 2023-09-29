#ifndef _LDPC_ENCODER_CYCLESHIFT_H
#define _LDPC_ENCODER_CYCLESHIFT_H

/*******************************************************************************
 * Include public/global header files
 ******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "xmmintrin.h" // SSE
#include "emmintrin.h" // SSE 2
#include "pmmintrin.h" // SSE 3
#include "tmmintrin.h" // SSSE 3
#include "smmintrin.h" // SSE 4 for media
#include <immintrin.h> // AVX


extern int32_t permuteTableFrom288to384[4][32];
extern int16_t permuteTableFrom144to256[8][32];
extern int16_t permuteTabUpto128[8][32];

extern inline __m512i cycle_bit_left_shift_from288to384(__m512i data, int16_t cycLeftShift, int16_t zcSize, int8_t zcIndex_, __m512i swapIdx0_);
extern inline __m512i cycle_bit_left_shift_from144to256(__m512i data, int16_t cycLeftShift, int16_t zcSize, int8_t zcIndex_, __m512i swapIdx0_);
extern inline __m512i cycle_bit_left_shift_from72to128(__m512i data, int16_t cycLeftShift, int16_t zcSize, int8_t zcIndex_, __m512i swapIdx0_);
extern inline __m512i cycle_bit_left_shift_less_than_64(__m512i data, int16_t cycLeftShift, int16_t zcSize, int8_t zcIndex_, __m512i swapIdx0_);
extern inline __m512i cycle_bit_left_shift_special(__m512i data, int16_t cycLeftShift, int16_t zcSize, int8_t zcIndex_, __m512i swapIdx0_);

typedef __m512i (* CYCLE_BIT_LEFT_SHIFT)(__m512i, int16_t, int16_t, int8_t, __m512i);
CYCLE_BIT_LEFT_SHIFT ldpc_select_left_shift_func(int16_t zcSize);
#endif
