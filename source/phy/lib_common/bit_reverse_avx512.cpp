/**********************************************************************
*
*
*  Copyright [2019 - 2023] [Intel Corporation]
* 
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  
*  You may obtain a copy of the License at
*  
*     http://www.apache.org/licenses/LICENSE-2.0 
*  
*  Unless required by applicable law or agreed to in writing, software 
*  distributed under the License is distributed on an "AS IS" BASIS, 
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and 
*  limitations under the License. 
*  
*  SPDX-License-Identifier: Apache-2.0 
*  
* 
*
**********************************************************************/

/*
 * @file   bit_reverse_avx512.cpp
 * @brief  Source code of conversion between float and int16, with agc gain, with AVX512 instructions.
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdint.h>

#include <immintrin.h>  /* AVX */

#include "bit_reverse.h"

#if defined (_BBLIB_AVX512_)

static inline __m512i BitBlend (__m512i a, __m512i b, int8_t choose) {
    return _mm512_ternarylogic_epi32(a, b, _mm512_set1_epi8(choose), 0xd8);
}

static inline __m512i shr(__m512i a, int offset) { return _mm512_srli_epi16(a, offset); }
static inline __m512i shl(__m512i a, int offset) { return _mm512_slli_epi16(a, offset); }

static __m512i bitInByteRev(__m512i bytes)
{
    const auto swapNibbles = BitBlend(shr(bytes, 4),       shl(bytes, 4),       0b11110000);
    const auto swapPairs =   BitBlend(shr(swapNibbles, 2), shl(swapNibbles, 2), 0b11001100);
    const auto swapBits =    BitBlend(shr(swapPairs, 1),   shl(swapPairs, 1),   0b10101010);
    return swapBits;
}


//! @{
/*! \brief Bit Reversion.
    \param [in] input Input buffer
    \param [in] num_data Number of data for conversion.
    \param [out] output Output buffer
    \return Return 0 for success, and -1 for error.
    \note Input/output is aligned with 512 bits. Only handles the tail processing
*/
void bblib_bit_reverse_avx512(int8_t* pInOut, int32_t bitLen)
{
    uint32_t bitmod512, bitDiv512;
    __m512i * pInOutOffset512 = (__m512i *)pInOut;
    __m512i temp;
    bitDiv512 = bitLen >> 9;
    bitmod512 = (bitLen - (bitDiv512 << 9) + 7) >> 3;
    for (uint32_t i = 0; i < bitDiv512; i++) {
        *pInOutOffset512 = bitInByteRev(*pInOutOffset512);
        pInOutOffset512++;
    }
    if (bitmod512 != 0) {
        temp = bitInByteRev(*pInOutOffset512);
        const auto tail_mask = ((uint64_t) 1 << bitmod512 ) - 1;
        _mm512_mask_storeu_epi8 (pInOut + (bitDiv512 << 6),
                tail_mask, temp);
    }
    return;
}

#else
void bblib_bit_reverse_avx512(int8_t* output, int32_t num_data)
{
    printf("This version of bblib_common requires AVX2 ISA support to run\n");
    exit(-1);
}

#endif
