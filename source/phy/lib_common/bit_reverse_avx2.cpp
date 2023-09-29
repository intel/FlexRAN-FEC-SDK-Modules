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
 * @file   bit_reverse_avx2.cpp
 * @brief  Source code of conversion between float and int16, with agc gain, with AVX2 instructions.
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdint.h>
#include <immintrin.h>  /* AVX */
#include "bit_reverse.h"

#if defined (_BBLIB_AVX2_) || defined (_BBLIB_AVX512_)

// reverse bits in each byte of 256 bit register
static inline __m256i reverse_bits_in_byte(const __m256i x)
{
    const auto luthigh = _mm256_setr_epi8(0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15,
                                          0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15);

    const auto lutlow  = _mm256_slli_epi16(luthigh, 4);
    const auto lowmask = _mm256_set1_epi8(15);
    const auto high    = _mm256_shuffle_epi8(lutlow, _mm256_and_si256(x, lowmask));
    const auto low     = _mm256_shuffle_epi8(luthigh, _mm256_and_si256(_mm256_srli_epi16(x, 4), lowmask));
    return _mm256_or_si256(low, high);
}

//! @{
/*! \brief Bit Reversion.
    \param [in] input Input buffer
    \param [in] num_data Number of data for conversion.
    \param [out] output Output buffer
    \return Return 0 for success, and -1 for error.
    \note Input and output is aligned with 512 bits.
*/
void bblib_bit_reverse_avx2(int8_t* pInOut, int32_t bitLen)
{
    uint32_t bitDiv32, bitMod32, byte;
    __m256i *pInOutOffset = (__m256i *)pInOut;
    uint8_t *pTail, *pTmp;
    __m256i temp;
    bitDiv32 = bitLen >> 8;
    bitMod32 = (bitLen - (bitDiv32 << 8) + 7) >> 3;
    for(uint32_t i = 0; i < bitDiv32; i ++) {
        *pInOutOffset = reverse_bits_in_byte(*pInOutOffset);
        pInOutOffset++;
    }
    if(bitMod32!=0) {
        temp = reverse_bits_in_byte(*pInOutOffset);
        pTail = (uint8_t *)pInOutOffset;
        pTmp = (uint8_t *) &temp;
        for (byte = 0; byte < bitMod32; byte++)
            pTail[byte] = pTmp[byte];
    }
    return;
}


#else
void bblib_bit_reverse_avx2(int8_t* output, int32_t num_data)
{
    printf("This version of bblib_common requires AVX2 ISA support to run\n");
    exit(-1);
}

#endif
