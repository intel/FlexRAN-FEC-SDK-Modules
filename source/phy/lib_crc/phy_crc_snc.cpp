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

/**
 * @file   phy_crc_avx512.cpp
 * @brief  Implementation of CRC algorithms CRC24A, CRC24B, CRC24C, CRC16, CRC11 & CRC6
 * all initialised with zeros and CRC24C initialised with ones (appending 1's to data),
 * in accordance with 3GPP TS 38.212 v15.1.1 spec.
 * This implementation is based on the efficient data folding & barrett reduction technique,
 * using optimised intrinsics.
 */

#ifdef _BBLIB_SNC_


/**
 * Include public/global header files
 */
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <functional>

#include "gcc_inc.h"
#include "phy_crc.h"
#include "phy_crc_internal.h"

struct init_crc_snc
{
    init_crc_snc()
    {
        bblib_print_crc_version();
    }
};

init_crc_snc do_constructor_crc_snc;

// Main CRC Generate Function
// Calculates CRC based on CRC type and message data
// Templates: PARAMS     - set of constant values for each CRC type
//            IS_ALIGNED - Indicates if data is byte aligned (ie. multipleof 8 bits)
// Params:    request    - pointer to request structure
//            response   - pointer to response structure




void crc24a_gen_snc(bblib_crc_request *request, bblib_crc_response *response)
{
    uint8_t *data = request->data;
    uint8_t *dataOut = response->data;
    uint8_t *temp_ptr;
    // len is passed in as bits, so turn into bytes
    uint32_t len_bytes = request->len/8;
    uint32_t rem_512bits = request->len&0x1FF;
    uint32_t rem_128bits = request->len&0x7F;
    uint32_t rem_16bytes = rem_128bits>>3;
    /* record the start of input data */
    int32_t    i = 0;
    int32_t    k;
    /* A= B mod C => A*K = B*K mod C*K, set K = 2^8, then compute CRC-32, the most significant */
    /* 24 bits is final 24 bits CRC. */
    const static uint64_t  CRC24APOLY = 0x1864CFB;   //CRC-24A polynomial
    const static uint64_t  CRC24APLUS8 = CRC24APOLY << 8;

    /* some pre-computed key constants */
    const static uint32_t k576   = 0x1F428700;   //t=512+64, x^578 mod CRC24APLUS8, verified
    const static uint32_t k512   = 0x467D2400;   //t=512, x^512 mod CRC24APLUS8, verified
    const static uint32_t k448   = 0x6C1C3500;
    const static uint32_t k384   = 0x5B703800;
    const static uint32_t k320   = 0x66DD1F00;
    const static uint32_t k256   = 0x9D89A200;
    const static uint32_t k192   = 0x2c8c9d00;   //t=128+64, x^192 mod CRC24APLUS8, verified
    const static uint32_t k128   = 0x64e4d700;   //t=128, x^128 mod CRC24APLUS8, verified
    const static uint32_t k96    = 0xfd7e0c00;   //t=96, x^96 mod CRC24APLUS8, verified
    const static uint32_t k64    = 0xd9fe8c00;   //t=64, x^64 mod CRC24APLUS8, verified
    const static uint64_t u      = 0x1f845fe24;  //u for crc24A * 256, floor(x^64 / CRC24APLUS8), verified
    const static __m128i ENDIA_SHUF_MASK_128bit = _mm_set_epi8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                                               0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
    const static __m512i ENDIA_SHUF_MASK_512bit = _mm512_set_epi8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                                                  0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
                                                                  0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                                                                  0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
                                                                  0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                                                                  0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F,
                                                                  0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
                                                                  0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F);
    __m128i xmm3, xmm2, xmm1, xmm0;
    __m128i k192k128;
    __m128i *pmm0, *pmm1, *pmm2;
    __m512i kmm0;
    __m512i kmm1;
    __m512i kmm2;
    __m512i k576k512;
    __m512i k448k384k320k256k192k128;
    int32_t num_fold4;
    int32_t num_fold1;
    num_fold4 = (request->len>>9)-1;
    pmm0 = (__m128i*)&kmm0;
    pmm1 = (__m128i*)&kmm1;
    pmm2 = (__m128i*)&kmm2;
    if(num_fold4 > 0)
    {
        k576k512 = _mm512_set_epi32(0, k576, 0, k512,0, k576, 0, k512,0, k576, 0, k512,0, k576, 0, k512);
        k192k128 = _mm_set_epi32(0, k192, 0, k128);
        num_fold1 = rem_512bits>>7;
        kmm0 = _mm512_load_si512((void const*)data);
        data += 64;
        kmm0 = _mm512_shuffle_epi8(kmm0, ENDIA_SHUF_MASK_512bit);
        for (i=0; i<num_fold4; i++)
        {
            kmm2 = _mm512_load_si512((void const*)data);
            data += 64;
            kmm2 = _mm512_shuffle_epi8(kmm2, ENDIA_SHUF_MASK_512bit);
            kmm1 = _mm512_clmulepi64_epi128(kmm0, k576k512, 0x00);
            kmm0 = _mm512_clmulepi64_epi128(kmm0, k576k512, 0x11);
            kmm0 = _mm512_ternarylogic_epi32(kmm0, kmm1, kmm2, 0x96);
        }
        for(i=0; i<3; i++)
        {
            pmm1[i] = _mm_clmulepi64_si128(pmm0[i], k192k128, 0x00);
            pmm0[i] = _mm_clmulepi64_si128(pmm0[i], k192k128, 0x11);
            pmm0[i+1] = _mm_ternarylogic_epi32(pmm0[i], pmm1[i], pmm0[i+1], 0x96);
        }
        xmm1 = pmm0[3];
    }
    else
    {
        k192k128 = _mm_set_epi32(0, k192, 0, k128);
        num_fold1 = (request->len>>7)-1;
        xmm1 = _mm_load_si128((__m128i *)data);
        data += 16;
        xmm1 = _mm_shuffle_epi8(xmm1, ENDIA_SHUF_MASK_128bit);
    }
    /* 1. fold by 128bit. remaining length <=2*128bits. */
    if(num_fold1 > 0)
    {
        for (i=0; i<num_fold1; i++)
        {
            xmm2 = xmm1;
            xmm0 = _mm_load_si128((__m128i *)data);
            data += 16;
            xmm0 = _mm_shuffle_epi8(xmm0, ENDIA_SHUF_MASK_128bit);
            xmm1 = _mm_clmulepi64_si128(xmm1, k192k128, 0x00);
            xmm2 = _mm_clmulepi64_si128(xmm2, k192k128, 0x11);
            xmm1 = _mm_ternarylogic_epi32(xmm0, xmm1, xmm2, 0x96);
        }
    }
    /* 2. if remaining length > 128 bits, then pad zero to the most-significant bit to grow to 256bits length,
     * then fold once to 128 bits. */
    if(num_fold1<0) /*data length less than 128bits*/
    {
        char byte_shift = (char)(16 - rem_16bytes);
        __m128i iota = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        __m128i shift_value = _mm_add_epi8(iota, _mm_set1_epi8(byte_shift));
        __m128i shifted_foldData = _mm_shuffle_epi8(xmm1, shift_value);
        __mmask16 next_mask = (__mmask16)(0xFFFF << (16-byte_shift));
        __mmask16 fold_mask = (__mmask16)~(next_mask);
        xmm0 = _mm_maskz_mov_epi8(fold_mask, shifted_foldData);
    }
    else
    {
        if(rem_128bits>0)
        {
            xmm0 = _mm_load_si128((__m128i *)data); //load remaining len%16 bytes and maybe some garbage bytes.
            xmm0 = _mm_shuffle_epi8(xmm0, ENDIA_SHUF_MASK_128bit);
            char byte_shift = (char)(16 - rem_16bytes);
            __m128i iota = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m128i shift_value = _mm_add_epi8(iota, _mm_set1_epi8(byte_shift));
            __m128i shifted_foldData = _mm_shuffle_epi8(xmm1, shift_value);
            __m128i shifted_nextData = _mm_shuffle_epi8(xmm0, shift_value);
            __mmask16 next_mask = (__mmask16)(0xFFFF << (16-byte_shift));
            xmm0 = _mm_mask_shuffle_epi8(shifted_nextData, next_mask, shifted_foldData, iota);
            __mmask16 fold_mask = (__mmask16)~(next_mask);
            xmm1 = _mm_maskz_mov_epi8(fold_mask, shifted_foldData);
            xmm2 = xmm1;
            xmm1 = _mm_clmulepi64_si128(xmm1, k192k128, 0x00);
            xmm2 = _mm_clmulepi64_si128(xmm2, k192k128, 0x11);
            xmm0 = _mm_ternarylogic_epi32(xmm0, xmm1, xmm2, 0x96);
        }
        else
            xmm0 = xmm1;
    }
    /* 3. Apply 64 bits fold to 64 bits + 32 bits crc(32 bits zero) */
    xmm3 =  _mm_set_epi32(0, 0, 0, k96);
    xmm1 = _mm_clmulepi64_si128(xmm0, xmm3, 0x01);
    xmm2 = _mm_slli_si128(xmm0, 8);
    xmm2 = _mm_srli_si128(xmm2, 4);
    xmm0 = _mm_xor_si128(xmm1, xmm2);


    /* 4. Apply 32 bits fold to 32 bits + 32 bits crc(32 bits zero) */
    xmm3 =  _mm_set_epi32(0, 0, 0, k64);
    xmm1 = _mm_clmulepi64_si128(xmm0, xmm3, 0x01);
    xmm0 = _mm_slli_si128(xmm0, 8);
    xmm0 = _mm_srli_si128(xmm0, 8);
    xmm0 = _mm_xor_si128(xmm1, xmm0);

    /* 5. Use Barrett Reduction Algorithm to calculate the 32 bits crc.
     * Output: C(x)  = R(x) mod P(x)
     * Step 1: T1(x) = floor(R(x)/x^32)) * u
     * Step 2: T2(x) = floor(T1(x)/x^32)) * P(x)
     * Step 3: C(x)  = R(x) xor T2(x) mod x^32 */
    xmm1 = _mm_set_epi32(0, 0, 1, (u & 0xFFFFFFFF));
    xmm2 = _mm_srli_si128(xmm0, 4);
    xmm1 = _mm_clmulepi64_si128(xmm1, xmm2, 0x00);
    xmm1 = _mm_srli_si128(xmm1, 4);
    xmm2 = _mm_set_epi32(0, 0, 1, (CRC24APLUS8 & 0xFFFFFFFF));
    xmm1 = _mm_clmulepi64_si128(xmm1, xmm2, 0x00);
    xmm0 = _mm_xor_si128(xmm0, xmm1);
    xmm1 = _mm_set_epi32(0, 0, 0, 0xFFFFFFFF);
    xmm0 = _mm_and_si128 (xmm0, xmm1);


    /* 6. Update Result
    /* add crc to last 3 bytes. */
    dataOut[len_bytes]   =  _mm_extract_epi8(xmm0, 3);
    dataOut[len_bytes+1] =  _mm_extract_epi8(xmm0, 2);
    dataOut[len_bytes+2] =  _mm_extract_epi8(xmm0, 1);
    response->len = (len_bytes + 3)*8;

    /* the most significant 24 bits of the 32 bits crc is the finial 24 bits crc. */
    response->crc_value = (((uint32_t)_mm_extract_epi32(xmm0, 0)) >> 8);
}


void crc24b_gen_snc(bblib_crc_request *request, bblib_crc_response *response)
{
    uint8_t *data = request->data;
    uint8_t *dataOut = response->data;
    uint8_t *temp_ptr;
    // len is passed in as bits, so turn into bytes
    uint32_t len_bytes = request->len/8;
    uint32_t rem_512bits = request->len&0x1FF;
    uint32_t rem_128bits = request->len&0x7F;
    uint32_t rem_16bytes = rem_128bits>>3;
    /* record the start of input data */
    int32_t    i = 0;
    int32_t    k;
    /* A= B mod C => A*K = B*K mod C*K, set K = 2^8, then compute CRC-32, the most significant */
    /* 24 bits is final 24 bits CRC. */
    const static uint64_t  CRC24BPOLY = 0x1800063;    //CRC24B Polynomial
    const static uint64_t  CRC24BPLUS8 = CRC24BPOLY << 8;

    /* some pre-computed key constants */
    const static uint32_t k576   = 0xB5015B00;   //t=512+64, x^578 mod CRC24APLUS8, verified
    const static uint32_t k512   = 0xA0660100;   //t=512, x^512 mod CRC24APLUS8, verified
    const static uint32_t k192   = 0x42000100;   //t=128+64, x^192 mod CRC24APLUS8, verified
    const static uint32_t k128   = 0x80140500;   //t=128, x^128 mod CRC24APLUS8, verified
    const static uint32_t k96    = 0x09000200;   //t=96, x^96 mod CRC24APLUS8, verified
    const static uint32_t k64    = 0x90042100;   //t=64, x^64 mod CRC24APLUS8, verified
    const static uint64_t u      = 0x1ffff83ff;  //u for crc24A * 256, floor(x^64 / CRC24APLUS8), verified
    const static __m128i ENDIA_SHUF_MASK_128bit = _mm_set_epi8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                                               0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
    const static __m512i ENDIA_SHUF_MASK_512bit = _mm512_set_epi8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                                                  0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
                                                                  0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                                                                  0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
                                                                  0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                                                                  0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F,
                                                                  0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
                                                                  0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F);
    __m128i xmm3, xmm2, xmm1, xmm0;
    __m128i k192k128;
    __m128i *pmm0, *pmm1, *pmm2;
    __m512i kmm0;
    __m512i kmm1;
    __m512i kmm2;
    __m512i k576k512;
    int32_t num_fold4;
    int32_t num_fold1;
    num_fold4 = (request->len>>9)-1;
    pmm0 = (__m128i*)&kmm0;
    pmm1 = (__m128i*)&kmm1;
    pmm2 = (__m128i*)&kmm2;
    if(num_fold4 > 0)
    {
        k576k512 = _mm512_set_epi32(0, k576, 0, k512,0, k576, 0, k512,0, k576, 0, k512,0, k576, 0, k512);
        k192k128 = _mm_set_epi32(0, k192, 0, k128);
        num_fold1 = rem_512bits>>7;
        kmm0 = _mm512_load_si512((void const*)data);
        data += 64;
        kmm0 = _mm512_shuffle_epi8(kmm0, ENDIA_SHUF_MASK_512bit);
        for (i=0; i<num_fold4; i++)
        {
            kmm2 = _mm512_load_si512((void const*)data);
            data += 64;
            kmm2 = _mm512_shuffle_epi8(kmm2, ENDIA_SHUF_MASK_512bit);
            kmm1 = _mm512_clmulepi64_epi128(kmm0, k576k512, 0x00);
            kmm0 = _mm512_clmulepi64_epi128(kmm0, k576k512, 0x11);
            kmm0 = _mm512_ternarylogic_epi32(kmm0, kmm1, kmm2, 0x96);
        }
        for(i=0; i<3; i++)
        {
            pmm1[i] = _mm_clmulepi64_si128(pmm0[i], k192k128, 0x00);
            pmm0[i] = _mm_clmulepi64_si128(pmm0[i], k192k128, 0x11);
            pmm0[i+1] = _mm_ternarylogic_epi32(pmm0[i], pmm1[i], pmm0[i+1], 0x96);
        }
        xmm1 = pmm0[3];
    }
    else
    {
        k192k128 = _mm_set_epi32(0, k192, 0, k128);
        num_fold1 = (request->len>>7)-1;
        xmm1 = _mm_load_si128((__m128i *)data);
        data += 16;
        xmm1 = _mm_shuffle_epi8(xmm1, ENDIA_SHUF_MASK_128bit);
    }
    /* 1. fold by 128bit. remaining length <=2*128bits. */
    if(num_fold1 > 0)
    {
        for (i=0; i<num_fold1; i++)
        {
            xmm2 = xmm1;
            xmm0 = _mm_load_si128((__m128i *)data);
            data += 16;
            xmm0 = _mm_shuffle_epi8(xmm0, ENDIA_SHUF_MASK_128bit);
            xmm1 = _mm_clmulepi64_si128(xmm1, k192k128, 0x00);
            xmm2 = _mm_clmulepi64_si128(xmm2, k192k128, 0x11);
            xmm1 = _mm_ternarylogic_epi32(xmm0, xmm1, xmm2, 0x96);
        }
    }
    /* 2. if remaining length > 128 bits, then pad zero to the most-significant bit to grow to 256bits length,
     * then fold once to 128 bits. */
    if(num_fold1<0) /*data length less than 128bits*/
    {
        char byte_shift = (char)(16 - rem_16bytes);
        __m128i iota = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        __m128i shift_value = _mm_add_epi8(iota, _mm_set1_epi8(byte_shift));
        __m128i shifted_foldData = _mm_shuffle_epi8(xmm1, shift_value);
        __mmask16 next_mask = (__mmask16)(0xFFFF << (16-byte_shift));
        __mmask16 fold_mask = (__mmask16)~(next_mask);
        xmm0 = _mm_maskz_mov_epi8(fold_mask, shifted_foldData);
    }
    else
    {
        if(rem_128bits>0)
        {
            xmm0 = _mm_load_si128((__m128i *)data); //load remaining len%16 bytes and maybe some garbage bytes.
            xmm0 = _mm_shuffle_epi8(xmm0, ENDIA_SHUF_MASK_128bit);
            char byte_shift = (char)(16 - rem_16bytes);
            __m128i iota = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m128i shift_value = _mm_add_epi8(iota, _mm_set1_epi8(byte_shift));
            __m128i shifted_foldData = _mm_shuffle_epi8(xmm1, shift_value);
            __m128i shifted_nextData = _mm_shuffle_epi8(xmm0, shift_value);
            __mmask16 next_mask = (__mmask16)(0xFFFF << (16-byte_shift));
            xmm0 = _mm_mask_shuffle_epi8(shifted_nextData, next_mask, shifted_foldData, iota);
            __mmask16 fold_mask = (__mmask16)~(next_mask);
            xmm1 = _mm_maskz_mov_epi8(fold_mask, shifted_foldData);
            xmm2 = xmm1;
            xmm1 = _mm_clmulepi64_si128(xmm1, k192k128, 0x00);
            xmm2 = _mm_clmulepi64_si128(xmm2, k192k128, 0x11);
            xmm0 = _mm_ternarylogic_epi32(xmm0, xmm1, xmm2, 0x96);
        }
        else
            xmm0 = xmm1;
    }
    /* 3. Apply 64 bits fold to 64 bits + 32 bits crc(32 bits zero) */
    xmm3 =  _mm_set_epi32(0, 0, 0, k96);
    xmm1 = _mm_clmulepi64_si128(xmm0, xmm3, 0x01);
    xmm2 = _mm_slli_si128(xmm0, 8);
    xmm2 = _mm_srli_si128(xmm2, 4);
    xmm0 = _mm_xor_si128(xmm1, xmm2);


    /* 4. Apply 32 bits fold to 32 bits + 32 bits crc(32 bits zero) */
    xmm3 =  _mm_set_epi32(0, 0, 0, k64);
    xmm1 = _mm_clmulepi64_si128(xmm0, xmm3, 0x01);
    xmm0 = _mm_slli_si128(xmm0, 8);
    xmm0 = _mm_srli_si128(xmm0, 8);
    xmm0 = _mm_xor_si128(xmm1, xmm0);

    /* 5. Use Barrett Reduction Algorithm to calculate the 32 bits crc.
     * Output: C(x)  = R(x) mod P(x)
     * Step 1: T1(x) = floor(R(x)/x^32)) * u
     * Step 2: T2(x) = floor(T1(x)/x^32)) * P(x)
     * Step 3: C(x)  = R(x) xor T2(x) mod x^32 */
    xmm1 = _mm_set_epi32(0, 0, 1, (u & 0xFFFFFFFF));
    xmm2 = _mm_srli_si128(xmm0, 4);
    xmm1 = _mm_clmulepi64_si128(xmm1, xmm2, 0x00);
    xmm1 = _mm_srli_si128(xmm1, 4);
    xmm2 = _mm_set_epi32(0, 0, 1, (CRC24BPLUS8 & 0xFFFFFFFF));
    xmm1 = _mm_clmulepi64_si128(xmm1, xmm2, 0x00);
    xmm0 = _mm_xor_si128(xmm0, xmm1);
    xmm1 = _mm_set_epi32(0, 0, 0, 0xFFFFFFFF);
    xmm0 = _mm_and_si128 (xmm0, xmm1);


    /* 6. Update Result
    /* add crc to last 3 bytes. */
    dataOut[len_bytes]   =  _mm_extract_epi8(xmm0, 3);
    dataOut[len_bytes+1] =  _mm_extract_epi8(xmm0, 2);
    dataOut[len_bytes+2] =  _mm_extract_epi8(xmm0, 1);
    response->len = (len_bytes + 3)*8;

    /* the most significant 24 bits of the 32 bits crc is the finial 24 bits crc. */
    response->crc_value = (((uint32_t)_mm_extract_epi32(xmm0, 0)) >> 8);
}


void crc24c_gen_snc(bblib_crc_request *request, bblib_crc_response *response)
{
    uint8_t *data = request->data;
    uint8_t *dataOut = response->data;
    uint8_t *temp_ptr;
    // len is passed in as bits, so turn into bytes
    uint32_t len_bytes = request->len/8;
    uint32_t rem_512bits = request->len&0x1FF;
    uint32_t rem_128bits = request->len&0x7F;
    uint32_t rem_16bytes = rem_128bits>>3;
    /* record the start of input data */
    int32_t    i = 0;
    int32_t    k;
    /* A= B mod C => A*K = B*K mod C*K, set K = 2^8, then compute CRC-32, the most significant */
    /* 24 bits is final 24 bits CRC. */
    const static uint64_t  CRC24CPOLY = 0x1B2B117;   //CRC-24C polynomial
    const static uint64_t  CRC24CPLUS8 = CRC24CPOLY << 8;

    /* some pre-computed key constants */
    const static uint32_t k576   = 0x1C70EC00;   //t=512+64, x^578 mod CRC24APLUS8, verified
    const static uint32_t k512   = 0x74665600;   //t=512, x^512 mod CRC24APLUS8, verified
    const static uint32_t k192   = 0x8cfa5500;   //t=128+64, x^192 mod CRC24APLUS8, verified
    const static uint32_t k128   = 0x6ccc8e00;   //t=128, x^128 mod CRC24APLUS8, verified
    const static uint32_t k96    = 0x13979900;   //t=96, x^96 mod CRC24APLUS8, verified
    const static uint32_t k64    = 0x74809300;   //t=64, x^64 mod CRC24APLUS8, verified
    const static uint64_t u      = 0x1c52cdcad;  //u for crc24A * 256, floor(x^64 / CRC24APLUS8), verified
    const static __m128i ENDIA_SHUF_MASK_128bit = _mm_set_epi8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                                               0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
    const static __m512i ENDIA_SHUF_MASK_512bit = _mm512_set_epi8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                                                  0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
                                                                  0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                                                                  0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
                                                                  0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                                                                  0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F,
                                                                  0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
                                                                  0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F);
    __m128i xmm3, xmm2, xmm1, xmm0;
    __m128i k192k128;
    __m128i *pmm0, *pmm1, *pmm2;
    __m512i kmm0;
    __m512i kmm1;
    __m512i kmm2;
    __m512i k576k512;
    int32_t num_fold4;
    int32_t num_fold1;
    num_fold4 = (request->len>>9)-1;
    pmm0 = (__m128i*)&kmm0;
    pmm1 = (__m128i*)&kmm1;
    pmm2 = (__m128i*)&kmm2;
    if(num_fold4 > 0)
    {
        k576k512 = _mm512_set_epi32(0, k576, 0, k512,0, k576, 0, k512,0, k576, 0, k512,0, k576, 0, k512);
        k192k128 = _mm_set_epi32(0, k192, 0, k128);
        num_fold1 = rem_512bits>>7;
        kmm0 = _mm512_load_si512((void const*)data);
        data += 64;
        kmm0 = _mm512_shuffle_epi8(kmm0, ENDIA_SHUF_MASK_512bit);
        for (i=0; i<num_fold4; i++)
        {
            kmm2 = _mm512_load_si512((void const*)data);
            data += 64;
            kmm2 = _mm512_shuffle_epi8(kmm2, ENDIA_SHUF_MASK_512bit);
            kmm1 = _mm512_clmulepi64_epi128(kmm0, k576k512, 0x00);
            kmm0 = _mm512_clmulepi64_epi128(kmm0, k576k512, 0x11);
            kmm0 = _mm512_ternarylogic_epi32(kmm0, kmm1, kmm2, 0x96);
        }
        for(i=0; i<3; i++)
        {
            pmm1[i] = _mm_clmulepi64_si128(pmm0[i], k192k128, 0x00);
            pmm0[i] = _mm_clmulepi64_si128(pmm0[i], k192k128, 0x11);
            pmm0[i+1] = _mm_ternarylogic_epi32(pmm0[i], pmm1[i], pmm0[i+1], 0x96);
        }
        xmm1 = pmm0[3];
    }
    else
    {
        k192k128 = _mm_set_epi32(0, k192, 0, k128);
        num_fold1 = (request->len>>7)-1;
        xmm1 = _mm_load_si128((__m128i *)data);
        data += 16;
        xmm1 = _mm_shuffle_epi8(xmm1, ENDIA_SHUF_MASK_128bit);
    }
    /* 1. fold by 128bit. remaining length <=2*128bits. */
    if(num_fold1 > 0)
    {
        for (i=0; i<num_fold1; i++)
        {
            xmm2 = xmm1;
            xmm0 = _mm_load_si128((__m128i *)data);
            data += 16;
            xmm0 = _mm_shuffle_epi8(xmm0, ENDIA_SHUF_MASK_128bit);
            xmm1 = _mm_clmulepi64_si128(xmm1, k192k128, 0x00);
            xmm2 = _mm_clmulepi64_si128(xmm2, k192k128, 0x11);
            xmm1 = _mm_ternarylogic_epi32(xmm0, xmm1, xmm2, 0x96);
        }
    }
    /* 2. if remaining length > 128 bits, then pad zero to the most-significant bit to grow to 256bits length,
     * then fold once to 128 bits. */
    if(num_fold1<0) /*data length less than 128bits*/
    {
        char byte_shift = (char)(16 - rem_16bytes);
        __m128i iota = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        __m128i shift_value = _mm_add_epi8(iota, _mm_set1_epi8(byte_shift));
        __m128i shifted_foldData = _mm_shuffle_epi8(xmm1, shift_value);
        __mmask16 next_mask = (__mmask16)(0xFFFF << (16-byte_shift));
        __mmask16 fold_mask = (__mmask16)~(next_mask);
        xmm0 = _mm_maskz_mov_epi8(fold_mask, shifted_foldData);
    }
    else
    {
        if(rem_128bits>0)
        {
            xmm0 = _mm_load_si128((__m128i *)data); //load remaining len%16 bytes and maybe some garbage bytes.
            xmm0 = _mm_shuffle_epi8(xmm0, ENDIA_SHUF_MASK_128bit);
            char byte_shift = (char)(16 - rem_16bytes);
            __m128i iota = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m128i shift_value = _mm_add_epi8(iota, _mm_set1_epi8(byte_shift));
            __m128i shifted_foldData = _mm_shuffle_epi8(xmm1, shift_value);
            __m128i shifted_nextData = _mm_shuffle_epi8(xmm0, shift_value);
            __mmask16 next_mask = (__mmask16)(0xFFFF << (16-byte_shift));
            xmm0 = _mm_mask_shuffle_epi8(shifted_nextData, next_mask, shifted_foldData, iota);
            __mmask16 fold_mask = (__mmask16)~(next_mask);
            xmm1 = _mm_maskz_mov_epi8(fold_mask, shifted_foldData);
            xmm2 = xmm1;
            xmm1 = _mm_clmulepi64_si128(xmm1, k192k128, 0x00);
            xmm2 = _mm_clmulepi64_si128(xmm2, k192k128, 0x11);
            xmm0 = _mm_ternarylogic_epi32(xmm0, xmm1, xmm2, 0x96);
        }
        else
            xmm0 = xmm1;
    }
    /* 3. Apply 64 bits fold to 64 bits + 32 bits crc(32 bits zero) */
    xmm3 =  _mm_set_epi32(0, 0, 0, k96);
    xmm1 = _mm_clmulepi64_si128(xmm0, xmm3, 0x01);
    xmm2 = _mm_slli_si128(xmm0, 8);
    xmm2 = _mm_srli_si128(xmm2, 4);
    xmm0 = _mm_xor_si128(xmm1, xmm2);


    /* 4. Apply 32 bits fold to 32 bits + 32 bits crc(32 bits zero) */
    xmm3 =  _mm_set_epi32(0, 0, 0, k64);
    xmm1 = _mm_clmulepi64_si128(xmm0, xmm3, 0x01);
    xmm0 = _mm_slli_si128(xmm0, 8);
    xmm0 = _mm_srli_si128(xmm0, 8);
    xmm0 = _mm_xor_si128(xmm1, xmm0);

    /* 5. Use Barrett Reduction Algorithm to calculate the 32 bits crc.
     * Output: C(x)  = R(x) mod P(x)
     * Step 1: T1(x) = floor(R(x)/x^32)) * u
     * Step 2: T2(x) = floor(T1(x)/x^32)) * P(x)
     * Step 3: C(x)  = R(x) xor T2(x) mod x^32 */
    xmm1 = _mm_set_epi32(0, 0, 1, (u & 0xFFFFFFFF));
    xmm2 = _mm_srli_si128(xmm0, 4);
    xmm1 = _mm_clmulepi64_si128(xmm1, xmm2, 0x00);
    xmm1 = _mm_srli_si128(xmm1, 4);
    xmm2 = _mm_set_epi32(0, 0, 1, (CRC24CPLUS8 & 0xFFFFFFFF));
    xmm1 = _mm_clmulepi64_si128(xmm1, xmm2, 0x00);
    xmm0 = _mm_xor_si128(xmm0, xmm1);
    xmm1 = _mm_set_epi32(0, 0, 0, 0xFFFFFFFF);
    xmm0 = _mm_and_si128 (xmm0, xmm1);


    /* 6. Update Result
    /* add crc to last 3 bytes. */
    dataOut[len_bytes]   =  _mm_extract_epi8(xmm0, 3);
    dataOut[len_bytes+1] =  _mm_extract_epi8(xmm0, 2);
    dataOut[len_bytes+2] =  _mm_extract_epi8(xmm0, 1);
    response->len = (len_bytes + 3)*8;

    /* the most significant 24 bits of the 32 bits crc is the finial 24 bits crc. */
    response->crc_value = (((uint32_t)_mm_extract_epi32(xmm0, 0)) >> 8);
}

void crc16_gen_snc(bblib_crc_request *request, bblib_crc_response *response)
{
    uint8_t *data = request->data;
    uint8_t *dataOut = response->data;
    uint8_t *temp_ptr;
    // len is passed in as bits, so turn into bytes
    uint32_t len_bytes = request->len/8;
    uint32_t rem_512bits = request->len&0x1FF;
    uint32_t rem_128bits = request->len&0x7F;
    uint32_t rem_16bytes = rem_128bits>>3;
    /* record the start of input data */
    int32_t    i = 0;
    int32_t    k;
    /* A= B mod C => A*K = B*K mod C*K, set K = 2^8, then compute CRC-32, the most significant */
    /* 24 bits is final 24 bits CRC. */
    const static uint64_t  CRC16POLY = 0x11021;    //CRC16 Polynomial
    const static uint64_t  CRC16PLUS16 = CRC16POLY << 16;  //pads poly to 32bits

    /* some pre-computed key constants */
    const static uint32_t k576   = 0x60190000;   //t=512+64, x^578 mod CRC24APLUS8, verified
    const static uint32_t k512   = 0x59B00000;   //t=512, x^512 mod CRC24APLUS8, verified
    const static uint32_t k192   = 0xD5F60000;   //t=128+64, x^192 mod CRC24APLUS8, verified
    const static uint32_t k128   = 0x45630000;   //t=128, x^128 mod CRC24APLUS8, verified
    const static uint32_t k96    = 0xEB230000;   //t=96, x^96 mod CRC24APLUS8, verified
    const static uint32_t k64    = 0xAA510000;   //t=64, x^64 mod CRC24APLUS8, verified
    const static uint64_t u      = 0x111303471;  //u for crc24A * 256, floor(x^64 / CRC24APLUS8), verified
    const static __m128i ENDIA_SHUF_MASK_128bit = _mm_set_epi8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                                               0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
    const static __m512i ENDIA_SHUF_MASK_512bit = _mm512_set_epi8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                                                  0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
                                                                  0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                                                                  0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
                                                                  0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                                                                  0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F,
                                                                  0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
                                                                  0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F);
    __m128i xmm3, xmm2, xmm1, xmm0;
    __m128i  k192k128;
    __m128i *pmm0, *pmm1, *pmm2;
    __m512i kmm0;
    __m512i kmm1;
    __m512i kmm2;
    __m512i k576k512;
    int32_t num_fold4;
    int32_t num_fold1;
    num_fold4 = (request->len>>9)-1;
    pmm0 = (__m128i*)&kmm0;
    pmm1 = (__m128i*)&kmm1;
    pmm2 = (__m128i*)&kmm2;
    if(num_fold4 > 0)
    {
        k576k512 = _mm512_set_epi32(0, k576, 0, k512,0, k576, 0, k512,0, k576, 0, k512,0, k576, 0, k512);
        k192k128 = _mm_set_epi32(0, k192, 0, k128);
        num_fold1 = rem_512bits>>7;
        kmm0 = _mm512_load_si512((void const*)data);
        data += 64;
        kmm0 = _mm512_shuffle_epi8(kmm0, ENDIA_SHUF_MASK_512bit);
        for (i=0; i<num_fold4; i++)
        {
            kmm2 = _mm512_load_si512((void const*)data);
            data += 64;
            kmm2 = _mm512_shuffle_epi8(kmm2, ENDIA_SHUF_MASK_512bit);
            kmm1 = _mm512_clmulepi64_epi128(kmm0, k576k512, 0x00);
            kmm0 = _mm512_clmulepi64_epi128(kmm0, k576k512, 0x11);
            kmm0 = _mm512_ternarylogic_epi32(kmm0, kmm1, kmm2, 0x96);
        }
        for(i=0; i<3; i++)
        {
            pmm1[i] = _mm_clmulepi64_si128(pmm0[i], k192k128, 0x00);
            pmm0[i] = _mm_clmulepi64_si128(pmm0[i], k192k128, 0x11);
            pmm0[i+1] = _mm_ternarylogic_epi32(pmm0[i], pmm1[i], pmm0[i+1], 0x96);
        }
        xmm1 = pmm0[3];
    }
    else
    {
        k192k128 = _mm_set_epi32(0, k192, 0, k128);
        num_fold1 = (request->len>>7)-1;
        xmm1 = _mm_load_si128((__m128i *)data);
        data += 16;
        xmm1 = _mm_shuffle_epi8(xmm1, ENDIA_SHUF_MASK_128bit);
    }
    /* 1. fold by 128bit. remaining length <=2*128bits. */
    if(num_fold1 > 0)
    {
        for (i=0; i<num_fold1; i++)
        {
            xmm2 = xmm1;
            xmm0 = _mm_load_si128((__m128i *)data);
            data += 16;
            xmm0 = _mm_shuffle_epi8(xmm0, ENDIA_SHUF_MASK_128bit);
            xmm1 = _mm_clmulepi64_si128(xmm1, k192k128, 0x00);
            xmm2 = _mm_clmulepi64_si128(xmm2, k192k128, 0x11);
            xmm1 = _mm_ternarylogic_epi32(xmm0, xmm1, xmm2, 0x96);
        }
    }
    /* 2. if remaining length > 128 bits, then pad zero to the most-significant bit to grow to 256bits length,
     * then fold once to 128 bits. */
    if(num_fold1<0) /*data length less than 128bits*/
    {
        char byte_shift = (char)(16 - rem_16bytes);
        __m128i iota = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        __m128i shift_value = _mm_add_epi8(iota, _mm_set1_epi8(byte_shift));
        __m128i shifted_foldData = _mm_shuffle_epi8(xmm1, shift_value);
        __mmask16 next_mask = (__mmask16)(0xFFFF << (16-byte_shift));
        __mmask16 fold_mask = (__mmask16)~(next_mask);
        xmm0 = _mm_maskz_mov_epi8(fold_mask, shifted_foldData);
    }
    else
    {
        if(rem_128bits>0)
        {
            xmm0 = _mm_load_si128((__m128i *)data); //load remaining len%16 bytes and maybe some garbage bytes.
            xmm0 = _mm_shuffle_epi8(xmm0, ENDIA_SHUF_MASK_128bit);
            char byte_shift = (char)(16 - rem_16bytes);
            __m128i iota = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m128i shift_value = _mm_add_epi8(iota, _mm_set1_epi8(byte_shift));
            __m128i shifted_foldData = _mm_shuffle_epi8(xmm1, shift_value);
            __m128i shifted_nextData = _mm_shuffle_epi8(xmm0, shift_value);
            __mmask16 next_mask = (__mmask16)(0xFFFF << (16-byte_shift));
            xmm0 = _mm_mask_shuffle_epi8(shifted_nextData, next_mask, shifted_foldData, iota);
            __mmask16 fold_mask = (__mmask16)~(next_mask);
            xmm1 = _mm_maskz_mov_epi8(fold_mask, shifted_foldData);
            xmm2 = xmm1;
            xmm1 = _mm_clmulepi64_si128(xmm1, k192k128, 0x00);
            xmm2 = _mm_clmulepi64_si128(xmm2, k192k128, 0x11);
            xmm0 = _mm_ternarylogic_epi32(xmm0, xmm1, xmm2, 0x96);
        }
        else
            xmm0 = xmm1;
    }
    /* 3. Apply 64 bits fold to 64 bits + 32 bits crc(32 bits zero) */
    xmm3 =  _mm_set_epi32(0, 0, 0, k96);
    xmm1 = _mm_clmulepi64_si128(xmm0, xmm3, 0x01);
    xmm2 = _mm_slli_si128(xmm0, 8);
    xmm2 = _mm_srli_si128(xmm2, 4);
    xmm0 = _mm_xor_si128(xmm1, xmm2);


    /* 4. Apply 32 bits fold to 32 bits + 32 bits crc(32 bits zero) */
    xmm3 =  _mm_set_epi32(0, 0, 0, k64);
    xmm1 = _mm_clmulepi64_si128(xmm0, xmm3, 0x01);
    xmm0 = _mm_slli_si128(xmm0, 8);
    xmm0 = _mm_srli_si128(xmm0, 8);
    xmm0 = _mm_xor_si128(xmm1, xmm0);

    /* 5. Use Barrett Reduction Algorithm to calculate the 32 bits crc.
     * Output: C(x)  = R(x) mod P(x)
     * Step 1: T1(x) = floor(R(x)/x^32)) * u
     * Step 2: T2(x) = floor(T1(x)/x^32)) * P(x)
     * Step 3: C(x)  = R(x) xor T2(x) mod x^32 */
    xmm1 = _mm_set_epi32(0, 0, 1, (u & 0xFFFFFFFF));
    xmm2 = _mm_srli_si128(xmm0, 4);
    xmm1 = _mm_clmulepi64_si128(xmm1, xmm2, 0x00);
    xmm1 = _mm_srli_si128(xmm1, 4);
    xmm2 = _mm_set_epi32(0, 0, 1, (CRC16PLUS16 & 0xFFFFFFFF));
    xmm1 = _mm_clmulepi64_si128(xmm1, xmm2, 0x00);
    xmm0 = _mm_xor_si128(xmm0, xmm1);
    xmm1 = _mm_set_epi32(0, 0, 0, 0xFFFFFFFF);
    xmm0 = _mm_and_si128 (xmm0, xmm1);

    /* 6. Update Result
    /* add crc to last 3 bytes. */
    dataOut[len_bytes]   =  _mm_extract_epi8(xmm0, 3);
    dataOut[len_bytes+1] =  _mm_extract_epi8(xmm0, 2);
//    dataOut[len_bytes+2] =  _mm_extract_epi8(xmm0, 1);
    response->len = (len_bytes + 2)*8;

    /* the most significant 24 bits of the 32 bits crc is the finial 24 bits crc. */
    response->crc_value = (((uint32_t)_mm_extract_epi32(xmm0, 0)) >> 16);
}

void crc11_gen_snc(bblib_crc_request *request, bblib_crc_response *response)
{
    uint8_t *data = request->data;
    uint8_t *dataOut = response->data;
    uint8_t *temp_ptr;
    // len is passed in as bits, so turn into bytes
    uint32_t len_bytes = request->len/8;
    uint32_t rem_512bits = request->len&0x1FF;
    uint32_t rem_128bits = request->len&0x7F;
    uint32_t rem_16bytes = rem_128bits>>3;
    /* record the start of input data */
    int32_t    i = 0;
    int32_t    k;
    /* A= B mod C => A*K = B*K mod C*K, set K = 2^8, then compute CRC-32, the most significant */
    /* 24 bits is final 24 bits CRC. */
    const static uint64_t  CRC11POLY = 0xe21;    //CRC11 Polynomial
    const static uint64_t  CRC11PLUS21 = CRC11POLY << 21;  //pads poly to 32bits

    /* some pre-computed key constants */
    const static uint32_t k576   = 0x9B800000;   //t=512+64, x^578 mod CRC24APLUS8, verified
    const static uint32_t k512   = 0x9D000000;   //t=512, x^512 mod CRC24APLUS8, verified
    const static uint32_t k192   = 0x8ea00000;   //t=128+64, x^192 mod CRC24APLUS8, verified
    const static uint32_t k128   = 0x47600000;   //t=128, x^128 mod CRC24APLUS8, verified
    const static uint32_t k96    = 0x5e600000;   //t=96, x^96 mod CRC24APLUS8, verified
    const static uint32_t k64    = 0xc9000000;   //t=64, x^64 mod CRC24APLUS8, verified
    const static uint64_t u      = 0x1b3fa1f48;  //u for crc24A * 256, floor(x^64 / CRC24APLUS8), verified
    const static __m128i ENDIA_SHUF_MASK_128bit = _mm_set_epi8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                                               0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
    const static __m512i ENDIA_SHUF_MASK_512bit = _mm512_set_epi8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                                                  0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
                                                                  0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                                                                  0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
                                                                  0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                                                                  0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F,
                                                                  0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
                                                                  0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F);
    __m128i xmm3, xmm2, xmm1, xmm0;
    __m128i k192k128;
    __m128i *pmm0, *pmm1, *pmm2;
    __m512i kmm0;
    __m512i kmm1;
    __m512i kmm2;
    __m512i k576k512;
    int32_t num_fold4;
    int32_t num_fold1;
    num_fold4 = (request->len>>9)-1;
    pmm0 = (__m128i*)&kmm0;
    pmm1 = (__m128i*)&kmm1;
    pmm2 = (__m128i*)&kmm2;
    if(num_fold4 > 0)
    {
        k576k512 = _mm512_set_epi32(0, k576, 0, k512,0, k576, 0, k512,0, k576, 0, k512,0, k576, 0, k512);
        k192k128 = _mm_set_epi32(0, k192, 0, k128);
        num_fold1 = rem_512bits>>7;
        kmm0 = _mm512_load_si512((void const*)data);
        data += 64;
        kmm0 = _mm512_shuffle_epi8(kmm0, ENDIA_SHUF_MASK_512bit);
        for (i=0; i<num_fold4; i++)
        {
            kmm2 = _mm512_load_si512((void const*)data);
            data += 64;
            kmm2 = _mm512_shuffle_epi8(kmm2, ENDIA_SHUF_MASK_512bit);
            kmm1 = _mm512_clmulepi64_epi128(kmm0, k576k512, 0x00);
            kmm0 = _mm512_clmulepi64_epi128(kmm0, k576k512, 0x11);
            kmm0 = _mm512_ternarylogic_epi32(kmm0, kmm1, kmm2, 0x96);
        }
        for(i=0; i<3; i++)
        {
            pmm1[i] = _mm_clmulepi64_si128(pmm0[i], k192k128, 0x00);
            pmm0[i] = _mm_clmulepi64_si128(pmm0[i], k192k128, 0x11);
            pmm0[i+1] = _mm_ternarylogic_epi32(pmm0[i], pmm1[i], pmm0[i+1], 0x96);
        }
        xmm1 = pmm0[3];
    }
    else
    {
        k192k128 = _mm_set_epi32(0, k192, 0, k128);
        num_fold1 = (request->len>>7)-1;
        xmm1 = _mm_load_si128((__m128i *)data);
        data += 16;
        xmm1 = _mm_shuffle_epi8(xmm1, ENDIA_SHUF_MASK_128bit);
    }
    /* 1. fold by 128bit. remaining length <=2*128bits. */
    if(num_fold1 > 0)
    {
        for (i=0; i<num_fold1; i++)
        {
            xmm2 = xmm1;
            xmm0 = _mm_load_si128((__m128i *)data);
            data += 16;
            xmm0 = _mm_shuffle_epi8(xmm0, ENDIA_SHUF_MASK_128bit);
            xmm1 = _mm_clmulepi64_si128(xmm1, k192k128, 0x00);
            xmm2 = _mm_clmulepi64_si128(xmm2, k192k128, 0x11);
            xmm1 = _mm_ternarylogic_epi32(xmm0, xmm1, xmm2, 0x96);
        }
    }
    /* 2. if remaining length > 128 bits, then pad zero to the most-significant bit to grow to 256bits length,
     * then fold once to 128 bits. */
    if(num_fold1<0) /*data length less than 128bits*/
    {
        char byte_shift = (char)(16 - rem_16bytes);
        __m128i iota = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        __m128i shift_value = _mm_add_epi8(iota, _mm_set1_epi8(byte_shift));
        __m128i shifted_foldData = _mm_shuffle_epi8(xmm1, shift_value);
        __mmask16 next_mask = (__mmask16)(0xFFFF << (16-byte_shift));
        __mmask16 fold_mask = (__mmask16)~(next_mask);
        xmm0 = _mm_maskz_mov_epi8(fold_mask, shifted_foldData);
    }
    else
    {
        if(rem_128bits>0)
        {
            xmm0 = _mm_load_si128((__m128i *)data); //load remaining len%16 bytes and maybe some garbage bytes.
            xmm0 = _mm_shuffle_epi8(xmm0, ENDIA_SHUF_MASK_128bit);
            char byte_shift = (char)(16 - rem_16bytes);
            __m128i iota = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m128i shift_value = _mm_add_epi8(iota, _mm_set1_epi8(byte_shift));
            __m128i shifted_foldData = _mm_shuffle_epi8(xmm1, shift_value);
            __m128i shifted_nextData = _mm_shuffle_epi8(xmm0, shift_value);
            __mmask16 next_mask = (__mmask16)(0xFFFF << (16-byte_shift));
            xmm0 = _mm_mask_shuffle_epi8(shifted_nextData, next_mask, shifted_foldData, iota);
            __mmask16 fold_mask = (__mmask16)~(next_mask);
            xmm1 = _mm_maskz_mov_epi8(fold_mask, shifted_foldData);
            xmm2 = xmm1;
            xmm1 = _mm_clmulepi64_si128(xmm1, k192k128, 0x00);
            xmm2 = _mm_clmulepi64_si128(xmm2, k192k128, 0x11);
            xmm0 = _mm_ternarylogic_epi32(xmm0, xmm1, xmm2, 0x96);
        }
        else
            xmm0 = xmm1;
    }
    /* 3. Apply 64 bits fold to 64 bits + 32 bits crc(32 bits zero) */
    xmm3 =  _mm_set_epi32(0, 0, 0, k96);
    xmm1 = _mm_clmulepi64_si128(xmm0, xmm3, 0x01);
    xmm2 = _mm_slli_si128(xmm0, 8);
    xmm2 = _mm_srli_si128(xmm2, 4);
    xmm0 = _mm_xor_si128(xmm1, xmm2);


    /* 4. Apply 32 bits fold to 32 bits + 32 bits crc(32 bits zero) */
    xmm3 =  _mm_set_epi32(0, 0, 0, k64);
    xmm1 = _mm_clmulepi64_si128(xmm0, xmm3, 0x01);
    xmm0 = _mm_slli_si128(xmm0, 8);
    xmm0 = _mm_srli_si128(xmm0, 8);
    xmm0 = _mm_xor_si128(xmm1, xmm0);

    /* 5. Use Barrett Reduction Algorithm to calculate the 32 bits crc.
     * Output: C(x)  = R(x) mod P(x)
     * Step 1: T1(x) = floor(R(x)/x^32)) * u
     * Step 2: T2(x) = floor(T1(x)/x^32)) * P(x)
     * Step 3: C(x)  = R(x) xor T2(x) mod x^32 */
    xmm1 = _mm_set_epi32(0, 0, 1, (u & 0xFFFFFFFF));
    xmm2 = _mm_srli_si128(xmm0, 4);
    xmm1 = _mm_clmulepi64_si128(xmm1, xmm2, 0x00);
    xmm1 = _mm_srli_si128(xmm1, 4);
    xmm2 = _mm_set_epi32(0, 0, 1, (CRC11PLUS21 & 0xFFFFFFFF));
    xmm1 = _mm_clmulepi64_si128(xmm1, xmm2, 0x00);
    xmm0 = _mm_xor_si128(xmm0, xmm1);
    xmm1 = _mm_set_epi32(0, 0, 0, 0xFFFFFFFF);
    xmm0 = _mm_and_si128 (xmm0, xmm1);


    /* 6. Update Result
    /* add crc to last 3 bytes. */
    dataOut[len_bytes]   =  _mm_extract_epi8(xmm0, 3);
    dataOut[len_bytes+1] =  _mm_extract_epi8(xmm0, 2);
    response->len = ((len_bytes + 2)*8) - 5;

    /* the most significant 24 bits of the 32 bits crc is the finial 24 bits crc. */
    response->crc_value = (((uint32_t)_mm_extract_epi32(xmm0, 0)) >> 21);
    _mm256_zeroupper();
}

void crc6_gen_snc(bblib_crc_request *request, bblib_crc_response *response)
{
    uint8_t *data = request->data;
    uint8_t *dataOut = response->data;
    uint8_t *temp_ptr;
    // len is passed in as bits, so turn into bytes
    uint32_t len_bytes = request->len/8;
    uint32_t rem_512bits = request->len&0x1FF;
    uint32_t rem_128bits = request->len&0x7F;
    uint32_t rem_16bytes = rem_128bits>>3;
    /* record the start of input data */
    int32_t    i = 0;
    int32_t    k;
    /* A= B mod C => A*K = B*K mod C*K, set K = 2^8, then compute CRC-32, the most significant */
    /* 24 bits is final 24 bits CRC. */
    const static uint64_t  CRC6POLY = 0x61;    //CRC6 Polynomial
    const static uint64_t  CRC6PLUS26 = CRC6POLY << 26;  //pads poly to 32bits

    /* some pre-computed key constants */
    const static uint32_t k576   = 0xAC000000;   //t=512+64, x^578 mod CRC24APLUS8, verified
    const static uint32_t k512   = 0x94000000;   //t=512, x^512 mod CRC24APLUS8, verified
    const static uint32_t k192   = 0x38000000;   //t=128+64, x^192 mod CRC24APLUS8, verified
    const static uint32_t k128   = 0x1c000000;   //t=128, x^128 mod CRC24APLUS8, verified
    const static uint32_t k96    = 0x8c000000;   //t=96, x^96 mod CRC24APLUS8, verified
    const static uint32_t k64    = 0xcc000000;   //t=64, x^64 mod CRC24APLUS8, verified
    const static uint64_t u      = 0x1fab37693;  //u for crc24A * 256, floor(x^64 / CRC24APLUS8), verified
    const static __m128i ENDIA_SHUF_MASK_128bit = _mm_set_epi8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                                               0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
    const static __m512i ENDIA_SHUF_MASK_512bit = _mm512_set_epi8(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                                                  0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
                                                                  0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
                                                                  0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
                                                                  0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                                                                  0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F,
                                                                  0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
                                                                  0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F);
    __m128i xmm3, xmm2, xmm1, xmm0;
    __m128i k192k128;
    __m128i *pmm0, *pmm1, *pmm2;
    __m512i kmm0;
    __m512i kmm1;
    __m512i kmm2;
    __m512i k576k512;
    int32_t num_fold4;
    int32_t num_fold1;
    num_fold4 = (request->len>>9)-1;
    pmm0 = (__m128i*)&kmm0;
    pmm1 = (__m128i*)&kmm1;
    pmm2 = (__m128i*)&kmm2;
    if(num_fold4 > 0)
    {
        k576k512 = _mm512_set_epi32(0, k576, 0, k512,0, k576, 0, k512,0, k576, 0, k512,0, k576, 0, k512);
        k192k128 = _mm_set_epi32(0, k192, 0, k128);
        num_fold1 = rem_512bits>>7;
        kmm0 = _mm512_load_si512((void const*)data);
        data += 64;
        kmm0 = _mm512_shuffle_epi8(kmm0, ENDIA_SHUF_MASK_512bit);
        for (i=0; i<num_fold4; i++)
        {
            kmm2 = _mm512_load_si512((void const*)data);
            data += 64;
            kmm2 = _mm512_shuffle_epi8(kmm2, ENDIA_SHUF_MASK_512bit);
            kmm1 = _mm512_clmulepi64_epi128(kmm0, k576k512, 0x00);
            kmm0 = _mm512_clmulepi64_epi128(kmm0, k576k512, 0x11);
            kmm0 = _mm512_ternarylogic_epi32(kmm0, kmm1, kmm2, 0x96);
        }
        for(i=0; i<3; i++)
        {
            pmm1[i] = _mm_clmulepi64_si128(pmm0[i], k192k128, 0x00);
            pmm0[i] = _mm_clmulepi64_si128(pmm0[i], k192k128, 0x11);
            pmm0[i+1] = _mm_ternarylogic_epi32(pmm0[i], pmm1[i], pmm0[i+1], 0x96);
        }
        xmm1 = pmm0[3];
    }
    else
    {
        k192k128 = _mm_set_epi32(0, k192, 0, k128);
        num_fold1 = (request->len>>7)-1;
        xmm1 = _mm_load_si128((__m128i *)data);
        data += 16;
        xmm1 = _mm_shuffle_epi8(xmm1, ENDIA_SHUF_MASK_128bit);
    }
    /* 1. fold by 128bit. remaining length <=2*128bits. */
    if(num_fold1 > 0)
    {
        for (i=0; i<num_fold1; i++)
        {
            xmm2 = xmm1;
            xmm0 = _mm_load_si128((__m128i *)data);
            data += 16;
            xmm0 = _mm_shuffle_epi8(xmm0, ENDIA_SHUF_MASK_128bit);
            xmm1 = _mm_clmulepi64_si128(xmm1, k192k128, 0x00);
            xmm2 = _mm_clmulepi64_si128(xmm2, k192k128, 0x11);
            xmm1 = _mm_ternarylogic_epi32(xmm0, xmm1, xmm2, 0x96);
        }
    }
    /* 2. if remaining length > 128 bits, then pad zero to the most-significant bit to grow to 256bits length,
     * then fold once to 128 bits. */
    if(num_fold1<0) /*data length less than 128bits*/
    {
        char byte_shift = (char)(16 - rem_16bytes);
        __m128i iota = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        __m128i shift_value = _mm_add_epi8(iota, _mm_set1_epi8(byte_shift));
        __m128i shifted_foldData = _mm_shuffle_epi8(xmm1, shift_value);
        __mmask16 next_mask = (__mmask16)(0xFFFF << (16-byte_shift));
        __mmask16 fold_mask = (__mmask16)~(next_mask);
        xmm0 = _mm_maskz_mov_epi8(fold_mask, shifted_foldData);
    }
    else
    {
        if(rem_128bits>0)
        {
            xmm0 = _mm_load_si128((__m128i *)data); //load remaining len%16 bytes and maybe some garbage bytes.
            xmm0 = _mm_shuffle_epi8(xmm0, ENDIA_SHUF_MASK_128bit);
            char byte_shift = (char)(16 - rem_16bytes);
            __m128i iota = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m128i shift_value = _mm_add_epi8(iota, _mm_set1_epi8(byte_shift));
            __m128i shifted_foldData = _mm_shuffle_epi8(xmm1, shift_value);
            __m128i shifted_nextData = _mm_shuffle_epi8(xmm0, shift_value);
            __mmask16 next_mask = (__mmask16)(0xFFFF << (16-byte_shift));
            xmm0 = _mm_mask_shuffle_epi8(shifted_nextData, next_mask, shifted_foldData, iota);
            __mmask16 fold_mask = (__mmask16)~(next_mask);
            xmm1 = _mm_maskz_mov_epi8(fold_mask, shifted_foldData);
            xmm2 = xmm1;
            xmm1 = _mm_clmulepi64_si128(xmm1, k192k128, 0x00);
            xmm2 = _mm_clmulepi64_si128(xmm2, k192k128, 0x11);
            xmm0 = _mm_ternarylogic_epi32(xmm0, xmm1, xmm2, 0x96);
        }
        else
            xmm0 = xmm1;
    }
    /* 3. Apply 64 bits fold to 64 bits + 32 bits crc(32 bits zero) */
    xmm3 =  _mm_set_epi32(0, 0, 0, k96);
    xmm1 = _mm_clmulepi64_si128(xmm0, xmm3, 0x01);
    xmm2 = _mm_slli_si128(xmm0, 8);
    xmm2 = _mm_srli_si128(xmm2, 4);
    xmm0 = _mm_xor_si128(xmm1, xmm2);


    /* 4. Apply 32 bits fold to 32 bits + 32 bits crc(32 bits zero) */
    xmm3 =  _mm_set_epi32(0, 0, 0, k64);
    xmm1 = _mm_clmulepi64_si128(xmm0, xmm3, 0x01);
    xmm0 = _mm_slli_si128(xmm0, 8);
    xmm0 = _mm_srli_si128(xmm0, 8);
    xmm0 = _mm_xor_si128(xmm1, xmm0);

    /* 5. Use Barrett Reduction Algorithm to calculate the 32 bits crc.
     * Output: C(x)  = R(x) mod P(x)
     * Step 1: T1(x) = floor(R(x)/x^32)) * u
     * Step 2: T2(x) = floor(T1(x)/x^32)) * P(x)
     * Step 3: C(x)  = R(x) xor T2(x) mod x^32 */
    xmm1 = _mm_set_epi32(0, 0, 1, (u & 0xFFFFFFFF));
    xmm2 = _mm_srli_si128(xmm0, 4);
    xmm1 = _mm_clmulepi64_si128(xmm1, xmm2, 0x00);
    xmm1 = _mm_srli_si128(xmm1, 4);
    xmm2 = _mm_set_epi32(0, 0, 1, (CRC6PLUS26 & 0xFFFFFFFF));
    xmm1 = _mm_clmulepi64_si128(xmm1, xmm2, 0x00);
    xmm0 = _mm_xor_si128(xmm0, xmm1);
    xmm1 = _mm_set_epi32(0, 0, 0, 0xFFFFFFFF);
    xmm0 = _mm_and_si128 (xmm0, xmm1);


    /* 6. Update Result
    /* add crc to last 3 bytes. */
    dataOut[len_bytes]   =  _mm_extract_epi8(xmm0, 3);
    response->len = ((len_bytes + 1)*8) - 2;

    /* the most significant 24 bits of the 32 bits crc is the finial 24 bits crc. */
    response->crc_value = (((uint32_t)_mm_extract_epi32(xmm0, 0)) >> 26);
    _mm256_zeroupper();
}

void crc24a_check_snc(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    if (request->data == NULL){
        printf("bblib_lte_crc24a_check input / output address error \n");
        response->check_passed = false;
        return;
    }

    // len is passed in as bits, so turn into bytes
    uint32_t len_bytes = request->len/8;

    /* CRC in the original sequence */
    uint32_t CRC_orig = 0;

    CRC_orig = ((request->data[len_bytes]<<16)&0x00FF0000) +
               ((request->data[len_bytes+1]<<8)&0x0000FF00) +
               (request->data[len_bytes+2]&0x000000FF);

    bblib_lte_crc24a_gen_snc(request, response);

    if (response->crc_value != CRC_orig)
        response->check_passed = false;
    else
        response->check_passed = true;
}

void crc24b_check_snc(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    if (request->data == NULL){
        printf("bblib_lte_crc24b_check input / output address error \n");
        response->check_passed = false;
        return;
    }

    // len is passed in as bits, so turn into bytes
    uint32_t len_bytes = request->len/8;

    /* CRC in the original sequence */
    uint32_t CRC_orig = 0;

    CRC_orig = ((request->data[len_bytes]<<16)&0x00FF0000) +
               ((request->data[len_bytes+1]<<8)&0x0000FF00) +
               (request->data[len_bytes+2]&0x000000FF);

    bblib_lte_crc24b_gen_snc(request, response);

    if (response->crc_value != CRC_orig)
        response->check_passed = false;
    else
        response->check_passed = true;
}

void crc24c_check_snc(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    if (request->data == NULL){
        printf("bblib_lte_crc24c_check input / output address error \n");
        response->check_passed = false;
        return;
    }

    // len is passed in as bits, so turn into bytes
    uint32_t len_bytes = request->len/8;

    /* CRC in the original sequence */
    uint32_t CRC_orig = 0;

    CRC_orig = ((request->data[len_bytes]<<16)&0x00FF0000) +
               ((request->data[len_bytes+1]<<8)&0x0000FF00) +
               (request->data[len_bytes+2]&0x000000FF);

    bblib_lte_crc24c_gen_snc(request, response);

    if (response->crc_value != CRC_orig)
        response->check_passed = false;
    else
        response->check_passed = true;
}

void crc16_check_snc(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    if (request->data == NULL){
        printf("bblib_lte_crc16_check input / output address error \n");
        response->check_passed = false;
        return;
    }

    // len is passed in as bits, so turn into bytes
    uint32_t len_bytes = request->len/8;

    /* CRC in the original sequence */
    uint32_t CRC_orig = 0;

    CRC_orig = ((request->data[len_bytes]<<8)&0x0000FF00) +
               (request->data[len_bytes+1]&0x000000FF);
    bblib_lte_crc16_gen_snc(request, response);

    if (response->crc_value != CRC_orig)
        response->check_passed = false;
    else
        response->check_passed = true;
}

void crc11_check_snc(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    if (request->data == NULL) {
        printf("bblib_lte_crc11_check input / output address error \n");
        response->check_passed = false;
        return;
    }
    // len is passed in as bits, so turn into bytes
    uint32_t len_bytes = request->len/8;

    /* CRC in the original sequence */
    uint32_t CRC_orig = 0;

    CRC_orig = (((request->data[len_bytes]<<8)&0x0000FF00) +
               ((request->data[len_bytes+1])&0x000000FF)) >> 5;
    bblib_lte_crc11_gen_snc(request, response);

    if (response->crc_value != CRC_orig)
        response->check_passed = false;
    else
        response->check_passed = true;
}

void crc6_check_snc(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    if (request->data == NULL) {
        printf("bblib_lte_crc6_check input / output address error \n");
        response->check_passed = false;
        return;
    }
    // len is passed in as bits, so turn into bytes
    uint32_t len_bytes = request->len/8;

    /* CRC in the original sequence */
    uint32_t CRC_orig = 0;

    CRC_orig = ((request->data[len_bytes])&0x000000FF)>>2;
    bblib_lte_crc6_gen_snc(request, response);

    if (response->crc_value != CRC_orig)
        response->check_passed = false;
    else
        response->check_passed = true;
}

void bblib_lte_crc24a_gen_snc(bblib_crc_request *request, bblib_crc_response *response)
{
    if((request->len&0x7)>0)
        bblib_lte_crc24a_gen_avx512(request, response);
    else
        crc24a_gen_snc(request, response);
}

void bblib_lte_crc24b_gen_snc(bblib_crc_request *request, bblib_crc_response *response)
{
    if((request->len&0x7)>0)
        bblib_lte_crc24b_gen_avx512(request, response);
    else
        crc24b_gen_snc(request, response);
}

void bblib_lte_crc24c_gen_snc(bblib_crc_request *request, bblib_crc_response *response)
{
    if((request->len&0x7)>0)
        bblib_lte_crc24c_gen_avx512(request, response);
    else
        crc24c_gen_snc(request, response);
}

void bblib_lte_crc16_gen_snc(bblib_crc_request *request, bblib_crc_response *response)
{
    if((request->len&0x7)>0)
        bblib_lte_crc16_gen_avx512(request, response);
    else
        crc16_gen_snc(request, response);
}
void bblib_lte_crc11_gen_snc(bblib_crc_request *request, bblib_crc_response *response)
{
    if((request->len&0x7)>0)
        bblib_lte_crc11_gen_avx512(request, response);
    else
        crc11_gen_snc(request, response);
}
void bblib_lte_crc6_gen_snc(bblib_crc_request *request, bblib_crc_response *response)
{
    if((request->len&0x7)>0)
        bblib_lte_crc6_gen_avx512(request, response);
    else
        crc6_gen_snc(request, response);
}

void bblib_lte_crc24a_check_snc(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    if((request->len&0x7)>0)
        bblib_lte_crc24a_check_avx512(request, response);
    else
        crc24a_check_snc(request, response);
}

void bblib_lte_crc24b_check_snc(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    if((request->len&0x7)>0)
        bblib_lte_crc24b_check_avx512(request, response);
    else
        crc24b_check_snc(request, response);
}

void bblib_lte_crc24c_check_snc(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    if((request->len&0x7)>0)
        bblib_lte_crc24c_check_avx512(request, response);
    else
        crc24c_check_snc(request, response);
}

void bblib_lte_crc16_check_snc(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    if((request->len&0x7)>0)
        bblib_lte_crc16_check_avx512(request, response);
    else
        crc16_check_snc(request, response);
}

void bblib_lte_crc11_check_snc(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    if((request->len&0x7)>0)
        bblib_lte_crc11_check_avx512(request, response);
    else
        crc11_check_snc(request, response);
}

void bblib_lte_crc6_check_snc(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    if((request->len&0x7)>0)
        bblib_lte_crc6_check_avx512(request, response);
    else
        crc6_check_snc(request, response);
}

#endif  /* #ifdef _BBLIB_SNC_ */

