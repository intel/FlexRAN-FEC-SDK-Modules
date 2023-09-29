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
#ifndef _PHY_DEMAPPER_
#define _PHY_DEMAPPER_

#pragma once
#include "simd_insts.hpp"
#include "bblib_common_const.h"
#include "common_typedef_sdk.h"

#ifdef _BBLIB_AVX512_

using namespace W_SDK;

static const __m512i alliZero = _mm512_setzero_epi32();
static const __m512 allfZero = _mm512_set1_ps(0.f);
static const __m512 allfOne = _mm512_set1_ps(1);
#define FLOAT16_SCALE (67108864.0)
#define FLOAT16_SCALE_SQRT (8192.0)

static const __m512i data_dmrs_type1[2] = {_mm512_set_epi32(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0),// port 2 3
                                           _mm512_set_epi32(14, 12, 10, 8, 6, 4, 2, 0, 15, 13, 11, 9, 7, 5, 3, 1)}; // port 0 1

static const __m512i data_dmrs_type2_cdms2_idx[3] = { _mm512_set_epi32(15, 14, 13, 12, 7, 6, 1, 0, 9, 8, 3, 2, 11, 10, 5, 4), //0, 4 re
                                                            _mm512_set_epi32(15, 14, 11, 10, 9, 8, 5, 4, 3, 2, 13, 12, 7, 6, 1, 0), //1, 6 re
                                                            _mm512_set_epi32(13, 12, 11, 10, 7, 6, 5, 4, 1, 0, 15, 14, 9, 8, 3, 2)}; //2, 6 re

static const __m512i data_dmrs_type2_cdms1_idx[3] = { _mm512_set_epi32(13, 12, 7, 6, 1, 0, 15, 14, 11, 10, 9, 8, 5, 4, 3, 2), //0, 10 re
                                                            _mm512_set_epi32(15, 14, 9, 8, 3, 2, 13, 12, 11, 10, 7, 6, 5, 4, 1, 0), //1, 10 re
                                                            _mm512_set_epi32(11, 10, 5, 4, 15, 14, 13, 12, 9, 8, 7, 6, 3, 2, 1, 0)}; //2,12 re


#define ptr_cast reinterpret_cast

#define BBLIB_INTERP_GRANS (5)  // Number of possible granularities for 2 DMRS CE symbols

static const int8_t DatanumType1 = 8;
static const int8_t DatanumType2[2][3] =
{
  {10, 10, 12},
  {4,   6,  6}
};


#define N_TX_1 1
#define N_TX_2 2
#define N_RX_2 2
#define N_RX_4 4
#define N_RX_16 16
#define N_TX_4 4
#define N_RX_8 8
#define N_TX_8 8
#define MAX_SC_TIME 205 //3276/16=205
#define MAX_TP_SC_TIME 75 //1200/16=75
#define POSTSNR_FXP_BASE 4 //at least 16S4 to ensure the output LLR not loose any integer digit
#define MAX_IDFT_SIZE 1200
#define PUSCH_MAX_DMRS_SYMBOL       (4)
#define PUSCH_MAX_DMRS_PORT_NUM     (4)


//#define SNR_SMOOTH //to enable the postSINR averaging to smooth the fxp quantized error

static const __m512i m512switchIQ = _mm512_set_epi8(
    61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 50,
    45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34,
    29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18,
    13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);

static const __m512i m512NegI = _mm512_set_epi16(
    0x1,0xFFFF,0x1,0xFFFF,0x1,0xFFFF,0x1,0xFFFF,
    0x1,0xFFFF,0x1,0xFFFF,0x1,0xFFFF,0x1,0xFFFF,
    0x1,0xFFFF,0x1,0xFFFF,0x1,0xFFFF,0x1,0xFFFF,
    0x1,0xFFFF,0x1,0xFFFF,0x1,0xFFFF,0x1,0xFFFF);

/* constants for LLR factor calculation, maximum 2*sqrt(2)=2.828 */
/* and consider to align the fxp with Y, set to 16S13 */

static const __m512i i_p_1_m_sqrt2 = _mm512_set1_epi16(11585);    // sqrt(2)
static const __m512i i_p_2_m_sqrt2 = _mm512_set1_epi16(23170);    // 2*sqrt(2)
static const __m512i i_p_1_d_sqrt10 = _mm512_set1_epi16(2591);   //1/sqrt(10)
static const __m512i i_p_2_d_sqrt10 = _mm512_set1_epi16(5181);   //2/sqrt(10)
static const __m512i i_p_4_d_sqrt10 = _mm512_set1_epi16(10362);    //4/sqrt(10)
static const __m512i i_p_8_d_sqrt10 = _mm512_set1_epi16(20724);    //8/sqrt(10)

static const __m512i i_p_1_d_sqrt42 = _mm512_set1_epi16(1264);   //1/sqrt(42)
static const __m512i i_p_2_d_sqrt42 = _mm512_set1_epi16(2528);   //2/sqrt(42)
static const __m512i i_p_3_d_sqrt42 = _mm512_set1_epi16(3792);   //3/sqrt(42)
static const __m512i i_p_4_d_sqrt42 = _mm512_set1_epi16(5056);   //4/sqrt(42)
static const __m512i i_p_5_d_sqrt42 = _mm512_set1_epi16(6320);   //5/sqrt(42)
static const __m512i i_p_6_d_sqrt42 = _mm512_set1_epi16(7584);   //6/sqrt(42)
static const __m512i i_p_8_d_sqrt42 = _mm512_set1_epi16(10112);    //8/sqrt(42)
static const __m512i i_p_12_d_sqrt42 = _mm512_set1_epi16(15169);    //12/sqrt(42)
static const __m512i i_p_16_d_sqrt42 = _mm512_set1_epi16(20225);    //16/sqrt(42)

static const __m512i i_p_1_d_sqrt170 = _mm512_set1_epi16(628);   //1/sqrt(170)
static const __m512i i_p_2_d_sqrt170 = _mm512_set1_epi16(1257);   //2/sqrt(170)
static const __m512i i_p_3_d_sqrt170 = _mm512_set1_epi16(1885);   //3/sqrt(170)
static const __m512i i_p_4_d_sqrt170 = _mm512_set1_epi16(2513);   //4/sqrt(170)
static const __m512i i_p_5_d_sqrt170 = _mm512_set1_epi16(3142);   //5/sqrt(170)
static const __m512i i_p_6_d_sqrt170 = _mm512_set1_epi16(3770);   //6/sqrt(170)
static const __m512i i_p_7_d_sqrt170 = _mm512_set1_epi16(4398);    //7/sqrt(170)
static const __m512i i_p_8_d_sqrt170 = _mm512_set1_epi16(5026);    //8/sqrt(170)
static const __m512i i_p_9_d_sqrt170 = _mm512_set1_epi16(5655);   //9/sqrt(170)
static const __m512i i_p_10_d_sqrt170 = _mm512_set1_epi16(6283);    //10/sqrt(170)
static const __m512i i_p_11_d_sqrt170 = _mm512_set1_epi16(6911);    //11/sqrt(170)
static const __m512i i_p_12_d_sqrt170 = _mm512_set1_epi16(7540);    //12/sqrt(170)
static const __m512i i_p_13_d_sqrt170 = _mm512_set1_epi16(8168);   //13/sqrt(170)
static const __m512i i_p_14_d_sqrt170 = _mm512_set1_epi16(8796);    //14/sqrt(170)
static const __m512i i_p_16_d_sqrt170 = _mm512_set1_epi16(10053);   //16/sqrt(170)
static const __m512i i_p_20_d_sqrt170 = _mm512_set1_epi16(12566);   //20/sqrt(170)
static const __m512i i_p_24_d_sqrt170 = _mm512_set1_epi16(15079);   //24/sqrt(170)
static const __m512i i_p_28_d_sqrt170 = _mm512_set1_epi16(17592);   //28/sqrt(170)
static const __m512i i_p_32_d_sqrt170 = _mm512_set1_epi16(20106);    //32/sqrt(170)

static const __mmask32 half_pi_bpsk_mask_even = 0x88888888;

static const __mmask32 half_pi_bpsk_mask_odd = 0x22222222;


static const __m512i half_pi_select_first_eight = _mm512_set_epi16(
                                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                    0,0,0,0,0,0,0,0,15,13,11,9,7,5,3,1);

static const __m512i half_pi_select_second_eight = _mm512_set_epi16(
                                    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                    0,0,0,0,0,0,0,0,31,29,27,25,23,21,19,17);

static const __m512i qpsk_perm_idx = _mm512_set_epi64(7,5,3,1,6,4,2,0);


static const __m512i qam16_shuffle_idx = _mm512_set_epi8(15,14,7,6,13,12,5,4,11,10,3,2,9,8,1,0,
                                     15,14,7,6,13,12,5,4,11,10,3,2,9,8,1,0,
                                     15,14,7,6,13,12,5,4,11,10,3,2,9,8,1,0,
                                     15,14,7,6,13,12,5,4,11,10,3,2,9,8,1,0);

static const __m512i qam64_first_64 = _mm512_set_epi16(22,18,49,21,17,48,20,16,43,15,11,42,14,10,41,13,
                                      9,40,12,8,35,7,3,34,6,2,33,5,1,32,4,0);
static const __m512i qam64_second_32 = _mm512_set_epi16(63,62,61,60,55,54,53,52,47,46,45,44,39,38,37,36,
                                      59,31,27,58,30,26,57,29,25,56,28,24,51,23,19,50);
static const __m512i qam64_third_32 = _mm512_set_epi16(13,44,40,12,39,35,7,38,34,6,37,33,5,36,32,4,
                                      27,26,25,24,19,18,17,16,11,10,9,8,3,2,1,0);
static const __m512i qam64_last_64 = _mm512_set_epi16(63,59,31,62,58,30,61,57,29,60,56,28,55,51,23,
                                      54,50,22,53,49,21,52,48,20,47,43,15,46,42,14,45,41);

static const __m512i qam64_index_lo_hi = _mm512_set_epi64(15,14,13,12,3,2,1,0);

static const __m512i qam256_first_64 = _mm512_set_epi16(47,43,15,11,46,42,14,10,45,41,13,9,44,40,12,8,
                                                        39,35,7,3,38,34,6,2,37,33,5,1,36,32,4,0);
static const __m512i qam256_second_64 = _mm512_set_epi16(63,59,31,27,62,58,30,26,61,57,29,25,60,56,28,24,
                                                         55,51,23,19,54,50,22,18,53,49,21,17,52,48,20,16);

static const __m512i m512_sw_r = _mm512_set_epi8(13,12,13,12,9,8,9,8,5,4,5,4,1,0,1,0, \
                                           13,12,13,12,9,8,9,8,5,4,5,4,1,0,1,0, \
                                           13,12,13,12,9,8,9,8,5,4,5,4,1,0,1,0, \
                                           13,12,13,12,9,8,9,8,5,4,5,4,1,0,1,0);

static const __m512i m512_sw_i = _mm512_set_epi8(15,14,15,14,11,10,11,10,7,6,7,6,3,2,3,2, \
                                           15,14,15,14,11,10,11,10,7,6,7,6,3,2,3,2, \
                                           15,14,15,14,11,10,11,10,7,6,7,6,3,2,3,2, \
                                           15,14,15,14,11,10,11,10,7,6,7,6,3,2,3,2);

/* Used for fxp beta and snr layer demapping */
static const __m512i duplicate_one_line = _mm512_set_epi16(
    31,27,30,26,29,25,28,24,23,19,22,18,21,17,20,16,15,11,14,10,13,9,12,8,7,3,6,2,5,1,4,0);

static const __m512i duplicate_low_half = _mm512_set_epi16(
    15,15,11,11,14,14,10,10,13,13,9,9,12,12,8,8,7,7,3,3,6,6,2,2,5,5,1,1,4,4,0,0);

static const __m512i duplicate_high_half = _mm512_set_epi16(
    31,31,27,27,30,30,26,26,29,29,25,25,28,28,24,24,23,23,19,19,22,22,18,18,21,21,17,17,20,20,16,16);

static const __m512i sinr_permute_index0 = _mm512_set_epi32(
    23,7,22,6,21,5,20,4,19,3,18,2,17,1,16,0);

static const __m512i sinr_permute_index1 = _mm512_set_epi32(
    31,15,30,14,29,13,28,12,27,11,26,10,25,9,24,8);

static const __m512i duplicate_low_half_fp16 = _mm512_set_epi16(
    15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8,7,7,6,6,5,5,4,4,3,3,2,2,1,1,0,0);

static const __m512i duplicate_high_half_fp16 = _mm512_set_epi16(
    31,31,30,30,29,29,28,28,27,27,26,26,25,25,24,24,23,23,22,22,21,21,20,20,19,19,18,18,17,17,16,16);

static const __m512i m512_permutex_each32_first256 = _mm512_set_epi32(23,7,22,6,21,5,20,4,19,3,18,2,17,1,16,0);

static const __m512i m512_permutex_each32_second256 = _mm512_set_epi32(31,15,30,14,29,13,28,12,27,11,26,10,25,9,24,8);

template<typename SIMD> struct DataType;
#ifdef _BBLIB_SPR_
template<> struct DataType<CF16vec16> {
    using FloatSimd = CF16vec16;
    using Float = float16;
    using procDataType = float16;
    using invType = float;
    const static FP16_E fp16Int16 = FP16_E::FP16;
};
#endif
template<> struct DataType<CI16vec16> {
    using FloatSimd = F32vec16;
    using Float = float;
    using procDataType = int16_t;
    using invType = float;
    const static FP16_E fp16Int16 = FP16_E::INT16;
};

//return high if a>b, otherwise return low
static  FORCE_INLINE inline __m512i select_high_low_epi16(__m512i a, __m512i b, __m512i high, __m512i low)
{
    return _mm512_mask_blend_epi16(_mm512_cmp_epi16_mask(b, a, _MM_CMPINT_LT), low, high);
}

// return the b value if a>b, other wise return a
static FORCE_INLINE inline __m512 select_low_float(__m512 a, __m512 b)
{
    return _mm512_mask_mov_ps(b,_mm512_cmple_ps_mask(a,b),a);
}

//limit the input epi16 data to requestd range: b(low)~c(high)
static FORCE_INLINE inline __m512i limit_to_saturated_range(__m512i a, int16_t b, int16_t c)
{
    __m512i low_range = _mm512_set1_epi16(b);
    __m512i high_range = _mm512_set1_epi16(c);
    // a = _mm512_mask_blend_epi16(_mm512_cmp_epi16_mask(a, low_range, _MM_CMPINT_LT), a, low_range);
    // return _mm512_mask_blend_epi16(_mm512_cmp_epi16_mask(high_range, a, _MM_CMPINT_LT), a, high_range);
    return _mm512_min_epi16(_mm512_max_epi16(low_range, a), high_range);
}

/*!
   \brief QPSK convert, pack, and permute for beta and 1/(1-beta).
    \param [in] ftempPostSINR is the post-SINR.
    \param [in] ftempGain is the Gain.
    \param [out] xRe is 1/(1-beta), 16Sx.
    \param [out] xIm is beta, 16S15 0~0.9999.
    \param [in] avxxTxSymbol is the Tx Symbol.
    \return null.
*/
template<size_t nLayerPerUe = 1, typename T = Is16vec32>
inline FORCE_INLINE
void cvt_pack_permute_QPSK(T* ftempPostSINR, T* ftempGain, __m512i xRe[nLayerPerUe], __m512i xIm[nLayerPerUe], Is16vec32 *avxxTxSymbol);

/*!
   \brief convert, pack, and permute for beta and 1/(1-beta).
    \param [in] ftempPostSINR is the post-SINR.
    \param [in] ftempGain is the Gain.
    \param [out] xRe is 1/(1-beta), 16Sx.
    \param [out] xIm is beta, 16S15 0~0.9999.
    \param [in] avxxTxSymbol is the Tx Symbol.
    \return null.
*/
template<size_t nLayerPerUe = 1, typename T = Is16vec32>
inline FORCE_INLINE
void cvt_pack_permute(T* ftempPostSINR, T* ftempGain, __m512i xRe[nLayerPerUe], __m512i xIm[nLayerPerUe], Is16vec32 *avxxTxSymbol);

/*!
   \brief QPSK convert, pack, and permute for beta and 1/(1-beta).
    \param [in] ftempPostSINR is the post-SINR.
    \param [in] ftempGain is the Gain.
    \param [out] xRe is 1/(1-beta), 16Sx.
    \param [out] xIm is beta, 16S15 0~0.9999.
    \param [in] avxxTxSymbol is the Tx Symbol.
    \return null.
*/
template<>
inline FORCE_INLINE
void cvt_pack_permute_QPSK<1, F32vec16>(F32vec16* ftempPostSINR, F32vec16* ftempGain, __m512i xRe[1], __m512i xIm[1], Is16vec32 *avxxTxSymbol)
{
    /*permute the beta and postSNR as well*/
    /*xRe[0]:1/(1-beta), 16Sx*/
    /*xIm[0]:beta,       16S15 0~0.9999*/
    auto xtemp1 = _mm512_cvtps_epi32(ftempPostSINR[0]);
    xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);
    xRe[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);
}

/*!
   \brief QPSK convert, pack, and permute for beta and 1/(1-beta).
    \param [in] ftempPostSINR is the post-SINR.
    \param [in] ftempGain is the Gain.
    \param [out] xRe is 1/(1-beta), 16Sx.
    \param [out] xIm is beta, 16S15 0~0.9999.
    \param [in] avxxTxSymbol is the Tx Symbol.
    \return null.
*/
template<>
inline FORCE_INLINE
void cvt_pack_permute_QPSK<2, F32vec16>(F32vec16* ftempPostSINR, F32vec16* ftempGain, __m512i xRe[2], __m512i xIm[2], Is16vec32 *avxxTxSymbol)
{
    /*permute the beta and postSNR as well*/
    /*xRe:1/(1-beta), 16Sx*/
    /*xIm:beta,       16S15 0~0.9999*/
    auto xtemp1 = _mm512_cvtps_epi32(ftempPostSINR[0]);
    auto xtemp2 = _mm512_cvtps_epi32(ftempPostSINR[1]);
    xtemp1 = _mm512_packs_epi32(xtemp1, xtemp2);
    xRe[0] = _mm512_permutexvar_epi16(duplicate_low_half,xtemp1);
    xRe[1] = _mm512_permutexvar_epi16(duplicate_high_half,xtemp1);

    auto xtemp3 = _mm512_permutex2var_epi32(avxxTxSymbol[0],m512_permutex_each32_first256,avxxTxSymbol[1]);
    avxxTxSymbol[1] = _mm512_permutex2var_epi32(avxxTxSymbol[0],m512_permutex_each32_second256,avxxTxSymbol[1]);
    avxxTxSymbol[0] = xtemp3;
}

/*!
   \brief convert, pack, and permute for beta and 1/(1-beta).
    \param [in] ftempPostSINR is the post-SINR.
    \param [in] ftempGain is the Gain.
    \param [out] xRe is 1/(1-beta), 16Sx.
    \param [out] xIm is beta, 16S15 0~0.9999.
    \param [in] avxxTxSymbol is the Tx Symbol.
    \return null.
*/
template<>
inline FORCE_INLINE
void cvt_pack_permute<1, F32vec16>(F32vec16* ftempPostSINR, F32vec16* ftempGain, __m512i xRe[1], __m512i xIm[1], Is16vec32 *avxxTxSymbol)
{
    /*permute the beta and postSNR as well*/
    /*xRe[0]:1/(1-beta), 16Sx*/
    /*xIm[0]:beta,       16S15 0~0.9999*/
    auto xtemp1 = _mm512_cvtps_epi32(ftempPostSINR[0]);
    xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);
    xRe[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);

    xtemp1 = _mm512_cvtps_epi32(ftempGain[0]);
    xtemp1 = _mm512_packs_epi32(xtemp1, xtemp1);
    xIm[0] = _mm512_permutexvar_epi16(duplicate_one_line,xtemp1);
}

/*!
   \brief convert, pack, and permute for beta and 1/(1-beta).
    \param [in] ftempPostSINR is the post-SINR.
    \param [in] ftempGain is the Gain.
    \param [out] xRe is 1/(1-beta), 16Sx.
    \param [out] xIm is beta, 16S15 0~0.9999.
    \param [in] avxxTxSymbol is the Tx Symbol.
    \return null.
*/
template<>
inline FORCE_INLINE
void cvt_pack_permute<2, F32vec16>(F32vec16* ftempPostSINR, F32vec16* ftempGain, __m512i xRe[2], __m512i xIm[2], Is16vec32 *avxxTxSymbol)
{
    /*permute the beta and postSNR as well*/
    /*xRe:1/(1-beta), 16Sx*/
    /*xIm:beta,       16S15 0~0.9999*/
    auto xtemp1 = _mm512_cvtps_epi32(ftempPostSINR[0]);
    auto xtemp2 = _mm512_cvtps_epi32(ftempPostSINR[1]);
    xtemp1 = _mm512_packs_epi32(xtemp1, xtemp2);
    xRe[0] = _mm512_permutexvar_epi16(duplicate_low_half,xtemp1);
    xRe[1] = _mm512_permutexvar_epi16(duplicate_high_half,xtemp1);

    xtemp1 = _mm512_cvtps_epi32(ftempGain[0]);
    xtemp2 = _mm512_cvtps_epi32(ftempGain[1]);
    xtemp1 = _mm512_packs_epi32(xtemp1, xtemp2);
    xIm[0] = _mm512_permutexvar_epi16(duplicate_low_half,xtemp1);
    xIm[1] = _mm512_permutexvar_epi16(duplicate_high_half,xtemp1);

    auto xtemp3 = _mm512_permutex2var_epi32(avxxTxSymbol[0],m512_permutex_each32_first256,avxxTxSymbol[1]);
    avxxTxSymbol[1] = _mm512_permutex2var_epi32(avxxTxSymbol[0],m512_permutex_each32_second256,avxxTxSymbol[1]);
    avxxTxSymbol[0] = xtemp3;
}
#ifdef _BBLIB_SPR_

/*!
   \brief permute_gain_sinr.
    \param [in] in is the Gain.
    \param [out] out is gain permuted .
    \return null.
*/
inline FORCE_INLINE
void permute_gain_sinr(CF16vec16 *in, __m512i *out)
{
    __m512i xtemp1 = _mm512_cvtph_epi16(in[0]);
    __m512i xtemp2 = _mm512_cvtph_epi16(in[1]);
    out[0] = _mm512_permutex2var_epi32 (xtemp1, sinr_permute_index0, xtemp2);
    out[1] = _mm512_permutex2var_epi32 (xtemp1, sinr_permute_index1, xtemp2);
}

/*!
   \brief QPSK convert, pack, and permute for beta and 1/(1-beta).
    \param [in] ftempPostSINR is the post-SINR.
    \param [in] ftempGain is the Gain.
    \param [out] xRe is 1/(1-beta), 16Sx.
    \param [out] xIm is beta, 16S15 0~0.9999.
    \param [in] avxxTxSymbol is the Tx Symbol.
    \return null.
*/
template<>
inline FORCE_INLINE
void cvt_pack_permute_QPSK<1, CF16vec16>(CF16vec16* ftempPostSINR, CF16vec16* ftempGain, __m512i xRe[1], __m512i xIm[1], Is16vec32 *avxxTxSymbol)
{
    /*permute the beta and postSNR as well*/
    /*xRe[0]:1/(1-beta), 16Sx*/
    /*xIm[0]:beta,       16S15 0~0.9999*/
    xRe[0] = _mm512_cvtph_epi16(ftempPostSINR[0]);
}

/*!
   \brief QPSK convert, pack, and permute for beta and 1/(1-beta).
    \param [in] ftempPostSINR is the post-SINR.
    \param [in] ftempGain is the Gain.
    \param [out] xRe is 1/(1-beta), 16Sx.
    \param [out] xIm is beta, 16S15 0~0.9999.
    \param [in] avxxTxSymbol is the Tx Symbol.
    \return null.
*/
template<>
inline FORCE_INLINE
void cvt_pack_permute_QPSK<2, CF16vec16>(CF16vec16* ftempPostSINR, CF16vec16* ftempGain, __m512i xRe[2], __m512i xIm[2], Is16vec32 *avxxTxSymbol)
{
    /*permute the beta and postSNR as well*/
    /*xRe:1/(1-beta), 16Sx*/
    /*xIm:beta,       16S15 0~0.9999*/
    permute_gain_sinr(ftempPostSINR, xRe);

    auto xtemp3 = _mm512_permutex2var_epi32(avxxTxSymbol[0],m512_permutex_each32_first256,avxxTxSymbol[1]);
    avxxTxSymbol[1] = _mm512_permutex2var_epi32(avxxTxSymbol[0],m512_permutex_each32_second256,avxxTxSymbol[1]);
    avxxTxSymbol[0] = xtemp3;
}

/*!
   \brief convert, pack, and permute for beta and 1/(1-beta).
    \param [in] ftempPostSINR is the post-SINR.
    \param [in] ftempGain is the Gain.
    \param [out] xRe is 1/(1-beta), 16Sx.
    \param [out] xIm is beta, 16S15 0~0.9999.
    \param [in] avxxTxSymbol is the Tx Symbol.
    \return null.
*/
template<>
inline FORCE_INLINE
void cvt_pack_permute<1, CF16vec16>(CF16vec16* ftempPostSINR, CF16vec16* ftempGain, __m512i xRe[1], __m512i xIm[1], Is16vec32 *avxxTxSymbol)
{
    /*permute the beta and postSNR as well*/
    /*xRe[0]:1/(1-beta), 16Sx*/
    /*xIm[0]:beta,       16S15 0~0.9999*/
    xRe[0] = _mm512_cvtph_epi16(ftempPostSINR[0]);
    xIm[0] = _mm512_cvtph_epi16(ftempGain[0]);
}

/*!
   \brief convert, pack, and permute for beta and 1/(1-beta).
    \param [in] ftempPostSINR is the post-SINR.
    \param [in] ftempGain is the Gain.
    \param [out] xRe is 1/(1-beta), 16Sx.
    \param [out] xIm is beta, 16S15 0~0.9999.
    \param [in] avxxTxSymbol is the Tx Symbol.
    \return null.
*/
template<>
inline FORCE_INLINE
void cvt_pack_permute<2, CF16vec16>(CF16vec16* ftempPostSINR, CF16vec16* ftempGain, __m512i xRe[2], __m512i xIm[2], Is16vec32 *avxxTxSymbol)
{
    /*permute the beta and postSNR as well*/
    /*xRe:1/(1-beta), 16Sx*/
    /*xIm:beta,       16S15 0~0.9999*/
    permute_gain_sinr(ftempPostSINR, xRe);
    permute_gain_sinr(ftempGain, xIm);

    auto xtemp3 = _mm512_permutex2var_epi32(avxxTxSymbol[0],m512_permutex_each32_first256,avxxTxSymbol[1]);
    avxxTxSymbol[1] = _mm512_permutex2var_epi32(avxxTxSymbol[0],m512_permutex_each32_second256,avxxTxSymbol[1]);
    avxxTxSymbol[0] = xtemp3;
}
#endif // _BBLIB_SPR_


//do LLR demapping store
//do LLR demapping store
/*!
   \brief do LLR demapping store.
    \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
    \param [in] xRe is the output soft bits.
    \param [in] xIm is the output soft bits.
    \return null.
*/
template<enum bblib_modulation_order modOrder = BBLIB_QAM16, size_t nLayerPerUe = 1>
inline FORCE_INLINE
void llr_store(int8_t * ppSoft, __m512i*  xRe, __m512i*  xIm);

/*!
   \brief do LLR demapping store.
    \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
    \param [in] xRe is the output soft bits.
    \param [in] xIm is the output soft bits.
    \return null.
*/
template<>
inline FORCE_INLINE
void llr_store<BBLIB_QPSK, 1>(int8_t * ppSoft, __m512i*  xRe, __m512i*  xIm)
{
    xRe[0] = _mm512_packs_epi16(xRe[0], xRe[0]);
    xRe[0] = _mm512_permutexvar_epi64(qpsk_perm_idx,xRe[0]);

    _mm256_storeu_si256 (reinterpret_cast<__m256i *>(ppSoft),_mm512_extracti32x8_epi32(xRe[0], 0));

    // ppSoft += 32;
}

/*!
   \brief do LLR demapping store.
    \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
    \param [in] xRe is the output soft bits.
    \param [in] xIm is the output soft bits.
    \return null.
*/
template<>
inline FORCE_INLINE
void llr_store<BBLIB_QPSK, 2>(int8_t * ppSoft, __m512i*  xRe, __m512i*  xIm)
{
    xRe[0] = _mm512_packs_epi16(xRe[0], xRe[1]);
    xRe[0] = _mm512_permutexvar_epi64(qpsk_perm_idx,xRe[0]);

    _mm512_storeu_si512 (ppSoft,xRe[0]);

    // ppSoft += 64;
}

/*!
   \brief do LLR demapping store.
    \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
    \param [in] xRe is the output soft bits.
    \param [in] xIm is the output soft bits.
    \return null.
*/
template<>
inline FORCE_INLINE
void llr_store<BBLIB_QAM16, 1>(int8_t * ppSoft, __m512i*  xRe, __m512i*  xIm)
{
    xRe[0] = _mm512_shuffle_epi8(xRe[0], qam16_shuffle_idx);
    _mm512_storeu_si512 (ppSoft,xRe[0]);

    // ppSoft += 64;
}

/*!
   \brief do LLR demapping store.
    \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
    \param [in] xRe is the output soft bits.
    \param [in] xIm is the output soft bits.
    \return null.
*/
template<>
inline FORCE_INLINE
void llr_store<BBLIB_QAM16, 2>(int8_t * ppSoft, __m512i*  xRe, __m512i*  xIm)
{
    xRe[0] = _mm512_shuffle_epi8(xRe[0], qam16_shuffle_idx);
    _mm512_storeu_si512 (ppSoft,xRe[0]);

    xRe[1] = _mm512_shuffle_epi8(xRe[1], qam16_shuffle_idx);
    _mm512_storeu_si512 (ppSoft+64,xRe[1]);

    // ppSoft += 128;
}

/*!
   \brief do LLR demapping store.
    \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
    \param [in] xRe is the output soft bits.
    \param [in] xIm is the output soft bits.
    \return null.
*/
template<>
inline FORCE_INLINE
void llr_store<BBLIB_QAM64, 1>(int8_t * ppSoft, __m512i*  xRe, __m512i*  xIm)
{
    auto avxtempCRe = _mm512_permutex2var_epi16(xRe[0], qam64_first_64, xIm[0]);
    _mm512_storeu_si512 (ppSoft,avxtempCRe);

    auto avxtempCIm= _mm512_permutex2var_epi16(xRe[0], qam64_second_32, xIm[0]);
    _mm256_storeu_si256 (reinterpret_cast<__m256i *>(ppSoft+64),_mm512_extracti32x8_epi32(avxtempCIm, 0));

    // ppSoft += 96;
}

/*!
   \brief do LLR demapping store.
    \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
    \param [in] xRe is the output soft bits.
    \param [in] xIm is the output soft bits.
    \return null.
*/
template<>
inline FORCE_INLINE
void llr_store<BBLIB_QAM64, 2>(int8_t * ppSoft, __m512i*  xRe, __m512i*  xIm)
{
    auto avxtempCRe = _mm512_permutex2var_epi16(xRe[0], qam64_first_64, xIm[0]);
    _mm512_storeu_si512 (ppSoft,avxtempCRe);

    auto avxtempCIm = _mm512_permutex2var_epi16(xRe[0], qam64_second_32, xIm[0]);
    auto avxtempDRe = _mm512_permutex2var_epi16(xRe[1], qam64_third_32, xIm[1]);
    auto avxtempDIm = _mm512_permutex2var_epi16(xRe[1], qam64_last_64, xIm[1]);
    _mm512_storeu_si512 (ppSoft+128,avxtempDIm);

    avxtempCIm = _mm512_permutex2var_epi64(avxtempCIm,qam64_index_lo_hi,avxtempDRe);
    _mm512_storeu_si512(ppSoft+64,avxtempCIm);

    // ppSoft += 192;
}

/*!
   \brief do LLR demapping store.
    \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
    \param [in] xRe is the output soft bits.
    \param [in] xIm is the output soft bits.
    \return null.
*/
template<>
inline FORCE_INLINE
void llr_store<BBLIB_QAM256, 1>(int8_t * ppSoft, __m512i*  xRe, __m512i*  xIm)
{
    auto avxtemp1 = _mm512_permutex2var_epi16(xRe[0], qam256_first_64, xIm[0]);
    _mm512_storeu_si512 (ppSoft,avxtemp1);

    auto avxtemp2= _mm512_permutex2var_epi16(xRe[0], qam256_second_64, xIm[0]);
    _mm512_storeu_si512 (ppSoft+64, avxtemp2);

    // ppSoft += 128;
}

/*!
   \brief do LLR demapping store.
    \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
    \param [in] xRe is the output soft bits.
    \param [in] xIm is the output soft bits.
    \return null.
*/
template<>
inline FORCE_INLINE
void llr_store<BBLIB_QAM256, 2>(int8_t * ppSoft, __m512i* xRe, __m512i* xIm)
{
    auto avxtempCRe = _mm512_permutex2var_epi16(xRe[0], qam256_first_64, xIm[0]);
    _mm512_storeu_si512(ppSoft,avxtempCRe);

    auto avxtempCIm= _mm512_permutex2var_epi16(xRe[0], qam256_second_64, xIm[0]);
    _mm512_storeu_si512(ppSoft+64,avxtempCIm);

    auto avxtempDRe = _mm512_permutex2var_epi16(xRe[1], qam256_first_64, xIm[1]);
    _mm512_storeu_si512(ppSoft+128,avxtempDRe);

    auto avxtempDIm= _mm512_permutex2var_epi16(xRe[1], qam256_second_64, xIm[1]);
    _mm512_storeu_si512 (ppSoft+192,avxtempDIm);

    // ppSoft += 256;
}


/*!
    \brief do LLR demapping store.
    \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
    \param [in] xRe is the output soft bits.
    \param [in] xIm is the output soft bits.
    \param [in] nSc is store subcarrier num.
    \return null.
*/
template<enum bblib_modulation_order modOrder = BBLIB_QPSK, size_t nLayerPerUe = 1>
inline FORCE_INLINE
void llr_store(int8_t * ppSoft, __m512i * xRe, __m512i * xIm, int32_t nSc);

/*!
    \brief do LLR demapping store.
    \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
    \param [in] xRe is the output soft bits.
    \param [in] xIm is the output soft bits.
    \param [in] nSc is store subcarrier num.
    \return null.
*/
template<>
inline FORCE_INLINE
void llr_store<BBLIB_QPSK, 1>(int8_t * ppSoft, __m512i*  xRe, __m512i*  xIm, int32_t nSc)
{
    xRe[0] = _mm512_packs_epi16(xRe[0], xRe[0]);
    xRe[0] = _mm512_permutexvar_epi64(qpsk_perm_idx,xRe[0]);
    auto mask = ((uint64_t) 1 << (nSc * BBLIB_QPSK)) - 1;
    _mm512_mask_storeu_epi8(ppSoft, mask, xRe[0]);
}

/*!
    \brief do LLR demapping store.
    \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
    \param [in] xRe is the output soft bits.
    \param [in] xIm is the output soft bits.
    \param [in] nSc is store subcarrier num.
    \return null.
*/
template<>
inline FORCE_INLINE
void llr_store<BBLIB_QAM16, 1>(int8_t * ppSoft, __m512i*  xRe, __m512i*  xIm, int32_t nSc)
{
    xRe[0] = _mm512_shuffle_epi8(xRe[0], qam16_shuffle_idx);
    auto mask = ((uint64_t) 1 << (nSc * BBLIB_QAM16)) - 1;
    _mm512_mask_storeu_epi8(ppSoft, mask, xRe[0]);
}

/*!
    \brief do LLR demapping store.
    \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
    \param [in] xRe is the output soft bits.
    \param [in] xIm is the output soft bits.
    \param [in] nSc is store subcarrier num.
    \return null.
*/
template<>
inline FORCE_INLINE
void llr_store<BBLIB_QAM64, 1>(int8_t * ppSoft, __m512i*  xRe, __m512i*  xIm, int32_t nSc)
{
    if (nSc == 8 || nSc == 4)
    {
        int64_t mask = ((uint64_t) 1 << (nSc * BBLIB_QAM64)) - 1;
        auto avxtempCRe = _mm512_permutex2var_epi16(xRe[0], qam64_first_64, xIm[0]);
        _mm512_mask_storeu_epi8(ppSoft, mask, avxtempCRe);
    }
    else if (nSc == 12) {
        auto avxtempCRe = _mm512_permutex2var_epi16(xRe[0], qam64_first_64, xIm[0]);
        // _mm512_storeu_epi16 (ppSoft, avxtempCRe);
        _mm512_storeu_si512 (ppSoft,avxtempCRe);

        int64_t mask = ((uint64_t) 1 << (nSc * BBLIB_QAM64 - 64)) - 1;
        auto avxtempCIm= _mm512_permutex2var_epi16(xRe[0], qam64_second_32, xIm[0]);
        _mm512_mask_storeu_epi8(ppSoft + 64, mask, avxtempCIm);
        // _mm512_mask_storeu_epi16(ppSoft+64,0xffff,avxtempCIm);
        // _mm256_storeu_si256 (reinterpret_cast<__m256i *>(ppSoft+64),_mm512_extracti32x8_epi32(avxtempCIm, 0));
    }
    // ppSoft += 96;
}

/*!
    \brief do LLR demapping store.
    \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
    \param [in] xRe is the output soft bits.
    \param [in] xIm is the output soft bits.
    \param [in] nSc is store subcarrier num.
    \return null.
*/
template<>
inline FORCE_INLINE
void llr_store<BBLIB_QAM256, 1>(int8_t * ppSoft, __m512i*  xRe, __m512i*  xIm, int32_t nSc)
{
    // first 64
    auto avxtemp1 = _mm512_permutex2var_epi16(xRe[0], qam256_first_64, xIm[0]);
    if(nSc == 8){
        _mm512_storeu_si512(ppSoft,avxtemp1);
    }
    else if(nSc == 4){
        constexpr int64_t mask = ((uint64_t) 1 << (32)) - 1;
        //rest 32
        _mm512_mask_storeu_epi8(ppSoft, mask, avxtemp1);
    }
    else if(nSc == 12){
        _mm512_storeu_si512(ppSoft,avxtemp1);
        auto avxtemp2= _mm512_permutex2var_epi16(xRe[0], qam256_second_64, xIm[0]);
        constexpr int64_t mask = ((uint64_t) 1 << (32)) - 1;
        _mm512_mask_storeu_epi8(ppSoft + 64, mask, avxtemp2);
    }
}

/*!
    \brief do LLR demapping store.
    \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
    \param [in] xRe is the output soft bits.
    \param [in] xIm is the output soft bits.
    \param [in] nSc is store subcarrier num
    \return null.
*/
template<>
inline FORCE_INLINE
void llr_store<BBLIB_QPSK, 2>(int8_t * ppSoft, __m512i*  xRe, __m512i*  xIm, int32_t nSc)
{
    xRe[0] = _mm512_packs_epi16(xRe[0], xRe[1]);
    xRe[0] = _mm512_permutexvar_epi64(qpsk_perm_idx,xRe[0]);
    auto mask = ((uint64_t) 1 << (nSc * BBLIB_QPSK * 2)) - 1;
    _mm512_mask_storeu_epi8(ppSoft, mask, xRe[0]);
}

/*!
    \brief do LLR demapping store.
    \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
    \param [in] xRe is the output soft bits.
    \param [in] xIm is the output soft bits.
    \param [in] nSc is store subcarrier num.
    \return null.
*/
template<>
inline FORCE_INLINE
void llr_store<BBLIB_QAM16, 2>(int8_t * ppSoft, __m512i*  xRe, __m512i*  xIm, int32_t nSc)
{
    xRe[0] = _mm512_shuffle_epi8(xRe[0], qam16_shuffle_idx);
    // _mm512_storeu_epi16 (ppSoft,xRe[0]);
    // _mm512_storeu_si512 (ppSoft,xRe[0]);

    if (nSc == 4) {
        // xRe[1] = _mm512_shuffle_epi8(xRe[1], qam16_shuffle_idx);
        auto mask = ((uint64_t) 1 << (nSc * BBLIB_QAM16 * 2)) - 1;
        //rest 32
        _mm512_mask_storeu_epi8(ppSoft, mask, xRe[0]);
    }
    else if (nSc == 12) {
        _mm512_storeu_si512 (ppSoft,xRe[0]);
        xRe[1] = _mm512_shuffle_epi8(xRe[1], qam16_shuffle_idx);
        // _mm512_storeu_epi16 (ppSoft+64,xRe[1]);
        // _mm512_storeu_si512 (ppSoft+64,xRe[1]);

        auto mask = ((uint64_t) 1 << ((nSc - 8) * BBLIB_QAM16 * 2)) - 1;
        _mm512_mask_storeu_epi8(ppSoft + 64, mask, xRe[1]);
    }
    else{
        _mm512_storeu_si512 (ppSoft,xRe[0]);
    }
    // ppSoft += 128;
}

/*!
    \brief do LLR demapping store.
    \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
    \param [in] xRe is the output soft bits.
    \param [in] xIm is the output soft bits.
    \param [in] nSc is store subcarrier num.
    \return null.
*/
template<>
inline FORCE_INLINE
void llr_store<BBLIB_QAM64, 2>(int8_t * ppSoft, __m512i*  xRe, __m512i*  xIm, int32_t nSc)
{
    auto avxtempCRe = _mm512_permutex2var_epi16(xRe[0], qam64_first_64, xIm[0]);
    // _mm512_storeu_epi16(ppSoft,avxtempCRe);
    //first 64
    if (nSc == 4) {
        constexpr int64_t mask = ((uint64_t) 1 << (BBLIB_QAM64 * 2 * 4)) - 1;
        //rest 32
        _mm512_mask_storeu_epi8(ppSoft, mask, avxtempCRe);
    } else {
        _mm512_storeu_si512 (ppSoft,avxtempCRe);

        auto avxtempCIm = _mm512_permutex2var_epi16(xRe[0], qam64_second_32, xIm[0]);
        auto avxtempDRe = _mm512_permutex2var_epi16(xRe[1], qam64_third_32, xIm[1]);
        auto avxtempDIm = _mm512_permutex2var_epi16(xRe[1], qam64_last_64, xIm[1]);

        avxtempCIm = _mm512_permutex2var_epi64(avxtempCIm,qam64_index_lo_hi,avxtempDRe);

        if (nSc == 8) {
            constexpr int64_t mask = ((uint64_t) 1 << (32)) - 1;
            //rest 32
            _mm512_mask_storeu_epi8(ppSoft + 64, mask, avxtempCIm);
        } else if (nSc == 12) {
            //second 64
            _mm512_storeu_si512(ppSoft+64,avxtempCIm);
            constexpr int64_t mask = ((uint64_t) 1 << (16)) - 1;
            //rest 16
            _mm512_mask_storeu_epi8(ppSoft + 128, mask, avxtempDIm);
        }
    }
}

/*!
    \brief do LLR demapping store.
    \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
    \param [in] xRe is the output soft bits.
    \param [in] xIm is the output soft bits.
    \param [in] nSc is store subcarrier num.
    \return null.
*/
template<>
inline FORCE_INLINE
void llr_store<BBLIB_QAM256, 2>(int8_t * ppSoft, __m512i* xRe, __m512i* xIm, int32_t nSc)
{
    auto avxtempCRe = _mm512_permutex2var_epi16(xRe[0], qam256_first_64, xIm[0]);
    _mm512_storeu_si512(ppSoft,avxtempCRe);

    if (nSc != 4) {
        auto avxtempCIm= _mm512_permutex2var_epi16(xRe[0], qam256_second_64, xIm[0]);
        _mm512_storeu_si512(ppSoft+64,avxtempCIm);

        if (nSc == 12) {
            auto avxtempDRe = _mm512_permutex2var_epi16(xRe[1], qam256_first_64, xIm[1]);
            _mm512_storeu_si512(ppSoft+128,avxtempDRe);
        }
    }
// ppSoft += 256;
}

template<size_t nLayerPerUe = 1,typename T = Is16vec32>
static void inline FORCE_INLINE llr_demap_layer_pi2bpsk(T * ftempPostSINR,
    T * ftempGain, Is16vec32 * avxxTxSymbol, int8_t * ppSoft, int16_t llr_range_low, int16_t llr_range_high, int32_t nSc)
    {
        __m512i xRe[nLayerPerUe], xIm[nLayerPerUe];
        // cvt_pack_permute_xx for pi/2 BPSK is same as QPSK
        cvt_pack_permute_QPSK<nLayerPerUe, T>(ftempPostSINR, ftempGain, xRe, xIm, avxxTxSymbol);
        /*
         * pi/2 BPSK is only used for tranform precoding
         * For curent delpoyment, nSC will surely be even
         * Then we hard code half_pi_bpsk_mask_even for LLR
         * leave the legacy code here for backward compatibility
          int32_t nLlrSequenceIdx = 1; //used for pi/2 llr demapping, 1 means start with even idx, -1 means start with odd idx
          __mmask32 halfPiSubMask = 0;
          if (nSc % 2 != 0)
          nLlrSequenceIdx = nLlrSequenceIdx*-1; //reverse the sign of the indication
          (nLlrSequenceIdx == 1) ? (halfPiSubMask = half_pi_bpsk_mask_even) : (halfPiSubMask = half_pi_bpsk_mask_odd);
        */
        __mmask32 halfPiSubMask = half_pi_bpsk_mask_even;
        Is16vec32 alliZero = _mm512_set1_epi16(0);
        __m512i x1 = alliZero;
        #pragma unroll(nLayerPerUe)
        for (size_t iLayer = 0; iLayer < nLayerPerUe; iLayer++)
        {
            /*Algorithm:*/
            /*even idx: LLR0 = (real(In)+imag(In))*sqrt(2)/(1-beta);*/
            /*odd idx:  LLR0 = (-real(In)+imag(In))*sqrt(2)/(1-beta)*/
            /*I0,Q0,I1,Q1,...I15,Q15 ---> x, I0,Q0,I1,Q1,...Q14,I15*/
            auto x0 = _mm512_bslli_epi128(avxxTxSymbol[iLayer], 2);
            /*if even: x,I0,Q0,-I1,Q1,I2,Q2,-I3,Q3...-I15*/
            /*if odd:  x,-I0,Q0,I1,Q1,-I2,Q2,I3,Q3...I15*/
            x0 = _mm512_mask_sub_epi16(x0, halfPiSubMask, alliZero, x0);
            /*if even: x+I0,I0+Q0,Q0+I1,-I1+Q1,Q1+I2,I2+Q2,Q2+I3,-I3+Q3,...-I15+Q15*/
            /*if odd:  x+I0,-I0+Q0,Q0+I1,I1+Q1,Q1+I2,-I2+Q2,Q2+I3,I3+Q3,...I15+Q15*/
            x0 = _mm512_adds_epi16(avxxTxSymbol[iLayer], x0);
            /*output: 16S13*(16S13*16Sx) -> 16S(nLlrFxpPoints)*/
            x0 = _mm512_mulhrs_epi16(x0, _mm512_mulhrs_epi16(xRe[0], i_p_1_m_sqrt2));
            x0 = limit_to_saturated_range(x0, llr_range_low, llr_range_high);

            x1 = _mm512_permutexvar_epi16(half_pi_select_first_eight, x0);
            auto x2 = _mm512_permutexvar_epi16(half_pi_select_second_eight, x0);
            x1 = _mm512_packs_epi16(x1, x2);
        }
        __mmask64 nHalfPiLlrStoreKs = 0;
        if(likely(nSc == 16)) {
            nHalfPiLlrStoreKs = 0xffff;
            _mm512_mask_storeu_epi8(ppSoft, nHalfPiLlrStoreKs, (__m512i)x1);
        } else {
            nHalfPiLlrStoreKs = 0xffffU >> (16 - nSc);
            _mm512_mask_storeu_epi8(ppSoft, nHalfPiLlrStoreKs, (__m512i)x1);
        }
 }

template<size_t nLayerPerUe = 1,typename T = Is16vec32>
static void inline FORCE_INLINE llr_demap_layer_qpsk(T * ftempPostSINR,
    T * ftempGain, Is16vec32 * avxxTxSymbol, int8_t * ppSoft, int16_t llr_range_low, int16_t llr_range_high, int32_t nSc) {
     __m512i xRe[nLayerPerUe], xIm[nLayerPerUe];
     cvt_pack_permute_QPSK<nLayerPerUe, T>(ftempPostSINR, ftempGain, xRe, xIm, avxxTxSymbol);
     #pragma unroll(nLayerPerUe)
     for (size_t iLayer = 0; iLayer < nLayerPerUe; iLayer ++)
     {
         /*Algorithm:*/
         /*LLR0 = real(In)*2*sqrt(2)/(1-beta)*/
         /*LLR1 = imag(In)*2*sqrt(2)/(1-beta)*/

         /*output 16S13*16S13->16S11*/
         auto x = _mm512_mulhrs_epi16(avxxTxSymbol[iLayer], i_p_2_m_sqrt2);
         /*output 16S11*16Sx->16S(nLlrFxpPoints)*/
         x = _mm512_mulhrs_epi16(x,xRe[iLayer]);
         xRe[iLayer] = limit_to_saturated_range(x, llr_range_low, llr_range_high);
     }
     if(likely(nSc == 16)) {
         llr_store<BBLIB_QPSK, nLayerPerUe>(ppSoft, xRe, xIm);
     } else {
         llr_store<BBLIB_QPSK, nLayerPerUe>(ppSoft, xRe, xIm, nSc);
     }
 }

 /*!
 \brief do LLR demapping for 16qam.
     \param [in] ftempPostSINR is the post-SINR.
     \param [in] ftempGain is the Gain.
     \param [in] avxxTxSymbol is the Tx Symbol.
     \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
     \param [in] llr_range_low is the lower bound of LLR output.
     \param [in] llr_range_high is the higher bound of LLR output.
     \param [in] nSc is store subcarrier num.
     \return null.
 */
 template<size_t nLayerPerUe = 1,typename T = Is16vec32>
 static void inline FORCE_INLINE llr_demap_layer_qam16(T*  ftempPostSINR,
    T*  ftempGain, Is16vec32*  avxxTxSymbol, int8_t * ppSoft, int16_t llr_range_low, int16_t llr_range_high, int32_t nSc) {
     __m512i xRe[nLayerPerUe], xIm[nLayerPerUe];

     cvt_pack_permute<nLayerPerUe, T>(ftempPostSINR, ftempGain, xRe, xIm, avxxTxSymbol);
     #pragma unroll(nLayerPerUe)
     #pragma ivdep
     for (size_t iLayer = 0; iLayer < nLayerPerUe; iLayer ++)
     {
         /*Algorithm
         LLR0 = 4/sqrt(10)*(2*real(In)+2*beta/sqrt(10)/(1-beta) when real(In)<-2/sqrt(10)*beta
         LLR0 = 4/sqrt(10)*real(In)/(1-beta) when -2/sqrt(10)*beta<real(In)<2/sqrt(10)*beta
         LLR0 = 4/sqrt(10)*(2*real(In)-2*beta/sqrt(10)/(1-beta) when real(In)>2/sqrt(10)*beta

         LLR1 = 4/sqrt(10)*(2*imag(In)+2*beta/sqrt(10)/(1-beta) when imag(In)<-2/sqrt(10)*beta
         LLR1 = 4/sqrt(10)*imag(In)/(1-beta) when -2/sqrt(10)*beta<imag(In)<2/sqrt(10)*beta
         LLR1 = 4/sqrt(10)*(2*imag(In)-2*beta/sqrt(10)/(1-beta) when imag(In)>2/sqrt(10)*beta

         LLR2 = 4/sqrt(10)*(real(In)+beta*2/sqrt(10)/(1-beta) when real(In)<0
         LLR2 = 4/sqrt(10)*(-real(In)+beta*2/sqrt(10)/(1-beta) when real(In)>0

         LLR3 = 4/sqrt(10)*(imag(In)+beta*2/sqrt(10)/(1-beta) when imag(In)<0
         LLR3 = 4/sqrt(10)*(-imag(In)+beta*2/sqrt(10)/(1-beta) when imag(In)>0*/

         /*first calculate the threshold 2*beta/sqrt(10)*/
         /*output: 16S15*16S13->16S13*/
         auto x0 = _mm512_mulhrs_epi16(xIm[iLayer],i_p_2_d_sqrt10);

         /*calculate the factor 4/sqrt(10)/(1-beta)*/
         /*output: 16S13*16Sx->16S(x-2)*/
         auto x1 = _mm512_mulhrs_epi16(xRe[iLayer],i_p_4_d_sqrt10);

         /*calculate the -+beta/sqrt(10) according to the sign of real(In) and imag(In)*/
         auto x2 = copy_inverted_sign_epi16(avxxTxSymbol[iLayer],_mm512_mulhrs_epi16(xIm[iLayer],i_p_1_d_sqrt10));

         /*calculate is it -8/sqrt(10)*(In+-beta/sqrt(10)) or 4/sqrt(10)*In, according to the abs of In*/
         /*output: 16S13*16S13->16S11*/
         auto x3 = _mm512_adds_epi16(avxxTxSymbol[iLayer],x2);
         x3 = _mm512_mulhrs_epi16(x3,i_p_8_d_sqrt10);
         x2 = select_high_low_epi16(_mm512_abs_epi16(avxxTxSymbol[iLayer]),x0,x3,_mm512_mulhrs_epi16(avxxTxSymbol[iLayer],i_p_4_d_sqrt10));

         /*calculate the final LLR0 and LLR1 by mulitply with factor 1/(1-beta)*/
         /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/
         x2 = _mm512_mulhrs_epi16(x2,xRe[iLayer]);
         x2 = limit_to_saturated_range(x2, llr_range_low, llr_range_high);

         /*calculate LLR2 and LLR3*/
         /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/
         auto x4 = _mm512_subs_epi16(x0,_mm512_abs_epi16(avxxTxSymbol[iLayer]));
         x4 = _mm512_mulhrs_epi16(x4,x1);
         x4 = limit_to_saturated_range(x4, llr_range_low, llr_range_high);

         /*pack to int8 and shuffle*/
         /*xRe[0]: LLR0_0,LLR1_0,LLR0_1,LLR1_1... LLR3_15*/
         xRe[iLayer] = _mm512_packs_epi16(x2,x4);
     }
     if(likely(nSc == 16)) {
         llr_store<BBLIB_QAM16, nLayerPerUe>(ppSoft, xRe, xIm);
     } else {
         llr_store<BBLIB_QAM16, nLayerPerUe>(ppSoft, xRe, xIm, nSc);
     }
 }

 /*!
 \brief do LLR demapping for 64qam.
     \param [in] ftempPostSINR is the post-SINR.
     \param [in] ftempGain is the Gain.
     \param [in] avxxTxSymbol is the Tx Symbol.
     \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
     \param [in] llr_range_low is the lower bound of LLR output.
     \param [in] llr_range_high is the higher bound of LLR output.
     \param [in] nSc is store subcarrier num.
     \return null.
 */
 template<size_t nLayerPerUe = 1,typename T = Is16vec32>
 static void inline FORCE_INLINE llr_demap_layer_qam64(T*  ftempPostSINR,
    T*  ftempGain, Is16vec32*  avxxTxSymbol, int8_t * ppSoft, int16_t llr_range_low, int16_t llr_range_high, int32_t nSc) {
     __m512i xRe[nLayerPerUe], xIm[nLayerPerUe];
     cvt_pack_permute<nLayerPerUe, T>(ftempPostSINR, ftempGain, xRe, xIm, avxxTxSymbol);
     #pragma unroll(nLayerPerUe)
     for (size_t iLayer = 0; iLayer < nLayerPerUe; iLayer ++)
     {
         /*Algorithm:
         LLR0 = InnerCompoundReal0/(1-beta)
         InnerCompoundReal0_0 = 16/sqrt(42)*(real(In)+InnerFactor0) when abs(real(In))>6*beta/sqrt(42)
         InnerCompoundReal0_1 = 12/sqrt(42)*(real(In)+InnerFactor1) when 4*beta/sqrt(42)<abs(real(In))<=6*beta/sqrt(42)
         InnerCompoundReal0_2 = 8/sqrt(42)*(real(In)r+InnerFactor2) when 2*beta/sqrt(42)<abs(real(In))<=4*beta/sqrt(42)
         InnerCompoundReal0_4 = 4/sqrt(42)*real(In) when abs(real(In))<=2*beta/sqrt(42)
         InnerFactor0 = 3*beta/(sqrt(42)); InnerFactor1 = 2*beta/(sqrt(42));InnerFactor2 = beta/(sqrt(42)),when real(In)<0
         InnerFactor0 = -3*beta/(sqrt(42)); InnerFactor1 = -2*beta/(sqrt(42));InnerFactor2 = -beta/(sqrt(42)),when real(In)>=0

         LLR1 = InnerCompoundImag0/(1-beta)

         LLR2 = InnerCompoundReal1/(1-beta)
         InnerCompoundReal1_0 = 8/sqrt(42)*(-abs(real(r))+5*beta/sqrt(42)) when abs(real(r))>6*beta/sqrt(42)
         InnerCompoundReal1_1 = 4/sqrt(42)*(-abs(real(r))+4*beta/sqrt(42)) when 2*beta/sqrt(42)<abs(real(r))<=6*beta/sqrt(42)
         InnerCompoundReal1_2 = 8/sqrt(42)*(-abs(real(r))+3*beta/sqrt(42)) when abs(real(r))<=2*beta/sqrt(42)

         LLR3 = InnerCompoundImag1/(1-beta)

         LLR4 = 4/sqrt(42)*InnerCompoundReal2/(1-beta)
         InnerCompoundReal2 = -1*abs(real(r))+6*beta/sqrt(42) when abs(real(r))>4*beta/sqrt(42)
         InnerCompoundReal2 = abs(real(r))-2*beta/sqrt(42) when abs(real(r))<=4*beta/sqrt(42)

         LLR5 = 4/sqrt(42)*InnerCompoundImag2/(1-beta)*/

         /*1.calculate the thresholds and scaling factors*/
         /*calculate the threshold 2*beta/sqrt(42)*/
         /*output 16S15*16S13->16S13*/
         auto x0 = _mm512_mulhrs_epi16(xIm[iLayer],i_p_2_d_sqrt42);

         /*calculate the threshold 4*beta/sqrt(42)*/
         auto x1 = _mm512_mulhrs_epi16(xIm[iLayer],i_p_4_d_sqrt42);

         /*calculate the threshold 6*beta/sqrt(42)*/
         auto x2 = _mm512_mulhrs_epi16(xIm[iLayer],i_p_6_d_sqrt42);

         /*calculate the common scaling factor 4/sqrt(42)/(1-beta)*/
         /*output 16S13*16Sx->16S(x-2)*/
         auto x3 = _mm512_mulhrs_epi16(xRe[iLayer],i_p_4_d_sqrt42);

         /*2.start to calculate LLR0*/
         /*abs: 16S13*/
         auto xtemp1 = _mm512_abs_epi16(avxxTxSymbol[iLayer]);

         /*calculate the the InnerCompoundReal0_0: 16/sqrt(42)*(abs(real(In))-3*beta/sqrt(42))*/
         /*output 16S13*16S13->16S11*/
         auto x4 = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[iLayer],i_p_3_d_sqrt42));
         x4 = _mm512_mulhrs_epi16(x4,i_p_16_d_sqrt42);

         /*calculate the the InnerCompoundReal0_1: 12/sqrt(42)*(abs(real(In))-2*beta/sqrt(42))*/
         auto x5 = _mm512_subs_epi16(xtemp1,x0);
         x5 = _mm512_mulhrs_epi16(x5,i_p_12_d_sqrt42);

         /*calculate the the InnerCompoundReal0_2: 8/sqrt(42)*(abs(real(In))-beta/sqrt(42))*/
         auto x6 = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[iLayer],i_p_1_d_sqrt42));
         x6 = _mm512_mulhrs_epi16(x6,i_p_8_d_sqrt42);

         /*if abs(real(In))>6*beta/sqrt(42), return InnerCompoundReal0_0, otherwise return InnerCompoundReal0_1*/
         auto x7 = select_high_low_epi16(xtemp1,x2,x4,x5);

         /*if abs(real(In))>2*beta/sqrt(42), return InnerCompoundReal0_2, otherwise return 4/sqrt(42)*abs(real(In))*/
         auto x8 = select_high_low_epi16(xtemp1,x0,x6,_mm512_mulhrs_epi16(xtemp1,i_p_4_d_sqrt42));

         /*if abs(real(In))>4*beta/sqrt(42), return x0, otherwise return x8*/
         x7 = select_high_low_epi16(xtemp1,x1,x7,x8);

         /*convert sign of InnferFactor according to the sign of real(In)*/
         x7 = copy_sign_epi16(avxxTxSymbol[iLayer],x7);

         /*multiply with the scaling factor 1/(1-beta) to get LLR0*/
         /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/
         x7 = _mm512_mulhrs_epi16(x7,xRe[iLayer]);
         x7 = limit_to_saturated_range(x7, llr_range_low, llr_range_high);

         /*3.start to calculate LLR2*/
         /*calculate the the InnerCompoundReal1_0: 8/sqrt(42)*(5*beta/sqrt(42)-abs(real(In)))*/
         /*output 16S13*16S13->16S11*/
         x4 = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[iLayer],i_p_5_d_sqrt42),xtemp1);
         x4 = _mm512_mulhrs_epi16(x4,i_p_8_d_sqrt42);

         /*calculate the the InnerCompoundReal1_1: 4/sqrt(42)*(4*beta/sqrt(42)-abs(real(In)))*/
         x5 = _mm512_subs_epi16(x1,xtemp1);
         x5 = _mm512_mulhrs_epi16(x5,i_p_4_d_sqrt42);

         /*calculate the the InnerCompoundReal1_2: 8/sqrt(42)*(3*beta/sqrt(42)-abs(real(In)))*/
         x6 = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[iLayer],i_p_3_d_sqrt42),xtemp1);
         x6 = _mm512_mulhrs_epi16(x6,i_p_8_d_sqrt42);

         /*if abs(real(In))>6*beta/sqrt(42), return InnerCompoundReal1_0, otherwise return InnerCompoundReal1_1*/
         auto x9 = select_high_low_epi16(xtemp1,x2,x4,x5);

         /*if abs(real(In))>2*beta/sqrt(42), return ftempBRe, otherwise return InnerCompoundReal1_2*/
         x9 = select_high_low_epi16(xtemp1,x0,x9,x6);

         /*multiply with the scaling factor 1/(1-beta) to get LLR2*/
         /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/
         x9 = _mm512_mulhrs_epi16(x9,xRe[iLayer]);
         x9 = limit_to_saturated_range(x9, llr_range_low, llr_range_high);

         /*4.start to calculate LLR4*/
         /*if abs(real(In))>4*beta/sqrt(42), return 6*beta/sqrt(42)-abs(real(In)), otherwise return abs(real(In))-2*beta/sqrt(42)*/
         /*output 16S13*/
         auto xtemp2 = select_high_low_epi16(xtemp1,x1,_mm512_subs_epi16(x2,xtemp1),_mm512_subs_epi16(xtemp1,x0));

         /*multiply with the scaling factor 1/(1-beta) to get LLR4*/
         /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/
         xtemp2 = _mm512_mulhrs_epi16(xtemp2,x3);
         xtemp2 = limit_to_saturated_range(xtemp2, llr_range_low, llr_range_high);

         /*8.saturated pack to int8*/
         if constexpr (1 == nLayerPerUe) {
             xRe[iLayer] = _mm512_packs_epi16(x7, x9);
             xIm[iLayer] = _mm512_packs_epi16(xtemp2, alliZero);
         } else {
             if (iLayer == 0) {
                 xRe[iLayer] = _mm512_packs_epi16(x7, x9);
                 xIm[iLayer] = _mm512_packs_epi16(xtemp2, alliZero);
             } else {
                 xRe[iLayer] =  _mm512_packs_epi16(alliZero, x7);
                 xIm[iLayer] = _mm512_packs_epi16(x9, xtemp2);
             }
         }
     }
     if(likely(nSc == 16)) {
         llr_store<BBLIB_QAM64, nLayerPerUe>(ppSoft, xRe, xIm);
     } else {
         llr_store<BBLIB_QAM64, nLayerPerUe>(ppSoft, xRe, xIm, nSc);
     }
 }

 /*!
 \brief do LLR demapping for 256qam.
     \param [in] ftempPostSINR is the post-SINR.
     \param [in] ftempGain is the Gain.
     \param [in] avxxTxSymbol is the Tx Symbol.
     \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
     \param [in] llr_range_low is the lower bound of LLR output.
     \param [in] llr_range_high is the higher bound of LLR output.
     \param [in] nSc is store subcarrier num.
     \return null.
 */
 template<size_t nLayerPerUe = 1,typename T = Is16vec32>
 static void inline FORCE_INLINE llr_demap_layer_qam256(T*  ftempPostSINR,
    T*  ftempGain, Is16vec32*  avxxTxSymbol, int8_t * ppSoft, int16_t llr_range_low, int16_t llr_range_high, int32_t nSc) {
     __m512i xRe[nLayerPerUe], xIm[nLayerPerUe];
     cvt_pack_permute<nLayerPerUe, T>(ftempPostSINR, ftempGain, xRe, xIm, avxxTxSymbol);
     #pragma unroll(nLayerPerUe)
     for (size_t iLayer = 0; iLayer < nLayerPerUe; iLayer ++)
     {
         /*Algorithm:
         LLR0 = InnerCompoundReal0/(1-beta)
         InnerCompoundReal0_0 = 32/sqrt(170)*(real(In)+InnerFactor0) when abs(real(In))>14*beta/sqrt(170)
         InnerCompoundReal0_1 = 28/sqrt(170)*(real(In)+InnerFactor1) when 14*beta/sqrt(170)<abs(real(In))<=12*beta/sqrt(170)
         InnerCompoundReal0_2 = 24/sqrt(170)*(real(In)r+InnerFactor2) when 12*beta/sqrt(170)<abs(real(In))<=10*beta/sqrt(170)
         InnerCompoundReal0_3 = 20/sqrt(170)*(real(In)r+InnerFactor3) when 10*beta/sqrt(170)<abs(real(In))<=8*beta/sqrt(170)
         InnerCompoundReal0_4 = 16/sqrt(170)*(real(In)r+InnerFactor4) when 8*beta/sqrt(170)<abs(real(In))<=6*beta/sqrt(170)
         InnerCompoundReal0_5 = 12/sqrt(170)*(real(In)r+InnerFactor5) when 6*beta/sqrt(170)<abs(real(In))<=4*beta/sqrt(170)
         InnerCompoundReal0_6 = 8/sqrt(170)*(real(In)r+InnerFactor6) when 4*beta/sqrt(170)<abs(real(In))<=2*beta/sqrt(170)
         InnerCompoundReal0_7 = 4/sqrt(170)*real(In) when abs(real(In))<=2*beta/sqrt(170)
         InnerFactor0 = 7*beta/(sqrt(170)); InnerFactor1 = 6*beta/(sqrt(170));InnerFactor2 = 5*beta/(sqrt(170)),when real(In)<0
         InnerFactor3 = 4*beta/(sqrt(170)); InnerFactor4 = 3*beta/(sqrt(170));InnerFactor5 = 2*beta/(sqrt(170)),when real(In)<0
         InnerFactor6 = beta/(sqrt(170)), when real(In)<0
         InnerFactor0 = -7*beta/(sqrt(170)); InnerFactor1 = -6*beta/(sqrt(170));InnerFactor2 = -5*beta/(sqrt(170)),when real(In)<0
         InnerFactor3 = -4*beta/(sqrt(170)); InnerFactor4 = -3*beta/(sqrt(170));InnerFactor5 = -2*beta/(sqrt(170)),when real(In)<0
         InnerFactor6 = -beta/(sqrt(170)), when real(In)<0

         LLR1 = InnerCompoundImag0/(1-beta)


         LLR2 = InnerCompoundReal1/(1-beta)
         InnerCompoundReal1_0 = 16/sqrt(170)*(-abs(real(r))+11*beta/sqrt(170)) when abs(real(r))>=14*beta/sqrt(170)
         InnerCompoundReal1_1 = 12/sqrt(170)*(-abs(real(r))+10*beta/sqrt(170)) when 12*beta/sqrt(170)<=abs(real(r))<=14*beta/sqrt(170)
         InnerCompoundReal1_2 = 8/sqrt(170)*(-abs(real(r))+9*beta/sqrt(170)) when 10*beta/sqrt(170)<=abs(real(r))<=12*beta/sqrt(170)
         InnerCompoundReal1_3 = 4/sqrt(170)*(-abs(real(r))+8*beta/sqrt(170)) when 8*beta/sqrt(170)<=abs(real(r))<=10*beta/sqrt(170)
         InnerCompoundReal1_4 = 8/sqrt(170)*(-abs(real(r))+7*beta/sqrt(170)) when 6*beta/sqrt(170)<=abs(real(r))<=8*beta/sqrt(170)
         InnerCompoundReal1_5 = 12/sqrt(170)*(-abs(real(r))+6*beta/sqrt(170)) when 4*beta/sqrt(170)<=abs(real(r))<=6*beta/sqrt(170)
         InnerCompoundReal1_6 = 16/sqrt(170)*(-abs(real(r))+5*beta/sqrt(170)) when 2*beta/sqrt(170)<=abs(real(r))

         LLR3 = InnerCompoundImag1/(1-beta)

         LLR4 = InnerCompoundReal2/(1-beta)
         InnerCompoundReal2_0 = 8/sqrt(170)*(-abs(real(r))+13*beta/sqrt(170) when abs(real(r))>=14*beta/sqrt(170)
         InnerCompoundReal2_1 = 4/sqrt(170)*(-abs(real(r))+12*beta/sqrt(170) when 10*beta/sqrt(170)<=abs(real(r))<=14*beta/sqrt(170)
         InnerCompoundReal2_2 = 8/sqrt(170)*(-abs(real(r))+11*beta/sqrt(170) when 8*beta/sqrt(170)<=abs(real(r))<=10*beta/sqrt(170)
         InnerCompoundReal2_3 = -8/sqrt(170)*(-abs(real(r))+5*beta/sqrt(170) when 6*beta/sqrt(170)<=abs(real(r))<=8*beta/sqrt(170)
         InnerCompoundReal2_4 = -4/sqrt(170)*(-abs(real(r))+4*beta/sqrt(170) when 2*beta/sqrt(170)<=abs(real(r))<=6*beta/sqrt(170)
         InnerCompoundReal2_5 = -8/sqrt(170)*(-abs(real(r))+3*beta/sqrt(170) when 2*beta/sqrt(170)>=abs(real(r))

         LLR5 = InnerCompoundImag2/(1-beta)

         LLR6 = 4/sqrt(170)*InnerCompoundReal3/(1-beta)
         InnerCompoundReal3_0 = -1*abs(real(r))+14*beta/sqrt(170) when abs(real(r))>=12*beta/sqrt(42)
         InnerCompoundReal3_1 = -1*abs(real(r))+10*beta/sqrt(170) when 8*beta/sqrt(170)<=abs(real(r))<12*beta/sqrt(42)
         InnerCompoundReal3_2 = -1*abs(real(r))+6*beta/sqrt(170) when 4*beta/sqrt(170)<=abs(real(r))<8*beta/sqrt(42)
         InnerCompoundReal3_3 = -1*abs(real(r))+2*beta/sqrt(170) when abs(real(r))<4*beta/sqrt(170)

         LLR7 = 4/sqrt(170)*InnerCompoundImag3/(1-beta)*/

         /*1.calculate the thresholds and scaling factors*/
         /*calculate the threshold 2*beta/sqrt(170)*/
         /*output 16S15*16S13->16S13*/
         auto x0 = _mm512_mulhrs_epi16(xIm[iLayer],i_p_2_d_sqrt170);

         /*calculate the threshold 4*beta/sqrt(170)*/
         auto x1 = _mm512_mulhrs_epi16(xIm[iLayer],i_p_4_d_sqrt170);

         /*calculate the threshold 6*beta/sqrt(170)*/
         auto x2 = _mm512_mulhrs_epi16(xIm[iLayer],i_p_6_d_sqrt170);

         /*calculate the threshold 8*beta/sqrt(170)*/
         auto x3 = _mm512_mulhrs_epi16(xIm[iLayer],i_p_8_d_sqrt170);

         /*calculate the threshold 10*beta/sqrt(170)*/
         auto x4 = _mm512_mulhrs_epi16(xIm[iLayer],i_p_10_d_sqrt170);

         /*calculate the threshold 12*beta/sqrt(170)*/
         auto x5 = _mm512_mulhrs_epi16(xIm[iLayer],i_p_12_d_sqrt170);

         /*calculate the threshold 14*beta/sqrt(170)*/
         auto x6 = _mm512_mulhrs_epi16(xIm[iLayer],i_p_14_d_sqrt170);

         /*calculate the common scaling factor 4/sqrt(170)/(1-beta)*/
         /*output 16S13*16Sx->16S(x-2)*/
         auto x7 = _mm512_mulhrs_epi16(xRe[iLayer],i_p_4_d_sqrt170);

         /*2.start to calculate LLR0*/
         /*abs: 16S13*/
         auto xtemp1 = _mm512_abs_epi16(avxxTxSymbol[iLayer]);

         /*InnerCompoundReal0_0 = 32/sqrt(170)*(abs(real(In))-7*beta/sqrt(170))*/
         /*output 16S13*16S13->16S11*/
         auto x8 = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[iLayer],i_p_7_d_sqrt170));
         x8 = _mm512_mulhrs_epi16(x8,i_p_32_d_sqrt170);

         /*calculate the the InnerCompoundReal0_1: 28/sqrt(170)*(abs(real(In))- 6*beta/sqrt(170))*/
         auto x9 = _mm512_subs_epi16(xtemp1,x2);
         x9 = _mm512_mulhrs_epi16(x9,i_p_28_d_sqrt170);

         /*calculate the the InnerCompoundReal0_2: 24/sqrt(170)*(abs(real(In))-5*beta/sqrt(170))*/
         auto x10 = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[iLayer],i_p_5_d_sqrt170));
         x10 = _mm512_mulhrs_epi16(x10,i_p_24_d_sqrt170);

         /*calculate the the InnerCompoundReal0_3: 20/sqrt(170)*(abs(real(In))-4*beta/sqrt(170))*/
         auto x11 = _mm512_subs_epi16(xtemp1,x1);
         x11 = _mm512_mulhrs_epi16(x11,i_p_20_d_sqrt170);

         /*calculate the the InnerCompoundReal0_4: 16/sqrt(170)*(abs(real(In))-3*beta/sqrt(170))*/
         auto x12 = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[iLayer],i_p_3_d_sqrt170));
         x12 = _mm512_mulhrs_epi16(x12,i_p_16_d_sqrt170);

         /*calculate the the InnerCompoundReal0_5: 12/sqrt(170)*(abs(real(In))-2*beta/sqrt(170))*/
         auto x13 = _mm512_subs_epi16(xtemp1,x0);
         x13 = _mm512_mulhrs_epi16(x13,i_p_12_d_sqrt170);

         /*calculate the the InnerCompoundReal0_6: 8/sqrt(170)*(abs(real(In))-1*beta/sqrt(170))*/
         auto x15 = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[iLayer],i_p_1_d_sqrt170));
         x15 = _mm512_mulhrs_epi16(x15,i_p_8_d_sqrt170);

         /*if abs(real(In))>14*beta/sqrt(170), return InnerCompoundReal0_0, otherwise return InnerCompoundReal0_1*/
         auto avxtemp1 = select_high_low_epi16(xtemp1,x6,x8,x9);

         /*if abs(real(In))>10*beta/sqrt(170), return InnerCompoundReal0_2, otherwise return InnerCompoundReal_0_1*/
         auto x16 = select_high_low_epi16(xtemp1,x4,x10,x11);

         /*if abs(real(In))>12*beta/sqrt(170), return x0, otherwise return InnerCompoundReal_0_1*/
         avxtemp1 = select_high_low_epi16(xtemp1,x5,avxtemp1,x16);

         /*if abs(real(In))>6*beta/sqrt(170), return InnerCompoundReal0_4, otherwise return InnerCompoundReal_0_5*/
         auto x17 = select_high_low_epi16(xtemp1,x2,x12,x13);

         /*if abs(real(In))>8*beta/sqrt(170), return x0, otherwise return x2*/
         avxtemp1 = select_high_low_epi16(xtemp1,x3,avxtemp1,x17);

         /*if abs(real(In))>2*beta/sqrt(170), return InnerCompoundReal0_6, otherwise return 4/sqrt(170)*abs(real(In))*/
         x16 = select_high_low_epi16(xtemp1,x0,x15,_mm512_mulhrs_epi16(xtemp1,i_p_4_d_sqrt170));

         /*if abs(real(In))>4*beta/sqrt(170), return x0, otherwise return x2*/
         avxtemp1 = select_high_low_epi16(xtemp1,x1,avxtemp1,x16);

         /*convert sign of InnferFactor according to the sign of real(In)*/
         avxtemp1 = copy_sign_epi16(avxxTxSymbol[iLayer],avxtemp1);

         /*multiply with the scaling factor 1/(1-beta) to get LLR0*/
         /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/
         avxtemp1 = _mm512_mulhrs_epi16(avxtemp1,xRe[iLayer]);
         avxtemp1 = limit_to_saturated_range(avxtemp1, llr_range_low, llr_range_high);

         /*3.start to calculate LLR2*/
         /*calculate the the InnerCompoundReal1_0: 16/sqrt(170)*(11*beta/sqrt(170)-abs(real(In)))*/
         /*output 16S13*16S13->16S11*/
         x8 = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[iLayer],i_p_11_d_sqrt170),xtemp1);
         x8 = _mm512_mulhrs_epi16(x8,i_p_16_d_sqrt170);

         /*calculate the the InnerCompoundReal1_1: 12/sqrt(170)*(10*beta/sqrt(170)-abs(real(In)))*/
         x9 = _mm512_subs_epi16(x4,xtemp1);
         x9 = _mm512_mulhrs_epi16(x9,i_p_12_d_sqrt170);

         /*calculate the the InnerCompoundReal1_2: 8/sqrt(170)*(9*beta/sqrt(170)-abs(real(In)))*/
         x10 = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[iLayer],i_p_9_d_sqrt170),xtemp1);
         x10 = _mm512_mulhrs_epi16(x10,i_p_8_d_sqrt170);

         /*calculate the the InnerCompoundReal1_3: 4/sqrt(170)*(8*beta/sqrt(170)-abs(real(In)))*/
         x11 = _mm512_subs_epi16(x3,xtemp1);
         x11 = _mm512_mulhrs_epi16(x11,i_p_4_d_sqrt170);

         /*calculate the the InnerCompoundReal1_4: 8/sqrt(170)*(7*beta/sqrt(170)-abs(real(In)))*/
         x12 = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[iLayer],i_p_7_d_sqrt170),xtemp1);
         x12 = _mm512_mulhrs_epi16(x12,i_p_8_d_sqrt170);

         /*calculate the the InnerCompoundReal1_5: 12/sqrt(170)*(6*beta/sqrt(170)-abs(real(In)))*/
         x13 = _mm512_subs_epi16(x2,xtemp1);
         x13 = _mm512_mulhrs_epi16(x13,i_p_12_d_sqrt170);

         /*calculate the the InnerCompoundReal1_5: 16/sqrt(170)*(5*beta/sqrt(170)-abs(real(In)))*/
         x15 = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[iLayer],i_p_5_d_sqrt170),xtemp1);
         x15 = _mm512_mulhrs_epi16(x15,i_p_16_d_sqrt170);

         /*if abs(real(In))>14*beta/sqrt(170), return InnerCompoundReal1_0, otherwise return InnerCompoundReal1_1*/
         auto avxtemp2 = select_high_low_epi16(xtemp1,x6,x8,x9);

         /*if abs(real(In))>10*beta/sqrt(170), return InnerCompoundReal1_2, otherwise return InnerCompoundReal_1_1*/
         x16 = select_high_low_epi16(xtemp1,x4,x10,x11);

         /*if abs(real(In))>12*beta/sqrt(170), return x0, otherwise return InnerCompoundReal_1_1*/
         avxtemp2 = select_high_low_epi16(xtemp1,x5,avxtemp2,x16);

         /*if abs(real(In))>4*beta/sqrt(170), return InnerCompoundReal1_4, otherwise return InnerCompoundReal_1_5*/
         x17 = select_high_low_epi16(xtemp1,x1,x12,x13);

         /*if abs(real(In))>6*beta/sqrt(170), return InnerCompoundReal1_4, otherwise return InnerCompoundReal_1_5*/
         avxtemp2 = select_high_low_epi16(xtemp1,x2,avxtemp2,x17);

         /*if abs(real(In))>2*beta/sqrt(170), return InnerCompoundReal1_5, otherwise return InnerCompoundReal1_6*/
         avxtemp2 = select_high_low_epi16(xtemp1,x0,avxtemp2,x15);

         /*multiply with the scaling factor 1/(1-beta) to get LLR2*/
         /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/
         avxtemp2 = _mm512_mulhrs_epi16(avxtemp2,xRe[iLayer]);
         avxtemp2 = limit_to_saturated_range(avxtemp2, llr_range_low, llr_range_high);

         /*4.start to calculate LLR4*/
         /*calculate the the InnerCompoundReal2_0: 8/sqrt(170)*(13*beta/sqrt(170)-abs(real(In)))*/
         /*output 16S13*/
         x8 = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[iLayer],i_p_13_d_sqrt170),xtemp1);
         x8 = _mm512_mulhrs_epi16(x8,i_p_8_d_sqrt170);

         /*calculate the the InnerCompoundReal2_1: 4/sqrt(170)*(12*beta/sqrt(170)-abs(real(In)))*/
         x9 = _mm512_subs_epi16(x5,xtemp1);
         x9 = _mm512_mulhrs_epi16(x9,i_p_4_d_sqrt170);

         /*calculate the the InnerCompoundReal2_2: 8/sqrt(170)*(11*beta/sqrt(170)-abs(real(In)))*/
         x10 = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[iLayer],i_p_11_d_sqrt170),xtemp1);
         x10 = _mm512_mulhrs_epi16(x10,i_p_8_d_sqrt170);

         /*calculate the the InnerCompoundReal2_3: 8/sqrt(170)*(abs(real(In)-5*beta/sqrt(170))*/
         x11 = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[iLayer],i_p_5_d_sqrt170));
         x11 = _mm512_mulhrs_epi16(x11,i_p_8_d_sqrt170);

         /*calculate the the InnerCompoundReal2_4: 4/sqrt(170)*(abs(real(In)-4*beta/sqrt(170))*/
         x12 = _mm512_subs_epi16(xtemp1,x1);
         x12 = _mm512_mulhrs_epi16(x12,i_p_4_d_sqrt170);

         /*calculate the the InnerCompoundReal2_5: 8/sqrt(170)*(abs(real(In)-3*beta/sqrt(170))*/
         x13 = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[iLayer],i_p_3_d_sqrt170));
         x13 = _mm512_mulhrs_epi16(x13,i_p_8_d_sqrt170);

         /*if abs(real(In))>14*beta/sqrt(170), return InnerCompoundReal2_0, otherwise return InnerCompoundReal2_1*/
         auto avxtemp3 = select_high_low_epi16(xtemp1,x6,x8,x9);

         /*if abs(real(In))>8*beta/sqrt(170), return InnerCompoundReal2_2, otherwise return InnerCompoundReal_2_1*/
         x16 = select_high_low_epi16(xtemp1,x3,x10,x11);

         /*if abs(real(In))>10*beta/sqrt(170), return x0, otherwise return InnerCompoundReal_1_1*/
         avxtemp3 = select_high_low_epi16(xtemp1,x4,avxtemp3,x16);

         /*if abs(real(In))>2*beta/sqrt(170), return InnerCompoundReal2_4, otherwise return InnerCompoundReal_2_5*/
         x17 = select_high_low_epi16(xtemp1,x0,x12,x13);

         /*if abs(real(In))>6*beta/sqrt(170), return x0, otherwise return InnerCompoundReal_1_1*/
         avxtemp3 = select_high_low_epi16(xtemp1,x2,avxtemp3,x17);

         /*multiply with the scaling factor 1/(1-beta) to get LLR4*/
         /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/
         avxtemp3 = _mm512_mulhrs_epi16(avxtemp3,xRe[iLayer]);
         avxtemp3 = limit_to_saturated_range(avxtemp3, llr_range_low, llr_range_high);

         /*4.start to calculate LLR6*/
         /*calculate the the InnerCompoundReal3_0: 4/sqrt(170)*(14*beta/sqrt(170)-abs(real(In)))*/
         /*output 16S13*/
         x8 = _mm512_subs_epi16(x6,xtemp1);

         /*calculate the the InnerCompoundReal3_1: 4/sqrt(170)*(abs(real(In)-10*beta/sqrt(170))*/
         x9 = _mm512_subs_epi16(xtemp1,x4);

         /*calculate the the InnerCompoundReal3_2: 4/sqrt(170)*(6*beta/sqrt(170)-abs(real(In)))*/
         x10 = _mm512_subs_epi16(x2,xtemp1);

         /*calculate the the InnerCompoundReal3_3: 4/sqrt(170)*(abs(real(In)-2*beta/sqrt(170))*/
         x11 = _mm512_subs_epi16(xtemp1,x0);

         /*if abs(real(In))>12*beta/sqrt(170), return InnerCompoundReal3_0, otherwise return InnerCompoundReal2_1*/
         auto avxtemp4 = select_high_low_epi16(xtemp1,x5,x8,x9);

         /*if abs(real(In))>4*beta/sqrt(170), return InnerCompoundReal3_2, otherwise return InnerCompoundReal_3_4*/
         x16 = select_high_low_epi16(xtemp1,x1,x10,x11);

         /*if abs(real(In))>8*beta/sqrt(170), return x6, otherwise return x16*/
         avxtemp4 = select_high_low_epi16(xtemp1,x3,avxtemp4,x16);

         /*multiply with the scaling factor 1/(1-beta) to get LLR6*/
         /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/
         avxtemp4 = _mm512_mulhrs_epi16(avxtemp4,x7);
         avxtemp4 = limit_to_saturated_range(avxtemp4, llr_range_low, llr_range_high);

         /*8.saturated pack to int8*/
         xRe[iLayer] = _mm512_packs_epi16(avxtemp1,
                                     avxtemp2);
         xIm[iLayer] = _mm512_packs_epi16(avxtemp3,
                                     avxtemp4);
     }
     if(likely(nSc == 16)) {
         llr_store<BBLIB_QAM256, nLayerPerUe>(ppSoft, xRe, xIm);
     } else {
         llr_store<BBLIB_QAM256, nLayerPerUe>(ppSoft, xRe, xIm, nSc);
     }
 }

// demapper
template<typename T, DATA_DMRS_MUX_E data_dmrs_mux_flag, uint8_t DMRSTYPE,uint8_t NROFCDMS>
struct DEMAPPER {
    // type define
    using FloatSimd = typename DataType<T>::FloatSimd;
    using Float = typename DataType<T>::Float;
    using procDataType = typename DataType<T>::procDataType;
    const static auto fp16Int16 = DataType<T>::fp16Int16;

    // method
    template<int32_t nLayerPerUe = 1>
    FORCE_INLINE inline
    static void demaper_llr(enum bblib_modulation_order modOrder, FloatSimd *pPostSINR, FloatSimd *pGain, Is16vec32 *pTx, int8_t *pSoft, const int16_t llr_range_low, const int16_t llr_range_high, int32_t nSc) {
        if (modOrder == BBLIB_HALF_PI_BPSK){
            llr_demap_layer_pi2bpsk<nLayerPerUe, FloatSimd>(pPostSINR, pGain, pTx, pSoft, llr_range_low, llr_range_high, nSc);
        }else if (modOrder == BBLIB_QPSK) {
            llr_demap_layer_qpsk<nLayerPerUe, FloatSimd>(pPostSINR, pGain, pTx, pSoft, llr_range_low, llr_range_high, nSc);
        } else if (modOrder == BBLIB_QAM16) {
            llr_demap_layer_qam16<nLayerPerUe, FloatSimd>(pPostSINR, pGain, pTx, pSoft, llr_range_low, llr_range_high, nSc);
        } else if (modOrder == BBLIB_QAM64) {
            llr_demap_layer_qam64<nLayerPerUe, FloatSimd>(pPostSINR, pGain, pTx, pSoft, llr_range_low, llr_range_high, nSc);
        } else if (modOrder == BBLIB_QAM256) {
            llr_demap_layer_qam256<nLayerPerUe, FloatSimd>(pPostSINR, pGain, pTx, pSoft, llr_range_low, llr_range_high, nSc);
        }
    }

    FORCE_INLINE inline
    static void permutexvar_DmrsData(size_t nLayerIdx,uint16_t portId, uint8_t nSc48Idx,FloatSimd *pPostSINR, FloatSimd *pGain,Is16vec32 *pTx) {
        if constexpr (1 == DMRSTYPE) {
            pPostSINR[nLayerIdx] = _mm512_permutexvar_ps(data_dmrs_type1[portId], pPostSINR[nLayerIdx]);
            pGain[nLayerIdx]     = _mm512_permutexvar_ps(data_dmrs_type1[portId], pGain[nLayerIdx]);
            pTx[nLayerIdx]       = _mm512_permutexvar_epi32(data_dmrs_type1[portId], pTx[nLayerIdx]);
        } else {//type2
            if constexpr (1 == NROFCDMS) {
                pPostSINR[nLayerIdx] = _mm512_permutexvar_ps(data_dmrs_type2_cdms1_idx[nSc48Idx], pPostSINR[nLayerIdx]);
                pGain[nLayerIdx]     = _mm512_permutexvar_ps(data_dmrs_type2_cdms1_idx[nSc48Idx], pGain[nLayerIdx]);
                pTx[nLayerIdx]       = _mm512_permutexvar_epi32(data_dmrs_type2_cdms1_idx[nSc48Idx], pTx[nLayerIdx]);
            }
            else{
                pPostSINR[nLayerIdx] = _mm512_permutexvar_ps(data_dmrs_type2_cdms2_idx[nSc48Idx], pPostSINR[nLayerIdx]);
                pGain[nLayerIdx]     = _mm512_permutexvar_ps(data_dmrs_type2_cdms2_idx[nSc48Idx], pGain[nLayerIdx]);
                pTx[nLayerIdx]       = _mm512_permutexvar_epi32(data_dmrs_type2_cdms2_idx[nSc48Idx], pTx[nLayerIdx]);
            }
        }
    }

    //template<DATA_DMRS_MUX_E data_dmrs_mux>
    FORCE_INLINE inline
    static void demaper(int8_t nMu, uint16_t *pLayerNum, uint16_t userLayer, int16_t nStartSC, enum bblib_modulation_order eModOrder[BBLIB_MAX_MU],uint16_t nTotalSubCarrier,
        int8_t *pLlr[BBLIB_MAX_MU], FloatSimd *ftempPostSINR, FloatSimd  *ftempGain, Is16vec32 *avxxTxSymbol, int32_t nSc, int32_t nSCIdx, const int16_t llr_range_low, const int16_t llr_range_high,
        int32_t *llrOffset, uint8_t nSc48Idx = 0, uint16_t nDmrsPortIdx = 0, int8_t *llrTmp = NULL) {

        for(int64_t imu = nMu - 1; imu >= 0; imu--) {
            auto nLayerPerUe = pLayerNum[imu];
            userLayer -= nLayerPerUe;
            int32_t llrStartOffset = nStartSC * nLayerPerUe * eModOrder[imu];
            const int16_t cTotalSubCarrier = nTotalSubCarrier;
            auto pSoftBase = pLlr[imu];
            auto pSoft = pSoftBase;
            auto p = pSoftBase;
            auto modOrder = eModOrder[imu];
            auto pPostSINR = &ftempPostSINR[userLayer];
            auto pGain = &ftempGain[userLayer];
            auto pTx = &avxxTxSymbol[userLayer];

            if constexpr (data_dmrs_mux_flag == DATA_DMRS_MUX_E::enable) {
                auto portId = (nDmrsPortIdx == 0) + (nDmrsPortIdx == 1);
                for (size_t iLayer = 0; iLayer < nLayerPerUe; iLayer++) {
                    permutexvar_DmrsData(iLayer,portId,nSc48Idx,pPostSINR,pGain,pTx);
                }
                pSoft = llrTmp;
            } else {
                pSoft = pSoftBase + llrStartOffset + llrOffset[imu] + nSCIdx * nLayerPerUe * modOrder;
                llrOffset[imu] += cTotalSubCarrier * nLayerPerUe * modOrder;
            }

            if (nLayerPerUe == 1) {
                demaper_llr<1>(modOrder, pPostSINR, pGain, pTx, pSoft, llr_range_low, llr_range_high, nSc);
            } else if (nLayerPerUe == 2) {
                demaper_llr<2>(modOrder, pPostSINR, pGain, pTx, pSoft, llr_range_low, llr_range_high, nSc);
            }

            if constexpr (data_dmrs_mux_flag == DATA_DMRS_MUX_E::enable)
            {
                int8_t nDataNum = 0;
                uint8_t nDatanumReGrp1 = 0;
                uint8_t nDatanumReGrp2 = 0;
                // interleaving type
                if(unlikely(nSc == 12))
                {
                    nDatanumReGrp1 = 8;
                    nDatanumReGrp2 = 4;
                }
                else if(unlikely(nSc == 8))
                {
                    nDatanumReGrp1 = 6;
                    nDatanumReGrp2 = 4;
                }
                else if(unlikely(nSc == 4))
                {
                    nDatanumReGrp1 = 4;
                    nDatanumReGrp2 = 2;
                }
                else //nSc == 16
                {
                    nDatanumReGrp1 = DatanumType2[0][nSc48Idx];
                    nDatanumReGrp2 = DatanumType2[1][nSc48Idx];
                }

                if constexpr (DMRSTYPE == 1)
                    nDataNum = nSc / 2;
                else {
                    if constexpr (NROFCDMS == 1)
                        nDataNum = nDatanumReGrp1;
                    else if constexpr (NROFCDMS == 2)
                        nDataNum = nDatanumReGrp2;
                }

                if constexpr (DMRSTYPE == 1)
                {
                    p = pSoftBase + llrOffset[imu] + ((llrStartOffset + nSCIdx * nLayerPerUe * modOrder) >> 1);
                    llrOffset[imu] += (cTotalSubCarrier * nLayerPerUe * modOrder) >> 1;
                }
                else
                {
                    int32_t nSCoffset = (nStartSC + nSCIdx) / 6;
                    int32_t reSCoffset = (nStartSC + nSCIdx) - nSCoffset * 6;
                    int32_t nSCoffset1 = 0;
                    if constexpr (NROFCDMS == 2)
                    {
                        nSCoffset1 = cTotalSubCarrier / 3;
                        nSCoffset = nStartSC + nSCIdx - nSCoffset * 4 - reSCoffset;
                    }
                    else
                    {
                        nSCoffset1 = cTotalSubCarrier * 2 / 3;
                        if (reSCoffset > 0)
                            nSCoffset = nStartSC + nSCIdx - nSCoffset * 2 - 2;
                        else
                            nSCoffset = nStartSC + nSCIdx - nSCoffset * 2;
                    }
                    p = pSoftBase + llrOffset[imu] + nSCoffset * nLayerPerUe * modOrder;
                    llrOffset[imu] += nSCoffset1 * nLayerPerUe * modOrder;
                }

                auto pllrTmp = llrTmp;
                auto llrNum = nDataNum * nLayerPerUe * modOrder; // data num Sc num for type 1
                auto intllrNum = llrNum & 0xffc0;//llrNum /64
                auto rellrNum = llrNum & 0x3f;//in 64
                auto llrMask = (((uint64_t) 1) << rellrNum) - 1;
                for (size_t illr = 0; illr < intllrNum; illr += 64) {
                    _mm512_storeu_si512(p + illr, _mm512_loadu_si512(pllrTmp));
                    pllrTmp += 64;
                }
                _mm512_mask_storeu_epi8 (p + intllrNum, llrMask, _mm512_loadu_si512(pllrTmp));
            }
        }
    }
};

//do LLR demapping for single layer
/*!
    \struct LLR
    \brief Struct of LLR demapping functions
*/
template<size_t nLayerPerUe = 2, typename T = Is16vec32>
struct LLR {
    /*!
    \brief do LLR demapping for qpsk.
    \param [in] ftempPostSINR is the post-SINR.
    \param [in] ftempGain is the Gain.
    \param [in] avxxTxSymbol is the Tx Symbol.\param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
    \param [in] llr_range_low is the lower bound of LLR output.
    \param [in] llr_range_high is the higher bound of LLR output.
    \param [in] ptrs_mask is PTRS mask.

    \return null.
    */
    static void inline FORCE_INLINE llr_demap_layer_qpsk_ptrs(F32vec16 *RESTRICT ftempPostSINR,
        F32vec16 *RESTRICT ftempGain, __m512i *RESTRICT avxxTxSymbol, int8_t *RESTRICT ppSoft, int16_t llr_range_low, int16_t llr_range_high, uint16_t ptrs_mask) {
        __m512i xRe[nLayerPerUe], xIm[nLayerPerUe];
        cvt_pack_permute_QPSK<nLayerPerUe>(ftempPostSINR, ftempGain, xRe, xIm, avxxTxSymbol);
#pragma unroll(nLayerPerUe)
        for (size_t iLayer = 0; iLayer < nLayerPerUe; iLayer ++)
        {
            /*Algorithm:*/
            /*LLR0 = real(In)*2*sqrt(2)/(1-beta)*/
            /*LLR1 = imag(In)*2*sqrt(2)/(1-beta)*/

            /*output 16S13*16S13->16S11*/
            auto x = _mm512_mulhrs_epi16(avxxTxSymbol[iLayer], i_p_2_m_sqrt2);
            /*output 16S11*16Sx->16S(nLlrFxpPoints)*/
            x = _mm512_mulhrs_epi16(x,xRe[iLayer]);
            xRe[iLayer] = limit_to_saturated_range(x, llr_range_low, llr_range_high);
        }
        llr_store<BBLIB_QPSK, nLayerPerUe>(ppSoft, xRe, xIm);
        //llr_store_ptrs<BBLIB_QPSK, nLayerPerUe>(ppSoft, xRe, xIm, ptrs_mask);

    } /*!< LLR demapping for QPSK */
// #ifndef _BBLIB_SPR_
    /*!
    \brief do LLR demapping for qpsk.
        \param [in] ftempPostSINR is the post-SINR.
        \param [in] ftempGain is the Gain.
        \param [in] avxxTxSymbol is the Tx Symbol.
        \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
        \param [in] llr_range_low is the lower bound of LLR output.
        \param [in] llr_range_high is the higher bound of LLR output.
        \param [in] nSc is store subcarrier num.
        \return null.
    */
    static void inline FORCE_INLINE llr_demap_layer_qpsk(T * ftempPostSINR,
        T * ftempGain, Is16vec32 * avxxTxSymbol, int8_t * ppSoft, int16_t llr_range_low, int16_t llr_range_high, int32_t nSc) {
        __m512i xRe[nLayerPerUe], xIm[nLayerPerUe];
        cvt_pack_permute_QPSK<nLayerPerUe, T>(ftempPostSINR, ftempGain, xRe, xIm, avxxTxSymbol);
        #pragma unroll(nLayerPerUe)
        for (size_t iLayer = 0; iLayer < nLayerPerUe; iLayer ++)
        {
            /*Algorithm:*/
            /*LLR0 = real(In)*2*sqrt(2)/(1-beta)*/
            /*LLR1 = imag(In)*2*sqrt(2)/(1-beta)*/

            /*output 16S13*16S13->16S11*/
            auto x = _mm512_mulhrs_epi16(avxxTxSymbol[iLayer], i_p_2_m_sqrt2);
            /*output 16S11*16Sx->16S(nLlrFxpPoints)*/
            x = _mm512_mulhrs_epi16(x,xRe[iLayer]);
            xRe[iLayer] = limit_to_saturated_range(x, llr_range_low, llr_range_high);
        }
        if(likely(nSc == 16)) {
            llr_store<BBLIB_QPSK, nLayerPerUe>(ppSoft, xRe, xIm);
        } else {
            llr_store<BBLIB_QPSK, nLayerPerUe>(ppSoft, xRe, xIm, nSc);
        }
    }

    /*!
    \brief do LLR demapping for 16qam.
        \param [in] ftempPostSINR is the post-SINR.
        \param [in] ftempGain is the Gain.
        \param [in] avxxTxSymbol is the Tx Symbol.
        \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
        \param [in] llr_range_low is the lower bound of LLR output.
        \param [in] llr_range_high is the higher bound of LLR output.
        \param [in] nSc is store subcarrier num.
        \return null.
    */
    static void inline FORCE_INLINE llr_demap_layer_qam16(T*  ftempPostSINR,
        T*  ftempGain, Is16vec32*  avxxTxSymbol, int8_t * ppSoft, int16_t llr_range_low, int16_t llr_range_high, int32_t nSc) {
        __m512i xRe[nLayerPerUe], xIm[nLayerPerUe];

        cvt_pack_permute<nLayerPerUe, T>(ftempPostSINR, ftempGain, xRe, xIm, avxxTxSymbol);
        #pragma unroll(nLayerPerUe)
        #pragma ivdep
        for (size_t iLayer = 0; iLayer < nLayerPerUe; iLayer ++)
        {
            /*Algorithm
            LLR0 = 4/sqrt(10)*(2*real(In)+2*beta/sqrt(10)/(1-beta) when real(In)<-2/sqrt(10)*beta
            LLR0 = 4/sqrt(10)*real(In)/(1-beta) when -2/sqrt(10)*beta<real(In)<2/sqrt(10)*beta
            LLR0 = 4/sqrt(10)*(2*real(In)-2*beta/sqrt(10)/(1-beta) when real(In)>2/sqrt(10)*beta

            LLR1 = 4/sqrt(10)*(2*imag(In)+2*beta/sqrt(10)/(1-beta) when imag(In)<-2/sqrt(10)*beta
            LLR1 = 4/sqrt(10)*imag(In)/(1-beta) when -2/sqrt(10)*beta<imag(In)<2/sqrt(10)*beta
            LLR1 = 4/sqrt(10)*(2*imag(In)-2*beta/sqrt(10)/(1-beta) when imag(In)>2/sqrt(10)*beta

            LLR2 = 4/sqrt(10)*(real(In)+beta*2/sqrt(10)/(1-beta) when real(In)<0
            LLR2 = 4/sqrt(10)*(-real(In)+beta*2/sqrt(10)/(1-beta) when real(In)>0

            LLR3 = 4/sqrt(10)*(imag(In)+beta*2/sqrt(10)/(1-beta) when imag(In)<0
            LLR3 = 4/sqrt(10)*(-imag(In)+beta*2/sqrt(10)/(1-beta) when imag(In)>0*/

            /*first calculate the threshold 2*beta/sqrt(10)*/
            /*output: 16S15*16S13->16S13*/
            auto x0 = _mm512_mulhrs_epi16(xIm[iLayer],i_p_2_d_sqrt10);

            /*calculate the factor 4/sqrt(10)/(1-beta)*/
            /*output: 16S13*16Sx->16S(x-2)*/
            auto x1 = _mm512_mulhrs_epi16(xRe[iLayer],i_p_4_d_sqrt10);

            /*calculate the -+beta/sqrt(10) according to the sign of real(In) and imag(In)*/
            auto x2 = copy_inverted_sign_epi16(avxxTxSymbol[iLayer],_mm512_mulhrs_epi16(xIm[iLayer],i_p_1_d_sqrt10));

            /*calculate is it -8/sqrt(10)*(In+-beta/sqrt(10)) or 4/sqrt(10)*In, according to the abs of In*/
            /*output: 16S13*16S13->16S11*/
            auto x3 = _mm512_adds_epi16(avxxTxSymbol[iLayer],x2);
            x3 = _mm512_mulhrs_epi16(x3,i_p_8_d_sqrt10);
            x2 = select_high_low_epi16(_mm512_abs_epi16(avxxTxSymbol[iLayer]),x0,x3,_mm512_mulhrs_epi16(avxxTxSymbol[iLayer],i_p_4_d_sqrt10));

            /*calculate the final LLR0 and LLR1 by mulitply with factor 1/(1-beta)*/
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/
            x2 = _mm512_mulhrs_epi16(x2,xRe[iLayer]);
            x2 = limit_to_saturated_range(x2, llr_range_low, llr_range_high);

            /*calculate LLR2 and LLR3*/
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/
            auto x4 = _mm512_subs_epi16(x0,_mm512_abs_epi16(avxxTxSymbol[iLayer]));
            x4 = _mm512_mulhrs_epi16(x4,x1);
            x4 = limit_to_saturated_range(x4, llr_range_low, llr_range_high);

            /*pack to int8 and shuffle*/
            /*xRe[0]: LLR0_0,LLR1_0,LLR0_1,LLR1_1... LLR3_15*/
            xRe[iLayer] = _mm512_packs_epi16(x2,x4);
        }
        if(likely(nSc == 16)) {
            llr_store<BBLIB_QAM16, nLayerPerUe>(ppSoft, xRe, xIm);
        } else {
            llr_store<BBLIB_QAM16, nLayerPerUe>(ppSoft, xRe, xIm, nSc);
        }
    }

    /*!
    \brief do LLR demapping for 64qam.
        \param [in] ftempPostSINR is the post-SINR.
        \param [in] ftempGain is the Gain.
        \param [in] avxxTxSymbol is the Tx Symbol.
        \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
        \param [in] llr_range_low is the lower bound of LLR output.
        \param [in] llr_range_high is the higher bound of LLR output.
        \param [in] nSc is store subcarrier num.
        \return null.
    */
    static void inline FORCE_INLINE llr_demap_layer_qam64(T*  ftempPostSINR,
        T*  ftempGain, Is16vec32*  avxxTxSymbol, int8_t * ppSoft, int16_t llr_range_low, int16_t llr_range_high, int32_t nSc) {
        __m512i xRe[nLayerPerUe], xIm[nLayerPerUe];
        cvt_pack_permute<nLayerPerUe, T>(ftempPostSINR, ftempGain, xRe, xIm, avxxTxSymbol);
        #pragma unroll(nLayerPerUe)
        for (size_t iLayer = 0; iLayer < nLayerPerUe; iLayer ++)
        {
            /*Algorithm:
            LLR0 = InnerCompoundReal0/(1-beta)
            InnerCompoundReal0_0 = 16/sqrt(42)*(real(In)+InnerFactor0) when abs(real(In))>6*beta/sqrt(42)
            InnerCompoundReal0_1 = 12/sqrt(42)*(real(In)+InnerFactor1) when 4*beta/sqrt(42)<abs(real(In))<=6*beta/sqrt(42)
            InnerCompoundReal0_2 = 8/sqrt(42)*(real(In)r+InnerFactor2) when 2*beta/sqrt(42)<abs(real(In))<=4*beta/sqrt(42)
            InnerCompoundReal0_4 = 4/sqrt(42)*real(In) when abs(real(In))<=2*beta/sqrt(42)
            InnerFactor0 = 3*beta/(sqrt(42)); InnerFactor1 = 2*beta/(sqrt(42));InnerFactor2 = beta/(sqrt(42)),when real(In)<0
            InnerFactor0 = -3*beta/(sqrt(42)); InnerFactor1 = -2*beta/(sqrt(42));InnerFactor2 = -beta/(sqrt(42)),when real(In)>=0

            LLR1 = InnerCompoundImag0/(1-beta)

            LLR2 = InnerCompoundReal1/(1-beta)
            InnerCompoundReal1_0 = 8/sqrt(42)*(-abs(real(r))+5*beta/sqrt(42)) when abs(real(r))>6*beta/sqrt(42)
            InnerCompoundReal1_1 = 4/sqrt(42)*(-abs(real(r))+4*beta/sqrt(42)) when 2*beta/sqrt(42)<abs(real(r))<=6*beta/sqrt(42)
            InnerCompoundReal1_2 = 8/sqrt(42)*(-abs(real(r))+3*beta/sqrt(42)) when abs(real(r))<=2*beta/sqrt(42)

            LLR3 = InnerCompoundImag1/(1-beta)

            LLR4 = 4/sqrt(42)*InnerCompoundReal2/(1-beta)
            InnerCompoundReal2 = -1*abs(real(r))+6*beta/sqrt(42) when abs(real(r))>4*beta/sqrt(42)
            InnerCompoundReal2 = abs(real(r))-2*beta/sqrt(42) when abs(real(r))<=4*beta/sqrt(42)

            LLR5 = 4/sqrt(42)*InnerCompoundImag2/(1-beta)*/

            /*1.calculate the thresholds and scaling factors*/
            /*calculate the threshold 2*beta/sqrt(42)*/
            /*output 16S15*16S13->16S13*/
            auto x0 = _mm512_mulhrs_epi16(xIm[iLayer],i_p_2_d_sqrt42);

            /*calculate the threshold 4*beta/sqrt(42)*/
            auto x1 = _mm512_mulhrs_epi16(xIm[iLayer],i_p_4_d_sqrt42);

            /*calculate the threshold 6*beta/sqrt(42)*/
            auto x2 = _mm512_mulhrs_epi16(xIm[iLayer],i_p_6_d_sqrt42);

            /*calculate the common scaling factor 4/sqrt(42)/(1-beta)*/
            /*output 16S13*16Sx->16S(x-2)*/
            auto x3 = _mm512_mulhrs_epi16(xRe[iLayer],i_p_4_d_sqrt42);

            /*2.start to calculate LLR0*/
            /*abs: 16S13*/
            auto xtemp1 = _mm512_abs_epi16(avxxTxSymbol[iLayer]);

            /*calculate the the InnerCompoundReal0_0: 16/sqrt(42)*(abs(real(In))-3*beta/sqrt(42))*/
            /*output 16S13*16S13->16S11*/
            auto x4 = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[iLayer],i_p_3_d_sqrt42));
            x4 = _mm512_mulhrs_epi16(x4,i_p_16_d_sqrt42);

            /*calculate the the InnerCompoundReal0_1: 12/sqrt(42)*(abs(real(In))-2*beta/sqrt(42))*/
            auto x5 = _mm512_subs_epi16(xtemp1,x0);
            x5 = _mm512_mulhrs_epi16(x5,i_p_12_d_sqrt42);

            /*calculate the the InnerCompoundReal0_2: 8/sqrt(42)*(abs(real(In))-beta/sqrt(42))*/
            auto x6 = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[iLayer],i_p_1_d_sqrt42));
            x6 = _mm512_mulhrs_epi16(x6,i_p_8_d_sqrt42);

            /*if abs(real(In))>6*beta/sqrt(42), return InnerCompoundReal0_0, otherwise return InnerCompoundReal0_1*/
            auto x7 = select_high_low_epi16(xtemp1,x2,x4,x5);

            /*if abs(real(In))>2*beta/sqrt(42), return InnerCompoundReal0_2, otherwise return 4/sqrt(42)*abs(real(In))*/
            auto x8 = select_high_low_epi16(xtemp1,x0,x6,_mm512_mulhrs_epi16(xtemp1,i_p_4_d_sqrt42));

            /*if abs(real(In))>4*beta/sqrt(42), return x0, otherwise return x8*/
            x7 = select_high_low_epi16(xtemp1,x1,x7,x8);

            /*convert sign of InnferFactor according to the sign of real(In)*/
            x7 = copy_sign_epi16(avxxTxSymbol[iLayer],x7);

            /*multiply with the scaling factor 1/(1-beta) to get LLR0*/
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/
            x7 = _mm512_mulhrs_epi16(x7,xRe[iLayer]);
            x7 = limit_to_saturated_range(x7, llr_range_low, llr_range_high);

            /*3.start to calculate LLR2*/
            /*calculate the the InnerCompoundReal1_0: 8/sqrt(42)*(5*beta/sqrt(42)-abs(real(In)))*/
            /*output 16S13*16S13->16S11*/
            x4 = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[iLayer],i_p_5_d_sqrt42),xtemp1);
            x4 = _mm512_mulhrs_epi16(x4,i_p_8_d_sqrt42);

            /*calculate the the InnerCompoundReal1_1: 4/sqrt(42)*(4*beta/sqrt(42)-abs(real(In)))*/
            x5 = _mm512_subs_epi16(x1,xtemp1);
            x5 = _mm512_mulhrs_epi16(x5,i_p_4_d_sqrt42);

            /*calculate the the InnerCompoundReal1_2: 8/sqrt(42)*(3*beta/sqrt(42)-abs(real(In)))*/
            x6 = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[iLayer],i_p_3_d_sqrt42),xtemp1);
            x6 = _mm512_mulhrs_epi16(x6,i_p_8_d_sqrt42);

            /*if abs(real(In))>6*beta/sqrt(42), return InnerCompoundReal1_0, otherwise return InnerCompoundReal1_1*/
            auto x9 = select_high_low_epi16(xtemp1,x2,x4,x5);

            /*if abs(real(In))>2*beta/sqrt(42), return ftempBRe, otherwise return InnerCompoundReal1_2*/
            x9 = select_high_low_epi16(xtemp1,x0,x9,x6);

            /*multiply with the scaling factor 1/(1-beta) to get LLR2*/
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/
            x9 = _mm512_mulhrs_epi16(x9,xRe[iLayer]);
            x9 = limit_to_saturated_range(x9, llr_range_low, llr_range_high);

            /*4.start to calculate LLR4*/
            /*if abs(real(In))>4*beta/sqrt(42), return 6*beta/sqrt(42)-abs(real(In)), otherwise return abs(real(In))-2*beta/sqrt(42)*/
            /*output 16S13*/
            auto xtemp2 = select_high_low_epi16(xtemp1,x1,_mm512_subs_epi16(x2,xtemp1),_mm512_subs_epi16(xtemp1,x0));

            /*multiply with the scaling factor 1/(1-beta) to get LLR4*/
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/
            xtemp2 = _mm512_mulhrs_epi16(xtemp2,x3);
            xtemp2 = limit_to_saturated_range(xtemp2, llr_range_low, llr_range_high);

            /*8.saturated pack to int8*/
            if constexpr (1 == nLayerPerUe) {
                xRe[iLayer] = _mm512_packs_epi16(x7, x9);
                xIm[iLayer] = _mm512_packs_epi16(xtemp2, alliZero);
            } else {
                if (iLayer == 0) {
                    xRe[iLayer] = _mm512_packs_epi16(x7, x9);
                    xIm[iLayer] = _mm512_packs_epi16(xtemp2, alliZero);
                } else {
                    xRe[iLayer] =  _mm512_packs_epi16(alliZero, x7);
                    xIm[iLayer] = _mm512_packs_epi16(x9, xtemp2);
                }
            }
        }
        if(likely(nSc == 16)) {
            llr_store<BBLIB_QAM64, nLayerPerUe>(ppSoft, xRe, xIm);
        } else {
            llr_store<BBLIB_QAM64, nLayerPerUe>(ppSoft, xRe, xIm, nSc);
        }
    }

    /*!
    \brief do LLR demapping for 256qam.
        \param [in] ftempPostSINR is the post-SINR.
        \param [in] ftempGain is the Gain.
        \param [in] avxxTxSymbol is the Tx Symbol.
        \param [out] ppSoft is the pointer to Output Buffer of LLRs, buffer should be 64 byte aligned, output format 8S(nLlrFxpPoints).
        \param [in] llr_range_low is the lower bound of LLR output.
        \param [in] llr_range_high is the higher bound of LLR output.
        \param [in] nSc is store subcarrier num.
        \return null.
    */
    static void inline FORCE_INLINE llr_demap_layer_qam256(T*  ftempPostSINR,
        T*  ftempGain, Is16vec32*  avxxTxSymbol, int8_t * ppSoft, int16_t llr_range_low, int16_t llr_range_high, int32_t nSc) {
        __m512i xRe[nLayerPerUe], xIm[nLayerPerUe];
        cvt_pack_permute<nLayerPerUe, T>(ftempPostSINR, ftempGain, xRe, xIm, avxxTxSymbol);
        #pragma unroll(nLayerPerUe)
        for (size_t iLayer = 0; iLayer < nLayerPerUe; iLayer ++)
        {
            /*Algorithm:
            LLR0 = InnerCompoundReal0/(1-beta)
            InnerCompoundReal0_0 = 32/sqrt(170)*(real(In)+InnerFactor0) when abs(real(In))>14*beta/sqrt(170)
            InnerCompoundReal0_1 = 28/sqrt(170)*(real(In)+InnerFactor1) when 14*beta/sqrt(170)<abs(real(In))<=12*beta/sqrt(170)
            InnerCompoundReal0_2 = 24/sqrt(170)*(real(In)r+InnerFactor2) when 12*beta/sqrt(170)<abs(real(In))<=10*beta/sqrt(170)
            InnerCompoundReal0_3 = 20/sqrt(170)*(real(In)r+InnerFactor3) when 10*beta/sqrt(170)<abs(real(In))<=8*beta/sqrt(170)
            InnerCompoundReal0_4 = 16/sqrt(170)*(real(In)r+InnerFactor4) when 8*beta/sqrt(170)<abs(real(In))<=6*beta/sqrt(170)
            InnerCompoundReal0_5 = 12/sqrt(170)*(real(In)r+InnerFactor5) when 6*beta/sqrt(170)<abs(real(In))<=4*beta/sqrt(170)
            InnerCompoundReal0_6 = 8/sqrt(170)*(real(In)r+InnerFactor6) when 4*beta/sqrt(170)<abs(real(In))<=2*beta/sqrt(170)
            InnerCompoundReal0_7 = 4/sqrt(170)*real(In) when abs(real(In))<=2*beta/sqrt(170)
            InnerFactor0 = 7*beta/(sqrt(170)); InnerFactor1 = 6*beta/(sqrt(170));InnerFactor2 = 5*beta/(sqrt(170)),when real(In)<0
            InnerFactor3 = 4*beta/(sqrt(170)); InnerFactor4 = 3*beta/(sqrt(170));InnerFactor5 = 2*beta/(sqrt(170)),when real(In)<0
            InnerFactor6 = beta/(sqrt(170)), when real(In)<0
            InnerFactor0 = -7*beta/(sqrt(170)); InnerFactor1 = -6*beta/(sqrt(170));InnerFactor2 = -5*beta/(sqrt(170)),when real(In)<0
            InnerFactor3 = -4*beta/(sqrt(170)); InnerFactor4 = -3*beta/(sqrt(170));InnerFactor5 = -2*beta/(sqrt(170)),when real(In)<0
            InnerFactor6 = -beta/(sqrt(170)), when real(In)<0

            LLR1 = InnerCompoundImag0/(1-beta)


            LLR2 = InnerCompoundReal1/(1-beta)
            InnerCompoundReal1_0 = 16/sqrt(170)*(-abs(real(r))+11*beta/sqrt(170)) when abs(real(r))>=14*beta/sqrt(170)
            InnerCompoundReal1_1 = 12/sqrt(170)*(-abs(real(r))+10*beta/sqrt(170)) when 12*beta/sqrt(170)<=abs(real(r))<=14*beta/sqrt(170)
            InnerCompoundReal1_2 = 8/sqrt(170)*(-abs(real(r))+9*beta/sqrt(170)) when 10*beta/sqrt(170)<=abs(real(r))<=12*beta/sqrt(170)
            InnerCompoundReal1_3 = 4/sqrt(170)*(-abs(real(r))+8*beta/sqrt(170)) when 8*beta/sqrt(170)<=abs(real(r))<=10*beta/sqrt(170)
            InnerCompoundReal1_4 = 8/sqrt(170)*(-abs(real(r))+7*beta/sqrt(170)) when 6*beta/sqrt(170)<=abs(real(r))<=8*beta/sqrt(170)
            InnerCompoundReal1_5 = 12/sqrt(170)*(-abs(real(r))+6*beta/sqrt(170)) when 4*beta/sqrt(170)<=abs(real(r))<=6*beta/sqrt(170)
            InnerCompoundReal1_6 = 16/sqrt(170)*(-abs(real(r))+5*beta/sqrt(170)) when 2*beta/sqrt(170)<=abs(real(r))

            LLR3 = InnerCompoundImag1/(1-beta)

            LLR4 = InnerCompoundReal2/(1-beta)
            InnerCompoundReal2_0 = 8/sqrt(170)*(-abs(real(r))+13*beta/sqrt(170) when abs(real(r))>=14*beta/sqrt(170)
            InnerCompoundReal2_1 = 4/sqrt(170)*(-abs(real(r))+12*beta/sqrt(170) when 10*beta/sqrt(170)<=abs(real(r))<=14*beta/sqrt(170)
            InnerCompoundReal2_2 = 8/sqrt(170)*(-abs(real(r))+11*beta/sqrt(170) when 8*beta/sqrt(170)<=abs(real(r))<=10*beta/sqrt(170)
            InnerCompoundReal2_3 = -8/sqrt(170)*(-abs(real(r))+5*beta/sqrt(170) when 6*beta/sqrt(170)<=abs(real(r))<=8*beta/sqrt(170)
            InnerCompoundReal2_4 = -4/sqrt(170)*(-abs(real(r))+4*beta/sqrt(170) when 2*beta/sqrt(170)<=abs(real(r))<=6*beta/sqrt(170)
            InnerCompoundReal2_5 = -8/sqrt(170)*(-abs(real(r))+3*beta/sqrt(170) when 2*beta/sqrt(170)>=abs(real(r))

            LLR5 = InnerCompoundImag2/(1-beta)

            LLR6 = 4/sqrt(170)*InnerCompoundReal3/(1-beta)
            InnerCompoundReal3_0 = -1*abs(real(r))+14*beta/sqrt(170) when abs(real(r))>=12*beta/sqrt(42)
            InnerCompoundReal3_1 = -1*abs(real(r))+10*beta/sqrt(170) when 8*beta/sqrt(170)<=abs(real(r))<12*beta/sqrt(42)
            InnerCompoundReal3_2 = -1*abs(real(r))+6*beta/sqrt(170) when 4*beta/sqrt(170)<=abs(real(r))<8*beta/sqrt(42)
            InnerCompoundReal3_3 = -1*abs(real(r))+2*beta/sqrt(170) when abs(real(r))<4*beta/sqrt(170)

            LLR7 = 4/sqrt(170)*InnerCompoundImag3/(1-beta)*/

            /*1.calculate the thresholds and scaling factors*/
            /*calculate the threshold 2*beta/sqrt(170)*/
            /*output 16S15*16S13->16S13*/
            auto x0 = _mm512_mulhrs_epi16(xIm[iLayer],i_p_2_d_sqrt170);

            /*calculate the threshold 4*beta/sqrt(170)*/
            auto x1 = _mm512_mulhrs_epi16(xIm[iLayer],i_p_4_d_sqrt170);

            /*calculate the threshold 6*beta/sqrt(170)*/
            auto x2 = _mm512_mulhrs_epi16(xIm[iLayer],i_p_6_d_sqrt170);

            /*calculate the threshold 8*beta/sqrt(170)*/
            auto x3 = _mm512_mulhrs_epi16(xIm[iLayer],i_p_8_d_sqrt170);

            /*calculate the threshold 10*beta/sqrt(170)*/
            auto x4 = _mm512_mulhrs_epi16(xIm[iLayer],i_p_10_d_sqrt170);

            /*calculate the threshold 12*beta/sqrt(170)*/
            auto x5 = _mm512_mulhrs_epi16(xIm[iLayer],i_p_12_d_sqrt170);

            /*calculate the threshold 14*beta/sqrt(170)*/
            auto x6 = _mm512_mulhrs_epi16(xIm[iLayer],i_p_14_d_sqrt170);

            /*calculate the common scaling factor 4/sqrt(170)/(1-beta)*/
            /*output 16S13*16Sx->16S(x-2)*/
            auto x7 = _mm512_mulhrs_epi16(xRe[iLayer],i_p_4_d_sqrt170);

            /*2.start to calculate LLR0*/
            /*abs: 16S13*/
            auto xtemp1 = _mm512_abs_epi16(avxxTxSymbol[iLayer]);

            /*InnerCompoundReal0_0 = 32/sqrt(170)*(abs(real(In))-7*beta/sqrt(170))*/
            /*output 16S13*16S13->16S11*/
            auto x8 = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[iLayer],i_p_7_d_sqrt170));
            x8 = _mm512_mulhrs_epi16(x8,i_p_32_d_sqrt170);

            /*calculate the the InnerCompoundReal0_1: 28/sqrt(170)*(abs(real(In))- 6*beta/sqrt(170))*/
            auto x9 = _mm512_subs_epi16(xtemp1,x2);
            x9 = _mm512_mulhrs_epi16(x9,i_p_28_d_sqrt170);

            /*calculate the the InnerCompoundReal0_2: 24/sqrt(170)*(abs(real(In))-5*beta/sqrt(170))*/
            auto x10 = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[iLayer],i_p_5_d_sqrt170));
            x10 = _mm512_mulhrs_epi16(x10,i_p_24_d_sqrt170);

            /*calculate the the InnerCompoundReal0_3: 20/sqrt(170)*(abs(real(In))-4*beta/sqrt(170))*/
            auto x11 = _mm512_subs_epi16(xtemp1,x1);
            x11 = _mm512_mulhrs_epi16(x11,i_p_20_d_sqrt170);

            /*calculate the the InnerCompoundReal0_4: 16/sqrt(170)*(abs(real(In))-3*beta/sqrt(170))*/
            auto x12 = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[iLayer],i_p_3_d_sqrt170));
            x12 = _mm512_mulhrs_epi16(x12,i_p_16_d_sqrt170);

            /*calculate the the InnerCompoundReal0_5: 12/sqrt(170)*(abs(real(In))-2*beta/sqrt(170))*/
            auto x13 = _mm512_subs_epi16(xtemp1,x0);
            x13 = _mm512_mulhrs_epi16(x13,i_p_12_d_sqrt170);

            /*calculate the the InnerCompoundReal0_6: 8/sqrt(170)*(abs(real(In))-1*beta/sqrt(170))*/
            auto x15 = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[iLayer],i_p_1_d_sqrt170));
            x15 = _mm512_mulhrs_epi16(x15,i_p_8_d_sqrt170);

            /*if abs(real(In))>14*beta/sqrt(170), return InnerCompoundReal0_0, otherwise return InnerCompoundReal0_1*/
            auto avxtemp1 = select_high_low_epi16(xtemp1,x6,x8,x9);

            /*if abs(real(In))>10*beta/sqrt(170), return InnerCompoundReal0_2, otherwise return InnerCompoundReal_0_1*/
            auto x16 = select_high_low_epi16(xtemp1,x4,x10,x11);

            /*if abs(real(In))>12*beta/sqrt(170), return x0, otherwise return InnerCompoundReal_0_1*/
            avxtemp1 = select_high_low_epi16(xtemp1,x5,avxtemp1,x16);

            /*if abs(real(In))>6*beta/sqrt(170), return InnerCompoundReal0_4, otherwise return InnerCompoundReal_0_5*/
            auto x17 = select_high_low_epi16(xtemp1,x2,x12,x13);

            /*if abs(real(In))>8*beta/sqrt(170), return x0, otherwise return x2*/
            avxtemp1 = select_high_low_epi16(xtemp1,x3,avxtemp1,x17);

            /*if abs(real(In))>2*beta/sqrt(170), return InnerCompoundReal0_6, otherwise return 4/sqrt(170)*abs(real(In))*/
            x16 = select_high_low_epi16(xtemp1,x0,x15,_mm512_mulhrs_epi16(xtemp1,i_p_4_d_sqrt170));

            /*if abs(real(In))>4*beta/sqrt(170), return x0, otherwise return x2*/
            avxtemp1 = select_high_low_epi16(xtemp1,x1,avxtemp1,x16);

            /*convert sign of InnferFactor according to the sign of real(In)*/
            avxtemp1 = copy_sign_epi16(avxxTxSymbol[iLayer],avxtemp1);

            /*multiply with the scaling factor 1/(1-beta) to get LLR0*/
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/
            avxtemp1 = _mm512_mulhrs_epi16(avxtemp1,xRe[iLayer]);
            avxtemp1 = limit_to_saturated_range(avxtemp1, llr_range_low, llr_range_high);

            /*3.start to calculate LLR2*/
            /*calculate the the InnerCompoundReal1_0: 16/sqrt(170)*(11*beta/sqrt(170)-abs(real(In)))*/
            /*output 16S13*16S13->16S11*/
            x8 = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[iLayer],i_p_11_d_sqrt170),xtemp1);
            x8 = _mm512_mulhrs_epi16(x8,i_p_16_d_sqrt170);

            /*calculate the the InnerCompoundReal1_1: 12/sqrt(170)*(10*beta/sqrt(170)-abs(real(In)))*/
            x9 = _mm512_subs_epi16(x4,xtemp1);
            x9 = _mm512_mulhrs_epi16(x9,i_p_12_d_sqrt170);

            /*calculate the the InnerCompoundReal1_2: 8/sqrt(170)*(9*beta/sqrt(170)-abs(real(In)))*/
            x10 = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[iLayer],i_p_9_d_sqrt170),xtemp1);
            x10 = _mm512_mulhrs_epi16(x10,i_p_8_d_sqrt170);

            /*calculate the the InnerCompoundReal1_3: 4/sqrt(170)*(8*beta/sqrt(170)-abs(real(In)))*/
            x11 = _mm512_subs_epi16(x3,xtemp1);
            x11 = _mm512_mulhrs_epi16(x11,i_p_4_d_sqrt170);

            /*calculate the the InnerCompoundReal1_4: 8/sqrt(170)*(7*beta/sqrt(170)-abs(real(In)))*/
            x12 = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[iLayer],i_p_7_d_sqrt170),xtemp1);
            x12 = _mm512_mulhrs_epi16(x12,i_p_8_d_sqrt170);

            /*calculate the the InnerCompoundReal1_5: 12/sqrt(170)*(6*beta/sqrt(170)-abs(real(In)))*/
            x13 = _mm512_subs_epi16(x2,xtemp1);
            x13 = _mm512_mulhrs_epi16(x13,i_p_12_d_sqrt170);

            /*calculate the the InnerCompoundReal1_5: 16/sqrt(170)*(5*beta/sqrt(170)-abs(real(In)))*/
            x15 = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[iLayer],i_p_5_d_sqrt170),xtemp1);
            x15 = _mm512_mulhrs_epi16(x15,i_p_16_d_sqrt170);

            /*if abs(real(In))>14*beta/sqrt(170), return InnerCompoundReal1_0, otherwise return InnerCompoundReal1_1*/
            auto avxtemp2 = select_high_low_epi16(xtemp1,x6,x8,x9);

            /*if abs(real(In))>10*beta/sqrt(170), return InnerCompoundReal1_2, otherwise return InnerCompoundReal_1_1*/
            x16 = select_high_low_epi16(xtemp1,x4,x10,x11);

            /*if abs(real(In))>12*beta/sqrt(170), return x0, otherwise return InnerCompoundReal_1_1*/
            avxtemp2 = select_high_low_epi16(xtemp1,x5,avxtemp2,x16);

            /*if abs(real(In))>4*beta/sqrt(170), return InnerCompoundReal1_4, otherwise return InnerCompoundReal_1_5*/
            x17 = select_high_low_epi16(xtemp1,x1,x12,x13);

            /*if abs(real(In))>6*beta/sqrt(170), return InnerCompoundReal1_4, otherwise return InnerCompoundReal_1_5*/
            avxtemp2 = select_high_low_epi16(xtemp1,x2,avxtemp2,x17);

            /*if abs(real(In))>2*beta/sqrt(170), return InnerCompoundReal1_5, otherwise return InnerCompoundReal1_6*/
            avxtemp2 = select_high_low_epi16(xtemp1,x0,avxtemp2,x15);

            /*multiply with the scaling factor 1/(1-beta) to get LLR2*/
            /*output: 16S11*16Sx->16S(nLlrFxpPoints)*/
            avxtemp2 = _mm512_mulhrs_epi16(avxtemp2,xRe[iLayer]);
            avxtemp2 = limit_to_saturated_range(avxtemp2, llr_range_low, llr_range_high);

            /*4.start to calculate LLR4*/
            /*calculate the the InnerCompoundReal2_0: 8/sqrt(170)*(13*beta/sqrt(170)-abs(real(In)))*/
            /*output 16S13*/
            x8 = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[iLayer],i_p_13_d_sqrt170),xtemp1);
            x8 = _mm512_mulhrs_epi16(x8,i_p_8_d_sqrt170);

            /*calculate the the InnerCompoundReal2_1: 4/sqrt(170)*(12*beta/sqrt(170)-abs(real(In)))*/
            x9 = _mm512_subs_epi16(x5,xtemp1);
            x9 = _mm512_mulhrs_epi16(x9,i_p_4_d_sqrt170);

            /*calculate the the InnerCompoundReal2_2: 8/sqrt(170)*(11*beta/sqrt(170)-abs(real(In)))*/
            x10 = _mm512_subs_epi16(_mm512_mulhrs_epi16(xIm[iLayer],i_p_11_d_sqrt170),xtemp1);
            x10 = _mm512_mulhrs_epi16(x10,i_p_8_d_sqrt170);

            /*calculate the the InnerCompoundReal2_3: 8/sqrt(170)*(abs(real(In)-5*beta/sqrt(170))*/
            x11 = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[iLayer],i_p_5_d_sqrt170));
            x11 = _mm512_mulhrs_epi16(x11,i_p_8_d_sqrt170);

            /*calculate the the InnerCompoundReal2_4: 4/sqrt(170)*(abs(real(In)-4*beta/sqrt(170))*/
            x12 = _mm512_subs_epi16(xtemp1,x1);
            x12 = _mm512_mulhrs_epi16(x12,i_p_4_d_sqrt170);

            /*calculate the the InnerCompoundReal2_5: 8/sqrt(170)*(abs(real(In)-3*beta/sqrt(170))*/
            x13 = _mm512_subs_epi16(xtemp1,_mm512_mulhrs_epi16(xIm[iLayer],i_p_3_d_sqrt170));
            x13 = _mm512_mulhrs_epi16(x13,i_p_8_d_sqrt170);

            /*if abs(real(In))>14*beta/sqrt(170), return InnerCompoundReal2_0, otherwise return InnerCompoundReal2_1*/
            auto avxtemp3 = select_high_low_epi16(xtemp1,x6,x8,x9);

            /*if abs(real(In))>8*beta/sqrt(170), return InnerCompoundReal2_2, otherwise return InnerCompoundReal_2_1*/
            x16 = select_high_low_epi16(xtemp1,x3,x10,x11);

            /*if abs(real(In))>10*beta/sqrt(170), return x0, otherwise return InnerCompoundReal_1_1*/
            avxtemp3 = select_high_low_epi16(xtemp1,x4,avxtemp3,x16);

            /*if abs(real(In))>2*beta/sqrt(170), return InnerCompoundReal2_4, otherwise return InnerCompoundReal_2_5*/
            x17 = select_high_low_epi16(xtemp1,x0,x12,x13);

            /*if abs(real(In))>6*beta/sqrt(170), return x0, otherwise return InnerCompoundReal_1_1*/
            avxtemp3 = select_high_low_epi16(xtemp1,x2,avxtemp3,x17);

            /*multiply with the scaling factor 1/(1-beta) to get LLR4*/
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/
            avxtemp3 = _mm512_mulhrs_epi16(avxtemp3,xRe[iLayer]);
            avxtemp3 = limit_to_saturated_range(avxtemp3, llr_range_low, llr_range_high);

            /*4.start to calculate LLR6*/
            /*calculate the the InnerCompoundReal3_0: 4/sqrt(170)*(14*beta/sqrt(170)-abs(real(In)))*/
            /*output 16S13*/
            x8 = _mm512_subs_epi16(x6,xtemp1);

            /*calculate the the InnerCompoundReal3_1: 4/sqrt(170)*(abs(real(In)-10*beta/sqrt(170))*/
            x9 = _mm512_subs_epi16(xtemp1,x4);

            /*calculate the the InnerCompoundReal3_2: 4/sqrt(170)*(6*beta/sqrt(170)-abs(real(In)))*/
            x10 = _mm512_subs_epi16(x2,xtemp1);

            /*calculate the the InnerCompoundReal3_3: 4/sqrt(170)*(abs(real(In)-2*beta/sqrt(170))*/
            x11 = _mm512_subs_epi16(xtemp1,x0);

            /*if abs(real(In))>12*beta/sqrt(170), return InnerCompoundReal3_0, otherwise return InnerCompoundReal2_1*/
            auto avxtemp4 = select_high_low_epi16(xtemp1,x5,x8,x9);

            /*if abs(real(In))>4*beta/sqrt(170), return InnerCompoundReal3_2, otherwise return InnerCompoundReal_3_4*/
            x16 = select_high_low_epi16(xtemp1,x1,x10,x11);

            /*if abs(real(In))>8*beta/sqrt(170), return x6, otherwise return x16*/
            avxtemp4 = select_high_low_epi16(xtemp1,x3,avxtemp4,x16);

            /*multiply with the scaling factor 1/(1-beta) to get LLR6*/
            /*output: 16S13*16S(x-2)->16S(nLlrFxpPoints)*/
            avxtemp4 = _mm512_mulhrs_epi16(avxtemp4,x7);
            avxtemp4 = limit_to_saturated_range(avxtemp4, llr_range_low, llr_range_high);

            /*8.saturated pack to int8*/
            xRe[iLayer] = _mm512_packs_epi16(avxtemp1,
                                        avxtemp2);
            xIm[iLayer] = _mm512_packs_epi16(avxtemp3,
                                        avxtemp4);
        }
        if(likely(nSc == 16)) {
            llr_store<BBLIB_QAM256, nLayerPerUe>(ppSoft, xRe, xIm);
        } else {
            llr_store<BBLIB_QAM256, nLayerPerUe>(ppSoft, xRe, xIm, nSc);
        }
    }
// #endif
};
#endif
#endif


