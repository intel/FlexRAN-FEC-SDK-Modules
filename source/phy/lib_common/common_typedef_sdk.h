/*******************************************************************************
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
*******************************************************************************/
/*! \file common_typedef_sdk.h
    \brief  This header file defines those data type both used by eNB and UE.
*/

#ifndef _COMMON_TYPEDEF_SDK_H_
#define _COMMON_TYPEDEF_SDK_H_

#include <stdint.h>

#include <stdio.h>
#include <stdbool.h>

/** complex type for int8_t */
typedef struct {
    int8_t re;                      /**< real part */
    int8_t im;                      /**< imaginary  part */
}complex_int8_t;

//! @{
/*!
    \struct complex_int16_t
    \brief Defines 32-bit complex structure; both real part and image part have 16 bit width.
*/
typedef struct {
    int16_t re; /*!< 16-bit real part */
    int16_t im; /*!< 16-bit image part */
}complex_int16_t;
//! @}

/*!
    \struct complex_int32_t
    \brief Defines 64-bit complex structure; both real part and image part have 32 bit width.
*/
typedef struct {
    int32_t re; /*!< 32-bit real part */
    int32_t im; /*!< 32-bit image part */
}complex_int32_t;

/*!
    \struct complex_float
    \brief Defines 64-bit complex structure; both real part and image part have 32 bit width.
*/
typedef struct {
    float re; /*!< 32-bit real part */
    float im; /*!< 32-bit image part */
}complex_float;

/*!
    \struct complex_double
    \brief Defines 128-bit complex structure; both real part and image part have 64 bit width.
*/
typedef struct {
    double re; /*!< 64-bit real part */
    double im; /*!< 64-bit image part */
}complex_double;

#ifdef _BBLIB_SPR_
/*!
    \struct complex_half
    \brief Defines 32-bit complex structure; both real part and image part have 16 bit width.
*/
typedef struct {
    _Float16 re; /*!< 16-bit real part */
    _Float16 im; /*!< 16-bit image part */
}complex_half;
#endif

/*!
    \enum instruction_cpu_support
    \brief Define instruction the CPU can support.
*/
typedef enum{
    CPU_GENERIC, /*!< C */
    SSE4_2,      /*!< SSE4_2 */
    AVX,         /*!< AVX */
    AVX2,        /*!< AVX2 */
    AVX_512,     /*!< AVX512 */
    SNC,         /*!< Sunny Cove Instructions (for ICX) */
}instruction_cpu_support;

/*!
    \enum bblib_modulation_order
    \brief Common enums for modulation order.
*/
enum bblib_modulation_order
{
    BBLIB_HALF_PI_BPSK   = 1, /*!< PI/2 BPSK */
    /*BBLIB_BPSK           = 1, *//*!< BPSK */
    BBLIB_QPSK           = 2, /*!< QPSK */
    BBLIB_PAM4           = 3, /*!< PAM4 */
    BBLIB_QAM16          = 4, /*!< QAM16 */
    BBLIB_PAM8           = 5, /*!< PAM8 */
    BBLIB_QAM64          = 6, /*!< QAM64 */
    BBLIB_PAM16          = 7, /*!< PAM16 */
    BBLIB_QAM256         = 8  /*!< QAM256 */
};

/*!
    \enum bblib_common_const_wireless_params
    \brief This enum contains the common wireless constants accross both LTE
    and 5G used throughout the bblib libraries.
*/
enum bblib_common_const_wireless_params {
    BBLIB_N_SC_PER_PRB = 12,    /*!< Number of subcarriers in a Physical Resource Block */
    BBLIB_N_SYMB_PER_SF = 14 /*!< Number of symbols in sub-frame */
};

/*!
    \enum mmse_mimo_constants
    \brief Constants used in MMSE MIMO
*/
enum mmse_mimo_constants {
        BBLIB_MAX_RX_ANT_NUM = 16, /*!< MAX number of Rx antennas */
        BBLIB_MAX_TX_LAYER_NUM = 16,   /*!< MAX number of Tx layers */
        BBLIB_RX_DATA_FIXED_POINT = 13,   /*!< Fixed point of Rx data */
        BBLIB_MMSE_X_LEFT_SHIFT = BBLIB_RX_DATA_FIXED_POINT, /*!< MMSE X left shift */
        BBLIB_MMSE_LEMMA_SCALING = ((BBLIB_RX_DATA_FIXED_POINT)*2), /*!< MMSE Lemma scaling */
        BBLIB_MAX_MU = 8          /*!< MAX number of multiple user pair */
};
/*!
    \enum byte_swap_flag
    \brief byte swap flag: indicate whether to do byte swap of output in BBLIB_FULL_FXP_16B_ACC_MODE or not
*/
typedef enum {
    BBLIB_NO_BYTE_SWAP, /*!< Don't do byte swap of precoding output. */
    BBLIB_BYTE_SWAP,    /*!< Enable byte swap of precoding output, specially for  BBLIB_FULL_FXP_16B_ACC_MODE*/
}byte_swap_flag;

/*!
    \enum bblib_srs_proc_gen_mode
    \brief Constants used in SRS processing
*/
enum bblib_srs_proc_gen_mode {
        BBLIB_4th_GEN = 0, /*!< SRS processing on 4th gen hardware */
        BBLIB_NEXT_GEN = 1 /*!< SRS processing on next gen hardware */
};

//! @{
/*!
    \struct bblib_pusch_xran_decomp
    \brief Defines bblib_pusch_xran_decomp for mmse and irc to call-back the xran decompress.
*/
typedef struct
{
    uint16_t nCellIdx; /*!< decompress subframe number */
    int32_t nSfIdx; /*!< decompress cell number */
    uint16_t nRBStart; /*!< decompress start RB */
    uint16_t nRBSize; /*!< decompress size RB */
    uint16_t antNum; /*!< decompress port number */
    uint16_t *pAnt;  /*!< decompress porit index */
    uint16_t symNum; /*!< decompress symbol number */
    uint16_t *pSymb; /*!< decompress symbol index */
    int16_t decompPRBStart; /*!< decompress PRB start for RB split */
    int16_t decompPRBNum; /*!< decompress PRB number at one time */
    int16_t nCvt; /*!< 0: no conversion; 1: data type conversion from int16_t to fp16 with fScale*/
    float fScale; /*!< scaling of data type conversion*/
    void *pDecompOut[BBLIB_MAX_RX_ANT_NUM][BBLIB_N_SYMB_PER_SF]; /*!< decompress output pointer */
    int32_t (*xran_decomp_func)(void* pDecomp); /*!< decompress xran callback */
} bblib_pusch_xran_decomp;
//! @}

#ifndef RESTRICT
#ifdef _WIN64
#define RESTRICT __restrict
#else
#if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC)
#define RESTRICT __restrict__
#else
#define RESTRICT __restrict
#endif
#endif
#endif

#ifndef __align
#ifdef _WIN64
#define __align(x) __declspec(align(x))
#else
#if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC)
#define __align(x) __attribute__((aligned(x)))
#else
#define __align(x) __declspec (align(x))
#endif
#endif
#endif

#ifndef _WIN64
#define _aligned_malloc(x,y) memalign(y,x)
#endif

#ifdef _BBLIB_SPR_
#define SPR_MODE2
#endif

/* Test time and loops for unit test */
#define TIME 40
#define LOOP 30

#endif /* #ifndef _COMMON_TYPEDEF_H_ */

