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
/*!
    \file   phy_ldpc_encoder_5gnr_internal.h
    \brief  Source code of External API for 5GNR LDPC Encoder functions

    __Overview:__

    The bblib_ldpc_encoder_5gnr kernel is a 5G NR LDPC Encoder function.
    It is implemeneted as defined in TS38212 5.3.2

    __Requirements and Test Coverage:__

    BaseGraph 1(.2). Lifting Factor >176.

    __Algorithm Guidance:__

    The algorithm is implemented as defined in TS38212 5.3.2.
*/
#ifndef _PHY_LDPC_ENCODER_5GNR_INTERNAL_H_
#define _PHY_LDPC_ENCODER_5GNR_INTERNAL_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h> // AVX
#include <stdint.h>

#include "common_typedef_sdk.h"

#ifdef __cplusplus
extern "C" {
#endif


#define PROC_BYTES 64
#define WAYS_144to256 2
#define WAYS_72to128 4
#define WAYS_36to64 8
#define WAYS_18to32 16
#define WAYS_2to16 16

void adapter_from288to384(int8_t **pBuff0, int8_t *pBuff1, uint16_t zcSize, uint32_t cbLen, int8_t direct);//1ways
void adapter_2ways_from144to256(int8_t **pbuff0, int8_t *pbuff1, uint16_t zcSize, uint32_t cbLen, int8_t direct);//2ways
void adapter_4ways_from72to128(int8_t **pbuff0, int8_t *pbuff1, uint16_t zcSize, uint32_t cbLen, int8_t direct);//4ways
void adapter_8ways_from36to64(int8_t **pbuff0, int8_t *pbuff1, uint16_t zcSize, uint32_t cbLen, int8_t direct);//8ways
void adapter_16ways_from18to32(int8_t **pbuff0, int8_t *pbuff1, uint16_t zcSize, uint32_t cbLen, int8_t direct);//16ways
void adapter_16ways_from2to16(int8_t **pbuff0, int8_t *pbuff1, uint16_t zcSize, uint32_t cbLen, int8_t direct);//16ways

typedef void (* LDPC_ADAPTER_P)(int8_t **, int8_t *, uint16_t , uint32_t , int8_t);
LDPC_ADAPTER_P ldpc_select_adapter_func(uint16_t zcSize);


#define PROC_BYTES 64
#define I_LS_NUM 8
#define ZC_MAX 384

#define BG1_COL_TOTAL 68
#define BG1_ROW_TOTAL 46
#define BG1_COL_INF_NUM 22
#define BG1_NONZERO_NUM 307

#define BG2_COL_TOTAL 52
#define BG2_ROW_TOTAL 42
#define BG2_COL_INF_NUM 10
#define BG2_NONZERO_NUM 188

void ldpc_enc_initial();

void ldpc_encoder_bg1(int8_t *pDataIn, int8_t *pDataOut, const int16_t *pMatrixNumPerCol, 
    const int16_t *pAddr, const int16_t *pShiftMatrix, int16_t zcSize);

extern int16_t Bg1MatrixNumPerCol[BG1_COL_TOTAL];
extern int16_t Bg1Address[BG1_NONZERO_NUM];
extern int16_t Bg1HShiftMatrix[BG1_NONZERO_NUM*I_LS_NUM];
extern int16_t Bg2MatrixNumPerCol[BG2_COL_TOTAL];
extern int16_t Bg2Address[BG2_NONZERO_NUM];
extern int16_t Bg2HShiftMatrix[BG2_NONZERO_NUM*I_LS_NUM];


#ifdef __cplusplus
}
#endif

#endif
