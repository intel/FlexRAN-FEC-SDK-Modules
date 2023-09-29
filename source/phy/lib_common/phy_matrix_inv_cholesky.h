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

/*******************************************************************************
* @file phy_matrix_inversion.h
* @brief this file defines the sturcture and functions for matrix inversion.
*******************************************************************************/

/*******************************************************************************
* Include public/global header files
********************************************************************************/
#ifndef _PHY_MATRIX_INVERSION_H_
#define _PHY_MATRIX_INVERSION_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE 2
#include <pmmintrin.h> // SSE 3
#include <tmmintrin.h> // SSSE 3
#include <smmintrin.h> // SSE 4
#include <immintrin.h> // AVX

#ifdef __cplusplus
#include <iostream>
#pragma once
using namespace std;
#endif

/*******************************************************************************
* Local macro/data declaration and definitions
*******************************************************************************/
#define NUM_MEM 100
#define N_2 2
#define N_3 3
#define N_4 4
#define N_5 5
#define N_6 6
#define N_7 7
#define N_8 8
#define N_9 9
#define N_10 10
#define N_11 11
#define N_12 12
#define N_13 13
#define N_14 14
#define N_15 15
#define N_16 16

void matrix_inv_cholesky_2x2(__m512 matBRe[N_2][N_2], __m512 matBIm[N_2][N_2],
    __m512 matInvBRe[N_2][N_2], __m512 matInvBIm[N_2][N_2]);

void matrix_inv_cholesky_3x3(__m512 matBRe[N_3][N_3], __m512 matBIm[N_3][N_3],
    __m512 matInvBRe[N_3][N_3], __m512 matInvBIm[N_3][N_3]);

void matrix_inv_cholesky_4x4(__m512 matBRe[N_4][N_4], __m512 matBIm[N_4][N_4],
    __m512 matInvBRe[N_4][N_4], __m512 matInvBIm[N_4][N_4]);

void matrix_inv_cholesky_5x5(__m512 matBRe[N_5][N_5], __m512 matBIm[N_5][N_5],
    __m512 matInvBRe[N_5][N_5], __m512 matInvBIm[N_5][N_5]);

void matrix_inv_cholesky_6x6(__m512 matBRe[N_6][N_6], __m512 matBIm[N_6][N_6],
    __m512 matInvBRe[N_6][N_6], __m512 matInvBIm[N_6][N_6]);

void matrix_inv_cholesky_7x7(__m512 matBRe[N_7][N_7], __m512 matBIm[N_7][N_7],
    __m512 matInvBRe[N_7][N_7], __m512 matInvBIm[N_7][N_7]);

void matrix_inv_cholesky_8x8(__m512 matBRe[N_8][N_8], __m512 matBIm[N_8][N_8],
    __m512 matInvBRe[N_8][N_8], __m512 matInvBIm[N_8][N_8]);

void matrix_inv_cholesky_9x9(__m512 matBRe[N_9][N_9], __m512 matBIm[N_9][N_9],
    __m512 matInvBRe[N_9][N_9], __m512 matInvBIm[N_9][N_9]);

void matrix_inv_cholesky_10x10(__m512 matBRe[N_10][N_10], __m512 matBIm[N_10][N_10],
    __m512 matInvBRe[N_10][N_10], __m512 matInvBIm[N_10][N_10]);

void matrix_inv_cholesky_11x11(__m512 matBRe[N_11][N_11], __m512 matBIm[N_11][N_11],
    __m512 matInvBRe[N_11][N_11], __m512 matInvBIm[N_11][N_11]);

void matrix_inv_cholesky_12x12(__m512 matBRe[N_12][N_12], __m512 matBIm[N_12][N_12],
    __m512 matInvBRe[N_12][N_12], __m512 matInvBIm[N_12][N_12]);

void matrix_inv_cholesky_13x13(__m512 matBRe[N_13][N_13], __m512 matBIm[N_13][N_13],
    __m512 matInvBRe[N_13][N_13], __m512 matInvBIm[N_13][N_13]);

void matrix_inv_cholesky_14x14(__m512 matBRe[N_14][N_14], __m512 matBIm[N_14][N_14],
    __m512 matInvBRe[N_14][N_14], __m512 matInvBIm[N_14][N_14]);

void matrix_inv_cholesky_15x15(__m512 matBRe[N_15][N_15], __m512 matBIm[N_15][N_15],
    __m512 matInvBRe[N_15][N_15], __m512 matInvBIm[N_15][N_15]);

void matrix_inv_cholesky_16x16(__m512 matBRe[N_16][N_16], __m512 matBIm[N_16][N_16],
    __m512 matInvBRe[N_16][N_16], __m512 matInvBIm[N_16][N_16]);

#endif

