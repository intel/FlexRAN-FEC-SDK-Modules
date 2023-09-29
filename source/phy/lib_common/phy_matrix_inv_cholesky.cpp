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
* @file matrix_inv_cholesky.cpp
* @brief cholesky based matrix inversion.
*******************************************************************************/

/*******************************************************************************
* Include private header files
*******************************************************************************/
#include "common_typedef_sdk.h"
#include "phy_matrix_inv_cholesky.h"
#if defined(_BBLIB_AVX512_)

static __m512 constZero = _mm512_setzero_ps();

// complex multiplication: A * A'
#define GET_AxAH(re, im, out)\
{\
    out = _mm512_add_ps(_mm512_mul_ps(re, re), _mm512_mul_ps(im, im));\
}

// complex multiplication: A * B'
#define GET_AxBH(are, aim, bre, bim, outre, outim)\
{\
    outre = _mm512_add_ps(_mm512_mul_ps(are, bre), _mm512_mul_ps(aim, bim));\
    outim = _mm512_sub_ps(_mm512_mul_ps(bre, aim), _mm512_mul_ps(are, bim));\
}

// complex multiplication: A * B
#define GET_AxB(are, aim, bre, bim, outre, outim)\
{\
    outre = _mm512_sub_ps(_mm512_mul_ps(are, bre), _mm512_mul_ps(aim, bim));\
    outim = _mm512_add_ps(_mm512_mul_ps(bre, aim), _mm512_mul_ps(are, bim));\
}

// complex multiplication: A * real(B)
#define GET_AxRealB(are, aim, bre, outre, outim)\
{\
    outre = _mm512_mul_ps(are, bre);\
    outim = _mm512_mul_ps(aim, bre);\
}

// get G00
#define GET_G00(matGRe, matBRe, matD, matND)\
{\
    matD[0] = _mm512_rsqrt14_ps(matBRe[0][0]);\
    matND[0] = _mm512_sub_ps(constZero, matD[0]);\
}

// get column 0 of matrix G
#define GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, j)\
{\
    matGRe[j][0] = _mm512_mul_ps(matBRe[j][0], matD[0]);\
    matGIm[j][0] = _mm512_mul_ps(matBIm[j][0], matD[0]);\
}

// get G11
#define GET_G11(matGRe, matGIm, matBRe, matD, matND, temp0)\
{\
    GET_AxAH(matGRe[1][0], matGIm[1][0], temp0);\
    matGRe[1][1] = _mm512_sub_ps(matBRe[1][1], temp0);\
    matD[1] = _mm512_rsqrt14_ps(matGRe[1][1]);\
    matND[1] = _mm512_sub_ps(constZero, matD[1]);\
}

// get column 1 of matrix G
#define GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, j, temp0, temp1)\
{\
    GET_AxBH(matGRe[j][0], matGIm[j][0], matGRe[1][0], matGIm[1][0], temp0, temp1);\
    matGRe[j][1] = _mm512_sub_ps(matBRe[j][1], temp0);\
    matGIm[j][1] = _mm512_sub_ps(matBIm[j][1], temp1);\
    matGRe[j][1] = _mm512_mul_ps(matGRe[j][1], matD[1]);\
    matGIm[j][1] = _mm512_mul_ps(matGIm[j][1], matD[1]);\
}

// get G22
#define GET_G22(matGRe, matGIm, matBRe, matD, matND, temp0, temp1)\
{\
    GET_AxAH(matGRe[2][0], matGIm[2][0], temp0);\
    GET_AxAH(matGRe[2][1], matGIm[2][1], temp1);\
    matGRe[2][2] = _mm512_sub_ps(matBRe[2][2], temp0);\
    matGRe[2][2] = _mm512_sub_ps(matGRe[2][2], temp1);\
    matD[2] = _mm512_rsqrt14_ps(matGRe[2][2]);\
    matND[2] = _mm512_sub_ps(constZero, matD[2]);\
}

// get column 2 of matrix G
#define GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, j, temp0, temp1)\
{\
    GET_AxBH(matGRe[j][0], matGIm[j][0], matGRe[2][0], matGIm[2][0], temp0, temp1);\
    matGRe[j][2] = _mm512_sub_ps(matBRe[j][2], temp0);\
    matGIm[j][2] = _mm512_sub_ps(matBIm[j][2], temp1);\
    GET_AxBH(matGRe[j][1], matGIm[j][1], matGRe[2][1], matGIm[2][1], temp0, temp1);\
    matGRe[j][2] = _mm512_sub_ps(matGRe[j][2], temp0);\
    matGIm[j][2] = _mm512_sub_ps(matGIm[j][2], temp1);\
    matGRe[j][2] = _mm512_mul_ps(matGRe[j][2], matD[2]);\
    matGIm[j][2] = _mm512_mul_ps(matGIm[j][2], matD[2]);\
}

// get G33
#define GET_G33(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2)\
{\
    GET_AxAH(matGRe[3][0], matGIm[3][0], temp0);\
    GET_AxAH(matGRe[3][1], matGIm[3][1], temp1);\
    GET_AxAH(matGRe[3][2], matGIm[3][2], temp2);\
    matGRe[3][3] = _mm512_sub_ps(matBRe[3][3], temp0);\
    matGRe[3][3] = _mm512_sub_ps(matGRe[3][3], temp1);\
    matGRe[3][3] = _mm512_sub_ps(matGRe[3][3], temp2);\
    matD[3] = _mm512_rsqrt14_ps(matGRe[3][3]);\
    matND[3] = _mm512_sub_ps(constZero, matD[3]);\
}

// get column 3 of matrix G
#define GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, j, temp0, temp1)\
{\
    GET_AxBH(matGRe[j][0], matGIm[j][0], matGRe[3][0], matGIm[3][0], temp0, temp1);\
    matGRe[j][3] = _mm512_sub_ps(matBRe[j][3], temp0);\
    matGIm[j][3] = _mm512_sub_ps(matBIm[j][3], temp1);\
    GET_AxBH(matGRe[j][1], matGIm[j][1], matGRe[3][1], matGIm[3][1], temp0, temp1);\
    matGRe[j][3] = _mm512_sub_ps(matGRe[j][3], temp0);\
    matGIm[j][3] = _mm512_sub_ps(matGIm[j][3], temp1);\
    GET_AxBH(matGRe[j][2], matGIm[j][2], matGRe[3][2], matGIm[3][2], temp0, temp1);\
    matGRe[j][3] = _mm512_sub_ps(matGRe[j][3], temp0);\
    matGIm[j][3] = _mm512_sub_ps(matGIm[j][3], temp1);\
    matGRe[j][3] = _mm512_mul_ps(matGRe[j][3], matD[3]);\
    matGIm[j][3] = _mm512_mul_ps(matGIm[j][3], matD[3]);\
}

// get G44
#define GET_G44(matGRe, matGIm, matBRe, matD, matND, temp0, temp1)\
{\
    GET_AxAH(matGRe[4][0], matGIm[4][0], temp0);\
    GET_AxAH(matGRe[4][1], matGIm[4][1], temp1);\
    matGRe[4][4] = _mm512_sub_ps(matBRe[4][4], temp0);\
    matGRe[4][4] = _mm512_sub_ps(matGRe[4][4], temp1);\
    GET_AxAH(matGRe[4][2], matGIm[4][2], temp0);\
    GET_AxAH(matGRe[4][3], matGIm[4][3], temp1);\
    matGRe[4][4] = _mm512_sub_ps(matGRe[4][4], temp0);\
    matGRe[4][4] = _mm512_sub_ps(matGRe[4][4], temp1);\
    matD[4] = _mm512_rsqrt14_ps(matGRe[4][4]);\
    matND[4] = _mm512_sub_ps(constZero, matD[4]);\
}

// get G55
#define GET_G55(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2)\
{\
    GET_AxAH(matGRe[5][0], matGIm[5][0], temp0);\
    GET_AxAH(matGRe[5][1], matGIm[5][1], temp1);\
    matGRe[5][5] = _mm512_sub_ps(matBRe[5][5], temp0);\
    matGRe[5][5] = _mm512_sub_ps(matGRe[5][5], temp1);\
    GET_AxAH(matGRe[5][2], matGIm[5][2], temp0);\
    GET_AxAH(matGRe[5][3], matGIm[5][3], temp1);\
    GET_AxAH(matGRe[5][4], matGIm[5][4], temp2);\
    matGRe[5][5] = _mm512_sub_ps(matGRe[5][5], temp0);\
    matGRe[5][5] = _mm512_sub_ps(matGRe[5][5], temp1);\
    matGRe[5][5] = _mm512_sub_ps(matGRe[5][5], temp2);\
    matD[5] = _mm512_rsqrt14_ps(matGRe[5][5]);\
    matND[5] = _mm512_sub_ps(constZero, matD[5]);\
}

// get G66
#define GET_G66(matGRe, matGIm, matBRe, matD, matND, temp0, temp1)\
{\
    GET_AxAH(matGRe[6][0], matGIm[6][0], temp0);\
    GET_AxAH(matGRe[6][1], matGIm[6][1], temp1);\
    matGRe[6][6] = _mm512_sub_ps(matBRe[6][6], temp0);\
    matGRe[6][6] = _mm512_sub_ps(matGRe[6][6], temp1);\
    GET_AxAH(matGRe[6][2], matGIm[6][2], temp0);\
    GET_AxAH(matGRe[6][3], matGIm[6][3], temp1);\
    matGRe[6][6] = _mm512_sub_ps(matGRe[6][6], temp0);\
    matGRe[6][6] = _mm512_sub_ps(matGRe[6][6], temp1);\
    GET_AxAH(matGRe[6][4], matGIm[6][4], temp0);\
    GET_AxAH(matGRe[6][5], matGIm[6][5], temp1);\
    matGRe[6][6] = _mm512_sub_ps(matGRe[6][6], temp0);\
    matGRe[6][6] = _mm512_sub_ps(matGRe[6][6], temp1);\
    matD[6] = _mm512_rsqrt14_ps(matGRe[6][6]);\
    matND[6] = _mm512_sub_ps(constZero, matD[6]);\
}

// get G77
#define GET_G77(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2)\
{\
    GET_AxAH(matGRe[7][0], matGIm[7][0], temp0);\
    GET_AxAH(matGRe[7][1], matGIm[7][1], temp1);\
    matGRe[7][7] = _mm512_sub_ps(matBRe[7][7], temp0);\
    matGRe[7][7] = _mm512_sub_ps(matGRe[7][7], temp1);\
    GET_AxAH(matGRe[7][2], matGIm[7][2], temp0);\
    GET_AxAH(matGRe[7][3], matGIm[7][3], temp1);\
    matGRe[7][7] = _mm512_sub_ps(matGRe[7][7], temp0);\
    matGRe[7][7] = _mm512_sub_ps(matGRe[7][7], temp1);\
    GET_AxAH(matGRe[7][4], matGIm[7][4], temp0);\
    GET_AxAH(matGRe[7][5], matGIm[7][5], temp1);\
    GET_AxAH(matGRe[7][6], matGIm[7][6], temp2);\
    matGRe[7][7] = _mm512_sub_ps(matGRe[7][7], temp0);\
    matGRe[7][7] = _mm512_sub_ps(matGRe[7][7], temp1);\
    matGRe[7][7] = _mm512_sub_ps(matGRe[7][7], temp2);\
    matD[7] = _mm512_rsqrt14_ps(matGRe[7][7]);\
    matND[7] = _mm512_sub_ps(constZero, matD[7]);\
}

// get Gii, odd diagonal element
#define GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, i)\
{\
    GET_AxAH(matGRe[i][0], matGIm[i][0], temp0);\
    matGRe[i][i] = _mm512_sub_ps(matBRe[i][i], temp0);\
    for (int32_t i1 = 1; i1 < i; i1+=2) \
    {\
        GET_AxAH(matGRe[i][i1], matGIm[i][i1], temp0);\
        GET_AxAH(matGRe[i][i1+1], matGIm[i][i1+1], temp1);\
        matGRe[i][i] = _mm512_sub_ps(matGRe[i][i], temp0);\
        matGRe[i][i] = _mm512_sub_ps(matGRe[i][i], temp1);\
    }\
    matD[i] = _mm512_rsqrt14_ps(matGRe[i][i]);\
    matND[i] = _mm512_sub_ps(constZero, matD[i]);\
}

// get Gii, even diagonal element
#define GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, i)\
{\
    GET_AxAH(matGRe[i][0], matGIm[i][0], temp0);\
    GET_AxAH(matGRe[i][1], matGIm[i][1], temp1);\
    matGRe[i][i] = _mm512_sub_ps(matBRe[i][i], temp0);\
    matGRe[i][i] = _mm512_sub_ps(matGRe[i][i], temp1);\
    for (int32_t i1 = 2; i1 < i; i1+=2) \
    {\
        GET_AxAH(matGRe[i][i1], matGIm[i][i1], temp0);\
        GET_AxAH(matGRe[i][i1+1], matGIm[i][i1+1], temp1);\
        matGRe[i][i] = _mm512_sub_ps(matGRe[i][i], temp0);\
        matGRe[i][i] = _mm512_sub_ps(matGRe[i][i], temp1);\
    }\
    matD[i] = _mm512_rsqrt14_ps(matGRe[i][i]);\
    matND[i] = _mm512_sub_ps(constZero, matD[i]);\
}

// get odd column n of matrix G, j is row index, i is col index
#define GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, j, i, temp0, temp1)\
{\
    GET_AxBH(matGRe[j][0], matGIm[j][0], matGRe[i][0], matGIm[i][0], temp0, temp1);\
    matGRe[j][i] = _mm512_sub_ps(matBRe[j][i], temp0);\
    matGIm[j][i] = _mm512_sub_ps(matBIm[j][i], temp1);\
    for (int32_t i1 = 1; i1 < i; i1+=2)\
    {\
        GET_AxBH(matGRe[j][i1], matGIm[j][i1], matGRe[i][i1], matGIm[i][i1], temp0, temp1);\
        matGRe[j][i] = _mm512_sub_ps(matGRe[j][i], temp0);\
        matGIm[j][i] = _mm512_sub_ps(matGIm[j][i], temp1);\
        GET_AxBH(matGRe[j][i1+1], matGIm[j][i1+1], matGRe[i][i1+1], matGIm[i][i1+1], temp0, temp1);\
        matGRe[j][i] = _mm512_sub_ps(matGRe[j][i], temp0);\
        matGIm[j][i] = _mm512_sub_ps(matGIm[j][i], temp1);\
    }\
    matGRe[j][i] = _mm512_mul_ps(matGRe[j][i], matD[i]);\
    matGIm[j][i] = _mm512_mul_ps(matGIm[j][i], matD[i]);\
}

// get even column n of matrix G, j is row index, i is col index
#define GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, j, i, temp0, temp1)\
{\
    GET_AxBH(matGRe[j][0], matGIm[j][0], matGRe[i][0], matGIm[i][0], temp0, temp1);\
    matGRe[j][i] = _mm512_sub_ps(matBRe[j][i], temp0);\
    matGIm[j][i] = _mm512_sub_ps(matBIm[j][i], temp1);\
    GET_AxBH(matGRe[j][1], matGIm[j][1], matGRe[i][1], matGIm[i][1], temp0, temp1);\
    matGRe[j][i] = _mm512_sub_ps(matGRe[j][i], temp0);\
    matGIm[j][i] = _mm512_sub_ps(matGIm[j][i], temp1);\
    for (int32_t i1 = 2; i1 < i; i1+=2)\
    {\
        GET_AxBH(matGRe[j][i1], matGIm[j][i1], matGRe[i][i1], matGIm[i][i1], temp0, temp1);\
        matGRe[j][i] = _mm512_sub_ps(matGRe[j][i], temp0);\
        matGIm[j][i] = _mm512_sub_ps(matGIm[j][i], temp1);\
        GET_AxBH(matGRe[j][i1+1], matGIm[j][i1+1], matGRe[i][i1+1], matGIm[i][i1+1], temp0, temp1);\
        matGRe[j][i] = _mm512_sub_ps(matGRe[j][i], temp0);\
        matGIm[j][i] = _mm512_sub_ps(matGIm[j][i], temp1);\
    }\
    matGRe[j][i] = _mm512_mul_ps(matGRe[j][i], matD[i]);\
    matGIm[j][i] = _mm512_mul_ps(matGIm[j][i], matD[i]);\
}

// set value for Lii, diagonal element
#define SET_Lii(matLRe, matLIm, matD, i)\
{\
    matLRe[i][i] = matD[i];\
    matLIm[i][i] = constZero;\
}

// get element L(i+1, i)
#define GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, j, i)\
{\
    GET_AxRealB(matGRe[j][i], matGIm[j][i], matLRe[i][i],  matLRe[j][i], matLIm[j][i]);\
    matLRe[j][i] = _mm512_mul_ps(matLRe[j][i], matND[j]);\
    matLIm[j][i] = _mm512_mul_ps(matLIm[j][i], matND[j]);\
}

// get element Lji, j is larger than i+1
#define GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, j, i, temp0, temp1)\
{\
    GET_AxRealB(matGRe[j][i], matGIm[j][i], matLRe[i][i],  matLRe[j][i], matLIm[j][i]);\
    for (int32_t i1 = i+1; i1 < j; i1++)\
    {\
        GET_AxB(matGRe[j][i1], matGIm[j][i1], matLRe[i1][i], matLIm[i1][i], temp0, temp1);\
        matLRe[j][i] = _mm512_add_ps(matLRe[j][i], temp0);\
        matLIm[j][i] = _mm512_add_ps(matLIm[j][i], temp1);\
    }\
    matLRe[j][i] = _mm512_mul_ps(matLRe[j][i], matND[j]);\
    matLIm[j][i] = _mm512_mul_ps(matLIm[j][i], matND[j]);\
}

void matrix_inv_cholesky_2x2(__m512 matBRe[N_2][N_2], __m512 matBIm[N_2][N_2],
    __m512 matInvBRe[N_2][N_2], __m512 matInvBIm[N_2][N_2])
{
    // temp matrix and variables for matrix inversion
    __m512 matGRe[N_2][N_2], matGIm[N_2][N_2];
    __m512 matLRe[N_2][N_2], matLIm[N_2][N_2];
    __m512 matD[N_2], matND[N_2];
    __m512 temp0, temp1;

    /////////////////////////////////// get G, B = G*G', G is a lower triangular matrix
    // Column 0
    GET_G00(matGRe, matBRe, matD, matND);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 1);

    // Column 1
    GET_G11(matGRe, matGIm, matBRe, matD, matND, temp0);


    /////////////////////////////////// get L, L=invG
    // Column 0
    SET_Lii(matLRe, matLIm, matD, 0);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 1, 0);

    // Column 1
    SET_Lii(matLRe, matLIm, matD, 1);


    /////////////////////////////////// get invB = L'*L
    matInvBRe[0][0] = _mm512_mul_ps(matLRe[0][0], matLRe[0][0]);
    temp1 = _mm512_add_ps(_mm512_mul_ps(matLRe[1][0], matLRe[1][0]), _mm512_mul_ps(matLIm[1][0], matLIm[1][0]));
    matInvBRe[0][0] = _mm512_add_ps(matInvBRe[0][0], temp1);
    matInvBIm[0][0] = _mm512_setzero_ps();

    matInvBRe[1][1] = _mm512_mul_ps(matLRe[1][1], matLRe[1][1]);
    matInvBIm[1][1] = _mm512_setzero_ps();

    matInvBRe[0][1] = _mm512_mul_ps(matLRe[1][0], matLRe[1][1]);
    matInvBIm[0][1] = _mm512_sub_ps(constZero, _mm512_mul_ps(matLIm[1][0], matLRe[1][1]));
    matInvBRe[1][0] = matInvBRe[0][1];
    matInvBIm[1][0] = _mm512_sub_ps(constZero, matInvBIm[0][1]);
}

void matrix_inv_cholesky_3x3(__m512 matBRe[N_3][N_3], __m512 matBIm[N_3][N_3],
    __m512 matInvBRe[N_3][N_3], __m512 matInvBIm[N_3][N_3])
{
    // temp matrix and variables for matrix inversion
    __m512 matGRe[N_3][N_3], matGIm[N_3][N_3];
    __m512 matLRe[N_3][N_3], matLIm[N_3][N_3];
    __m512 matD[N_3], matND[N_3];
    __m512 temp0, temp1, temp2;
    int32_t i, j, k;

    /////////////////////////////////// get G, B = G*G', G is a lower triangular matrix
    // Column 0
    GET_G00(matGRe, matBRe, matD, matND);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 1);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 2);

    // Column 1
    GET_G11(matGRe, matGIm, matBRe, matD, matND, temp0);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 2, temp0, temp1);

    // Column 2
    GET_G22(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);


    /////////////////////////////////// get L, L=invG
    // Column 0
    SET_Lii(matLRe, matLIm, matD, 0);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 1, 0);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 2, 0, temp0, temp1);

    // Column 1
    SET_Lii(matLRe, matLIm, matD, 1);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 2, 1);

    // Column 2
    SET_Lii(matLRe, matLIm, matD, 2);


    /////////////////////////////////// get invB = L'*L
    for(i = 0; i < N_3; i ++)
    {
        matInvBRe[i][i] = _mm512_mul_ps(matLRe[i][i], matLRe[i][i]);
        for (k = (i+1); k < N_3; k++)
        {
            temp1 = _mm512_add_ps(_mm512_mul_ps(matLRe[k][i], matLRe[k][i]), _mm512_mul_ps(matLIm[k][i], matLIm[k][i]));
            matInvBRe[i][i] = _mm512_add_ps(matInvBRe[i][i], temp1);
        }
        matInvBIm[i][i] = _mm512_setzero_ps();
    }

    for(i = 0; i < N_3; i ++)
    {
        for(j = i+1; j < N_3; j ++)
        {
            matInvBRe[i][j] = _mm512_mul_ps(matLRe[j][i], matLRe[j][j]);
            matInvBIm[i][j] = _mm512_sub_ps(constZero, _mm512_mul_ps(matLIm[j][i], matLRe[j][j]));

            for (k = (j+1); k < N_3; k++)
            {
                GET_AxBH(matLRe[k][j], matLIm[k][j], matLRe[k][i], matLIm[k][i], temp1, temp2);
                matInvBRe[i][j] = _mm512_add_ps(matInvBRe[i][j], temp1);
                matInvBIm[i][j] = _mm512_add_ps(matInvBIm[i][j], temp2);
            }

            // Hermite matrix
            matInvBRe[j][i] = matInvBRe[i][j];
            matInvBIm[j][i] = _mm512_sub_ps(constZero, matInvBIm[i][j]);
        }
    }
}

void matrix_inv_cholesky_4x4(__m512 matBRe[N_4][N_4], __m512 matBIm[N_4][N_4],
    __m512 matInvBRe[N_4][N_4], __m512 matInvBIm[N_4][N_4])
{
    // temp matrix and variables for matrix inversion
    __m512 matGRe[N_4][N_4], matGIm[N_4][N_4];
    __m512 matLRe[N_4][N_4], matLIm[N_4][N_4];
    __m512 matD[N_4], matND[N_4];
    __m512 temp0, temp1, temp2;
    int32_t i, j, k;

    /////////////////////////////////// get G, B = G*G', G is a lower triangular matrix
    // Column 0
    GET_G00(matGRe, matBRe, matD, matND);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 1);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 2);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 3);

    // Column 1
    GET_G11(matGRe, matGIm, matBRe, matD, matND, temp0);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 2, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);

    // Column 2
    GET_G22(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);

    // Column 3
    GET_G33(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);


    /////////////////////////////////// get L, L=invG
    // Column 0
    SET_Lii(matLRe, matLIm, matD, 0);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 1, 0);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 2, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 0, temp0, temp1);

    // Column 1
    SET_Lii(matLRe, matLIm, matD, 1);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 2, 1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 1, temp0, temp1);

    // Column 2
    SET_Lii(matLRe, matLIm, matD, 2);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 3, 2);

    // Column 3
    SET_Lii(matLRe, matLIm, matD, 3);


    /////////////////////////////////// get invB = L'*L
    for(i = 0; i < N_4; i ++)
    {
        matInvBRe[i][i] = _mm512_mul_ps(matLRe[i][i], matLRe[i][i]);
        for (k = (i+1); k < N_4; k++)
        {
            temp1 = _mm512_add_ps(_mm512_mul_ps(matLRe[k][i], matLRe[k][i]), _mm512_mul_ps(matLIm[k][i], matLIm[k][i]));
            matInvBRe[i][i] = _mm512_add_ps(matInvBRe[i][i], temp1);
        }
        matInvBIm[i][i] = _mm512_setzero_ps();
    }

    for(i = 0; i < N_4; i ++)
    {
        for(j = i+1; j < N_4; j ++)
        {
            matInvBRe[i][j] = _mm512_mul_ps(matLRe[j][i], matLRe[j][j]);
            matInvBIm[i][j] = _mm512_sub_ps(constZero, _mm512_mul_ps(matLIm[j][i], matLRe[j][j]));

            for (k = (j+1); k < N_4; k++)
            {
                GET_AxBH(matLRe[k][j], matLIm[k][j], matLRe[k][i], matLIm[k][i], temp1, temp2);
                matInvBRe[i][j] = _mm512_add_ps(matInvBRe[i][j], temp1);
                matInvBIm[i][j] = _mm512_add_ps(matInvBIm[i][j], temp2);
            }

            // Hermite matrix
            matInvBRe[j][i] = matInvBRe[i][j];
            matInvBIm[j][i] = _mm512_sub_ps(constZero, matInvBIm[i][j]);
        }
    }
}

void matrix_inv_cholesky_5x5(__m512 matBRe[N_5][N_5], __m512 matBIm[N_5][N_5],
    __m512 matInvBRe[N_5][N_5], __m512 matInvBIm[N_5][N_5])
{
    // temp matrix and variables for matrix inversion
    __m512 matGRe[N_5][N_5], matGIm[N_5][N_5];
    __m512 matLRe[N_5][N_5], matLIm[N_5][N_5];
    __m512 matD[N_5], matND[N_5];
    __m512 temp0, temp1, temp2;
    int32_t i, j, k;

    /////////////////////////////////// get G, B = G*G', G is a lower triangular matrix
    // Column 0
    GET_G00(matGRe, matBRe, matD, matND);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 1);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 2);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 3);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 4);

    // Column 1
    GET_G11(matGRe, matGIm, matBRe, matD, matND, temp0);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 2, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);

    // Column 2
    GET_G22(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);

    // Column 3
    GET_G33(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);

    // Column 4
    GET_G44(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);


    /////////////////////////////////// get L, L=invG
    // Column 0
    SET_Lii(matLRe, matLIm, matD, 0);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 1, 0);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 2, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 0, temp0, temp1);

    // Column 1
    SET_Lii(matLRe, matLIm, matD, 1);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 2, 1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 1, temp0, temp1);

    // Column 2
    SET_Lii(matLRe, matLIm, matD, 2);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 3, 2);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 2, temp0, temp1);

    // Column 3
    SET_Lii(matLRe, matLIm, matD, 3);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 4, 3);

    // Column 4
    SET_Lii(matLRe, matLIm, matD, 4);


    /////////////////////////////////// get invB = L'*L
    for(i = 0; i < N_5; i ++)
    {
        matInvBRe[i][i] = _mm512_mul_ps(matLRe[i][i], matLRe[i][i]);
        for (k = (i+1); k < N_5; k++)
        {
            temp1 = _mm512_add_ps(_mm512_mul_ps(matLRe[k][i], matLRe[k][i]), _mm512_mul_ps(matLIm[k][i], matLIm[k][i]));
            matInvBRe[i][i] = _mm512_add_ps(matInvBRe[i][i], temp1);
        }
        matInvBIm[i][i] = _mm512_setzero_ps();
    }

    for(i = 0; i < N_5; i ++)
    {
        for(j = i+1; j < N_5; j ++)
        {
            matInvBRe[i][j] = _mm512_mul_ps(matLRe[j][i], matLRe[j][j]);
            matInvBIm[i][j] = _mm512_sub_ps(constZero, _mm512_mul_ps(matLIm[j][i], matLRe[j][j]));

            for (k = (j+1); k < N_5; k++)
            {
                GET_AxBH(matLRe[k][j], matLIm[k][j], matLRe[k][i], matLIm[k][i], temp1, temp2);
                matInvBRe[i][j] = _mm512_add_ps(matInvBRe[i][j], temp1);
                matInvBIm[i][j] = _mm512_add_ps(matInvBIm[i][j], temp2);
            }

            // Hermite matrix
            matInvBRe[j][i] = matInvBRe[i][j];
            matInvBIm[j][i] = _mm512_sub_ps(constZero, matInvBIm[i][j]);
        }
    }
}

void matrix_inv_cholesky_6x6(__m512 matBRe[N_6][N_6], __m512 matBIm[N_6][N_6],
    __m512 matInvBRe[N_6][N_6], __m512 matInvBIm[N_6][N_6])
{
    // temp matrix and variables for matrix inversion
    __m512 matGRe[N_6][N_6], matGIm[N_6][N_6];
    __m512 matLRe[N_6][N_6], matLIm[N_6][N_6];
    __m512 matD[N_6], matND[N_6];
    __m512 temp0, temp1, temp2;
    int32_t i, j, k;

    /////////////////////////////////// get G, B = G*G', G is a lower triangular matrix
    // Column 0
    GET_G00(matGRe, matBRe, matD, matND);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 1);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 2);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 3);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 4);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 5);

    // Column 1
    GET_G11(matGRe, matGIm, matBRe, matD, matND, temp0);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 2, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);

    // Column 2
    GET_G22(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);

    // Column 3
    GET_G33(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);

    // Column 4
    GET_G44(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 5, 4, temp0, temp1);

    // Column 5
    GET_G55(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);


    /////////////////////////////////// get L, L=invG
    // Column 0
    SET_Lii(matLRe, matLIm, matD, 0);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 1, 0);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 2, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 0, temp0, temp1);

    // Column 1
    SET_Lii(matLRe, matLIm, matD, 1);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 2, 1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 1, temp0, temp1);

    // Column 2
    SET_Lii(matLRe, matLIm, matD, 2);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 3, 2);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 2, temp0, temp1);

    // Column 3
    SET_Lii(matLRe, matLIm, matD, 3);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 4, 3);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 3, temp0, temp1);

    // Column 4
    SET_Lii(matLRe, matLIm, matD, 4);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 5, 4);

    // Column 5
    SET_Lii(matLRe, matLIm, matD, 5);


    /////////////////////////////////// get invB = L'*L
    for(i = 0; i < N_6; i ++)
    {
        matInvBRe[i][i] = _mm512_mul_ps(matLRe[i][i], matLRe[i][i]);
        for (k = (i+1); k < N_6; k++)
        {
            temp1 = _mm512_add_ps(_mm512_mul_ps(matLRe[k][i], matLRe[k][i]), _mm512_mul_ps(matLIm[k][i], matLIm[k][i]));
            matInvBRe[i][i] = _mm512_add_ps(matInvBRe[i][i], temp1);
        }
        matInvBIm[i][i] = _mm512_setzero_ps();
    }

    for(i = 0; i < N_6; i ++)
    {
        for(j = i+1; j < N_6; j ++)
        {
            matInvBRe[i][j] = _mm512_mul_ps(matLRe[j][i], matLRe[j][j]);
            matInvBIm[i][j] = _mm512_sub_ps(constZero, _mm512_mul_ps(matLIm[j][i], matLRe[j][j]));

            for (k = (j+1); k < N_6; k++)
            {
                GET_AxBH(matLRe[k][j], matLIm[k][j], matLRe[k][i], matLIm[k][i], temp1, temp2);
                matInvBRe[i][j] = _mm512_add_ps(matInvBRe[i][j], temp1);
                matInvBIm[i][j] = _mm512_add_ps(matInvBIm[i][j], temp2);
            }

            // Hermite matrix
            matInvBRe[j][i] = matInvBRe[i][j];
            matInvBIm[j][i] = _mm512_sub_ps(constZero, matInvBIm[i][j]);
        }
    }
}

void matrix_inv_cholesky_7x7(__m512 matBRe[N_7][N_7], __m512 matBIm[N_7][N_7],
    __m512 matInvBRe[N_7][N_7], __m512 matInvBIm[N_7][N_7])
{
    // temp matrix and variables for matrix inversion
    __m512 matGRe[N_7][N_7], matGIm[N_7][N_7];
    __m512 matLRe[N_7][N_7], matLIm[N_7][N_7];
    __m512 matD[N_7], matND[N_7];
    __m512 temp0, temp1, temp2;
    int32_t i, j, k;

    /////////////////////////////////// get G, B = G*G', G is a lower triangular matrix
    // Column 0
    GET_G00(matGRe, matBRe, matD, matND);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 1);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 2);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 3);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 4);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 5);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 6);

    // Column 1
    GET_G11(matGRe, matGIm, matBRe, matD, matND, temp0);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 2, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);

    // Column 2
    GET_G22(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);

    // Column 3
    GET_G33(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);

    // Column 4
    GET_G44(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 5, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 6, 4, temp0, temp1);

    // Column 5
    GET_G55(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 6, 5, temp0, temp1);

    // Column 6
    GET_G66(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);


    /////////////////////////////////// get L, L=invG
    // Column 0
    SET_Lii(matLRe, matLIm, matD, 0);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 1, 0);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 2, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 0, temp0, temp1);

    // Column 1
    SET_Lii(matLRe, matLIm, matD, 1);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 2, 1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 1, temp0, temp1);

    // Column 2
    SET_Lii(matLRe, matLIm, matD, 2);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 3, 2);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 2, temp0, temp1);

    // Column 3
    SET_Lii(matLRe, matLIm, matD, 3);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 4, 3);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 3, temp0, temp1);

    // Column 4
    SET_Lii(matLRe, matLIm, matD, 4);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 5, 4);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 4, temp0, temp1);

    // Column 5
    SET_Lii(matLRe, matLIm, matD, 5);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 6, 5);

    // Column 6
    SET_Lii(matLRe, matLIm, matD, 6);


    /////////////////////////////////// get invB = L'*L
    for(i = 0; i < N_7; i ++)
    {
        matInvBRe[i][i] = _mm512_mul_ps(matLRe[i][i], matLRe[i][i]);
        for (k = (i+1); k < N_7; k++)
        {
            temp1 = _mm512_add_ps(_mm512_mul_ps(matLRe[k][i], matLRe[k][i]), _mm512_mul_ps(matLIm[k][i], matLIm[k][i]));
            matInvBRe[i][i] = _mm512_add_ps(matInvBRe[i][i], temp1);
        }
        matInvBIm[i][i] = _mm512_setzero_ps();
    }

    for(i = 0; i < N_7; i ++)
    {
        for(j = i+1; j < N_7; j ++)
        {
            matInvBRe[i][j] = _mm512_mul_ps(matLRe[j][i], matLRe[j][j]);
            matInvBIm[i][j] = _mm512_sub_ps(constZero, _mm512_mul_ps(matLIm[j][i], matLRe[j][j]));

            for (k = (j+1); k < N_7; k++)
            {
                GET_AxBH(matLRe[k][j], matLIm[k][j], matLRe[k][i], matLIm[k][i], temp1, temp2);
                matInvBRe[i][j] = _mm512_add_ps(matInvBRe[i][j], temp1);
                matInvBIm[i][j] = _mm512_add_ps(matInvBIm[i][j], temp2);
            }

            // Hermite matrix
            matInvBRe[j][i] = matInvBRe[i][j];
            matInvBIm[j][i] = _mm512_sub_ps(constZero, matInvBIm[i][j]);
        }
    }
}

void matrix_inv_cholesky_8x8(__m512 matBRe[N_8][N_8], __m512 matBIm[N_8][N_8],
    __m512 matInvBRe[N_8][N_8], __m512 matInvBIm[N_8][N_8])
{
    // temp matrix and variables for matrix inversion
    __m512 matGRe[N_8][N_8], matGIm[N_8][N_8];
    __m512 matLRe[N_8][N_8], matLIm[N_8][N_8];
    __m512 matD[N_8], matND[N_8];
    __m512 temp0, temp1, temp2;
    int32_t i, j, k;

    /////////////////////////////////// get G, B = G*G', G is a lower triangular matrix
    // Column 0
    GET_G00(matGRe, matBRe, matD, matND);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 1);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 2);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 3);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 4);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 5);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 6);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 7);

    // Column 1
    GET_G11(matGRe, matGIm, matBRe, matD, matND, temp0);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 2, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);

    // Column 2
    GET_G22(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);

    // Column 3
    GET_G33(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);

    // Column 4
    GET_G44(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 5, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 6, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 7, 4, temp0, temp1);

    // Column 5
    GET_G55(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 6, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 7, 5, temp0, temp1);

    // Column 6
    GET_G66(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 7, 6, temp0, temp1);

    // Column 7
    GET_G77(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);


    /////////////////////////////////// get L, L=invG
    // Column 0
    SET_Lii(matLRe, matLIm, matD, 0);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 1, 0);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 2, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 0, temp0, temp1);

    // Column 1
    SET_Lii(matLRe, matLIm, matD, 1);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 2, 1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 1, temp0, temp1);

    // Column 2
    SET_Lii(matLRe, matLIm, matD, 2);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 3, 2);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 2, temp0, temp1);

    // Column 3
    SET_Lii(matLRe, matLIm, matD, 3);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 4, 3);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 3, temp0, temp1);

    // Column 4
    SET_Lii(matLRe, matLIm, matD, 4);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 5, 4);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 4, temp0, temp1);

    // Column 5
    SET_Lii(matLRe, matLIm, matD, 5);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 6, 5);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 5, temp0, temp1);

    // Column 6
    SET_Lii(matLRe, matLIm, matD, 6);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 7, 6);

    // Column 7
    SET_Lii(matLRe, matLIm, matD, 7);


    /////////////////////////////////// get invB = L'*L
    for(i = 0; i < N_8; i ++)
    {
        matInvBRe[i][i] = _mm512_mul_ps(matLRe[i][i], matLRe[i][i]);
        for (k = (i+1); k < N_8; k++)
        {
            temp1 = _mm512_add_ps(_mm512_mul_ps(matLRe[k][i], matLRe[k][i]), _mm512_mul_ps(matLIm[k][i], matLIm[k][i]));
            matInvBRe[i][i] = _mm512_add_ps(matInvBRe[i][i], temp1);
        }
        matInvBIm[i][i] = _mm512_setzero_ps();
    }

    for(i = 0; i < N_8; i ++)
    {
        for(j = i+1; j < N_8; j ++)
        {
            matInvBRe[i][j] = _mm512_mul_ps(matLRe[j][i], matLRe[j][j]);
            matInvBIm[i][j] = _mm512_sub_ps(constZero, _mm512_mul_ps(matLIm[j][i], matLRe[j][j]));

            for (k = (j+1); k < N_8; k++)
            {
                GET_AxBH(matLRe[k][j], matLIm[k][j], matLRe[k][i], matLIm[k][i], temp1, temp2);
                matInvBRe[i][j] = _mm512_add_ps(matInvBRe[i][j], temp1);
                matInvBIm[i][j] = _mm512_add_ps(matInvBIm[i][j], temp2);
            }

            // Hermite matrix
            matInvBRe[j][i] = matInvBRe[i][j];
            matInvBIm[j][i] = _mm512_sub_ps(constZero, matInvBIm[i][j]);
        }
    }
}
void matrix_inv_cholesky_9x9(__m512 matBRe[N_9][N_9], __m512 matBIm[N_9][N_9],
    __m512 matInvBRe[N_9][N_9], __m512 matInvBIm[N_9][N_9])
{
    // temp matrix and variables for matrix inversion
    __m512 matGRe[N_9][N_9], matGIm[N_9][N_9];
    __m512 matLRe[N_9][N_9], matLIm[N_9][N_9];
    __m512 matD[N_9], matND[N_9];
    __m512 temp0, temp1, temp2;
    int32_t i, j, k;

    /////////////////////////////////// get G, B = G*G', G is a lower triangular matrix
    // Column 0
    GET_G00(matGRe, matBRe, matD, matND);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 1);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 2);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 3);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 4);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 5);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 6);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 7);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 8);

    // Column 1
    GET_G11(matGRe, matGIm, matBRe, matD, matND, temp0);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 2, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);

    // Column 2
    GET_G22(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);

    // Column 3
    GET_G33(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);

    // Column 4
    GET_G44(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 5, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 6, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 7, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 8, 4, temp0, temp1);

    // Column 5
    GET_G55(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 6, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 7, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 8, 5, temp0, temp1);

    // Column 6
    GET_G66(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 7, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 8, 6, temp0, temp1);

    // Column 7
    GET_G77(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 8, 7, temp0, temp1);

    // Column 8
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 8);

    /////////////////////////////////// get L, L=invG
    // Column 0
    SET_Lii(matLRe, matLIm, matD, 0);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 1, 0);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 2, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 0, temp0, temp1);

    // Column 1
    SET_Lii(matLRe, matLIm, matD, 1);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 2, 1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 1, temp0, temp1);
	GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 1, temp0, temp1);

    // Column 2
    SET_Lii(matLRe, matLIm, matD, 2);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 3, 2);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 2, temp0, temp1);
	GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 2, temp0, temp1);

    // Column 3
    SET_Lii(matLRe, matLIm, matD, 3);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 4, 3);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 3, temp0, temp1);
	GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 3, temp0, temp1);

    // Column 4
    SET_Lii(matLRe, matLIm, matD, 4);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 5, 4);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 4, temp0, temp1);
	GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 4, temp0, temp1);

    // Column 5
    SET_Lii(matLRe, matLIm, matD, 5);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 6, 5);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 5, temp0, temp1);
	GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 5, temp0, temp1);

    // Column 6
    SET_Lii(matLRe, matLIm, matD, 6);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 7, 6);
	GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 6, temp0, temp1);

    // Column 7
    SET_Lii(matLRe, matLIm, matD, 7);
	GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 8, 7);

	//Column 8
	SET_Lii(matLRe, matLIm, matD, 8);


    /////////////////////////////////// get invB = L'*L
    for(i = 0; i < N_9; i ++)
    {
        matInvBRe[i][i] = _mm512_mul_ps(matLRe[i][i], matLRe[i][i]);
        for (k = (i+1); k < N_9; k++)
        {
            temp1 = _mm512_add_ps(_mm512_mul_ps(matLRe[k][i], matLRe[k][i]), _mm512_mul_ps(matLIm[k][i], matLIm[k][i]));
            matInvBRe[i][i] = _mm512_add_ps(matInvBRe[i][i], temp1);
        }
        matInvBIm[i][i] = _mm512_setzero_ps();
    }

    for(i = 0; i < N_9; i ++)
    {
        for(j = i+1; j < N_9; j ++)
        {
            matInvBRe[i][j] = _mm512_mul_ps(matLRe[j][i], matLRe[j][j]);
            matInvBIm[i][j] = _mm512_sub_ps(constZero, _mm512_mul_ps(matLIm[j][i], matLRe[j][j]));

            for (k = (j+1); k < N_9; k++)
            {
                GET_AxBH(matLRe[k][j], matLIm[k][j], matLRe[k][i], matLIm[k][i], temp1, temp2);
                matInvBRe[i][j] = _mm512_add_ps(matInvBRe[i][j], temp1);
                matInvBIm[i][j] = _mm512_add_ps(matInvBIm[i][j], temp2);
            }

            // Hermite matrix
            matInvBRe[j][i] = matInvBRe[i][j];
            matInvBIm[j][i] = _mm512_sub_ps(constZero, matInvBIm[i][j]);
        }
    }
}

void matrix_inv_cholesky_10x10(__m512 matBRe[N_10][N_10], __m512 matBIm[N_10][N_10],
    __m512 matInvBRe[N_10][N_10], __m512 matInvBIm[N_10][N_10])
{
    // temp matrix and variables for matrix inversion
    __m512 matGRe[N_10][N_10], matGIm[N_10][N_10];
    __m512 matLRe[N_10][N_10], matLIm[N_10][N_10];
    __m512 matD[N_10], matND[N_10];
    __m512 temp0, temp1, temp2;
    int32_t i, j, k;

    /////////////////////////////////// get G, B = G*G', G is a lower triangular matrix
    // Column 0
    GET_G00(matGRe, matBRe, matD, matND);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 1);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 2);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 3);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 4);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 5);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 6);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 7);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 8);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 9);

    // Column 1
    GET_G11(matGRe, matGIm, matBRe, matD, matND, temp0);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 2, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 9, temp0, temp1);

    // Column 2
    GET_G22(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 9, temp0, temp1);

    // Column 3
    GET_G33(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 9, temp0, temp1);

    // Column 4
    GET_G44(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 5, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 6, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 7, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 8, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 9, 4, temp0, temp1);

    // Column 5
    GET_G55(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 6, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 7, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 8, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 9, 5, temp0, temp1);

    // Column 6
    GET_G66(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 7, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 8, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 9, 6, temp0, temp1);

    // Column 7
    GET_G77(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 8, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 9, 7, temp0, temp1);

    // Column 8
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 8);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 9, 8, temp0, temp1);

    //Column 9
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 9);

    /////////////////////////////////// get L, L=invG
    // Column 0
    SET_Lii(matLRe, matLIm, matD, 0);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 1, 0);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 2, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 0, temp0, temp1);

    // Column 1
    SET_Lii(matLRe, matLIm, matD, 1);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 2, 1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 1, temp0, temp1);

    // Column 2
    SET_Lii(matLRe, matLIm, matD, 2);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 3, 2);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 2, temp0, temp1);
	GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 2, temp0, temp1);

    // Column 3
    SET_Lii(matLRe, matLIm, matD, 3);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 4, 3);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 3, temp0, temp1);
	GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 3, temp0, temp1);

    // Column 4
    SET_Lii(matLRe, matLIm, matD, 4);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 5, 4);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 4, temp0, temp1);
	GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 4, temp0, temp1);

    // Column 5
    SET_Lii(matLRe, matLIm, matD, 5);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 6, 5);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 5, temp0, temp1);
	GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 5, temp0, temp1);

    // Column 6
    SET_Lii(matLRe, matLIm, matD, 6);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 7, 6);
	GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 6, temp0, temp1);

    // Column 7
    SET_Lii(matLRe, matLIm, matD, 7);
	GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 8, 7);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 7, temp0, temp1);

	//Column 8
	SET_Lii(matLRe, matLIm, matD, 8);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 9, 8);

    //Column 9
    SET_Lii(matLRe, matLIm, matD, 9);

    /////////////////////////////////// get invB = L'*L
    for(i = 0; i < N_10; i ++)
    {
        matInvBRe[i][i] = _mm512_mul_ps(matLRe[i][i], matLRe[i][i]);
        for (k = (i+1); k < N_10; k++)
        {
            temp1 = _mm512_add_ps(_mm512_mul_ps(matLRe[k][i], matLRe[k][i]), _mm512_mul_ps(matLIm[k][i], matLIm[k][i]));
            matInvBRe[i][i] = _mm512_add_ps(matInvBRe[i][i], temp1);
        }
        matInvBIm[i][i] = _mm512_setzero_ps();
    }

    for(i = 0; i < N_10; i ++)
    {
        for(j = i+1; j < N_10; j ++)
        {
            matInvBRe[i][j] = _mm512_mul_ps(matLRe[j][i], matLRe[j][j]);
            matInvBIm[i][j] = _mm512_sub_ps(constZero, _mm512_mul_ps(matLIm[j][i], matLRe[j][j]));

            for (k = (j+1); k < N_10; k++)
            {
                GET_AxBH(matLRe[k][j], matLIm[k][j], matLRe[k][i], matLIm[k][i], temp1, temp2);
                matInvBRe[i][j] = _mm512_add_ps(matInvBRe[i][j], temp1);
                matInvBIm[i][j] = _mm512_add_ps(matInvBIm[i][j], temp2);
            }

            // Hermite matrix
            matInvBRe[j][i] = matInvBRe[i][j];
            matInvBIm[j][i] = _mm512_sub_ps(constZero, matInvBIm[i][j]);
        }
    }
}

void matrix_inv_cholesky_11x11(__m512 matBRe[N_11][N_11], __m512 matBIm[N_11][N_11],
    __m512 matInvBRe[N_11][N_11], __m512 matInvBIm[N_11][N_11])
{
    // temp matrix and variables for matrix inversion
    __m512 matGRe[N_11][N_11], matGIm[N_11][N_11];
    __m512 matLRe[N_11][N_11], matLIm[N_11][N_11];
    __m512 matD[N_11], matND[N_11];
    __m512 temp0, temp1, temp2;
    int32_t i, j, k;

    /////////////////////////////////// get G, B = G*G', G is a lower triangular matrix
    // Column 0
    GET_G00(matGRe, matBRe, matD, matND);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 1);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 2);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 3);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 4);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 5);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 6);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 7);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 8);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 9);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 10);

    // Column 1
    GET_G11(matGRe, matGIm, matBRe, matD, matND, temp0);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 2, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 9, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 10, temp0, temp1);

    // Column 2
    GET_G22(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 9, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 10, temp0, temp1);

    // Column 3
    GET_G33(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 9, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 10, temp0, temp1);

    // Column 4
    GET_G44(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 5, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 6, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 7, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 8, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 9, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 10, 4, temp0, temp1);

    // Column 5
    GET_G55(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 6, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 7, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 8, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 9, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 10, 5, temp0, temp1);

    // Column 6
    GET_G66(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 7, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 8, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 9, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 10, 6, temp0, temp1);

    // Column 7
    GET_G77(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 8, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 9, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 10, 7, temp0, temp1);

    // Column 8
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 8);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 9, 8, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 10, 8, temp0, temp1);

    // Column 9
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 9);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 10, 9, temp0, temp1);

    // Column 10
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 10);

    /////////////////////////////////// get L, L=invG
    // Column 0
    SET_Lii(matLRe, matLIm, matD, 0);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 1, 0);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 2, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 0, temp0, temp1);

    // Column 1
    SET_Lii(matLRe, matLIm, matD, 1);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 2, 1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 1, temp0, temp1);

    // Column 2
    SET_Lii(matLRe, matLIm, matD, 2);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 3, 2);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 2, temp0, temp1);

    // Column 3
    SET_Lii(matLRe, matLIm, matD, 3);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 4, 3);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 3, temp0, temp1);

    // Column 4
    SET_Lii(matLRe, matLIm, matD, 4);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 5, 4);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 4, temp0, temp1);

    // Column 5
    SET_Lii(matLRe, matLIm, matD, 5);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 6, 5);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 5, temp0, temp1);

    // Column 6
    SET_Lii(matLRe, matLIm, matD, 6);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 7, 6);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 6, temp0, temp1);

    // Column 7
    SET_Lii(matLRe, matLIm, matD, 7);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 8, 7);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 7, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 7, temp0, temp1);

    // Column 8
    SET_Lii(matLRe, matLIm, matD, 8);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 9, 8);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 8, temp0, temp1);

    // Column 9
    SET_Lii(matLRe, matLIm, matD, 9);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 10, 9);

    // Column 10
    SET_Lii(matLRe, matLIm, matD, 10);

    /////////////////////////////////// get invB = L'*L
    for(i = 0; i < N_11; i ++)
    {
        matInvBRe[i][i] = _mm512_mul_ps(matLRe[i][i], matLRe[i][i]);
        for (k = (i+1); k < N_11; k++)
        {
            temp1 = _mm512_add_ps(_mm512_mul_ps(matLRe[k][i], matLRe[k][i]), _mm512_mul_ps(matLIm[k][i], matLIm[k][i]));
            matInvBRe[i][i] = _mm512_add_ps(matInvBRe[i][i], temp1);
        }
        matInvBIm[i][i] = _mm512_setzero_ps();
    }

    for(i = 0; i < N_11; i ++)
    {
        for(j = i+1; j < N_11; j ++)
        {
            matInvBRe[i][j] = _mm512_mul_ps(matLRe[j][i], matLRe[j][j]);
            matInvBIm[i][j] = _mm512_sub_ps(constZero, _mm512_mul_ps(matLIm[j][i], matLRe[j][j]));

            for (k = (j+1); k < N_11; k++)
            {
                GET_AxBH(matLRe[k][j], matLIm[k][j], matLRe[k][i], matLIm[k][i], temp1, temp2);
                matInvBRe[i][j] = _mm512_add_ps(matInvBRe[i][j], temp1);
                matInvBIm[i][j] = _mm512_add_ps(matInvBIm[i][j], temp2);
            }

            // Hermite matrix
            matInvBRe[j][i] = matInvBRe[i][j];
            matInvBIm[j][i] = _mm512_sub_ps(constZero, matInvBIm[i][j]);
        }
    }
}

void matrix_inv_cholesky_12x12(__m512 matBRe[N_12][N_12], __m512 matBIm[N_12][N_12],
    __m512 matInvBRe[N_12][N_12], __m512 matInvBIm[N_12][N_12])
{
    // temp matrix and variables for matrix inversion
    __m512 matGRe[N_12][N_12], matGIm[N_12][N_12];
    __m512 matLRe[N_12][N_12], matLIm[N_12][N_12];
    __m512 matD[N_12], matND[N_12];
    __m512 temp0, temp1, temp2;
    int32_t i, j, k;

    /////////////////////////////////// get G, B = G*G', G is a lower triangular matrix
    // Column 0
    GET_G00(matGRe, matBRe, matD, matND);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 1);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 2);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 3);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 4);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 5);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 6);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 7);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 8);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 9);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 10);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 11);

    // Column 1
    GET_G11(matGRe, matGIm, matBRe, matD, matND, temp0);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 2, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 9, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 10, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 11, temp0, temp1);

    // Column 2
    GET_G22(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 9, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 10, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 11, temp0, temp1);

    // Column 3
    GET_G33(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 9, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 10, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 11, temp0, temp1);

    // Column 4
    GET_G44(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 5, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 6, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 7, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 8, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 9, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 10, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 11, 4, temp0, temp1);

    // Column 5
    GET_G55(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 6, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 7, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 8, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 9, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 10, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 11, 5, temp0, temp1);

    // Column 6
    GET_G66(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 7, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 8, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 9, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 10, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 11, 6, temp0, temp1);

    // Column 7
    GET_G77(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 8, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 9, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 10, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 11, 7, temp0, temp1);

    // Column 8
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 8);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 9, 8, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 10, 8, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 11, 8, temp0, temp1);

    // Column 9
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 9);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 10, 9, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 11, 9, temp0, temp1);

    // Column 10
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 10);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 11, 10, temp0, temp1);

    //Column 11
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 11);

    /////////////////////////////////// get L, L=invG
    // Column 0
    SET_Lii(matLRe, matLIm, matD, 0);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 1, 0);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 2, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 0, temp0, temp1);

    // Column 1
    SET_Lii(matLRe, matLIm, matD, 1);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 2, 1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 1, temp0, temp1);

    // Column 2
    SET_Lii(matLRe, matLIm, matD, 2);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 3, 2);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 2, temp0, temp1);

    // Column 3
    SET_Lii(matLRe, matLIm, matD, 3);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 4, 3);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 3, temp0, temp1);

    // Column 4
    SET_Lii(matLRe, matLIm, matD, 4);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 5, 4);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 4, temp0, temp1);

    // Column 5
    SET_Lii(matLRe, matLIm, matD, 5);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 6, 5);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 5, temp0, temp1);

    // Column 6
    SET_Lii(matLRe, matLIm, matD, 6);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 7, 6);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 6, temp0, temp1);

    // Column 7
    SET_Lii(matLRe, matLIm, matD, 7);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 8, 7);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 7, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 7, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 7, temp0, temp1);

    // Column 8
    SET_Lii(matLRe, matLIm, matD, 8);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 9, 8);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 8, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 8, temp0, temp1);

    // Column 9
    SET_Lii(matLRe, matLIm, matD, 9);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 10, 9);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 9, temp0, temp1);

    // Column 10
    SET_Lii(matLRe, matLIm, matD, 10);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 11, 10);

    //Column 11 
    SET_Lii(matLRe, matLIm, matD, 11);

    /////////////////////////////////// get invB = L'*L
    for(i = 0; i < N_12; i ++)
    {
        matInvBRe[i][i] = _mm512_mul_ps(matLRe[i][i], matLRe[i][i]);
        for (k = (i+1); k < N_12; k++)
        {
            temp1 = _mm512_add_ps(_mm512_mul_ps(matLRe[k][i], matLRe[k][i]), _mm512_mul_ps(matLIm[k][i], matLIm[k][i]));
            matInvBRe[i][i] = _mm512_add_ps(matInvBRe[i][i], temp1);
        }
        matInvBIm[i][i] = _mm512_setzero_ps();
    }

    for(i = 0; i < N_12; i ++)
    {
        for(j = i+1; j < N_12; j ++)
        {
            matInvBRe[i][j] = _mm512_mul_ps(matLRe[j][i], matLRe[j][j]);
            matInvBIm[i][j] = _mm512_sub_ps(constZero, _mm512_mul_ps(matLIm[j][i], matLRe[j][j]));

            for (k = (j+1); k < N_12; k++)
            {
                GET_AxBH(matLRe[k][j], matLIm[k][j], matLRe[k][i], matLIm[k][i], temp1, temp2);
                matInvBRe[i][j] = _mm512_add_ps(matInvBRe[i][j], temp1);
                matInvBIm[i][j] = _mm512_add_ps(matInvBIm[i][j], temp2);
            }

            // Hermite matrix
            matInvBRe[j][i] = matInvBRe[i][j];
            matInvBIm[j][i] = _mm512_sub_ps(constZero, matInvBIm[i][j]);
        }
    }
}

void matrix_inv_cholesky_13x13(__m512 matBRe[N_13][N_13], __m512 matBIm[N_13][N_13],
    __m512 matInvBRe[N_13][N_13], __m512 matInvBIm[N_13][N_13])
{
    // temp matrix and variables for matrix inversion
    __m512 matGRe[N_13][N_13], matGIm[N_13][N_13];
    __m512 matLRe[N_13][N_13], matLIm[N_13][N_13];
    __m512 matD[N_13], matND[N_13];
    __m512 temp0, temp1, temp2;
    int32_t i, j, k;

    /////////////////////////////////// get G, B = G*G', G is a lower triangular matrix
    // Column 0
    GET_G00(matGRe, matBRe, matD, matND);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 1);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 2);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 3);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 4);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 5);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 6);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 7);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 8);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 9);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 10);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 11);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 12);

    // Column 1
    GET_G11(matGRe, matGIm, matBRe, matD, matND, temp0);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 2, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 9, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 10, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 11, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 12, temp0, temp1);

    // Column 2
    GET_G22(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 9, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 10, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 11, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 12, temp0, temp1);

    // Column 3
    GET_G33(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 9, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 10, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 11, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 12, temp0, temp1);

    // Column 4
    GET_G44(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 5, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 6, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 7, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 8, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 9, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 10, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 11, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 12, 4, temp0, temp1);

    // Column 5
    GET_G55(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 6, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 7, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 8, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 9, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 10, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 11, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 12, 5, temp0, temp1);

    // Column 6
    GET_G66(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 7, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 8, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 9, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 10, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 11, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 12, 6, temp0, temp1);

    // Column 7
    GET_G77(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 8, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 9, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 10, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 11, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 12, 7, temp0, temp1);

    // Column 8
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 8);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 9, 8, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 10, 8, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 11, 8, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 12, 8, temp0, temp1);

    // Column 9
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 9);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 10, 9, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 11, 9, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 12, 9, temp0, temp1);

    // Column 10
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 10);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 11, 10, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 12, 10, temp0, temp1);

    //Column 11
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 11);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 12, 11, temp0, temp1);

    //Column 12
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 12);

    /////////////////////////////////// get L, L=invG
    // Column 0
    SET_Lii(matLRe, matLIm, matD, 0);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 1, 0);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 2, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 0, temp0, temp1);

    // Column 1
    SET_Lii(matLRe, matLIm, matD, 1);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 2, 1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 1, temp0, temp1);

    // Column 2
    SET_Lii(matLRe, matLIm, matD, 2);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 3, 2);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 2, temp0, temp1);

    // Column 3
    SET_Lii(matLRe, matLIm, matD, 3);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 4, 3);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 3, temp0, temp1);

    // Column 4
    SET_Lii(matLRe, matLIm, matD, 4);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 5, 4);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 4, temp0, temp1);

    // Column 5
    SET_Lii(matLRe, matLIm, matD, 5);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 6, 5);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 5, temp0, temp1);

    // Column 6
    SET_Lii(matLRe, matLIm, matD, 6);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 7, 6);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 6, temp0, temp1);

    // Column 7
    SET_Lii(matLRe, matLIm, matD, 7);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 8, 7);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 7, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 7, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 7, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 7, temp0, temp1);

    // Column 8
    SET_Lii(matLRe, matLIm, matD, 8);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 9, 8);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 8, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 8, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 8, temp0, temp1);

    // Column 9
    SET_Lii(matLRe, matLIm, matD, 9);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 10, 9);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 9, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 9, temp0, temp1);

    // Column 10
    SET_Lii(matLRe, matLIm, matD, 10);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 11, 10);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 10, temp0, temp1);

    //Column 11 
    SET_Lii(matLRe, matLIm, matD, 11);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 12, 11);

    //Column 12
    SET_Lii(matLRe, matLIm, matD, 12);

    /////////////////////////////////// get invB = L'*L
    for(i = 0; i < N_13; i ++)
    {
        matInvBRe[i][i] = _mm512_mul_ps(matLRe[i][i], matLRe[i][i]);
        for (k = (i+1); k < N_13; k++)
        {
            temp1 = _mm512_add_ps(_mm512_mul_ps(matLRe[k][i], matLRe[k][i]), _mm512_mul_ps(matLIm[k][i], matLIm[k][i]));
            matInvBRe[i][i] = _mm512_add_ps(matInvBRe[i][i], temp1);
        }
        matInvBIm[i][i] = _mm512_setzero_ps();
    }

    for(i = 0; i < N_13; i ++)
    {
        for(j = i+1; j < N_13; j ++)
        {
            matInvBRe[i][j] = _mm512_mul_ps(matLRe[j][i], matLRe[j][j]);
            matInvBIm[i][j] = _mm512_sub_ps(constZero, _mm512_mul_ps(matLIm[j][i], matLRe[j][j]));

            for (k = (j+1); k < N_13; k++)
            {
                GET_AxBH(matLRe[k][j], matLIm[k][j], matLRe[k][i], matLIm[k][i], temp1, temp2);
                matInvBRe[i][j] = _mm512_add_ps(matInvBRe[i][j], temp1);
                matInvBIm[i][j] = _mm512_add_ps(matInvBIm[i][j], temp2);
            }

            // Hermite matrix
            matInvBRe[j][i] = matInvBRe[i][j];
            matInvBIm[j][i] = _mm512_sub_ps(constZero, matInvBIm[i][j]);
        }
    }
}

void matrix_inv_cholesky_14x14(__m512 matBRe[N_14][N_14], __m512 matBIm[N_14][N_14],
    __m512 matInvBRe[N_14][N_14], __m512 matInvBIm[N_14][N_14])
{
    // temp matrix and variables for matrix inversion
    __m512 matGRe[N_14][N_14], matGIm[N_14][N_14];
    __m512 matLRe[N_14][N_14], matLIm[N_14][N_14];
    __m512 matD[N_14], matND[N_14];
    __m512 temp0, temp1, temp2;
    int32_t i, j, k;

    /////////////////////////////////// get G, B = G*G', G is a lower triangular matrix
    // Column 0
    GET_G00(matGRe, matBRe, matD, matND);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 1);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 2);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 3);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 4);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 5);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 6);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 7);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 8);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 9);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 10);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 11);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 12);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 13);

    // Column 1
    GET_G11(matGRe, matGIm, matBRe, matD, matND, temp0);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 2, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 9, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 10, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 11, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 12, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 13, temp0, temp1);

    // Column 2
    GET_G22(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 9, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 10, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 11, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 12, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 13, temp0, temp1);

    // Column 3
    GET_G33(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 9, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 10, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 11, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 12, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 13, temp0, temp1);

    // Column 4
    GET_G44(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 5, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 6, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 7, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 8, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 9, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 10, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 11, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 12, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 13, 4, temp0, temp1);

    // Column 5
    GET_G55(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 6, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 7, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 8, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 9, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 10, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 11, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 12, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 13, 5, temp0, temp1);

    // Column 6
    GET_G66(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 7, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 8, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 9, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 10, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 11, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 12, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 13, 6, temp0, temp1);

    // Column 7
    GET_G77(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 8, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 9, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 10, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 11, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 12, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 13, 7, temp0, temp1);

    // Column 8
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 8);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 9, 8, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 10, 8, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 11, 8, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 12, 8, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 13, 8, temp0, temp1);

    // Column 9
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 9);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 10, 9, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 11, 9, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 12, 9, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 13, 9, temp0, temp1);

    // Column 10
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 10);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 11, 10, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 12, 10, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 13, 10, temp0, temp1);

    //Column 11
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 11);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 12, 11, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 13, 11, temp0, temp1);

    //Column 12
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 12);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 13, 12, temp0, temp1);

    //Column 13
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 13);

    /////////////////////////////////// get L, L=invG
    // Column 0
    SET_Lii(matLRe, matLIm, matD, 0);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 1, 0);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 2, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 0, temp0, temp1);

    // Column 1
    SET_Lii(matLRe, matLIm, matD, 1);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 2, 1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 1, temp0, temp1);

    // Column 2
    SET_Lii(matLRe, matLIm, matD, 2);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 3, 2);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 2, temp0, temp1);

    // Column 3
    SET_Lii(matLRe, matLIm, matD, 3);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 4, 3);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 3, temp0, temp1);

    // Column 4
    SET_Lii(matLRe, matLIm, matD, 4);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 5, 4);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 4, temp0, temp1);

    // Column 5
    SET_Lii(matLRe, matLIm, matD, 5);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 6, 5);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 5, temp0, temp1);

    // Column 6
    SET_Lii(matLRe, matLIm, matD, 6);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 7, 6);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 6, temp0, temp1);

    // Column 7
    SET_Lii(matLRe, matLIm, matD, 7);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 8, 7);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 7, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 7, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 7, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 7, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 7, temp0, temp1);

    // Column 8
    SET_Lii(matLRe, matLIm, matD, 8);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 9, 8);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 8, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 8, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 8, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 8, temp0, temp1);

    // Column 9
    SET_Lii(matLRe, matLIm, matD, 9);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 10, 9);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 9, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 9, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 9, temp0, temp1);

    // Column 10
    SET_Lii(matLRe, matLIm, matD, 10);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 11, 10);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 10, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 10, temp0, temp1);

    //Column 11 
    SET_Lii(matLRe, matLIm, matD, 11);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 12, 11);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 11, temp0, temp1);

    //Column 12
    SET_Lii(matLRe, matLIm, matD, 12);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 13, 12);

    //Column 13
    SET_Lii(matLRe, matLIm, matD, 13);

    /////////////////////////////////// get invB = L'*L
    for(i = 0; i < N_14; i ++)
    {
        matInvBRe[i][i] = _mm512_mul_ps(matLRe[i][i], matLRe[i][i]);
        for (k = (i+1); k < N_14; k++)
        {
            temp1 = _mm512_add_ps(_mm512_mul_ps(matLRe[k][i], matLRe[k][i]), _mm512_mul_ps(matLIm[k][i], matLIm[k][i]));
            matInvBRe[i][i] = _mm512_add_ps(matInvBRe[i][i], temp1);
        }
        matInvBIm[i][i] = _mm512_setzero_ps();
    }

    for(i = 0; i < N_14; i ++)
    {
        for(j = i+1; j < N_14; j ++)
        {
            matInvBRe[i][j] = _mm512_mul_ps(matLRe[j][i], matLRe[j][j]);
            matInvBIm[i][j] = _mm512_sub_ps(constZero, _mm512_mul_ps(matLIm[j][i], matLRe[j][j]));

            for (k = (j+1); k < N_14; k++)
            {
                GET_AxBH(matLRe[k][j], matLIm[k][j], matLRe[k][i], matLIm[k][i], temp1, temp2);
                matInvBRe[i][j] = _mm512_add_ps(matInvBRe[i][j], temp1);
                matInvBIm[i][j] = _mm512_add_ps(matInvBIm[i][j], temp2);
            }

            // Hermite matrix
            matInvBRe[j][i] = matInvBRe[i][j];
            matInvBIm[j][i] = _mm512_sub_ps(constZero, matInvBIm[i][j]);
        }
    }
}

void matrix_inv_cholesky_15x15(__m512 matBRe[N_15][N_15], __m512 matBIm[N_15][N_15],
    __m512 matInvBRe[N_15][N_15], __m512 matInvBIm[N_15][N_15])
{
    // temp matrix and variables for matrix inversion
    __m512 matGRe[N_15][N_15], matGIm[N_15][N_15];
    __m512 matLRe[N_15][N_15], matLIm[N_15][N_15];
    __m512 matD[N_15], matND[N_15];
    __m512 temp0, temp1, temp2;
    int32_t i, j, k;

    /////////////////////////////////// get G, B = G*G', G is a lower triangular matrix
    // Column 0
    GET_G00(matGRe, matBRe, matD, matND);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 1);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 2);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 3);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 4);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 5);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 6);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 7);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 8);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 9);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 10);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 11);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 12);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 13);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 14);

    // Column 1
    GET_G11(matGRe, matGIm, matBRe, matD, matND, temp0);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 2, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 9, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 10, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 11, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 12, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 13, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 14, temp0, temp1);

    // Column 2
    GET_G22(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 9, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 10, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 11, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 12, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 13, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 14, temp0, temp1);

    // Column 3
    GET_G33(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 9, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 10, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 11, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 12, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 13, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 14, temp0, temp1);

    // Column 4
    GET_G44(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 5, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 6, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 7, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 8, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 9, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 10, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 11, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 12, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 13, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 14, 4, temp0, temp1);

    // Column 5
    GET_G55(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 6, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 7, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 8, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 9, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 10, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 11, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 12, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 13, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 14, 5, temp0, temp1);

    // Column 6
    GET_G66(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 7, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 8, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 9, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 10, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 11, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 12, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 13, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 14, 6, temp0, temp1);

    // Column 7
    GET_G77(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 8, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 9, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 10, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 11, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 12, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 13, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 14, 7, temp0, temp1);

    // Column 8
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 8);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 9, 8, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 10, 8, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 11, 8, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 12, 8, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 13, 8, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 14, 8, temp0, temp1);

    // Column 9
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 9);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 10, 9, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 11, 9, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 12, 9, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 13, 9, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 14, 9, temp0, temp1);

    // Column 10
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 10);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 11, 10, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 12, 10, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 13, 10, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 14, 10, temp0, temp1);

    //Column 11
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 11);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 12, 11, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 13, 11, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 14, 11, temp0, temp1);

    //Column 12
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 12);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 13, 12, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 14, 12, temp0, temp1);

    //Column 13
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 13);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 14, 13, temp0, temp1);

    //Column 14
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 14);

    /////////////////////////////////// get L, L=invG
    // Column 0
    SET_Lii(matLRe, matLIm, matD, 0);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 1, 0);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 2, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 0, temp0, temp1);

    // Column 1
    SET_Lii(matLRe, matLIm, matD, 1);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 2, 1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 1, temp0, temp1);

    // Column 2
    SET_Lii(matLRe, matLIm, matD, 2);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 3, 2);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 2, temp0, temp1);

    // Column 3
    SET_Lii(matLRe, matLIm, matD, 3);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 4, 3);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 3, temp0, temp1);

    // Column 4
    SET_Lii(matLRe, matLIm, matD, 4);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 5, 4);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 4, temp0, temp1);

    // Column 5
    SET_Lii(matLRe, matLIm, matD, 5);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 6, 5);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 5, temp0, temp1);

    // Column 6
    SET_Lii(matLRe, matLIm, matD, 6);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 7, 6);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 6, temp0, temp1);

    // Column 7
    SET_Lii(matLRe, matLIm, matD, 7);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 8, 7);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 7, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 7, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 7, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 7, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 7, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 7, temp0, temp1);

    // Column 8
    SET_Lii(matLRe, matLIm, matD, 8);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 9, 8);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 8, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 8, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 8, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 8, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 8, temp0, temp1);

    // Column 9
    SET_Lii(matLRe, matLIm, matD, 9);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 10, 9);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 9, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 9, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 9, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 9, temp0, temp1);

    // Column 10
    SET_Lii(matLRe, matLIm, matD, 10);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 11, 10);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 10, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 10, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 10, temp0, temp1);

    //Column 11 
    SET_Lii(matLRe, matLIm, matD, 11);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 12, 11);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 11, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 11, temp0, temp1);

    //Column 12
    SET_Lii(matLRe, matLIm, matD, 12);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 13, 12);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 12, temp0, temp1);

    //Column 13
    SET_Lii(matLRe, matLIm, matD, 13);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 14, 13);

    //Column 14
    SET_Lii(matLRe, matLIm, matD, 14);

    /////////////////////////////////// get invB = L'*L
    for(i = 0; i < N_15; i ++)
    {
        matInvBRe[i][i] = _mm512_mul_ps(matLRe[i][i], matLRe[i][i]);
        for (k = (i+1); k < N_15; k++)
        {
            temp1 = _mm512_add_ps(_mm512_mul_ps(matLRe[k][i], matLRe[k][i]), _mm512_mul_ps(matLIm[k][i], matLIm[k][i]));
            matInvBRe[i][i] = _mm512_add_ps(matInvBRe[i][i], temp1);
        }
        matInvBIm[i][i] = _mm512_setzero_ps();
    }

    for(i = 0; i < N_15; i ++)
    {
        for(j = i+1; j < N_15; j ++)
        {
            matInvBRe[i][j] = _mm512_mul_ps(matLRe[j][i], matLRe[j][j]);
            matInvBIm[i][j] = _mm512_sub_ps(constZero, _mm512_mul_ps(matLIm[j][i], matLRe[j][j]));

            for (k = (j+1); k < N_15; k++)
            {
                GET_AxBH(matLRe[k][j], matLIm[k][j], matLRe[k][i], matLIm[k][i], temp1, temp2);
                matInvBRe[i][j] = _mm512_add_ps(matInvBRe[i][j], temp1);
                matInvBIm[i][j] = _mm512_add_ps(matInvBIm[i][j], temp2);
            }

            // Hermite matrix
            matInvBRe[j][i] = matInvBRe[i][j];
            matInvBIm[j][i] = _mm512_sub_ps(constZero, matInvBIm[i][j]);
        }
    }
}
void matrix_inv_cholesky_16x16(__m512 matBRe[N_16][N_16], __m512 matBIm[N_16][N_16],
    __m512 matInvBRe[N_16][N_16], __m512 matInvBIm[N_16][N_16])
{
    // temp matrix and variables for matrix inversion
    __m512 matGRe[N_16][N_16], matGIm[N_16][N_16];
    __m512 matLRe[N_16][N_16], matLIm[N_16][N_16];
    __m512 matD[N_16], matND[N_16];
    __m512 temp0, temp1, temp2;
    int32_t i, j, k;

    /////////////////////////////////// get G, B = G*G', G is a lower triangular matrix
    // Column 0
    GET_G00(matGRe, matBRe, matD, matND);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 1);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 2);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 3);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 4);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 5);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 6);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 7);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 8);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 9);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 10);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 11);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 12);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 13);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 14);
    GET_G_COL0(matGRe, matGIm, matBRe, matBIm, matD, 15);

    // Column 1
    GET_G11(matGRe, matGIm, matBRe, matD, matND, temp0);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 2, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 9, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 10, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 11, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 12, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 13, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 14, temp0, temp1);
    GET_G_COL1(matGRe, matGIm, matBRe, matBIm, matD, 15, temp0, temp1);

    // Column 2
    GET_G22(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 3, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 9, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 10, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 11, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 12, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 13, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 14, temp0, temp1);
    GET_G_COL2(matGRe, matGIm, matBRe, matBIm, matD, 15, temp0, temp1);

    // Column 3
    GET_G33(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 4, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 5, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 6, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 7, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 8, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 9, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 10, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 11, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 12, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 13, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 14, temp0, temp1);
    GET_G_COL3(matGRe, matGIm, matBRe, matBIm, matD, 15, temp0, temp1);

    // Column 4
    GET_G44(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 5, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 6, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 7, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 8, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 9, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 10, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 11, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 12, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 13, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 14, 4, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 15, 4, temp0, temp1);

    // Column 5
    GET_G55(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 6, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 7, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 8, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 9, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 10, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 11, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 12, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 13, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 14, 5, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 15, 5, temp0, temp1);

    // Column 6
    GET_G66(matGRe, matGIm, matBRe, matD, matND, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 7, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 8, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 9, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 10, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 11, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 12, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 13, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 14, 6, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 15, 6, temp0, temp1);

    // Column 7
    GET_G77(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 8, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 9, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 10, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 11, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 12, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 13, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 14, 7, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 15, 7, temp0, temp1);

    // Column 8
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 8);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 9, 8, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 10, 8, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 11, 8, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 12, 8, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 13, 8, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 14, 8, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 15, 8, temp0, temp1);

    // Column 9
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 9);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 10, 9, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 11, 9, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 12, 9, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 13, 9, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 14, 9, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 15, 9, temp0, temp1);

    // Column 10
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 10);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 11, 10, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 12, 10, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 13, 10, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 14, 10, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 15, 10, temp0, temp1);

    // Column 11
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 11);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 12, 11, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 13, 11, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 14, 11, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 15, 11, temp0, temp1);

    // Column 12
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 12);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 13, 12, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 14, 12, temp0, temp1);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 15, 12, temp0, temp1);

    // Column 13
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 13);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 14, 13, temp0, temp1);
    GET_G_COL_ODD(matGRe, matGIm, matBRe, matBIm, matD, 15, 13, temp0, temp1);

    // Column 14
    GET_Gii_EVEN(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 14);
    GET_G_COL_EVEN(matGRe, matGIm, matBRe, matBIm, matD, 15, 14, temp0, temp1);

    // Column 15
    GET_Gii_ODD(matGRe, matGIm, matBRe, matD, matND, temp0, temp1, temp2, 15);


    /////////////////////////////////// get L, L=invG
    // Column 0
    SET_Lii(matLRe, matLIm, matD, 0);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 1, 0);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 2, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 0, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 15, 0, temp0, temp1);

    // Column 1
    SET_Lii(matLRe, matLIm, matD, 1);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 2, 1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 3, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 1, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 15, 1, temp0, temp1);

    // Column 2
    SET_Lii(matLRe, matLIm, matD, 2);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 3, 2);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 4, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 2, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 15, 2, temp0, temp1);

    // Column 3
    SET_Lii(matLRe, matLIm, matD, 3);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 4, 3);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 5, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 3, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 15, 3, temp0, temp1);

    // Column 4
    SET_Lii(matLRe, matLIm, matD, 4);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 5, 4);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 6, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 4, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 15, 4, temp0, temp1);

    // Column 5
    SET_Lii(matLRe, matLIm, matD, 5);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 6, 5);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 7, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 5, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 15, 5, temp0, temp1);

    // Column 6
    SET_Lii(matLRe, matLIm, matD, 6);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 7, 6);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 8, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 6, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 15, 6, temp0, temp1);

    // Column 7
    SET_Lii(matLRe, matLIm, matD, 7);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 8, 7);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 9, 7, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 7, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 7, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 7, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 7, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 7, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 15, 7, temp0, temp1);

    // Column 8
    SET_Lii(matLRe, matLIm, matD, 8);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 9, 8);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 10, 8, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 8, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 8, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 8, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 8, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 15, 8, temp0, temp1);

    // Column 9
    SET_Lii(matLRe, matLIm, matD, 9);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 10, 9);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 11, 9, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 9, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 9, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 9, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 15, 9, temp0, temp1);

    // Column 10
    SET_Lii(matLRe, matLIm, matD, 10);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 11, 10);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 12, 10, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 10, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 10, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 15, 10, temp0, temp1);

    // Column 11
    SET_Lii(matLRe, matLIm, matD, 11);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 12, 11);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 13, 11, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 11, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 15, 11, temp0, temp1);

    // Column 12
    SET_Lii(matLRe, matLIm, matD, 12);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 13, 12);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 14, 12, temp0, temp1);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 15, 12, temp0, temp1);

    // Column 13
    SET_Lii(matLRe, matLIm, matD, 13);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 14, 13);
    GET_L_ji(matLRe, matLIm, matGRe, matGIm, matND, 15, 13, temp0, temp1);

    // Column 14
    SET_Lii(matLRe, matLIm, matD, 14);
    GET_L_i1i(matLRe, matLIm, matGRe, matGIm, matND, 15, 14);

    // Column 15
    SET_Lii(matLRe, matLIm, matD, 15);


    /////////////////////////////////// get invB = L'*L
    for(i = 0; i < N_16; i ++)
    {
        matInvBRe[i][i] = _mm512_mul_ps(matLRe[i][i], matLRe[i][i]);
        for (k = (i+1); k < N_16; k++)
        {
            temp1 = _mm512_add_ps(_mm512_mul_ps(matLRe[k][i], matLRe[k][i]), _mm512_mul_ps(matLIm[k][i], matLIm[k][i]));
            matInvBRe[i][i] = _mm512_add_ps(matInvBRe[i][i], temp1);
        }
        matInvBIm[i][i] = _mm512_setzero_ps();
    }

    for(i = 0; i < N_16; i ++)
    {
        for(j = i+1; j < N_16; j ++)
        {
            matInvBRe[i][j] = _mm512_mul_ps(matLRe[j][i], matLRe[j][j]);
            matInvBIm[i][j] = _mm512_sub_ps(constZero, _mm512_mul_ps(matLIm[j][i], matLRe[j][j]));

            for (k = (j+1); k < N_16; k++)
            {
                GET_AxBH(matLRe[k][j], matLIm[k][j], matLRe[k][i], matLIm[k][i], temp1, temp2);
                matInvBRe[i][j] = _mm512_add_ps(matInvBRe[i][j], temp1);
                matInvBIm[i][j] = _mm512_add_ps(matInvBIm[i][j], temp2);
            }

            // Hermite matrix
            matInvBRe[j][i] = matInvBRe[i][j];
            matInvBIm[j][i] = _mm512_sub_ps(constZero, matInvBIm[i][j]);
        }
    }
}
#else
void matrix_inv_cholesky_2x2(__m512 matBRe[N_2][N_2], __m512 matBIm[N_2][N_2],
    __m512 matInvBRe[N_2][N_2], __m512 matInvBIm[N_2][N_2])
{
	printf("bblib_matrix_inverse requires AVX512 ISA support to run\n");
}

void matrix_inv_cholesky_3x3(__m512 matBRe[N_3][N_3], __m512 matBIm[N_3][N_3],
    __m512 matInvBRe[N_3][N_3], __m512 matInvBIm[N_3][N_3])
{
	printf("bblib_matrix_inverse requires AVX512 ISA support to run\n");
}

void matrix_inv_cholesky_4x4(__m512 matBRe[N_4][N_4], __m512 matBIm[N_4][N_4],
    __m512 matInvBRe[N_4][N_4], __m512 matInvBIm[N_4][N_4])
{
	printf("bblib_matrix_inverse requires AVX512 ISA support to run\n");
}

void matrix_inv_cholesky_5x5(__m512 matBRe[N_5][N_5], __m512 matBIm[N_5][N_5],
    __m512 matInvBRe[N_5][N_5], __m512 matInvBIm[N_5][N_5])
{
	printf("bblib_matrix_inverse requires AVX512 ISA support to run\n");
}

void matrix_inv_cholesky_6x6(__m512 matBRe[N_6][N_6], __m512 matBIm[N_6][N_6],
    __m512 matInvBRe[N_6][N_6], __m512 matInvBIm[N_6][N_6])
{
	printf("bblib_matrix_inverse requires AVX512 ISA support to run\n");
}

void matrix_inv_cholesky_7x7(__m512 matBRe[N_7][N_7], __m512 matBIm[N_7][N_7],
    __m512 matInvBRe[N_7][N_7], __m512 matInvBIm[N_7][N_7])
{
	printf("bblib_matrix_inverse requires AVX512 ISA support to run\n");
}

void matrix_inv_cholesky_8x8(__m512 matBRe[N_8][N_8], __m512 matBIm[N_8][N_8],
    __m512 matInvBRe[N_8][N_8], __m512 matInvBIm[N_8][N_8])
{
	printf("bblib_matrix_inverse requires AVX512 ISA support to run\n");
}

void matrix_inv_cholesky_9x9(__m512 matBRe[N_9][N_9], __m512 matBIm[N_9][N_9],
    __m512 matInvBRe[N_9][N_9], __m512 matInvBIm[N_9][N_9])
{
	printf("bblib_matrix_inverse requires AVX512 ISA support to run\n");
}

void matrix_inv_cholesky_10x10(__m512 matBRe[N_10][N_10], __m512 matBIm[N_10][N_10],
    __m512 matInvBRe[N_10][N_10], __m512 matInvBIm[N_10][N_10])
{
	printf("bblib_matrix_inverse requires AVX512 ISA support to run\n");
}

void matrix_inv_cholesky_11x11(__m512 matBRe[N_11][N_11], __m512 matBIm[N_11][N_11],
    __m512 matInvBRe[N_11][N_11], __m512 matInvBIm[N_11][N_11])
{
	printf("bblib_matrix_inverse requires AVX512 ISA support to run\n");
}

void matrix_inv_cholesky_12x12(__m512 matBRe[N_12][N_12], __m512 matBIm[N_12][N_12],
    __m512 matInvBRe[N_12][N_12], __m512 matInvBIm[N_12][N_12])
{
	printf("bblib_matrix_inverse requires AVX512 ISA support to run\n");
}

void matrix_inv_cholesky_13x13(__m512 matBRe[N_13][N_13], __m512 matBIm[N_13][N_13],
    __m512 matInvBRe[N_13][N_13], __m512 matInvBIm[N_13][N_13])
{
	printf("bblib_matrix_inverse requires AVX512 ISA support to run\n");
}

void matrix_inv_cholesky_14x14(__m512 matBRe[N_14][N_14], __m512 matBIm[N_14][N_14],
    __m512 matInvBRe[N_14][N_14], __m512 matInvBIm[N_14][N_14])
{
	printf("bblib_matrix_inverse requires AVX512 ISA support to run\n");
}

void matrix_inv_cholesky_15x15(__m512 matBRe[N_15][N_15], __m512 matBIm[N_15][N_15],
    __m512 matInvBRe[N_15][N_15], __m512 matInvBIm[N_15][N_15])
{
	printf("bblib_matrix_inverse requires AVX512 ISA support to run\n");
}

void matrix_inv_cholesky_16x16(__m512 matBRe[N_16][N_16], __m512 matBIm[N_16][N_16],
    __m512 matInvBRe[N_16][N_16], __m512 matInvBIm[N_16][N_16])
{
	printf("bblib_matrix_inverse requires AVX512 ISA support to run\n");
}


#endif

