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
#pragma once
#include "simd_insts.hpp"
#include <map>
#include <tuple>
#include "phy_matrix_inv_cholesky.h"

namespace W_SDK {
#ifdef _BBLIB_AVX512_
namespace PUSCH_SYMBOL_PROCESS {
template<int N = 16>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[N][N], F32vec16 matBIm[N][N],
    F32vec16 matInvBRe[N][N], F32vec16 matInvBIm[N][N]);

template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[16][16], F32vec16 matBIm[16][16],
    F32vec16 matInvBRe[16][16], F32vec16 matInvBIm[16][16]) {
    #define type_cast reinterpret_cast<__m512 (*)[16]>
    matrix_inv_cholesky_16x16(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm));
    #undef type_cast
}
template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[8][8], F32vec16 matBIm[8][8],
    F32vec16 matInvBRe[8][8], F32vec16 matInvBIm[8][8]) {
    #define type_cast reinterpret_cast<__m512 (*)[8]>
    matrix_inv_cholesky_8x8(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm));
    #undef type_cast
}
template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[7][7], F32vec16 matBIm[7][7],
    F32vec16 matInvBRe[7][7], F32vec16 matInvBIm[7][7]) {
    #define type_cast reinterpret_cast<__m512 (*)[7]>
    matrix_inv_cholesky_7x7(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm));
    #undef type_cast
}
template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[6][6], F32vec16 matBIm[6][6],
    F32vec16 matInvBRe[6][6], F32vec16 matInvBIm[6][6]) {
    #define type_cast reinterpret_cast<__m512 (*)[6]>
    matrix_inv_cholesky_6x6(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm));
    #undef type_cast
}
template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[5][5], F32vec16 matBIm[5][5],
    F32vec16 matInvBRe[5][5], F32vec16 matInvBIm[5][5]) {
    #define type_cast reinterpret_cast<__m512 (*)[5]>
    matrix_inv_cholesky_5x5(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm));
    #undef type_cast
}

template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[3][3], F32vec16 matBIm[3][3],
    F32vec16 matInvBRe[3][3], F32vec16 matInvBIm[3][3]) {
    #define type_cast reinterpret_cast<__m512 (*)[3]>
    matrix_inv_cholesky_3x3(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm));
    #undef type_cast
}
template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[2][2], F32vec16 matBIm[2][2],
    F32vec16 matInvBRe[2][2], F32vec16 matInvBIm[2][2]) {
    /*
    #define type_cast reinterpret_cast<__m512 (*)[2]>
    matrix_inv_cholesky_2x2(type_cast(matBRe), type_cast(matBIm), type_cast(matInvBRe), type_cast(matInvBIm));
    #undef type_cast
    */
    // 2. invA = inv(H' * H + Sigma2*I), 1x1 matrix inversion
    matInvBRe[0][0] = matBRe[1][1];
    matInvBRe[0][1] = _mm512_sub_ps(F32vec16(0.0), matBRe[1][0]);
    matInvBRe[1][0] = _mm512_sub_ps(F32vec16(0.0), matBRe[0][1]);
    matInvBRe[1][1] = matBRe[0][0];

    matInvBIm[0][1] = matBIm[1][0];
    matInvBIm[1][0] = matBIm[0][1];

    // 2) calculate the determinant of A, det(A) = a00*a11 - a01*a10;
    auto avxfdetARe = _mm512_mul_ps(matBRe[0][0], matBRe[1][1]);
    avxfdetARe = _mm512_fnmadd_ps(matBRe[0][1], matBRe[0][1], avxfdetARe);
    avxfdetARe = _mm512_fnmadd_ps(matBIm[0][1], matBIm[0][1], avxfdetARe);

    // 3) detA = 1 / detA
    avxfdetARe = _mm512_rcp14_ps(avxfdetARe);

    // 4) invA = (A*) * detA
    matInvBRe[0][0] = _mm512_mul_ps(matInvBRe[0][0], avxfdetARe);
    matInvBRe[0][1] = _mm512_mul_ps(matInvBRe[0][1], avxfdetARe);
    matInvBRe[1][0] = _mm512_mul_ps(matInvBRe[1][0], avxfdetARe);
    matInvBRe[1][1] = _mm512_mul_ps(matInvBRe[1][1], avxfdetARe);

    matInvBIm[0][0] = F32vec16(0.0);
    matInvBIm[1][1] = F32vec16(0.0);
    matInvBIm[0][1] = _mm512_mul_ps(matInvBIm[0][1], avxfdetARe);
    matInvBIm[1][0] = _mm512_mul_ps(matInvBIm[1][0], avxfdetARe);
}
template<>
inline FORCE_INLINE
void matrix_inverse(F32vec16 matBRe[1][1], F32vec16 matBIm[1][1],
    F32vec16 matInvBRe[1][1], F32vec16 matInvBIm[1][1]) {
        matInvBRe[0][0] = rcp(matBRe[0][0]);
        matInvBIm[0][0] = matBIm[0][0];
}
}
#endif

#ifdef _BBLIB_SPR_
// #define PRINT_DEBUG
template<typename T, size_t N = 16> inline FORCE_INLINE
void matrix_print(T A[N][N], std::string str) {
  std::cout << "------------------------" << str << "-------------------------" << std::endl;
  for (int i = 0; i < N; i++) {
    for (int j =0; j < N; j++) {
      std::cout<<"----row = " << i << " col = " << j << ":" << A[i][j] <<std::endl;
      // std::cout << A[i][j] << std::endl;
      // for(size_t k = 0; k < 16; k ++) {
      //   std::cout <<  float(A[i][j][k].real()) << "+" << float(A[i][j][k].imag()) << "j ";
      // }
      // std::cout << "..." << std::endl;
    }
  }
}
// vector A * vector B^H
template<typename T, size_t N = 16> inline FORCE_INLINE
T dot (T x0[N], T x1[N]) {
    auto sum0 = T();
    auto sum1 = T();
    if constexpr (N % 2) 
    {
        #pragma unroll(N >> 1)
        for (size_t j = 0; j < N-1; j = j + 2) {
            sum0 = fma(x0[j], x1[j], sum0);
            sum1 = fma(x0[j + 1], x1[j + 1], sum1);
        }
        sum0 = fma(x0[N-1], x1[N-1], sum0);
    }
    else 
    {
        #pragma unroll(N >> 1)
        for (size_t j = 0; j < N; j = j + 2) {
            sum0 = fma(x0[j], x1[j], sum0);
            sum1 = fma(x0[j + 1], x1[j + 1], sum1);
        }
    }
    return sum0 + sum1;
}
template<> inline FORCE_INLINE
CF16vec16 dot<CF16vec16, 1> (CF16vec16 x0[1], CF16vec16 x1[1]) {
    CF16vec16 sum0 = fmul(x0[0], x1[0]);
    return sum0;
}
// vector A * vector B^T
template<typename T, size_t N = 16> inline FORCE_INLINE
T dotC (T x0[N], T x1[N]) {
    auto sum0 = T();
    auto sum1 = T();
    #pragma unroll(N >> 1)
    for (size_t j = 0; j < N; j = j + 2) {
        sum0 = fmaconj(x0[j], x1[j], sum0);
        sum1 = fmaconj(x0[j + 1], x1[j + 1], sum1);
    }
    return sum0 + sum1; // + sum2 + sum3;
}

template<> inline FORCE_INLINE
CF16vec16 dotC<CF16vec16, 1> (CF16vec16 x0[1], CF16vec16 x1[1]) {
    CF16vec16 sum0 = fmulconj(x0[0], x1[0]);
    return sum0;
}

template<typename T, size_t N = 16> inline FORCE_INLINE
T dotCReal (T x[N]) {
    auto sum0 = T();
    auto sum1 = T();
    #pragma unroll(N >> 1)
    for (size_t j = 0; j < N; j = j + 2) {
        sum0 = _mm512_fmadd_ph(x[j], x[j], sum0);
        sum1 = _mm512_fmadd_ph(x[j + 1], x[j + 1], sum1);
    }
    auto sum = sum0 + sum1; // + sum2 + sum3;
    return addRealImag(sum);
}
template<> inline FORCE_INLINE
CF16vec16 dotCReal<CF16vec16, 1> (CF16vec16 x[1]) {
    auto sum = _mm512_mul_ph(x[0], x[0]);
    return addRealImag(sum);
}

///\brief Left Auto-Covariance of input matrix: A^H * A
  ///
  /// Four dot-product accumulators
  ///
  ///\param TC SIMD vector class
  ///\param IfT IO matrix precision
  ///\param doUpper If set will populate upper triangle with conjugated lower triangle
  ///\param matrixIn Input matrix struct
  ///\param matrixOut Pointer to external TC buffer
  ///\param offset SIMD batch offset
  template<typename T, size_t N, bool doUpper>
  inline
  void AutoCov(T in[N][N],
               T out[N][N]) {
    // Main Diagonal
    #pragma unroll(N)
    for (size_t c = 0; c < N; c++) {
      // Initialise accumulators
      auto acc = T();
#ifdef __linux__
#pragma unroll(N)
#endif
      for (int i = 0; i < N; i++) {
        acc = fmanorm(in[i][c], acc);
      }
      out[c][c] = acc;
    }
    // Lower triangle
    for (size_t c = 0; c < N; c++) {
      for (size_t r = c + 1; r < N; r++) {
        // Initialise accumulators
        auto acc = T();
#ifdef __linux__
#pragma unroll(N)
#endif
        for (int i = 0; i < N; i++) {
          acc = fmaconj(in[i][c], in[i][r], acc);
        }
        out[r][c] = acc;
        if constexpr(doUpper == true)
            out[c][r] = conj(acc);
      }
    }
}

///\brief Left Auto-Covariance of input matrix: A^H * A
  ///
  ///
  ///\param TC SIMD vector class
  ///\param IfT IO matrix precision
  ///\param doUpper If set will populate upper triangle with conjugated lower triangle
  ///\param matrixIn Input matrix struct
  ///\param matrixOut Pointer to external TC buffer
template<typename T, size_t M, size_t N, bool doUpper>
inline void AutoCovZF(T in[M][N], T out[N][N]) {
    // Main Diagonal
    #pragma unroll(N)
    for (size_t c = 0; c < N; c++) {
        auto sum0 = T();
        auto sum1 = T();
        auto sum2 = T();
        auto sum3 = T();
        #pragma unroll(M >> 2)
        for (size_t i = 0; i < M; i = i + 4) {
            sum0 = fmaconj(in[i][c], in[i][c], sum0);
            sum1 = fmaconj(in[i+1][c], in[i+1][c], sum1);
            sum2 = fmaconj(in[i+2][c], in[i+2][c], sum2);
            sum3 = fmaconj(in[i+3][c], in[i+3][c], sum3);
        }
        out[c][c] = sum0 + sum1 + sum2 + sum3;
    }
    // Lower triangle
    for (size_t c = 0; c < N; c++) {
      for (size_t r = c + 1; r < N; r++) {
        auto sum0 = T();
        auto sum1 = T();
        auto sum2 = T();
        auto sum3 = T();
        #pragma unroll(M >> 2)
        for (size_t i = 0; i < M; i = i + 4) {
            sum0 = fmaconj(in[i][c], in[i][r], sum0);
            sum1 = fmaconj(in[i+1][c], in[i+1][r], sum1);
            sum2 = fmaconj(in[i+2][c], in[i+2][r], sum2);
            sum3 = fmaconj(in[i+3][c], in[i+3][r], sum3);
        }
        out[r][c] = sum0 + sum1 + sum2 + sum3;

        if constexpr(doUpper == true) {
          const auto k_imagSignBitMask = _mm512_set1_epi32(0x80000000);
          out[c][r] = _mm512_castsi512_ph(_mm512_xor_si512(k_imagSignBitMask, _mm512_castph_si512(out[r][c])));;
        }
      }
    }
}

///\brief Right Auto-Covariance of input matrix: A * A^H
///
///
///\param TC SIMD vector class
///\param IfT IO matrix precision
///\param doUpper If set will populate upper triangle with conjugated lower triangle
///\param matrixIn Input matrix struct
///\param matrixOut Pointer to external TC buffer
template<typename T, size_t M, size_t N, bool doUpper>
inline void AutoCovRight(T in[M][N], T out[M][M]) {
    // Main Diagonal
    #pragma unroll(M)
    for (size_t c = 0; c < M; c++) {
        auto sum0 = T();
        auto sum1 = T();
        auto sum2 = T();
        auto sum3 = T();
        #pragma unroll(N >> 2)
        for (size_t i = 0; i < N; i = i + 4) {
            sum0 = fmaconj(in[c][i], in[c][i], sum0);
            sum1 = fmaconj(in[c][i+1], in[c][i+1], sum1);
            sum2 = fmaconj(in[c][i+2], in[c][i+2], sum2);
            sum3 = fmaconj(in[c][i+3], in[c][i+3], sum3);
        }
        out[c][c] = sum0 + sum1 + sum2 + sum3;
    }

    // Lower triangle
    for (size_t c = 0; c < M; c++) {
      for (size_t r = c + 1; r < M; r++) {
        auto sum0 = T();
        auto sum1 = T();
        auto sum2 = T();
        auto sum3 = T();
        #pragma unroll(N >> 2)
        for (size_t i = 0; i < N; i = i + 4) {
            sum0 = fmaconj(in[r][i], in[c][i], sum0);
            sum1 = fmaconj(in[r][i+1], in[c][i+1], sum1);
            sum2 = fmaconj(in[r][i+2], in[c][i+2], sum2);
            sum3 = fmaconj(in[r][i+3], in[c][i+3], sum3);
        }
        out[r][c] = sum0 + sum1 + sum2 + sum3;

        if constexpr(doUpper == true) {
          const auto k_imagSignBitMask = _mm512_set1_epi32(0x80000000);    
          out[c][r] = _mm512_castsi512_ph(_mm512_xor_si512(k_imagSignBitMask, _mm512_castph_si512(out[r][c])));;
        }
      }
    }
}

///\brief matrix multiplication: in1 * in2^H
///
///
///\param TC SIMD vector class
///\param in1, in2 Input matrix struct
///\param out Pointer to external TC buffer
template<typename T, size_t M1, size_t M2, size_t N>
inline void matMatConjMul(T in1[M1][N], T in2[M2][N], T out[M1][M2]) {
    for (size_t c = 0; c < M2; c++) {
      for (size_t r = 0; r < M1; r++) {
        auto sum0 = T();
        auto sum1 = T();
        auto sum2 = T();
        auto sum3 = T();
        #pragma unroll(N >> 2)
        for (size_t i = 0; i < N; i = i + 4) {
            sum0 = fmaconj(in1[r][i], in2[c][i], sum0);
            sum1 = fmaconj(in1[r][i+1], in2[c][i+1], sum1);
            sum2 = fmaconj(in1[r][i+2], in2[c][i+2], sum2);
            sum3 = fmaconj(in1[r][i+3], in2[c][i+3], sum3);
        }
        out[r][c] = sum0 + sum1 + sum2 + sum3;
      }
    }
}

///\brief matrix multiplication:: in1^H * in2
///
///
///\param TC SIMD vector class
///\param in1, in2 Input matrix struct
///\param out Pointer to external TC buffer
template<typename T, size_t M1, size_t M2, size_t N>
inline void conjMatMatMul(T in1[N][M1], T in2[M2][N], T out[M1][M2]) {
    for (size_t c = 0; c < M2; c++) {
      for (size_t r = 0; r < M1; r++) {
        auto sum0 = T();
        auto sum1 = T();
        auto sum2 = T();
        auto sum3 = T();
        #pragma unroll(N >> 2)
        for (size_t i = 0; i < N; i = i + 4) {
            sum0 = fmaconj(in2[i][c], in1[i][r], sum0);
            sum1 = fmaconj(in2[i+1][c], in1[i+1][r], sum1);
            sum2 = fmaconj(in2[i+2][c], in1[i+2][r], sum2);
            sum3 = fmaconj(in2[i+3][c], in1[i+3][r], sum3);
        }
        out[r][c] = sum0 + sum1+ sum2 + sum3;
      }
    }
}


  ///\brief Cholesky Factorisation
  ///
  /// Factorise input matrix and populate Cholesky factor in lower triangle
  /// Main diagonal is kept in reciprocal form to help later steps
  /// Result size is [matrixA.numCols x matrixA.numCols]
  ///
  ///\param TC dvec SIMD type
  ///\param order Size of input matrix
  ///\param A Pointer to input matrix
  template<typename T, size_t N = 16>
  inline FORCE_INLINE
  void Factorise(T A[N][N]) {
    // Work down each column starting from the main diagonal
    #pragma unroll(N)
    for (int c = 0; c < N; c++) {

    {
      auto temp0 = A[c][c];
      for (int k = 0; k < c; k++) {
        temp0-=fmulconj(A[c][k],A[c][k]);
      }
      // std::cout<<temp<<std::endl;
      A[c][c] = rsqrt(temp0);
      // L[c][c] = A[c][c];
      // std::cout<<A[c][c]<<std::endl;
    }
    // A[c][c] = rsqrt(A[c][c] - acc);

      // Compute L(r,c)
      for (int r = c + 1; r < N; r++) {
        auto temp1 = A[r][c];
#ifdef __linux__
#pragma unroll(2)
#endif
        for (int k = 0; k < c; k++)
          temp1-=fmulconj(A[r][k],A[c][k]);
        A[r][c] = temp1 * A[c][c];
        // L[r][c] = A[r][c];
      }
    }
  }
    ///\brief Cholesky Factor Invertion
  ///
  /// Invert the Cholesky factor populating lower triangle of input matrix
  /// Invertion overwrites Cholesky factor in lower triangle
  /// Result size is [matrixA.numCols x matrixA.numCols]
  ///
  ///\param TC dvec SIMD type
  ///\param order Size of input matrix
  ///\param A Pointer to input matrix containing Cholesky factor in lower triangle
  template<typename T, size_t N = 16>
  inline FORCE_INLINE
  void InvertFactor(T A[N][N]) {
    // Note that the main diagonal is aleady inverted due to reciprocal square root
    // used in the Cholesky factorisation step
    // Work down each column starting from the main diagonal
    for (int c = 0; c < N; c++) {
      for (int r = c + 1; r < N; r++) {
        T accICF = {T()};
        for (int k = c; k < r; k++) {
          accICF = fma(A[r][k], A[k][c], accICF);
        }
        // A[r][c] = accICF - A[r][r];
        A[r][c] = accICF * -A[r][r];
      }
    }
  }

  
  ///\brief Channel Covariance invertion using inverted Cholesky factor
  ///
  /// Compute W^-1 = L^-H * L^-1;
  /// Result size is [matrixA.numCols x matrixA.numCols]
  /// Result is computed in-place and overwrites input matrixA.
  ///
  ///\param TC dvec SIMD type
  ///\param order Size of input matrix
  ///\param A Pointer to input matrix containing inverted Cholesky factor in lower triangle
  template<typename T, size_t N = 16>
  inline FORCE_INLINE
  void InvFactMult(T A[N][N]) {
    // Triangular matrix multiplication of lower triangular with its Hermitian transpose
    for (int c = 0; c < N; c++) {
      // Main diagonal first
      T accHHHMD = A[c][c] * A[c][c];
      for (int r = c + 1; r < N; r++) {
        accHHHMD = fmanorm(A[r][c], accHHHMD);
        T accHHH = {T()};
        for (int k = r; k < N; k++) {
          accHHH = fmaconj(A[k][c], A[k][r], accHHH);
        }
        A[r][c] = accHHH;
        A[c][r] = conj(accHHH);
      }
      A[c][c] = clamp(accHHHMD, T(-65504.0), T(65504.0));
    }
  }
  ///\brief Complex conjugate vector outer product (innerDim = 1)
  ///
  /// Performs multiply of LHS input with the conjugate tranpose of RHS input
  /// matrixOut = matrixL * matrixR^H
  ///
  ///\param TC SIMD vector class
  ///\param IfT IO matrix precision
  ///\param innerDim matrix multiply inner dimension, templated to allow loop unrolling
  ///\param matrixL Pointer to external TC buffer
  ///\param matrixR RHS matrix struct
  ///\param matrixOut Output matrix struct
  ///\param offset SIMD batch offset
  template<typename T, size_t N>
  inline
  void AxConjB(T matrixL[N][N],
                T matrixR[N][N],
                T out[N][N]) {
    for (size_t c = 0; c < N; c++) {
      for (size_t r = 0; r < N; r++) {
        auto acc = T();
#ifdef __linux__
#pragma unroll(N)
#endif
        for (size_t i = 0; i < N; i++)
        {
          acc = fmaconj(matrixL[r][i], matrixR[c][i], acc);
        }
        out[r][c] = acc;
      }
    }
  }
///\brief Left Pseudo Inverse: A^+ * A = I
///
/// A^+ = (A^H * A)^-1 * A^H
/// This function computes a single batch left pseudo inverse of matIn
/// at the provided batch offset
///
///\param TC SIMD vector class
///\param IfT IO matrix precision
///\param nRows Number of rows of input matrix
///\param nCols Number of columns of input matrix
///\param matIn Input matrix. Matrix size is [nRows][nCols]
///\param matOut The pseudo inverse result. Matrix size is [nCols][nRows]
template<typename T, size_t N = 16> inline
void matrix_inverse(T in[N][N]) {

  // Compute input matrix covariance A^H * A
  #ifdef PRINT_DEBUG
  matrix_print(in, "input");
  #endif


  // Compute inverse covariance
  // Cholesky Factorisation (L * L^H = A^H * A)
  // Note L is computed in place and stored to lower triangle of covBuf
  Factorise<T, N>(in);
  #ifdef PRINT_DEBUG
  matrix_print(in, "Factorise");
  #endif

  // Inverse Cholesky factor L^-1 (overwrites L)
  InvertFactor<T, N>(in);
  #ifdef PRINT_DEBUG
  matrix_print(in, "InvertFactor");
  #endif

  // Inverse Covariance: L^-1 * L^-H = inv(A^H * A)
  InvFactMult<T, N>(in);

  // Final step of Pseudo Inverse: inv(A^H * A) * A^H
  //AxConjB<T, N>(covBuf, in, out)
  #ifdef PRINT_DEBUG
  matrix_print(in, "output");
  #endif

}
template<> inline FORCE_INLINE
void matrix_inverse<CF16vec16, 1>(CF16vec16 in[1][1]) {
  in[0][0] = max(in[0][0], CF16vec16(1.529e-05, 0));
  in[0][0] = rcp(in[0][0]);
}
#endif
}
