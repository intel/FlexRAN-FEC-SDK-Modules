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

#include "common_typedef_sdk.h"
#include "simd_insts.hpp"
#ifdef _BBLIB_SPR_
#include "matrix.hpp"
#endif
#ifdef _BBLIB_AVX512_
namespace W_SDK {

enum class FO_E {
    disable = 0,
    enable = 1
};

enum class PTRS_E {
    disable = 0,
    enable = 1
};

enum class DATA_DMRS_MUX_E {
    disable = 0,
    enable = 1
};

enum class FP16_E {
    FP16 = 0,
    INT16 = 1
};

enum class INTERP_E {
    disable = 0,
    enable = 1
};

static const auto m512shuffleIQ = I8vec64(
    15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0,
    15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0,
    15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0,
    15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);

/* Threshold of the MMSE gain, to eiminate the error in _mm512_rcp14_ps, which has 10^-5 precision. */

static const __m512 m_gain_threshold = _mm512_set1_ps(0.9999);  // 0.9999
#ifdef _BBLIB_SPR_
static const __m512h m_gain_threshold_f = _mm512_set1_ph(0.9990);  //% 1 - 1/1024 = 0.9990
#endif

const auto use_1st_half = _mm512_set_epi32(
    7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
const auto use_2nd_half = _mm512_set_epi32(
    15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8);

const auto m512LoadGran2 = _mm512_set_epi32(
    30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);

/* all fxp is deliberately designed, to directly get output llr fxp 8S(LLR_FXP)*/
    static const __m512 llr_beta_fxp = _mm512_set1_ps(1<<15); //beta is 16S15
#ifdef _BBLIB_SPR_
    static const CF16vec16 llr_beta_fxp_fp16 = CF16vec16((float16)(1<<15)); //beta is 16S15
    static const CF16vec16 value_one = CF16vec16(1.0);
    static const CF16vec16 max_value = CF16vec16(32759.0);
    // For MMSE-IRC variation: x = inv(H’ *inv(4*Rnn)* H + 0.25*I) * H’*inv(4*Rnn) * Y
    static const float16 rnn_fp16_scale_unit = 0.25; // 2 ^ RNN_FP16_SCALE = 1/rnn_fp16_scale_unit
#endif

// acc_sum HxH
template<size_t N = 16>
inline FORCE_INLINE
F32vec16 acc_sum (Is16vec32 x0[N], Is16vec32 x1[N]) {
    auto sum = mul_add(x0[0], x1[0]);
    #pragma unroll(N - 1)
    for (size_t j = 1; j < N; j++) {
        sum = sum + (mul_add(x0[j], x1[j]));
    }
    return cvt(sum);
}
template<>
inline FORCE_INLINE
F32vec16 acc_sum<4> (Is16vec32 x0[4], Is16vec32 x1[4]) {
    auto a0 = mul_add(x0[0], x1[0]);
    auto a1 = mul_add(x0[1], x1[1]);
    auto a2 = mul_add(x0[2], x1[2]);
    auto a3 = mul_add(x0[3], x1[3]);
    auto sum0 = a0 + a1;
    auto sum1 = a2 + a3;
    auto sum = sum0 + sum1;
    return cvt(sum);
}

template<>
inline FORCE_INLINE
F32vec16 acc_sum<8> (Is16vec32 x0[8], Is16vec32 x1[8]) {
    auto sumf = acc_sum<4> (x0, x1);
    return sumf + acc_sum<4> (&x0[4], &x1[4]);
}
template<>
inline FORCE_INLINE
F32vec16 acc_sum<16> (Is16vec32 x0[16], Is16vec32 x1[16]) {
    auto sumf = acc_sum<8> (x0, x1);
    return sumf + acc_sum<8> (&x0[8], &x1[8]);
}
// acc_sum HxRx
template<size_t N = 16>
inline FORCE_INLINE
F32vec16 acc_sum (Is16vec32 x0[N], Is16vec32 *x1[N]) {
    Is16vec32 temp0 = loadu(x1[0]);
    auto sum = mul_add(x0[0], temp0);
    #pragma unroll(N - 1)
    for (size_t j = 1; j < N; j++) {
        temp0 = loadu(x1[j]);
        sum = sum + (mul_add(x0[j], temp0));
    }
    return cvt(sum);
}
template<>
inline FORCE_INLINE
F32vec16 acc_sum<8> (Is16vec32 x0[8], Is16vec32 *x1[8]) {
    auto sumf = acc_sum<4>(x0, x1);
    return sumf + acc_sum<4>(&x0[4], &x1[4]);
}
template<>
inline FORCE_INLINE
F32vec16 acc_sum<16> (Is16vec32 x0[16], Is16vec32 *x1[16]) {
    auto sumf = acc_sum<8>(x0, x1);
    return sumf + acc_sum<8>(&x0[8], &x1[8]);
}
//real part H hermit mul H
template<size_t N_TX = 16, size_t N_RX = 16>
inline FORCE_INLINE
void HxHReal (F32vec16 ftempARe[N_TX][N_TX],
               F32vec16 ftempBRe[N_TX][N_TX],
               Is16vec32 ChIn[N_TX][N_RX]) {
    if constexpr (N_TX > 1) {
        for (size_t i = 0; i < N_TX; i ++) {
            for (size_t j = i + 1; j < N_TX; j ++) {
                ftempARe[i][j] = acc_sum<N_RX>(ChIn[i], ChIn[j]);
                // Hermite matrix
                ftempARe[j][i] = ftempARe[i][j];
                ftempBRe[i][j] = ftempARe[i][j];
                ftempBRe[j][i] = ftempARe[j][i];
            }
        }
    }
}
//image part H hermit mul H
template<size_t N_TX = 16, size_t N_RX = 16>
inline FORCE_INLINE
void HxHImage (F32vec16 ftempAIm[N_TX][N_TX],
               F32vec16 ftempBIm[N_TX][N_TX],
               Is16vec32 ChImNegRe[N_TX][N_RX],
               Is16vec32 ChIn[N_TX][N_RX]) {
    // calculate the imag part of H' * H
    if constexpr (N_TX > 1) {
        for (size_t i = 0; i < N_TX; i ++) {
            for (size_t j = i + 1; j < N_TX; j ++) {
                ftempAIm[i][j] = acc_sum<N_RX>(ChImNegRe[i], ChIn[j]);
                // Hermite matrix
                ftempAIm[j][i] = F32vec16(0.0) - ftempAIm[i][j];
                ftempBIm[i][j] = ftempAIm[i][j];
                ftempBIm[j][i] = ftempAIm[j][i];
            }
        }
    }
}
template<size_t N_TX = 16, size_t N_RX = 16, typename T = Is16vec32>
inline FORCE_INLINE
void HxH (F32vec16 ftempARe[N_TX][N_TX],
        F32vec16 ftempBRe[N_TX][N_TX],
        T ChIn[N_TX][N_RX],
        F32vec16 ftempAIm[N_TX][N_TX],
        F32vec16 ftempBIm[N_TX][N_TX],
        T ChImNegRe[N_TX][N_RX],
        F32vec16 avxfSigma2) noexcept {
    for (size_t i = 0; i < N_TX; i ++) {
        ftempARe[i][i] = acc_sum<N_RX>(ChIn[i], ChIn[i]);
        ftempAIm[i][i] = F32vec16(0.0);
        // B = A + sigma2
        ftempBRe[i][i] = ftempARe[i][i] + avxfSigma2;
        ftempBIm[i][i] = F32vec16(0.0);
        // #pragma loop_count min(1), max(N_TX - 1)
        if constexpr (N_TX > 1) {
            for (size_t j = i + 1; j < N_TX; j ++) {
                ftempARe[i][j] = acc_sum<N_RX>(ChIn[i], ChIn[j]);
                // Hermite matrix
                ftempARe[j][i] = ftempARe[i][j];
                ftempBRe[i][j] = ftempARe[i][j];
                ftempBRe[j][i] = ftempARe[j][i];

                ftempAIm[i][j] = acc_sum<N_RX>(ChImNegRe[i], ChIn[j]);
                // Hermite matrix
                ftempAIm[j][i] = F32vec16(0.0) - ftempAIm[i][j];
                ftempBIm[i][j] = ftempAIm[i][j];
                ftempBIm[j][i] = ftempAIm[j][i];
            }
        }
    }
}
// tx calculate
template<size_t N_TX = 16>
inline FORCE_INLINE
I8vec64 txCalc (size_t i,
    const F32vec16 &avxShift,
    F32vec16 finvAIm[N_TX][N_TX],
    F32vec16 finvARe[N_TX][N_TX],
    F32vec16 ftempZRe[N_TX],
    F32vec16 ftempZIm[N_TX]) {
    // real part
    auto temp = finvAIm[i][0] * ftempZIm[0];
    temp = fmsub(finvARe[i][0], ftempZRe[0], temp);
    #pragma unroll(N_TX - 1)
    for(size_t j = 1; j < N_TX; j++) {
        const auto temp0 = finvAIm[i][j] * ftempZIm[j];
        temp += fmsub(finvARe[i][j], ftempZRe[j], temp0);
    }
    temp = temp * avxShift;
    const auto reTemp = cvt(temp);

    // imag part
    temp = finvAIm[i][0] * ftempZRe[0];
    temp = fmadd(finvARe[i][0], ftempZIm[0], temp);
    #pragma unroll(N_TX - 1)
    for(size_t j = 1; j < N_TX; j++) {
        const auto temp0 = finvAIm[i][j] * ftempZRe[j];
        temp += fmadd(finvARe[i][j], ftempZIm[j], temp0);
    }
    temp = temp * avxShift;
    const auto imTemp = cvt(temp);
    // combine the real part and imag part
    auto tx = pack_sat(reTemp, imTemp);
    tx = shuffle(tx, m512shuffleIQ);
    return tx;
}
// postSINR calculate
// calculate the gain = real(diag(inv(H'H+sigma2)*H'*H)) = real(diag(invA * A))
template<size_t N_TX = 16>
inline FORCE_INLINE
F32vec16 postSINRCalc (size_t i,
    const F32vec16 &avxGainShift,
    F32vec16 finvAIm[N_TX][N_TX],
    F32vec16 finvARe[N_TX][N_TX],
    F32vec16 ftempAIm[N_TX][N_TX],
    F32vec16 ftempARe[N_TX][N_TX]) {

    // calculate the gain = real(diag(inv(H'H+sigma2)*H'*H)) = real(diag(invA * A))
    auto temp = finvAIm[i][0] * ftempAIm[0][i];
    auto ftempGain = fmsub(finvARe[i][0], ftempARe[0][i], temp);
    #pragma unroll(N_TX-1)
    for (size_t j = 1; j < N_TX; j++) {
        const auto temp = finvAIm[i][j] * ftempAIm[j][i];
        ftempGain += fmsub(finvARe[i][j], ftempARe[j][i], temp);
    }
    ftempGain = ftempGain * avxGainShift;
    // calculate the post SINR = gain ./ (1-gain)
    temp = ftempGain * rcp(F32vec16(1.0) - ftempGain);
    return temp;
}

// gain calculate
template<size_t N_TX = 16>
inline FORCE_INLINE
F32vec16 gainCalc (size_t i,
    const F32vec16 &avxGainShift,
    F32vec16 finvAIm[N_TX][N_TX],
    F32vec16 finvARe[N_TX][N_TX],
    F32vec16 ftempAIm[N_TX][N_TX],
    F32vec16 ftempARe[N_TX][N_TX]) {
    // calculate the gain = real(diag(inv(H'H+sigma2)*H'*H)) = real(diag(invA * A))
    auto temp = finvAIm[i][0] * ftempAIm[0][i];
    auto ftempGain = fmsub(finvARe[i][0], ftempARe[0][i], temp);
    #pragma unroll(N_TX-1)
    for (size_t j = 1; j < N_TX; j++) {
        const auto temp = finvAIm[i][j] * ftempAIm[j][i];
        ftempGain += fmsub(finvARe[i][j], ftempARe[j][i], temp);
    }
    ftempGain = ftempGain * avxGainShift;
    ftempGain = select_low_float(m_gain_threshold, ftempGain);
    return ftempGain;
}

template<size_t N_TX = 16>
inline FORCE_INLINE
F32vec16 gainCalc (size_t i,
    F32vec16 finvAIm[N_TX][N_TX],
    F32vec16 finvARe[N_TX][N_TX],
    F32vec16 ftempAIm[N_TX][N_TX],
    F32vec16 ftempARe[N_TX][N_TX]) {
    // calculate the gain = real(diag(inv(H'H+sigma2)*H'*H)) = real(diag(invA * A))
    // minus 1 due to add I at HTransMulInvRnnMulHPlusI
    ftempARe[i][i] = ftempARe[i][i] - F32vec16(1.0);

    auto temp = finvAIm[i][0] * ftempAIm[0][i];
    auto ftempGain = fmsub(finvARe[i][0], ftempARe[0][i], temp);
    #pragma unroll(N_TX-1)
    for (size_t j = 1; j < N_TX; j++) {
        const auto temp = finvAIm[i][j] * ftempAIm[j][i];
        ftempGain += fmsub(finvARe[i][j], ftempARe[j][i], temp);
    }
    ftempGain = select_low_float(m_gain_threshold, ftempGain);
    return ftempGain;
}

template<size_t N_TX = 16, size_t POST_SINR_FLAG = 0>
inline FORCE_INLINE
void gainCalc (
    F32vec16 gain[N_TX],
    F32vec16 postSINR[N_TX], F32vec16 fsumPostSINR[N_TX],
    F32vec16 &avxGainShift,
    F32vec16 finvAIm[N_TX][N_TX],
    F32vec16 finvARe[N_TX][N_TX],
    F32vec16 ftempAIm[N_TX][N_TX],
    F32vec16 ftempARe[N_TX][N_TX],
    float llr_postsnr_fxp_dynamic) {

    #pragma unroll(N_TX)
    for (size_t i = 0; i < N_TX; i++) {
        // calculate the gain = real(diag(inv(H'H+sigma2)*H'*H)) = real(diag(invA * A))
        auto tempGain = gainCalc (i, avxGainShift, finvAIm, finvARe, ftempAIm, ftempARe);
        gain[i] = _mm512_mul_ps(tempGain, llr_beta_fxp);
        auto temp = rcp(F32vec16(1.0) - tempGain);
        if constexpr (POST_SINR_FLAG == 1) {
            fsumPostSINR[i] += _mm512_mul_ps(tempGain, temp);
        }
        //store ftempGain as early as possbile to improve port utilization
        // ftempPostSINR[i] = temp*llr_postsnr_fxp_dynamic;
        postSINR[i] = _mm512_mul_ps (temp, F32vec16(llr_postsnr_fxp_dynamic));
    }
}

// calc H'*inv(Rnn) (N_TX * N_RX) * (N_RX * N_RX)
template<typename T, typename FLOAT, size_t N_RX = 16, size_t N_TX = 8>
inline FORCE_INLINE
void HTransMulInvRnn2(T chRe[N_RX][N_TX], T chIm[N_RX][N_TX],
                     FLOAT finvRnnRe[N_RX][N_RX], FLOAT finvRnnIm[N_RX][N_RX],
                     FLOAT finvRnnRe2[N_RX][N_RX], FLOAT finvRnnIm2[N_RX][N_RX],
                     T hTansInvRnnRe[N_TX][N_RX], T hTansInvRnnIm[N_TX][N_RX], size_t Mode){
if (Mode == 0){
    for (size_t i = 0; i < N_TX; i ++) {
        for (size_t j = 0; j < N_RX; j ++) {
            T sumRe = T(0.0);
            T sumIm = T(0.0);
            T sumRe1 = T(0.0);
            T sumIm1 = T(0.0);
            #pragma unroll(N_RX>>1)
            for (size_t k = 0; k < N_RX; k +=2) {
                auto rnnRe = T(finvRnnRe[k][j]);
                auto rnnIm = T(finvRnnIm[k][j]);
                auto rnnRe1 = T(finvRnnRe[k+1][j]);
                auto rnnIm1 = T(finvRnnIm[k+1][j]);
                sumRe = fmadd(chRe[k][i], rnnRe, sumRe);
                sumRe = fmadd(chIm[k][i], rnnIm, sumRe);

                sumIm = fmadd(chRe[k][i], rnnIm, sumIm);
                sumIm = fnmadd(chIm[k][i], rnnRe, sumIm);

                sumRe1 = fmadd(chRe[k+1][i], rnnRe1, sumRe1);
                sumRe1 = fmadd(chIm[k+1][i], rnnIm1, sumRe1);

                sumIm1 = fmadd(chRe[k+1][i], rnnIm1, sumIm1);
                sumIm1 = fnmadd(chIm[k+1][i], rnnRe1, sumIm1);
            }
            hTansInvRnnRe[i][j] = sumRe + sumRe1;
            hTansInvRnnIm[i][j] = sumIm + sumIm1;
        }
    }
}
else {
    for (size_t i = 0; i < N_TX; i ++) {
        for (size_t j = 0; j < N_RX; j ++) {
            T sumRe = T(0.0);
            T sumIm = T(0.0);
            T sumRe1 = T(0.0);
            T sumIm1 = T(0.0);
            #pragma unroll(N_RX>>1)
            for (size_t k = 0; k < N_RX; k +=2) {
                auto rnnRe = _mm512_setr_ps(
                    finvRnnRe[k][j], finvRnnRe[k][j], finvRnnRe[k][j], finvRnnRe[k][j],
                    finvRnnRe[k][j], finvRnnRe[k][j], finvRnnRe[k][j], finvRnnRe[k][j],
                    finvRnnRe2[k][j], finvRnnRe2[k][j], finvRnnRe2[k][j], finvRnnRe2[k][j],
                    finvRnnRe2[k][j], finvRnnRe2[k][j], finvRnnRe2[k][j], finvRnnRe2[k][j]);
                auto rnnIm = _mm512_setr_ps(
                    finvRnnIm[k][j], finvRnnIm[k][j], finvRnnIm[k][j], finvRnnIm[k][j],
                    finvRnnIm[k][j], finvRnnIm[k][j], finvRnnIm[k][j], finvRnnIm[k][j],
                    finvRnnIm2[k][j], finvRnnIm2[k][j], finvRnnIm2[k][j], finvRnnIm2[k][j],
                    finvRnnIm2[k][j], finvRnnIm2[k][j], finvRnnIm2[k][j], finvRnnIm2[k][j]);
                auto rnnRe1 = _mm512_setr_ps(
                    finvRnnRe[k+1][j], finvRnnRe[k+1][j], finvRnnRe[k+1][j], finvRnnRe[k+1][j],
                    finvRnnRe[k+1][j], finvRnnRe[k+1][j], finvRnnRe[k+1][j], finvRnnRe[k+1][j],
                    finvRnnRe2[k+1][j], finvRnnRe2[k+1][j], finvRnnRe2[k+1][j], finvRnnRe2[k+1][j],
                    finvRnnRe2[k+1][j], finvRnnRe2[k+1][j], finvRnnRe2[k+1][j], finvRnnRe2[k+1][j]);
                auto rnnIm1 = _mm512_setr_ps(
                    finvRnnIm[k+1][j], finvRnnIm[k+1][j], finvRnnIm[k+1][j], finvRnnIm[k+1][j],
                    finvRnnIm[k+1][j], finvRnnIm[k+1][j], finvRnnIm[k+1][j], finvRnnIm[k+1][j],
                    finvRnnIm2[k+1][j], finvRnnIm2[k+1][j], finvRnnIm2[k+1][j], finvRnnIm2[k+1][j],
                    finvRnnIm2[k+1][j], finvRnnIm2[k+1][j], finvRnnIm2[k+1][j], finvRnnIm2[k+1][j]);

                sumRe = fmadd(chRe[k][i], rnnRe, sumRe);
                sumRe = fmadd(chIm[k][i], rnnIm, sumRe);

                sumIm = fmadd(chRe[k][i], rnnIm, sumIm);
                sumIm = fnmadd(chIm[k][i], rnnRe, sumIm);

                sumRe1 = fmadd(chRe[k+1][i], rnnRe1, sumRe1);
                sumRe1 = fmadd(chIm[k+1][i], rnnIm1, sumRe1);

                sumIm1 = fmadd(chRe[k+1][i], rnnIm1, sumIm1);
                sumIm1 = fnmadd(chIm[k+1][i], rnnRe1, sumIm1);
            }
            hTansInvRnnRe[i][j] = sumRe + sumRe1;
            hTansInvRnnIm[i][j] = sumIm + sumIm1;
        }
    }
}
}


// calc H'*inv(Rnn) (N_TX * N_RX) * (N_RX * N_RX)
template<typename T, typename FLOAT, size_t N_RX = 16, size_t N_TX = 8>
inline FORCE_INLINE
void HTransMulInvRnn(T chRe[N_RX][N_TX], T chIm[N_RX][N_TX],
                     FLOAT finvRnnRe[N_RX][N_RX], FLOAT finvRnnIm[N_RX][N_RX],
                     T hTansInvRnnRe[N_TX][N_RX], T hTansInvRnnIm[N_TX][N_RX]){
    for (size_t i = 0; i < N_TX; i ++) {
        for (size_t j = 0; j < N_RX; j ++) {
            T sumRe0 = T(0.0);
            T sumIm0 = T(0.0);
            T sumRe1 = T(0.0);
            T sumIm1 = T(0.0);
            #pragma unroll(N_RX>>1)
            for (size_t k = 0; k < N_RX; k +=2) {
                auto rnnRe0 = T(finvRnnRe[k][j]);
                auto rnnIm0 = T(finvRnnIm[k][j]);
                auto rnnRe1 = T(finvRnnRe[k+1][j]);
                auto rnnIm1 = T(finvRnnIm[k+1][j]);
                sumRe0 = fmadd(chRe[k][i], rnnRe0, sumRe0);
                sumRe0 = fmadd(chIm[k][i], rnnIm0, sumRe0);

                sumIm0 = fmadd(chRe[k][i], rnnIm0, sumIm0);
                sumIm0 = fnmadd(chIm[k][i], rnnRe0, sumIm0);

                sumRe1 = fmadd(chRe[k+1][i], rnnRe1, sumRe1);
                sumRe1 = fmadd(chIm[k+1][i], rnnIm1, sumRe1);

                sumIm1 = fmadd(chRe[k+1][i], rnnIm1, sumIm1);
                sumIm1 = fnmadd(chIm[k+1][i], rnnRe1, sumIm1);
            }
            hTansInvRnnRe[i][j] = sumRe0 + sumRe1;
            hTansInvRnnIm[i][j] = sumIm0 + sumIm1;
        }
    }
}

template<>
inline FORCE_INLINE
void HTransMulInvRnn<F32vec16, float, 1, 1>(F32vec16 chRe[1][1], F32vec16 chIm[1][1],
                     float finvRnnRe[1][1], float finvRnnIm[1][1],
                     F32vec16 hTansInvRnnRe[1][1], F32vec16 hTansInvRnnIm[1][1]){
    F32vec16 sumRe0 = F32vec16(0.0);
    F32vec16 sumIm0 = F32vec16(0.0);

    auto rnnRe0 = F32vec16(finvRnnRe[0][0]);
    sumRe0 = fmadd(chRe[0][0], rnnRe0, sumRe0);
    sumIm0 = fnmadd(chIm[0][0], rnnRe0, sumIm0);

    hTansInvRnnRe[0][0] = sumRe0;
    hTansInvRnnIm[0][0] = sumIm0;
}

#define F32VEC_COMPLEX_MUL(a_re, a_im, b_re, b_im, sum_re, sum_im)\
                sum_re = fmadd(a_re, b_re, sum_re);\
                sum_re = fnmadd(a_im, b_im, sum_re);\
                sum_im = fmadd(a_im, b_re, sum_im);\
                sum_im = fmadd(a_re, b_im, sum_im);

#define F32VEC_COMPLEX_MUL_REAL(a_re, a_im, b_re, b_im, sum_re)\
                sum_re = fmadd(a_re, b_re, sum_re);\
                sum_re = fnmadd(a_im, b_im, sum_re);


// H' * inv(Rnn) * H + I (N_TX * N_RX) * (N_RX * N_TX)
template<typename T, size_t N_RX = 16, size_t N_TX = 8>
inline FORCE_INLINE
void HTransMulInvRnnMulHPlusI(
    T hTansInvRnnRe[N_TX][N_RX], T hTansInvRnnIm[N_TX][N_RX],
    T chRe[N_RX][N_TX], T chIm[N_RX][N_TX],
    T ftempARe[N_TX][N_TX], T ftempAIm[N_TX][N_TX]) {
    for (size_t i = 0; i < N_TX; i ++) {
        auto sumRe0 = T(0.0);
        auto sumRe1 = T(0.0);
        #pragma unroll(N_RX>>1)
        for (size_t k = 0; k < N_RX; k +=2) {
            F32VEC_COMPLEX_MUL_REAL(hTansInvRnnRe[i][k], hTansInvRnnIm[i][k], chRe[k][i], chIm[k][i], sumRe0)
            F32VEC_COMPLEX_MUL_REAL(hTansInvRnnRe[i][k+1], hTansInvRnnIm[i][k+1], chRe[k+1][i], chIm[k+1][i], sumRe1)
        }
        ftempARe[i][i] = sumRe0 + sumRe1  + T(1.0);
        ftempAIm[i][i] = T(0.0);

        if constexpr (N_TX > 1) {
            for (size_t j = i + 1; j < N_TX; j ++) {
                auto sumRe0 = T(0.0);
                auto sumIm0 = T(0.0);
                auto sumRe1 = T(0.0);
                auto sumIm1 = T(0.0);
                #pragma unroll(N_RX>>1)
                for (size_t k = 0; k < N_RX; k +=2) {
                    F32VEC_COMPLEX_MUL(hTansInvRnnRe[i][k], hTansInvRnnIm[i][k], chRe[k][j], chIm[k][j], sumRe0, sumIm0)
                    F32VEC_COMPLEX_MUL(hTansInvRnnRe[i][k+1], hTansInvRnnIm[i][k+1], chRe[k+1][j], chIm[k+1][j], sumRe1, sumIm1)
                }
                ftempARe[i][j] = sumRe0 + sumRe1;
                ftempAIm[i][j] = sumIm0 + sumIm1;

                //ftempARe[j][i] = sumRe0 + sumRe1;
                //ftempAIm[j][i] = T(0.0) - (sumIm0 + sumIm1);
                ftempARe[j][i] = ftempARe[i][j];
                ftempAIm[j][i] = T(0.0) - ftempAIm[i][j];
            }
        }
    }
}

template<>
inline FORCE_INLINE
void HTransMulInvRnnMulHPlusI<F32vec16, 1, 1>(
    F32vec16 hTansInvRnnRe[1][1], F32vec16 hTansInvRnnIm[1][1],
    F32vec16 chRe[1][1], F32vec16 chIm[1][1],
    F32vec16 ftempARe[1][1], F32vec16 ftempAIm[1][1]) {
    auto sumRe0 = F32vec16(0.0);
    F32VEC_COMPLEX_MUL_REAL(hTansInvRnnRe[0][0], hTansInvRnnIm[0][0], chRe[0][0], chIm[0][0], sumRe0)

    ftempARe[0][0] = sumRe0 + F32vec16(1.0);
    ftempAIm[0][0] = F32vec16(0.0);
}
// inverse(H' * inv(Rnn) * H + I) * H' * inv(Rnn) (N_TX * N_TX) * (N_TX * N_RX)
template<typename T, size_t N_RX = 16, size_t N_TX = 8>
inline FORCE_INLINE
void invMatrixMulHTransMulInvRnn2(
    T finvARe[N_TX][N_TX], T finvAIm[N_TX][N_TX],
    T hTansInvRnnRe[N_TX][N_RX], T hTansInvRnnIm[N_TX][N_RX],
    T invMatrixHTansInvRnnRe[N_TX][N_RX], T invMatrixHTansInvRnnIm[N_TX][N_RX],
    T invMatrixHTansInvRnnRe2[N_TX][N_RX], T invMatrixHTansInvRnnIm2[N_TX][N_RX]) {

    if (N_TX % 2){
        for (size_t i = 0; i < N_TX; i ++) {
            for (size_t j = 0; j < N_RX; j ++) {
                auto sumRe0 = T(0.0);
                auto sumIm0 = T(0.0);
                auto sumRe1 = T(0.0);
                auto sumIm1 = T(0.0);
                #pragma unroll(N_TX>>1)
                for (size_t k = 0; k < N_TX-1; k +=2) {
                    F32VEC_COMPLEX_MUL(finvARe[i][k], finvAIm[i][k], hTansInvRnnRe[k][j], hTansInvRnnIm[k][j], sumRe0, sumIm0)
                    F32VEC_COMPLEX_MUL(finvARe[i][k+1], finvAIm[i][k+1], hTansInvRnnRe[k+1][j], hTansInvRnnIm[k+1][j], sumRe1, sumIm1)

                }
                F32VEC_COMPLEX_MUL(finvARe[i][N_TX-1], finvAIm[i][N_TX-1], hTansInvRnnRe[N_TX-1][j], hTansInvRnnIm[N_TX-1][j], sumRe0, sumIm0)
                invMatrixHTansInvRnnRe[i][j] = sumRe0 + sumRe1;
                invMatrixHTansInvRnnIm[i][j] = sumIm0 + sumIm1;


                invMatrixHTansInvRnnRe2[i][j] = _mm512_permutex2var_epi32(invMatrixHTansInvRnnRe[i][j], use_2nd_half, invMatrixHTansInvRnnRe[i][j]);
                invMatrixHTansInvRnnRe[i][j] = _mm512_permutex2var_epi32(invMatrixHTansInvRnnRe[i][j], use_1st_half, invMatrixHTansInvRnnRe[i][j]);
                invMatrixHTansInvRnnIm2[i][j] = _mm512_permutex2var_epi32(invMatrixHTansInvRnnIm[i][j], use_2nd_half, invMatrixHTansInvRnnIm[i][j]);
                invMatrixHTansInvRnnIm[i][j] = _mm512_permutex2var_epi32(invMatrixHTansInvRnnIm[i][j], use_1st_half, invMatrixHTansInvRnnIm[i][j]);
            }
        }
    }
    else{
        for (size_t i = 0; i < N_TX; i ++) {
            for (size_t j = 0; j < N_RX; j ++) {
                auto sumRe0 = T(0.0);
                auto sumIm0 = T(0.0);
                auto sumRe1 = T(0.0);
                auto sumIm1 = T(0.0);
                #pragma unroll(N_TX>>1)
                for (size_t k = 0; k < N_TX; k +=2) {
                    F32VEC_COMPLEX_MUL(finvARe[i][k], finvAIm[i][k], hTansInvRnnRe[k][j], hTansInvRnnIm[k][j], sumRe0, sumIm0)
                    F32VEC_COMPLEX_MUL(finvARe[i][k+1], finvAIm[i][k+1], hTansInvRnnRe[k+1][j], hTansInvRnnIm[k+1][j], sumRe1, sumIm1)

                }
                invMatrixHTansInvRnnRe[i][j] = sumRe0 + sumRe1;
                invMatrixHTansInvRnnIm[i][j] = sumIm0 + sumIm1;


                invMatrixHTansInvRnnRe2[i][j] = _mm512_permutex2var_epi32(invMatrixHTansInvRnnRe[i][j], use_2nd_half, invMatrixHTansInvRnnRe[i][j]);
                invMatrixHTansInvRnnRe[i][j] = _mm512_permutex2var_epi32(invMatrixHTansInvRnnRe[i][j], use_1st_half, invMatrixHTansInvRnnRe[i][j]);
                invMatrixHTansInvRnnIm2[i][j] = _mm512_permutex2var_epi32(invMatrixHTansInvRnnIm[i][j], use_2nd_half, invMatrixHTansInvRnnIm[i][j]);
                invMatrixHTansInvRnnIm[i][j] = _mm512_permutex2var_epi32(invMatrixHTansInvRnnIm[i][j], use_1st_half, invMatrixHTansInvRnnIm[i][j]);
            }
        }
    }
}

// inverse(H' * inv(Rnn) * H + I) * H' * inv(Rnn) (N_TX * N_TX) * (N_TX * N_RX)
template<typename T, size_t N_RX = 16, size_t N_TX = 8>
inline FORCE_INLINE
void invMatrixMulHTransMulInvRnn(
    T finvARe[N_TX][N_TX], T finvAIm[N_TX][N_TX],
    T hTansInvRnnRe[N_TX][N_RX], T hTansInvRnnIm[N_TX][N_RX],
    T invMatrixHTansInvRnnRe[N_TX][N_RX], T invMatrixHTansInvRnnIm[N_TX][N_RX]) {

    if (N_TX % 2){
        for (size_t i = 0; i < N_TX; i ++) {
            for (size_t j = 0; j < N_RX; j ++) {
                auto sumRe0 = T(0.0);
                auto sumIm0 = T(0.0);
                auto sumRe1 = T(0.0);
                auto sumIm1 = T(0.0);
                #pragma unroll(N_TX>>1)
                for (size_t k = 0; k < N_TX-1; k +=2) {
                    F32VEC_COMPLEX_MUL(finvARe[i][k], finvAIm[i][k], hTansInvRnnRe[k][j], hTansInvRnnIm[k][j], sumRe0, sumIm0)
                    F32VEC_COMPLEX_MUL(finvARe[i][k+1], finvAIm[i][k+1], hTansInvRnnRe[k+1][j], hTansInvRnnIm[k+1][j], sumRe1, sumIm1)

                }
                F32VEC_COMPLEX_MUL(finvARe[i][N_TX-1], finvAIm[i][N_TX-1], hTansInvRnnRe[N_TX-1][j], hTansInvRnnIm[N_TX-1][j], sumRe0, sumIm0)
                invMatrixHTansInvRnnRe[i][j] = sumRe0 + sumRe1;
                invMatrixHTansInvRnnIm[i][j] = sumIm0 + sumIm1;
            }
        }
    }
    else{
        for (size_t i = 0; i < N_TX; i ++) {
            for (size_t j = 0; j < N_RX; j ++) {
                auto sumRe0 = T(0.0);
                auto sumIm0 = T(0.0);
                auto sumRe1 = T(0.0);
                auto sumIm1 = T(0.0);
                #pragma unroll(N_TX>>1)
                for (size_t k = 0; k < N_TX; k +=2) {
                    F32VEC_COMPLEX_MUL(finvARe[i][k], finvAIm[i][k], hTansInvRnnRe[k][j], hTansInvRnnIm[k][j], sumRe0, sumIm0)
                    F32VEC_COMPLEX_MUL(finvARe[i][k+1], finvAIm[i][k+1], hTansInvRnnRe[k+1][j], hTansInvRnnIm[k+1][j], sumRe1, sumIm1)

                }
                invMatrixHTansInvRnnRe[i][j] = sumRe0 + sumRe1;
                invMatrixHTansInvRnnIm[i][j] = sumIm0 + sumIm1;
            }
        }
    }
}

// inverse(H' * invRnn * H + I) * H' * invRnn * Y
template<typename T, size_t N_RX = 16, size_t N_TX = 8, FO_E foFlag = FO_E::disable>
inline FORCE_INLINE
void txCalc(
    T invMatrixMulHTansInvRnnRe[N_TX][N_RX], T invMatrixMulHTansInvRnnIm[N_TX][N_RX],
    T yRe[N_RX], T yIm[N_RX],
    T avxShift, Is16vec32 txBuf[N_TX], CI16vec16 *FoTable = NULL, bool chUpdateFLag = 1) {

    if constexpr (N_RX == 1) {
        auto sumRe0 = F32vec16(0.0);
        auto sumIm0 = F32vec16(0.0);

        F32VEC_COMPLEX_MUL(invMatrixMulHTansInvRnnRe[0][0], invMatrixMulHTansInvRnnIm[0][0], yRe[0], yIm[0], sumRe0, sumIm0)

        sumRe0 = sumRe0 * avxShift;
        sumIm0 = sumIm0 * avxShift;
        const auto re = cvt(sumRe0);
        const auto im = cvt(sumIm0);
        // combine the real part and imag part
        CI16vec16 tx = _mm512_packs_epi32(re, im);
        tx = _mm512_shuffle_epi8(tx, m512shuffleIQ);

        if constexpr (foFlag == FO_E::enable) {
            tx = fmulconj(tx, FoTable[0]);
        }

        txBuf[0] = tx;
    } else {
        for (size_t i = 0; i < N_TX; i ++) {
            auto sumRe0 = T(0.0);
            auto sumIm0 = T(0.0);
            auto sumRe1 = T(0.0);
            auto sumIm1 = T(0.0);
            #pragma unroll(N_RX>>1)
            for (size_t k = 0; k < N_RX; k +=2) {
                F32VEC_COMPLEX_MUL(invMatrixMulHTansInvRnnRe[i][k], invMatrixMulHTansInvRnnIm[i][k], yRe[k], yIm[k], sumRe0, sumIm0)
                F32VEC_COMPLEX_MUL(invMatrixMulHTansInvRnnRe[i][k+1], invMatrixMulHTansInvRnnIm[i][k+1], yRe[k+1], yIm[k+1], sumRe1, sumIm1)

            }
            sumRe0 = (sumRe0 + sumRe1) * avxShift;
            sumIm0 = (sumIm0 + sumIm1) * avxShift;
            const auto re = cvt(sumRe0);
            const auto im = cvt(sumIm0);
            // combine the real part and imag part
            auto tx = _mm512_packs_epi32(re, im);
            tx = _mm512_shuffle_epi8(tx, m512shuffleIQ);

            if constexpr (foFlag == FO_E::enable) {
                tx = fmulconj(tx, FoTable[i]);
            }

            txBuf[i] = tx;
        }
    }
}

#undef F32VEC_COMPLEX_MUL
#undef F32VEC_COMPLEX_MUL_REAL

#ifdef _BBLIB_SPR_
// H'xH
template<typename T, size_t N_TX = 16, size_t N_RX = 16>
inline FORCE_INLINE
void HxH (T ftempARe[N_TX][N_TX],
          T ftempBRe[N_TX][N_TX],
          T ChIn[N_TX][N_RX],
          T avxfSigma2) {
    //calculate the real part of H' * H
    for (int32_t i = N_TX - 1; i >= 0; i --) {
        ftempARe[i][i] = dotCReal<T, N_RX>(ChIn[i]);
        ftempBRe[i][i] = ftempARe[i][i] + avxfSigma2;
        //theta = max(ftempARe[i][i], theta);
        if constexpr (N_TX > 1) {
            for (size_t j = i + 1; j < N_TX; j ++) {
                ftempARe[i][j] = dotC<T, N_RX>(ChIn[j], ChIn[i]);
                // Hermite matrix
                ftempARe[j][i] = negImag(ftempARe[i][j]);
                ftempBRe[i][j] = ftempARe[i][j];
                ftempBRe[j][i] = ftempARe[j][i];
            }
        }
    }
    #if 0
    theta = theta * static_cast<float16>(0.001953125); //(1.0/512.0 = 0.001953125);
    for (size_t i = 0; i < N_TX; i ++) {
        ftempBRe[i][i] = ftempBRe[i][i] + theta;
    }
    #endif
}
// acc_sum HxRx
template<typename T, size_t N = 16>
inline FORCE_INLINE
T acc_sum (T x0[N], T *x1[N]) {
    T sum0 = T();
    T sum1 = T();
    #pragma unroll(N >> 1)
    for (size_t j = 0; j < N; j = j + 2) {
        auto temp0 = loadu(x1[j]);
        sum0 = fmaconj(temp0, x0[j], sum0);
        auto temp1 = loadu(x1[j + 1]);
        sum1 = fmaconj(temp1, x0[j + 1], sum1);
    }
    return sum0 + sum1;
}

template<>
inline FORCE_INLINE
CF16vec16 acc_sum<CF16vec16, 1> (CF16vec16 x0[1], CF16vec16 *x1[1]) {
    auto temp0 = loadu(x1[0]);
    auto sum0 = fmulconj(temp0, x0[0]);
    return sum0;
}


template<size_t N = 16>
inline FORCE_INLINE
CF16vec16 acc_sum (CF16vec16 x0[N], CI16vec16 *x1[N], F16vec32 x1Scale) {
    CF16vec16 sum0 = CF16vec16();
    CF16vec16 sum1 = CF16vec16();
    #pragma unroll(N >> 1)
    for (size_t j = 0; j < N; j = j + 2) {
        CF16vec16 temp0 = _mm512_cvtepi16_ph(loadu(x1[j]));
        sum0 = _mm512_fcmadd_pch(_mm512_mul_ph(temp0 ,x1Scale), x0[j], sum0);
        CF16vec16 temp1 = _mm512_cvtepi16_ph(loadu(x1[j + 1]));
        sum1 = _mm512_fcmadd_pch(_mm512_mul_ph(temp1 ,x1Scale), x0[j + 1], sum1);
    }
    return sum0 + sum1;
}

template<>
inline FORCE_INLINE
CF16vec16 acc_sum<1> (CF16vec16 x0[1], CI16vec16 *x1[1], F16vec32 x1Scale) {
    CF16vec16 temp0 = _mm512_cvtepi16_ph(loadu(x1[0]));
    CF16vec16 sum0 = fmulconj(_mm512_mul_ph(temp0, x1Scale), x0[0]);
    return sum0;
}

template<typename T, size_t N = 16>
inline FORCE_INLINE
T acc_sum (T x0[N], T x1[N]) {
    T sum0 = T();
    T sum1 = T();
    #pragma unroll(N >> 1)
    for (size_t j = 0; j < N; j = j + 2) {
        sum0 = fmaconj(x1[j], x0[j], sum0);
        sum1 = fmaconj(x1[j + 1], x0[j + 1], sum1);
    }
    return sum0 + sum1;
}

template<>
inline FORCE_INLINE
CF16vec16 acc_sum<CF16vec16, 1> (CF16vec16 x0[1], CF16vec16 x1[1]) {
    auto sum0 = fmulconj(x1[0], x0[0]);
    return sum0;
}

template<size_t N = 16>
inline FORCE_INLINE
CF16vec16 acc_sum (CF16vec16 x0[N], CI16vec16 x1[N], F16vec32 x1Scale) {
    CF16vec16 sum0 = CF16vec16();
    CF16vec16 sum1 = CF16vec16();

    #pragma unroll(N >> 1)
    for (size_t j = 0; j < N; j = j + 2) {
        CF16vec16 temp0 = _mm512_cvtepi16_ph(x1[j]);
        sum0 = _mm512_fcmadd_pch(_mm512_mul_ph(temp0 ,x1Scale), x0[j], sum0);
        CF16vec16 temp1 = _mm512_cvtepi16_ph(x1[j + 1]);
        sum1 = _mm512_fcmadd_pch(_mm512_mul_ph(temp1 ,x1Scale), x0[j + 1], sum1);
    }
    return sum0 + sum1;
}

template<>
inline FORCE_INLINE
CF16vec16 acc_sum<1> (CF16vec16 x0[1], CI16vec16 x1[1], F16vec32 x1Scale) {
    CF16vec16 temp0 = _mm512_cvtepi16_ph(x1[0]);
    CF16vec16 sum0 = fmulconj(_mm512_mul_ph(temp0 ,x1Scale), x0[0]);
    return sum0;
}

// postSINR calculate
// calculate the gain = real(diag(inv(H'H+sigma2)*H'*H)) = real(diag(invA * A))
template<typename T, size_t N_TX = 16>
inline FORCE_INLINE
T postSINRCalc (size_t i,
    T finvARe[N_TX][N_TX],
    T ftempARe[N_TX][N_TX]) {

    // calculate the gain = real(diag(inv(H'H+sigma2)*H'*H)) = real(diag(invA * A))
    auto gain0 = T();
    auto gain1 = T();
    // auto gain = fmul(finvARe[i][0], ftempARe[0][i]);
    // #pragma unroll(N_TX >> 1)
    #pragma unroll(N_TX >> 1)
    for (size_t j = 0; j < N_TX; j = j + 2) {
        gain0 = _mm512_fmadd_ph(finvARe[i][j], ftempARe[j][i], gain0);
        gain1 = _mm512_fmadd_ph(finvARe[i][j + 1], ftempARe[j + 1][i], gain1);
    }
    auto gain = min(gain0 + gain1, m_gain_threshold_f); //% 1 - 1/1024 = 0.9990
    gain = subRealImag(gain);
    // calculate the post SINR = gain ./ (1-gain)
    T temp = T(1.0) - gain;
    gain = gain * rcp(temp);
    return gain;
}

template<>
inline FORCE_INLINE
CF16vec16 postSINRCalc <CF16vec16, 1> (size_t i,
    CF16vec16 finvARe[1][1],
    CF16vec16 ftempARe[1][1]) {
    // calculate the gain = real(diag(inv(H'H+sigma2)*H'*H)) = real(diag(invA * A))
    CF16vec16 gain0 = _mm512_mul_ph(finvARe[i][0], ftempARe[0][i]);
    gain0 = subRealImag(gain0);

    // calculate the post SINR = gain ./ (1-gain)
    auto gain = min(gain0, m_gain_threshold_f); //% 1 - 1/1024 = 0.9990
    CF16vec16 temp = CF16vec16(1.0) - gain;
    gain = gain * rcp(temp);
    return gain;
}

// gainCalc calculate
// calculate the gain = real(diag(inv(H'H+sigma2)*H'*H)) = real(diag(invA * A))
template<typename T, size_t N_TX = 16>
inline FORCE_INLINE
T gainCalc (size_t i,
    T finvARe[N_TX][N_TX],
    T ftempARe[N_TX][N_TX]) {

    // calculate the gain = real(diag(inv(H'H+sigma2)*H'*H)) = real(diag(invA * A))
    auto gain0 = T();
    auto gain1 = T();

    if constexpr (N_TX % 2)
    {
        // auto gain = fmul(finvARe[i][0], ftempARe[0][i]);
        #pragma unroll(N_TX >> 1)
        for (size_t j = 0; j < N_TX-1; j += 2) {
            gain0 = fma(finvARe[i][j], ftempARe[j][i], gain0);
            gain1 = fma(finvARe[i][j + 1], ftempARe[j + 1][i], gain1);
        }
        gain0 = fma(finvARe[i][N_TX-1], ftempARe[N_TX-1][i], gain0);
    }
    else
    {
        // auto gain = fmul(finvARe[i][0], ftempARe[0][i]);
        #pragma unroll(N_TX >> 1)
        for (size_t j = 0; j < N_TX; j += 2) {
            gain0 = fma(finvARe[i][j], ftempARe[j][i], gain0);
            gain1 = fma(finvARe[i][j + 1], ftempARe[j + 1][i], gain1);
        }
    }
    auto gain = min(gain0 + gain1, CF16vec16((float16)0.9990 )); //% 1 - 1/1024 = 0.9990

    // calculate the post SINR = gain ./ (1-gain)
    return gain;
}

template<>
inline FORCE_INLINE
CF16vec16 gainCalc <CF16vec16, 1> (size_t i,
    CF16vec16 finvARe[1][1],
    CF16vec16 ftempARe[1][1]) {
    // calculate the gain = real(diag(inv(H'H+sigma2)*H'*H)) = real(diag(invA * A))
    auto gain0 = fmul(finvARe[i][0], ftempARe[0][i]);

    // calculate the post SINR = gain ./ (1-gain)
    auto gain = min(gain0, CF16vec16((float16)0.9990 )); //% 1 - 1/1024 = 0.9990

    return gain;
}

template<size_t N_TX = 16, size_t POST_SINR_FLAG = 0>
inline FORCE_INLINE
void gainCalc (CF16vec16 gain[N_TX],
    CF16vec16 postSINR[N_TX], F32vec16 fsumPostSINR[N_TX],
    CF16vec16 ftempBRe[N_TX][N_TX],
    CF16vec16 ftempARe[N_TX][N_TX],
    float &llr_postsnr_fxp_dynamic) {
    #pragma unroll(N_TX)
    for (size_t i = 0; i < N_TX; i++) {
        CF16vec16 gainTmp = gainCalc(i,  ftempBRe, ftempARe);
        gainTmp =  duplicateReal(gainTmp);
        CF16vec16 postSinrTmp = _mm512_rcp_ph(value_one - gainTmp);
        if constexpr (POST_SINR_FLAG == 1) {
            CF16vec16 postSinrOut = _mm512_mul_ph(gainTmp, postSinrTmp);
            fsumPostSINR[i] += _mm512_cvtph_ps(postSinrOut.real());
        }
        gain[i] = _mm512_mul_ph(gainTmp, llr_beta_fxp_fp16);
        postSinrTmp = _mm512_mul_ph (postSinrTmp, CF16vec16((float16)llr_postsnr_fxp_dynamic));
        postSINR[i] = min(postSinrTmp, CF16vec16((float16)32759.0));
    }
}

// calc H'*inv(Rnn) (N_TX * N_RX) * (N_RX * N_RX)
template<typename T, typename invType, size_t N_RX = 16, size_t N_TX = 8>
inline FORCE_INLINE
void HTransMulInvRnn2(CF16vec16 ChIn[N_RX][N_TX], invType finvRnnC[N_RX][N_RX], invType finvRnnC2[N_RX][N_RX],
                     CF16vec16 hTansInvRnnRe[N_TX][N_RX], size_t Mode){
if (Mode == 0)
{
    // #pragma unroll(N_TX)
    for (size_t i = 0; i < N_TX; i ++) {
        // #pragma unroll(N_RX)
        for (size_t j = 0; j < N_RX; j ++) {
            T sum0 = T();
            T sum1 = T();
            #pragma unroll(N_RX>>1)
            for (size_t k = 0; k < N_RX; k += 2) {
                T rnnRe0 = _mm512_castsi512_ph(_mm512_set1_ps(finvRnnC[k][j]));
                T rnnRe1 = _mm512_castsi512_ph(_mm512_set1_ps(finvRnnC[k+1][j]));
                sum0 = fmaconj(rnnRe0, ChIn[k][i], sum0);
                sum1 = fmaconj(rnnRe1, ChIn[k+1][i], sum1);
            }
            hTansInvRnnRe[i][j] = sum0 + sum1;
        }
    }
}
else
{
    // #pragma unroll(N_TX)
    for (size_t i = 0; i < N_TX; i ++) {
        // #pragma unroll(N_RX)
        for (size_t j = 0; j < N_RX; j ++) {
            T sum0 = T(0.0);
            T sum1 = T(0.0);
            #pragma unroll(N_RX>>1)
            for (size_t k = 0; k < N_RX; k +=2) {
                T rnnRe0 = _mm512_castsi512_ph(_mm512_setr_ps(
                    finvRnnC[k][j], finvRnnC[k][j], finvRnnC[k][j], finvRnnC[k][j],
                    finvRnnC[k][j], finvRnnC[k][j], finvRnnC[k][j], finvRnnC[k][j],
                    finvRnnC2[k][j], finvRnnC2[k][j], finvRnnC2[k][j], finvRnnC2[k][j],
                    finvRnnC2[k][j], finvRnnC2[k][j], finvRnnC2[k][j], finvRnnC2[k][j]));
                T rnnRe1 = _mm512_castsi512_ph(_mm512_setr_ps(
                    finvRnnC[k+1][j], finvRnnC[k+1][j], finvRnnC[k+1][j], finvRnnC[k+1][j],
                    finvRnnC[k+1][j], finvRnnC[k+1][j], finvRnnC[k+1][j], finvRnnC[k+1][j],
                    finvRnnC2[k+1][j], finvRnnC2[k+1][j], finvRnnC2[k+1][j], finvRnnC2[k+1][j],
                    finvRnnC2[k+1][j], finvRnnC2[k+1][j], finvRnnC2[k+1][j], finvRnnC2[k+1][j]));

                sum0 = fmaconj(rnnRe0, ChIn[k][i], sum0);
                sum1 = fmaconj(rnnRe1, ChIn[k+1][i], sum1);
            }
            hTansInvRnnRe[i][j] = sum0 + sum1;
        }
    }
}
}

// calc H'*inv(Rnn) (N_TX * N_RX) * (N_RX * N_RX)
template<typename T, typename invType, size_t N_RX = 16, size_t N_TX = 8>
inline FORCE_INLINE
void HTransMulInvRnn(CF16vec16 ChIn[N_RX][N_TX], invType finvRnnC[N_RX][N_RX],
                     CF16vec16 hTansInvRnnRe[N_TX][N_RX]){
    // #pragma unroll(N_TX)
    for (size_t i = 0; i < N_TX; i ++) {
        for (size_t j = 0; j < N_RX; j ++) {
            T sum0 = T();
            T sum1 = T();
            #pragma unroll(N_RX>>1)
            for (size_t k = 0; k < N_RX; k += 2) {
                T rnnRe0 = _mm512_castsi512_ph(_mm512_set1_ps(finvRnnC[k][j]));
                T rnnRe1 = _mm512_castsi512_ph(_mm512_set1_ps(finvRnnC[k+1][j]));
                sum0 = fmaconj(rnnRe0, ChIn[k][i], sum0);
                sum1 = fmaconj(rnnRe1, ChIn[k+1][i], sum1);
            }
            hTansInvRnnRe[i][j] = sum0 + sum1;
        }
    }
}

template<>
inline FORCE_INLINE
void HTransMulInvRnn<CF16vec16, float, 1, 1>(CF16vec16 ChIn[1][1], float finvRnnC[1][1],
                     CF16vec16 hTansInvRnnRe[1][1]){
    auto rnnRe = _mm512_castsi512_ph(_mm512_set1_ps(finvRnnC[0][0]));
    hTansInvRnnRe[0][0] = fmulconj(rnnRe, ChIn[0][0]);
}

// H' * inv(4 * Rnn) * H + 0.25 * I (N_TX * N_RX) * (N_RX * N_TX)
template<typename T, size_t N_RX = 16, size_t N_TX = 8>
inline FORCE_INLINE
void HTransMulInvRnnMulHPlusI(
    T hTansInvRnnRe[N_TX][N_RX], T chRe[N_RX][N_TX], T ftempARe[N_TX][N_TX], T ftempBRe[N_TX][N_TX]) {
    // #pragma unroll(N_TX)
    for (size_t i = 0; i < N_TX; i ++) {
        T sum0 = T();
        T sum1 = T();
        #pragma unroll(N_RX>>1)
        for (size_t k = 0; k < N_RX; k += 2) {
            sum0 = fma(hTansInvRnnRe[i][k], chRe[k][i], sum0);
            sum1 = fma(hTansInvRnnRe[i][k+1], chRe[k+1][i], sum1);
        }
        ftempARe[i][i] = sum0 + sum1;
        ftempBRe[i][i] = ftempARe[i][i] + T(rnn_fp16_scale_unit, 0.0);

        if constexpr (N_TX > 1) {
            for (size_t j = i + 1; j < N_TX; j ++) {
                T sum0 = T();
                T sum1 = T();
                #pragma unroll(N_RX>>1)
                for (size_t k = 0; k < N_RX; k += 2) {
                    sum0 = fma(hTansInvRnnRe[i][k], chRe[k][j], sum0);
                    sum1 = fma(hTansInvRnnRe[i][k+1], chRe[k+1][j], sum1);
                }
                ftempARe[i][j] = sum0 + sum1;
                ftempARe[j][i] = negImag(ftempARe[i][j]);
                ftempBRe[i][j] = ftempARe[i][j];
                ftempBRe[j][i] = ftempARe[j][i];
            }
        }
    }
}

template<>
inline FORCE_INLINE
void HTransMulInvRnnMulHPlusI<CF16vec16, 1, 1>(
    CF16vec16 hTansInvRnnRe[1][1], CF16vec16 chRe[1][1], CF16vec16 ftempARe[1][1], CF16vec16 ftempBRe[1][1]) {
    ftempARe[0][0] = fmul(hTansInvRnnRe[0][0], chRe[0][0]);
    ftempBRe[0][0] = ftempARe[0][0] + CF16vec16(rnn_fp16_scale_unit, 0.0);
}

template<typename T, size_t N_RX = 16, size_t N_TX = 8>
inline FORCE_INLINE
void invMatrixMulHTransMulInvRnn(
    T finvARe[N_TX][N_TX], T hTansInvRnnRe[N_TX][N_RX], T invMatrixHTansInvRnnRe[N_TX][N_RX]) {
    if(N_TX % 2){
        // #pragma unroll(N_TX)
        for (size_t i = 0; i < N_TX; i ++) {
            #pragma unroll(N_RX)
            for (size_t j = 0; j < N_RX; j ++) {
                T sum0 = T();
                T sum1 = T();
                #pragma unroll(N_TX>>1)
                for (size_t k = 0; k < N_TX-1; k += 2) {
                    sum0 = fma(finvARe[i][k], hTansInvRnnRe[k][j], sum0);
                    sum1 = fma(finvARe[i][k+1], hTansInvRnnRe[k+1][j], sum1);
                }
                sum0 = fma(finvARe[i][N_TX-1], hTansInvRnnRe[N_TX-1][j], sum0);
                invMatrixHTansInvRnnRe[i][j] = sum0 + sum1;
            }
        }
    }
    else{
        // #pragma unroll(N_TX)
        for (size_t i = 0; i < N_TX; i ++) {
            #pragma unroll(N_RX)
            for (size_t j = 0; j < N_RX; j ++) {
                T sum0 = T();
                T sum1 = T();
                #pragma unroll(N_TX>>1)
                for (size_t k = 0; k < N_TX; k += 2) {
                    sum0 = fma(finvARe[i][k], hTansInvRnnRe[k][j], sum0);
                    sum1 = fma(finvARe[i][k+1], hTansInvRnnRe[k+1][j], sum1);
                }
                invMatrixHTansInvRnnRe[i][j] = sum0 + sum1;
            }
        }
    }
}

template<typename T, size_t N_RX = 16, size_t N_TX = 8>
inline FORCE_INLINE
void invMatrixMulHTransMulInvRnn2(
    T finvARe[N_TX][N_TX], T hTansInvRnnRe[N_TX][N_RX], T invMatrixHTansInvRnnRe[N_TX][N_RX], T invMatrixHTansInvRnnRe2[N_TX][N_RX]) {
    if(N_TX % 2){
        // #pragma unroll(N_TX)
        for (size_t i = 0; i < N_TX; i ++) {
            #pragma unroll(N_RX)
            for (size_t j = 0; j < N_RX; j ++) {
                T sum0 = T();
                T sum1 = T();
                #pragma unroll(N_TX>>1)
                for (size_t k = 0; k < N_TX-1; k += 2) {
                    sum0 = fma(finvARe[i][k], hTansInvRnnRe[k][j], sum0);
                    sum1 = fma(finvARe[i][k+1], hTansInvRnnRe[k+1][j], sum1);
                }
                sum0 = fma(finvARe[i][N_TX-1], hTansInvRnnRe[N_TX-1][j], sum0);
                invMatrixHTansInvRnnRe[i][j] = sum0 + sum1;

                invMatrixHTansInvRnnRe2[i][j] = _mm512_permutex2var_epi32(invMatrixHTansInvRnnRe[i][j], use_2nd_half, invMatrixHTansInvRnnRe[i][j]);
                invMatrixHTansInvRnnRe[i][j] = _mm512_permutex2var_epi32(invMatrixHTansInvRnnRe[i][j], use_1st_half, invMatrixHTansInvRnnRe[i][j]);
            }
        }
    }
    else{
        // #pragma unroll(N_TX)
        for (size_t i = 0; i < N_TX; i ++) {
            #pragma unroll(N_RX)
            for (size_t j = 0; j < N_RX; j ++) {
                T sum0 = T();
                T sum1 = T();
                #pragma unroll(N_TX>>1)
                for (size_t k = 0; k < N_TX; k += 2) {
                    sum0 = fma(finvARe[i][k], hTansInvRnnRe[k][j], sum0);
                    sum1 = fma(finvARe[i][k+1], hTansInvRnnRe[k+1][j], sum1);
                }
                invMatrixHTansInvRnnRe[i][j] = sum0 + sum1;

                invMatrixHTansInvRnnRe2[i][j] = _mm512_permutex2var_epi32(invMatrixHTansInvRnnRe[i][j], use_2nd_half, invMatrixHTansInvRnnRe[i][j]);
                invMatrixHTansInvRnnRe[i][j] = _mm512_permutex2var_epi32(invMatrixHTansInvRnnRe[i][j], use_1st_half, invMatrixHTansInvRnnRe[i][j]);
            }
        }
    }
}

// inverse(H' * invRnn * H + I) * H' * invRnn * Y
template<typename T, size_t N_RX = 16, size_t N_TX = 8, FO_E foFlag = FO_E::disable>
inline FORCE_INLINE
void txCalc(
    T invMatrixMulHTansInvRnnRe[N_TX][N_RX], T yRe[N_RX],
    T avxShift, Is16vec32 txBuf[N_TX], T FoTable[N_TX] = {NULL}, bool chUpdateFLag = 1) {
    // #pragma unroll(N_TX)
    for (size_t i = 0; i < N_TX; i ++) {
        auto sum = T();
        sum = dot<T, N_RX>(invMatrixMulHTansInvRnnRe[i], yRe);
        if constexpr (foFlag == FO_E::enable) {
            sum = fmulconj(sum, FoTable[i]);
        }
        sum = sum * avxShift;
        txBuf[i] = _mm512_cvtph_epi16(sum);
    }
}
#endif

    template<typename Treq, typename Trx, size_t N_RX>
    inline void xran_decomp(Treq *request, Trx *pRxIn[BBLIB_N_SYMB_PER_SF][N_RX], int16_t iSc, int16_t nSc, int16_t iChSymb) {

        auto pDecomp = request->pPuschDecomp;
        auto nDataSymb = request->nSymb;

        if (pDecomp != NULL && iSc % (pDecomp->decompPRBNum * BBLIB_N_SC_PER_PRB) == 0 && iChSymb == 0) {
            uint16_t decompPRBNum = pDecomp->decompPRBNum;

            uint16_t nScDiv16 = (decompPRBNum * BBLIB_N_SC_PER_PRB) / 16;

            if(iSc != 0) {
                for (int32_t iSym = 0; iSym < nDataSymb; iSym++) {
                    auto nDataSymbIdx = *(request->pSymbIndex + iSym);
                    for (size_t iAnt = 0; iAnt < N_RX; iAnt++) {
                        pRxIn[nDataSymbIdx][iAnt] -= nScDiv16;
                    }
                }
            }

            uint16_t iPRB = iSc / BBLIB_N_SC_PER_PRB;
            int16_t decompTotalPRB = nSc / BBLIB_N_SC_PER_PRB;
            uint32_t procPRBNum = (iPRB + decompPRBNum < decompTotalPRB) ? decompPRBNum : decompTotalPRB - iPRB;
            for (int32_t iSymb = 0; iSymb < nDataSymb; iSymb++)
            {
                pDecomp->nRBStart = pDecomp->decompPRBStart + iPRB;
                pDecomp->nRBSize = procPRBNum;
            }
            pDecomp->xran_decomp_func(pDecomp);
        }
    }
}
#endif
