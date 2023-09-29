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

#include "dvec_com.hpp"
#include <complex>

namespace W_SDK {
    #ifdef _BBLIB_AVX512_
    class CI16vec16;

    inline CI16vec16 swapRealImag(const CI16vec16 a);
    inline CI16vec16 negateReal(const CI16vec16 a);
    inline CI16vec16 negateImag(const CI16vec16 a);

    class CI16vec16 : public Is16vec32 {
    public:
        CI16vec16() = default;

        CI16vec16(__m512i v) : Is16vec32(v) {}

        CI16vec16(const std::complex<short>& e) {
            vec = _mm512_set1_epi32(*((int*)&e));
        }
        /*! \brief common function to (int32_t ) to register */
        inline FORCE_INLINE
        CI16vec16(int32_t a) { vec = _mm512_set1_epi32(a); }
        /*! \brief common function to (int16_t ) to register */
        inline FORCE_INLINE
        CI16vec16(int16_t a) { vec = _mm512_set1_epi16(a); }

        inline FORCE_INLINE
        CI16vec16& operator+=(const CI16vec16 &rhs) {
            vec = _mm512_adds_epi16(vec, rhs);
            return *this;
        }

        inline FORCE_INLINE
        CI16vec16& operator-=(const CI16vec16 &rhs) {
            vec = _mm512_subs_epi16(vec, rhs);
            return *this;
        }

        inline FORCE_INLINE
        CI16vec16 operator*=(const CI16vec16 &rhs) {
            const __m512i m512_sw_r = _mm512_set_epi8(
                61, 60, 61, 60, 57, 56, 57, 56, 53, 52, 53, 52, 49, 48, 49, 48,
                45, 44, 45, 44, 41, 40, 41, 40, 37, 36, 37, 36, 33, 32, 33, 32,
                29, 28, 29, 28, 25, 24, 25, 24, 21, 20, 21, 20, 17, 16, 17, 16,
                13, 12, 13, 12, 9, 8, 9, 8, 5, 4, 5, 4, 1, 0, 1, 0);
            const __m512i m512_sw_i = _mm512_set_epi8(
                63, 62, 63, 62, 59, 58, 59, 58, 55, 54, 55, 54, 51, 50, 51, 50,
                47, 46, 47, 46, 43, 42, 43, 42, 39, 38, 39, 38, 35, 34, 35, 34,
                31, 30, 31, 30, 27, 26, 27, 26, 23, 22, 23, 22, 19, 18, 19, 18,
                15, 14, 15, 14, 11, 10, 11, 10, 7, 6, 7, 6, 3, 2, 3, 2);
            const auto  nMaskNegQ =0x55555555;

            // Select real or image part from a complex value
            __m512i ReRe = _mm512_shuffle_epi8(vec, m512_sw_r);
            __m512i ImIm = _mm512_shuffle_epi8(vec, m512_sw_i);

            // Swap real or image part and negative image part from a complex value
            // switch IQ
            __m512i tmp1 =    _mm512_rol_epi32(rhs,16);/* t1,t0,t3,t2,t5,t4,t7,t6 */

            // Negative the Q part
            __m512i negImPosRe = _mm512_mask_subs_epi16(tmp1, nMaskNegQ, _mm512_setzero_si512(), tmp1); /* -t1,t0,-t3,t2,-t5,t4,-t7,t6 */

            // Multiply complex
            tmp1 = _mm512_mulhrs_epi16(ReRe, rhs);
            __m512i tmp2 = _mm512_mulhrs_epi16(ImIm, negImPosRe);
            vec = _mm512_adds_epi16(tmp1, tmp2);
            return *this;
        }

        inline FORCE_INLINE
        CI16vec16& operator>>=(const int &rhs) {
            vec = _mm512_srai_epi16(vec, rhs);
            return *this;
        }
#ifndef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
        inline FORCE_INLINE
            CI16vec16& operator<<=(const int &rhs) {
            vec = _mm512_slli_epi16(vec, rhs);
            return *this;
        }
#endif

    };

    /* multiply complex: (A + iB)*(C+iD) = AC-BD + i(AD+BC) */
    inline FORCE_INLINE
    CI16vec16 fmul(const CI16vec16& a, const CI16vec16& b)
    {
        auto t = a;
        t *= b;
        return t;
    }

    inline FORCE_INLINE
    CI16vec16 operator+(const CI16vec16 &a, const CI16vec16 &b) {
        auto t = a;
        t += b;
        return t;
    }

    inline FORCE_INLINE
    CI16vec16 operator-(const CI16vec16 &a, const CI16vec16 &b) {
        auto t = a;
        t -= b;
        return t;
    }

    inline FORCE_INLINE
    CI16vec16 operator*(const CI16vec16 &a, const CI16vec16 &b) {
        auto t = a;
        t *= b;
        return t;
    }

    inline FORCE_INLINE
    CI16vec16 operator>>(const CI16vec16 &a, const int &b) {
        auto t = a;
        t >>= b;
        return t;
    }

    inline FORCE_INLINE
    CI16vec16 operator<<(const CI16vec16 &a, const int &b) {
        auto t = a;
        t <<= b;
        return t;
    }

    inline FORCE_INLINE
    CI16vec16 swapRealImag(const CI16vec16 a) {
        // Shuffle provides better ports utilization for arithmetic heavy applications.
        return _mm512_rol_epi32(a,16);
    }

    inline FORCE_INLINE
    CI16vec16 negateReal(const CI16vec16 a) {
        return _mm512_mask_sub_epi16(a, 0x55555555, _mm512_setzero_si512(), a);
    }

    inline FORCE_INLINE
    CI16vec16 negateImag(const CI16vec16 a) {
        return _mm512_mask_sub_epi16(a, 0xAAAAAAAA, _mm512_setzero_si512(), a);
    }

    // a + bi -> bi - a
    FORCE_INLINE inline
    CI16vec16 imagNegReal(const CI16vec16 a) {
        // CI16vec16 b = _mm512_ror_epi32 (a, 16);
        // __m512i index = _mm512_set_epi8(
        // 61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 50,
        // 45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34,
        // 29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18,
        // 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);
        // CI16vec16 b = _mm512_shuffle_epi8(a, index);
        CI16vec16 b  =  _mm512_rol_epi32(a,16);

        b = _mm512_mask_sub_epi16 (b, 0x55555555, CI16vec16(), b);
        return b;

    }

    inline FORCE_INLINE
    CI16vec16 mulhrs(const CI16vec16 a, const CI16vec16 b) { return (_mm512_mulhrs_epi16( a, b)); }
    inline FORCE_INLINE
    CI16vec16 mullo(const CI16vec16 a, const CI16vec16 b) { return _mm512_mullo_epi16(a, b); }

    inline FORCE_INLINE CI16vec16 loadu(CI16vec16 *p) {return _mm512_loadu_si512(p);}

    inline FORCE_INLINE CI16vec16 fmulconj(const CI16vec16& input0, const CI16vec16& input1)
    {
        const __m512i m512_sw_r = _mm512_set_epi8(
            61, 60, 61, 60, 57, 56, 57, 56, 53, 52, 53, 52, 49, 48, 49, 48,
            45, 44, 45, 44, 41, 40, 41, 40, 37, 36, 37, 36, 33, 32, 33, 32,
            29, 28, 29, 28, 25, 24, 25, 24, 21, 20, 21, 20, 17, 16, 17, 16,
            13, 12, 13, 12, 9, 8, 9, 8, 5, 4, 5, 4, 1, 0, 1, 0);
        const __m512i m512_sw_i = _mm512_set_epi8(
            63, 62, 63, 62, 59, 58, 59, 58, 55, 54, 55, 54, 51, 50, 51, 50,
            47, 46, 47, 46, 43, 42, 43, 42, 39, 38, 39, 38, 35, 34, 35, 34,
            31, 30, 31, 30, 27, 26, 27, 26, 23, 22, 23, 22, 19, 18, 19, 18,
            15, 14, 15, 14, 11, 10, 11, 10, 7, 6, 7, 6, 3, 2, 3, 2);
        const Mask32  nMaskNegI =0xaaaaaaaa;

        // Select real or image part from a complex value
        __m512i ReRe = _mm512_shuffle_epi8(input0, m512_sw_r);
        __m512i ImIm = _mm512_shuffle_epi8(input0, m512_sw_i);

        // Swap real or image part and negative image part from a complex value
        // switch IQ
        __m512i tmp1 =  _mm512_rol_epi32(input1,16);

        // Negative the Q part
        __m512i negImPosRe = _mm512_mask_subs_epi16(input1, nMaskNegI, _mm512_setzero_si512(), input1);

        // Multiply complex
        tmp1 = _mm512_mulhrs_epi16(ImIm, tmp1);
        __m512i tmp2 = _mm512_mulhrs_epi16(ReRe, negImPosRe);

        return _mm512_adds_epi16(tmp1, tmp2);
    }
    #endif

    namespace dvec {
        /// Return the number of SIMD elements
        template<typename T>
        struct num_elements_helper { static constexpr unsigned value = 1; };

        template<class T>
        struct num_elements { static constexpr unsigned value = dvec::num_elements_helper<typename std::decay<T>::type>::value; };

        #ifdef _BBLIB_AVX512_
        template<>
        struct num_elements_helper<CI16vec16> { static constexpr unsigned value = 16; };
        #endif
    }
}
