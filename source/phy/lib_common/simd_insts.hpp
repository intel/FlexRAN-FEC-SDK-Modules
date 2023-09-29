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

/**
 * @file simd_insts.h
 * @brief This header file used to define instrinsic instruction with dvec.h
 * used global.
 */

#pragma once
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>
#include <type_traits>

#ifdef _BBLIB_SPR_
#include "dvec_fp16.hpp"
#endif
#include "dvec_int16.hpp"

namespace W_SDK {
/*! \brief common function to define mask
 *
 * mask declare
 */
#define ZEROS_512I  _mm512_setzero_si512 ()
#define ZEROS_256I  _mm256_setzero_si256 ()
#define ZEROS_512F  _mm512_setzero_ps ()
#define R_CAST reinterpret_cast
#define S_CAST static_cast


/*! \brief common function to load to register */
#define INST(OP) \
inline FORCE_INLINE M512 OP (M512 *p) {return _mm512_##OP##_si512(reinterpret_cast<void const*>(p)); } \
inline FORCE_INLINE M256 OP (M256 *p) {return _mm256_##OP##_si256(reinterpret_cast<__m256i const *>(p)); } \
inline FORCE_INLINE M128 OP (M128 *p) {return _mm_##OP##_si128(reinterpret_cast<__m128i const *>(p)); }
INST(load)
INST(loadu)
#undef INST

/*! \brief common function to mask load unalign to register*/
inline FORCE_INLINE I8vec64 loadu(const Mask64& k, void const *p) { return _mm512_maskz_loadu_epi8 (k, reinterpret_cast<void const*>(p)); }
inline FORCE_INLINE M512 loadu(const M512& src, Mask32 k, void const *p) {return _mm512_mask_loadu_epi16(src, k, p);}
inline FORCE_INLINE M512 loadu(const M512& src, Mask16 k, void const *p) {return _mm512_mask_loadu_epi32(src, k, p);}
inline FORCE_INLINE F32vec16 loadu(void const *p) {return _mm512_loadu_ps(p);}
inline FORCE_INLINE Is16vec32 loadu(Is16vec32 *p) {return _mm512_loadu_si512(p);}
inline FORCE_INLINE Is16vec32 loadu(Mask32 &k, Is16vec32 *p) {return _mm512_maskz_loadu_epi16(k, p);}
inline FORCE_INLINE Is16vec32 loadu(Mask16 &k, Is16vec32 *p) {return _mm512_maskz_loadu_epi32(k, p);}//[R23.03] Add


/*! \brief common function to store to register */
inline FORCE_INLINE void store(M512 *p, const M512& a) {_mm512_store_si512(reinterpret_cast<void *>(p), a);}
inline FORCE_INLINE void store(float *p, const F32vec16 &a) { _mm512_store_ps(p, a); }
inline FORCE_INLINE void store(M256 *p, const M256& a) {_mm256_store_si256(reinterpret_cast<__m256i *>(p), a);}
inline FORCE_INLINE void store(M128 *p,  const M128& a) {_mm_store_si128(reinterpret_cast<__m128i *>(p), a); }

inline FORCE_INLINE void storeu(M512 *p, const M512& a) {_mm512_storeu_si512(reinterpret_cast<void *>(p), a);}
inline FORCE_INLINE void storeu(M256 *p, const M256& a) {_mm256_storeu_si256(reinterpret_cast<__m256i *>(p), a);}
inline FORCE_INLINE void storeu(M128 *p,  const M128& a) {_mm_storeu_si128(reinterpret_cast<__m128i *>(p), a); }
inline FORCE_INLINE void storeu(F32vec16 *p, const F32vec16& a) {_mm512_storeu_ps(reinterpret_cast<void *>(p), a);}
inline FORCE_INLINE void storeu(void *p, Mask64 &k, const M512 &src) {_mm512_mask_storeu_epi8(p, k, src);}
inline FORCE_INLINE void storeu(void *p, Mask32 &k, const M512 &src) {_mm512_mask_storeu_epi16(p, k, src);}
inline FORCE_INLINE void storeu(void *p, Mask16 &k, const M512 &src) {_mm512_mask_storeu_epi32(p, k, src);}
inline FORCE_INLINE void storeu(void *p, Mask32 &k, const M256 &src) {_mm256_mask_storeu_epi8(p, k, src);}
inline FORCE_INLINE void storeu(float *p, Mask16 &k, const F32vec16 &src) {_mm512_mask_storeu_ps(p, k, src);}

/*! \brief common function to movepi to register */
inline FORCE_INLINE Mask64 movepi(const I8vec64& a) {
    return _mm512_movepi8_mask (a); }
inline FORCE_INLINE Mask32 movepi(const I8vec32& a) {
    return _mm256_movepi8_mask (a); }
/*! \brief common function to movm to register */
inline FORCE_INLINE I8vec32 movm(const Mask32 &mask) {
    return _mm256_movm_epi8(mask); }
inline FORCE_INLINE I8vec64 movm(const Mask64 &mask) {
    return _mm512_movm_epi8(mask); }
inline FORCE_INLINE I8vec64 movm(const int64_t &mask) {
    return _mm512_movm_epi8(mask); }
/*! \brief common function to shuffle to register */
inline FORCE_INLINE I8vec64 shuffle(const I8vec64 &a, const I8vec64 &b) {
    return _mm512_shuffle_epi8(a, b); }
/*! \brief common function to permutexvar to register */
inline FORCE_INLINE I16vec32 permutexvar(const I16vec32 &index, const I16vec32 &a) {
    return _mm512_permutexvar_epi16(index, a); }
inline FORCE_INLINE I32vec16 permutexvar(const I32vec16 &index, const I32vec16 &a) {
    return _mm512_permutexvar_epi32(index, a); }
inline FORCE_INLINE I16vec32 permutexvar(const I32vec16 &index, const I16vec32 &a) {
    return _mm512_permutexvar_epi32(index, a); }
inline FORCE_INLINE I32vec16 permutexvar(const Mask16 mask, const I32vec16 &index, const I32vec16 &a) {
    return _mm512_maskz_permutexvar_epi32 (mask, index, a); }
inline FORCE_INLINE I16vec32 permutexvar(const Mask32 mask, const I16vec32 &index, const I16vec32 &a) {
    return _mm512_maskz_permutexvar_epi16 (mask, index, a); }
/*! \brief common function to permutex2var to register */
inline FORCE_INLINE Is16vec32 permutex2var( const Is16vec32 &a, const I16vec32 &index, const Is16vec32 &b) {
    return _mm512_permutex2var_epi16(a, index, b);}
inline FORCE_INLINE Is16vec32 permutex2var( const Is16vec32 &a, const I32vec16 &index, const Is16vec32 &b) {
    return _mm512_permutex2var_epi32(a, index, b);}
/*! \brief common function to converte to register */
inline FORCE_INLINE I16vec32 cvt(const I8vec32 &a) { return _mm512_cvtepi8_epi16 (a); }
inline FORCE_INLINE I8vec32 cvt(const I16vec32 &a) { return _mm512_cvtepi16_epi8 (a); }
inline FORCE_INLINE F32vec16 cvt(const I32vec16 &a) { return _mm512_cvtepi32_ps (a); }
inline FORCE_INLINE Is32vec16 cvt(const F32vec16 &a) { return _mm512_cvtps_epi32 (a); }
/*! \brief common function to xor to register */
inline FORCE_INLINE I64vec8 maskxor(const I64vec8& a, const Mask8 k, const I64vec8& b, const I64vec8& c) {
   return _mm512_mask_xor_epi64 (a, k, b, c); }
/*! \brief common function to sub to register */
inline FORCE_INLINE I64vec8 masksub(const I16vec32& a, const Mask32 k, const I16vec32& b, const I16vec32& c) {
   return _mm512_mask_sub_epi16 (a, k, b, c); }
/*! \brief common function to fmadd to register */
inline FORCE_INLINE F32vec16 fmadd(const F32vec16& a, const F32vec16& b, const F32vec16& c) {
   return _mm512_fmadd_ps (a, b, c); }
//[R23.03] Add: a*b/32768+c
inline FORCE_INLINE Is16vec32 fmadd(const Is16vec32& a, const Is16vec32& b, const Is16vec32& c) {
   return _mm512_adds_epi16(c, _mm512_mulhrs_epi16(a, b)); }
/*! \brief common function to fnmadd to register */
inline FORCE_INLINE F32vec16 fnmadd(const F32vec16& a, const F32vec16& b, const F32vec16& c) {
   return _mm512_fnmadd_ps (a, b, c); }
/*! \brief common function to fmsub to register */
inline FORCE_INLINE F32vec16 fmsub(const F32vec16& a, const F32vec16& b, const F32vec16& c) {
   return _mm512_fmsub_ps (a, b, c); }
/*! \brief common function to fnmsub to register */
inline FORCE_INLINE F32vec16 fnmsub(const F32vec16& a, const F32vec16& b, const F32vec16& c) {
   return _mm512_fnmsub_ps (a, b, c); }
/*! \brief common function to muladd to register */
inline FORCE_INLINE Is32vec16 mul_add(const Is16vec32 &a, const Is16vec32 &b) {
    return _mm512_madd_epi16(a, b);}
inline FORCE_INLINE Is32vec16 mul_add(const Mask16 &a, const Is16vec32 &b, const Is16vec32 &c) {
    return _mm512_maskz_madd_epi16(a, b, c);}

/*! \brief common function to mul to register */
inline FORCE_INLINE Iu64vec8 mul(const Iu64vec8 &a, const Iu64vec8 &b) {
    return _mm512_mul_epu32(a, b);}

/*! \brief common function to mulhr to register */
inline FORCE_INLINE Is16vec32 mulhrs(const Is16vec32 &a, const Is16vec32 &b) {
    return _mm512_mulhrs_epi16(a, b);}
/*! \brief common function to mullo to register */
inline FORCE_INLINE Is16vec32 mullo(const Is16vec32 &a, const Is16vec32 &b) {
    return _mm512_mullo_epi16(a, b);}
/*! \brief common function to add to register */
inline FORCE_INLINE F32vec16 add(const F32vec16 &a, const Mask16 &b, const F32vec16 &c, const F32vec16 &d) {
    return _mm512_mask_add_ps(a, b, c, d);}
inline FORCE_INLINE Is16vec32 add(const Is16vec32 &a, const Mask32 &b, const Is16vec32 &c, const Is16vec32 &d) {
    return _mm512_mask_add_epi16(a, b, c, d);}

inline FORCE_INLINE F32vec16 add(const F32vec16 &a, const F32vec16 &c) {
    return _mm512_add_ps(a, c);}
/*! \brief common function to adds to register */
inline FORCE_INLINE Is32vec16 add(const Is32vec16 &a, const Is32vec16 &b) {
    return _mm512_add_epi32(a, b);}

/*! \brief common function to adds to register */
inline FORCE_INLINE Is16vec32 adds(const Is16vec32 &a, const Is16vec32 &b) {
    return _mm512_adds_epi16(a, b);}
/*! \brief common function to subs to register */
inline FORCE_INLINE Is16vec32 subs(const Is16vec32 &a, const Is16vec32 &b) {
    return _mm512_subs_epi16(a, b);}

inline FORCE_INLINE Is16vec32 subs(const Mask32 &c, const Is16vec32 &a, const Is16vec32 &b) {
    return _mm512_maskz_subs_epi16(c, a, b);}
/*! \brief common function to set a to register */
inline FORCE_INLINE Is16vec32 set(const int16_t &a) {
    return _mm512_set1_epi16 (a);}
inline FORCE_INLINE Is32vec16 set(const int32_t &a) {
    return _mm512_set1_epi32 (a);}
inline FORCE_INLINE F32vec16 set(const float &a) {
    return _mm512_set1_ps (a);}

inline FORCE_INLINE Is32vec16 set(const Is32vec16 &a, const Mask16 &b, const int32_t c) {
    return _mm512_mask_set1_epi32 (a, b, c);}

/*! \brief common function to srli to register */
inline FORCE_INLINE Is16vec32 srli(const Is16vec32 &a, const unsigned int b) {
    //printf("srli epi16 call\n");
    return _mm512_srli_epi16 (a, b);}
inline FORCE_INLINE Is32vec16 srli32(const Is32vec16 &a, const unsigned int b) {
   // printf("srli epi32 call\n");
    return _mm512_srli_epi32 (a, b);}
inline FORCE_INLINE Iu64vec8 srli64(const Iu64vec8 &a, const unsigned int b) {
   // printf("srli epi64 call\n");
    return _mm512_srli_epi64 (a, b);}
/*! \brief common function to srai to register */
inline FORCE_INLINE Is16vec32 srai(const Is16vec32 &a, const unsigned int b) {
    return _mm512_srai_epi16 (a, b);}
inline FORCE_INLINE Is32vec16 srai(const Is32vec16 &a, const unsigned int b) {
    return _mm512_srai_epi32 (a, b);}
/*! \brief common function to slli to register */
inline FORCE_INLINE Is16vec32 slli( const Is16vec32 &a,  const uint8_t &b) {
    return _mm512_slli_epi16(a, b);}
/*! \brief common function to reduce add to register */
inline FORCE_INLINE float reduce_add(const F32vec16 &a) {
    return _mm512_reduce_add_ps (a);}
inline FORCE_INLINE float reduce_add(const Mask16 &a, const F32vec16 &b) {
    return _mm512_mask_reduce_add_ps (a, b);}
inline FORCE_INLINE int32_t reduce_add(const Is32vec16 &a) {
    return _mm512_reduce_add_epi32 (a);}
inline FORCE_INLINE int32_t reduce_add(const Mask16 &a, const Is32vec16 &b) {
    return _mm512_mask_reduce_add_epi32 (a, b);}
inline FORCE_INLINE
Is32vec16 reduce_add_half(Is32vec16 a) {
    a += _mm512_shuffle_i32x4(a, a, _MM_PERM_CDAB); // 11,10,9,8, 15,14,13,12, 3,2,1,0, 7,6,5,4
    a += _mm512_shuffle_epi32(a, 0x4E); // 3,2,1,0 -> 1,0,3,2
    a += _mm512_shuffle_epi32(a, 0x11); // 3,2,1,0 -> 2,3,0,1
    return a;
}

inline FORCE_INLINE
F32vec16 reduce_add_half(F32vec16 &a) {
    a += _mm512_shuffle_f32x4(a, a, _MM_PERM_CDAB); // 11,10,9,8, 15,14,13,12, 3,2,1,0, 7,6,5,4
    a += _mm512_shuffle_ps(a, a, 0x4E); // 3,2,1,0 -> 1,0,3,2
    a += _mm512_shuffle_ps(a, a, 0x11); // 3,2,1,0 -> 2,3,0,1
    return a;
}

/*! \brief common function to blend to register */
inline FORCE_INLINE Is32vec16 blend(const Mask16 &a, const Is32vec16 &b, const Is32vec16 &c) {
    return _mm512_mask_blend_epi32 (a, b, c);}

/*! \brief common function to blend to register */
inline FORCE_INLINE F32vec16 blend(const Mask16 &a, const F32vec16 &b, const F32vec16 &c) {
    return _mm512_mask_blend_ps (a, b, c);}

/*! \brief common function to cmp to register */
inline FORCE_INLINE Mask16 cmpge(const Is32vec16 &a, const Is32vec16 &b){
    return _mm512_cmpge_epi32_mask (a, b);}
/*! \brief common function to scatter to register */
inline FORCE_INLINE void scatter(void *addr, const Mask16 k, const Is32vec16 &index, const Is16vec32 &a) {
    _mm512_mask_i32scatter_epi32(addr, k, index, a, 4);}

/*! \brief some define function */
/*! \brief common function to permutex to register */
#define permutex(mask, a, imm8) return _mm512_maskz_permutex_epi64 (mask, a, imm8)


/*! \brief icc specail instructions, will use replacement in ICX */
#ifdef __ICC
/*! \brief common function to permute4f128 to register */
inline FORCE_INLINE I32vec16 permute4f128 (const I64vec8 &a, const _MM_PERM_ENUM&& imm8) {
    return _mm512_permute4f128_epi32 (a, imm8); }
#endif



/* multiply complex: (A + iB)*(C+iD) = AC-BD + i(AD+BC) */
inline FORCE_INLINE Is16vec32 fmul(const Is16vec32& input0, const Is16vec32& input1)
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
    const Mask32  nMaskNegQ =0x55555555;

    // Select real or image part from a complex value
    __m512i ReRe = _mm512_shuffle_epi8(input0, m512_sw_r);
    __m512i ImIm = _mm512_shuffle_epi8(input0, m512_sw_i);

    // Swap real or image part and negative image part from a complex value
    // switch IQ
    __m512i tmp1 =    _mm512_rol_epi32(input1,16);/* t1,t0,t3,t2,t5,t4,t7,t6 */

    // Negative the Q part
    __m512i negImPosRe = _mm512_mask_subs_epi16(tmp1, nMaskNegQ, _mm512_setzero_si512(), tmp1); /* -t1,t0,-t3,t2,-t5,t4,-t7,t6 */

    // Multiply complex
    tmp1 = _mm512_mulhrs_epi16(ReRe, input1);
    __m512i tmp2 = _mm512_mulhrs_epi16(ImIm, negImPosRe);
    return _mm512_adds_epi16(tmp1, tmp2);
}


inline FORCE_INLINE Is16vec32 fmulconj(const Is16vec32& input0, const Is16vec32& input1)
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
    __m512i tmp1 =  _mm512_rol_epi32(input1,16);/* t1,t0,t3,t2,t5,t4,t7,t6 */

    // Negative the Q part
    __m512i negImPosRe = _mm512_mask_subs_epi16(input1, nMaskNegI, _mm512_setzero_si512(), input1); /* t0,-t1,t2,-t3,t4,-t5,t6,-t7 */

    // Multiply complex
    tmp1 = _mm512_mulhrs_epi16(ImIm, tmp1);
    __m512i tmp2 = _mm512_mulhrs_epi16(ReRe, negImPosRe);
    return _mm512_adds_epi16(tmp1, tmp2);
}

}
