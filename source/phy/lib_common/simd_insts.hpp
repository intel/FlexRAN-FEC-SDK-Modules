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
// #include <dvec.h>
#include <type_traits>

#ifndef _WIN32

#ifdef __ICC
#if __ICC >= 1910
#include <dvec.h>
#else
#include "dvec_icc.h"
#endif
#else
#include <dvec.h>
#endif

#else
#include "dvec_icc.h"
#endif

namespace W_SDK {
/*! \brief common function to define mask
 *
 * mask declare
 */
#ifdef _WIN32
#define FORCE_INLINE
#else
#define FORCE_INLINE  __attribute__((always_inline))
#endif
using Mask64 = __mmask64;
using Mask32 = __mmask32;
using Mask16 = __mmask16;
using Mask8  = __mmask8;
using M512 = M512vec;

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

/*! \brief common function to store to register */
inline FORCE_INLINE void store(M512 *p, const M512& a) {_mm512_store_si512(reinterpret_cast<void *>(p), a);}
inline FORCE_INLINE void store(float *p, const F32vec16 &a) { _mm512_store_ps(p, a); }
inline FORCE_INLINE void store(M256 *p, const M256& a) {_mm256_store_si256(reinterpret_cast<__m256i *>(p), a);}
inline FORCE_INLINE void store(M128 *p,  const M128& a) {_mm_store_si128(reinterpret_cast<__m128i *>(p), a); }

inline FORCE_INLINE void storeu(M512 *p, const M512& a) {_mm512_storeu_si512(reinterpret_cast<void *>(p), a);}
inline FORCE_INLINE void storeu(M256 *p, const M256& a) {_mm256_storeu_si256(reinterpret_cast<__m256i *>(p), a);} 
inline FORCE_INLINE void storeu(M128 *p,  const M128& a) {_mm_storeu_si128(reinterpret_cast<__m128i *>(p), a); }
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
inline FORCE_INLINE I32vec16 permutexvar(const Mask16 mask, const I32vec16 &index, const I32vec16 &a) {
    return _mm512_maskz_permutexvar_epi32 (mask, index, a); }
/*! \brief common function to permutex2var to register */    
inline FORCE_INLINE Is16vec32 permutex2var( const Is16vec32 &a, const I16vec32 &index, const Is16vec32 &b){
    return _mm512_permutex2var_epi16(a, index, b);}

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
/*! \brief common function to fmsub to register */
inline FORCE_INLINE F32vec16 fmsub(const F32vec16& a, const F32vec16& b, const F32vec16& c) {
   return _mm512_fmsub_ps (a, b, c); }
/*! \brief common function to muladd to register */
inline FORCE_INLINE Is32vec16 mul_add(const Is16vec32 &a, const Is16vec32 &b) {
    return _mm512_madd_epi16(a, b);}
inline FORCE_INLINE Is32vec16 mul_add(const Mask16 &a, const Is16vec32 &b, const Is16vec32 &c) {
    return _mm512_maskz_madd_epi16(a, b, c);}
/*! \brief common function to mulhr to register */
inline FORCE_INLINE Is16vec32 mulhrs(const Is16vec32 &a, const Is16vec32 &b) {
    return _mm512_mulhrs_epi16(a, b);}
/*! \brief common function to mullo to register */
inline FORCE_INLINE Is16vec32 mullo(const Is16vec32 &a, const Is16vec32 &b) {
    return _mm512_mullo_epi16(a, b);}
/*! \brief common function to slli to register */
inline FORCE_INLINE Is16vec32 slli( const Is16vec32 &a,  const uint8_t &b){
    return _mm512_slli_epi16(a, b);}

/*! \brief common function to add to register */
inline FORCE_INLINE F32vec16 add(const F32vec16 &a, const Mask16 &b, const F32vec16 &c, const F32vec16 &d) {
    return _mm512_mask_add_ps(a, b, c, d);}
/*! \brief common function to adds to register */
inline FORCE_INLINE Is16vec32 adds(const Is16vec32 &a, const Is16vec32 &b) {
    return _mm512_adds_epi16(a, b);}
/*! \brief common function to subs to register */
inline FORCE_INLINE Is16vec32 subs(const Is16vec32 &a, const Is16vec32 &b) {
    return _mm512_subs_epi16(a, b);}
/*! \brief common function to set a to register */
inline FORCE_INLINE Is16vec32 set(const int16_t &a) {
    return _mm512_set1_epi16 (a);}
inline FORCE_INLINE Is32vec16 set(const int32_t &a) {
    return _mm512_set1_epi32 (a);}
/*! \brief common function to srai to register */
inline FORCE_INLINE Is16vec32 srai(const Is16vec32 &a, const unsigned int b) {
    return _mm512_srai_epi16 (a, b);}
inline FORCE_INLINE Is32vec16 srai(const Is32vec16 &a, const unsigned int b) {
    return _mm512_srai_epi32 (a, b);}
/*! \brief common function to reduce add to register */
inline FORCE_INLINE float reduce_add(const F32vec16 &a) {
    return _mm512_reduce_add_ps (a);}
inline FORCE_INLINE int32_t reduce_add(const Is32vec16 &a) {
    return _mm512_reduce_add_epi32 (a);}
inline FORCE_INLINE int32_t reduce_add(const Mask16 &a, const Is32vec16 &b) {
    return _mm512_mask_reduce_add_epi32 (a, b);}
/*! \brief common function to blend to register */
inline FORCE_INLINE Is32vec16 blend(const Mask16 &a, const Is32vec16 &b, const Is32vec16 &c) {
    return _mm512_mask_blend_epi32 (a, b, c);}
/*! \brief common function to scatter to register */
inline FORCE_INLINE void scatter(void *addr, const Mask16 k, const Is32vec16 &index, const Is16vec32 &a){
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
}
