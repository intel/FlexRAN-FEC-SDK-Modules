//  Copyright (c) 2020 Intel Corporation.
//
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.

/*
 *  Definition of a C++ class interface to MMX(TM) instruction intrinsics.
 *
 */

#if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC)

#ifndef _GCC_INC_H_
#define _GCC_INC_H_

#include <x86intrin.h>
#include <immintrin.h>

// Some defines are needed for GCC for some reason
typedef long long __int64;
typedef unsigned long long __uint64;

inline __m512i _mm512_permutevar_epi32(__m512i idx, __m512i a)
{
    return _mm512_permutex2var_epi32(a, idx, a);
}

//inline __m256i _mm_broadcastsi128_si256(__m128i a)
//{
//    return _mm256_set_m128(a, a);
//}

//inline int _mm512_cvtsi512_si32(__m512i a)
//{
//    int *b = (int*) &a;
//    return *b;
//}

inline __m256 _mm256_invsqrt_ps(__m256 a)
{
    return _mm256_rsqrt_ps(a);
}

#endif // _GCC_INC_H_
#endif // #if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC)

