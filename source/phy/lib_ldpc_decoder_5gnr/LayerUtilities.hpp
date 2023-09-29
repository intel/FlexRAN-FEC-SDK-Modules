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

#include "dvec_inc.h"
#include <cstdint>
#include <iostream>

//////////////////////////////////////////////////////////////////////////////////////////////////////
// Dvec Extensions
//////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef USE_PRIVATE_DVEC
#undef USE_PRIVATE_DVEC
#endif

#ifdef _WIN32

#if (__INTEL_COMPILER < 1910)
#define USE_PRIVATE_DVEC
#endif

#else

#ifdef __ICC
#if (__ICC < 1910)
#define USE_PRIVATE_DVEC
#endif
#endif

#endif


#ifdef USE_PRIVATE_DVEC

// Dvec in ISS 2019 doesn't provide a Is16vec16 class. This was reported to the compiler team as issue
// CMPLRIL0-30948 in JIRA

class Is16vec16
{
public:

  Is16vec16() = default;

  Is16vec16(int16_t v) : value(_mm256_set1_epi16(v)) { }
  Is16vec16(__m256i v) : value(v) { }

  operator __m256i() const { return value; }

  friend Is16vec16 abs(Is16vec16 v) { return _mm256_abs_epi16(v); }
  friend Is16vec16 sat_add(Is16vec16 lhs, Is16vec16 rhs) { return _mm256_adds_epi16(lhs, rhs); }
  friend Is16vec16 sat_sub(Is16vec16 lhs, Is16vec16 rhs) { return _mm256_subs_epi16(lhs, rhs); }
  friend Is16vec16 sat_sub_unsigned(Is16vec16 lhs, Is16vec16 rhs) { return _mm256_subs_epu16(lhs, rhs); }

  friend Is16vec16 simd_min(Is16vec16 lhs, Is16vec16 rhs) { return _mm256_min_epi16(lhs, rhs); }
  friend Is16vec16 simd_max(Is16vec16 lhs, Is16vec16 rhs) { return _mm256_max_epi16(lhs, rhs); }

  friend Is16vec16 operator^(Is16vec16 lhs, Is16vec16 rhs) { return _mm256_xor_si256(lhs, rhs); }
  friend Is16vec16 operator|(Is16vec16 lhs, Is16vec16 rhs) { return _mm256_or_si256(lhs, rhs); }

  Is16vec16 operator>>(int count) const { return _mm256_srai_epi16(value, count); }

  __m256i value;
};

// dvec-like operators for choosing one of two element values based a comparison.
inline Is16vec16 select_eq(Is16vec16 a, Is16vec16 b, Is16vec16 c, Is16vec16 d)
{
  return _mm256_blendv_epi8(d, c, _mm256_cmpeq_epi16(a, b));
}

inline Is16vec16 select_lt(Is16vec16 a, Is16vec16 b, Is16vec16 c, Is16vec16 d)
{
    return _mm256_blendv_epi8(d, c, _mm256_cmpgt_epi16(b, a));
}


// Some functions are missing from dvec for AVX-512.
#ifdef _BBLIB_AVX512_
inline Is16vec32 abs(Is16vec32 v) { return _mm512_abs_epi16(v); }
inline Is16vec32 sat_sub_unsigned(Is16vec32 lhs, Is16vec32 rhs) { return _mm512_subs_epu16(lhs, rhs); }
#endif

#else

// dvec-like operators for choosing one of two element values based a comparison.
inline Is16vec16 select_eq(Is16vec16 a, Is16vec16 b, Is16vec16 c, Is16vec16 d)
{
    return _mm256_blendv_epi8(d, c, _mm256_cmpeq_epi16(a, b));
}

inline Is16vec16 select_lt(Is16vec16 a, Is16vec16 b, Is16vec16 c, Is16vec16 d)
{
    return _mm256_blendv_epi8(d, c, _mm256_cmpgt_epi16(b, a));
}

// Some functions are missing from dvec for AVX-512.
#ifdef _BBLIB_AVX512_
inline Is16vec32 abs(Is16vec32 v) { return _mm512_abs_epi16(v); }
inline Is16vec32 sat_sub_unsigned(Is16vec32 lhs, Is16vec32 rhs) { return _mm512_subs_epu16(lhs, rhs); }
#endif
inline Is16vec16 abs(Is16vec16 v) { return _mm256_abs_epi16(v); }
inline Is16vec16 sat_sub_unsigned(Is16vec16 lhs, Is16vec16 rhs) { return _mm256_subs_epu16(lhs, rhs); }


#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////
// Misc
//////////////////////////////////////////////////////////////////////////////////////////////////////


// Equivalent to Ceil(a/b)
static inline int RoundUpDiv(int a, int b)
{
  return ((a + b - 1) / b);
}

/// Expand the set of bits into complete 16-bit elements which are either all 1 or all 0.
static Is16vec16 ExpandMsb(int16_t msb)
{
  // There are 16 bits, each of which needs to be expanded into a complete 16-bit integer of 1's and
  // 0's.

  // Start by broadcasting all bits to every position.
  const auto dupBits = _mm256_set1_epi16(msb);

  // AND with a mask bit for each position. 1 for the first element, 2 for the next, 4 and so
  // on. That effectively creates a single bit in each element; its the MSB but in the wrong place.
  const auto k_bitForElementMask =
    _mm256_setr_epi16(0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 0x0080,
                      0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x2000, 0x4000, 0x8000);
  const auto justElementBits = _mm256_and_si256(dupBits, k_bitForElementMask);

  // Now that each element has just one bit that is set or cleared, comparing back against the same
  // bit mask that was used above will turn that single bit into a complete set of 0's and 1's which
  // fills the element.
  return _mm256_cmpeq_epi16(justElementBits, k_bitForElementMask);
}

template<typename SIMD_TYPE>
static int GetNumSimdLoops(int zExpansion)
{
  // The number of 16-bit elements in the given SIMD type.
  constexpr int k_numElements = sizeof(SIMD_TYPE) / sizeof(int16_t);
  const int numLoops = zExpansion / k_numElements;

  if (zExpansion != (numLoops * k_numElements))
  {
      std::cout << "This Expansion-Factor Z does not divide by the number of elements\n";
      return -1;
  }

  return numLoops;
}

template<typename SIMD_TYPE>
static int GetNumAlignedSimdLoops(int zExpansion)
{
  // The number of 16-bit elements in the given SIMD type.
  constexpr int k_numElements = sizeof(SIMD_TYPE) / sizeof(int16_t);
  const int numLoops = (zExpansion + (k_numElements - 1)) / k_numElements;

  return numLoops;
}

/// Broadcast a single int16 to the given type. Unfortunately this is needed because Is16vec32 from
/// dvec doesn't provide its own scalar broadcast.
template<typename T> T BroadcastInt16(int16_t);
template<> Is16vec16 inline BroadcastInt16(int16_t v) { return _mm256_set1_epi16(v); }

#ifdef _BBLIB_AVX512_
template<> Is16vec32 inline BroadcastInt16(int16_t v) { return _mm512_set1_epi16(v); }
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////
// AVX2 / AVX512 Overloaded functions
//////////////////////////////////////////////////////////////////////////////////////////////////////

/// Get a mask of the negative elements. One bit for each element.
static int16_t GetNegativeMask(Is16vec16 simd)
{
  // Extract all the MSBs of each byte. There is no 16-bit equivalent
  // so this 8-bit version will give too many bits to start with.
  const auto byteMsbs = _mm256_movemask_epi8(simd);

  // Reduce the number of bits down by extracting every other bit, which corresponds to just the
  // epi16 MSBs.
  const auto bits = _pext_u32(byteMsbs, 0b10101010101010101010101010101010);

  return int16_t(bits);
}

/// Get a mask of the negative elements. One bit for each element.
#ifdef _BBLIB_AVX512_
static int32_t GetNegativeMask(Is16vec32 simd)
{
  // Read the MSB by comparing to zero. This is a 3/0.5 instruction. The alternative is to use
  // movepi16_mask instead, which extracts the MSB bit into a mask directly. That instruction is
  // 1/1. For LDPC we are very compute limited though, so the higher throughput of the comparison
  // below gives better performance (3% at time of writing).
  return _mm512_cmp_epi16_mask(simd, _mm512_setzero_si512(), _MM_CMPINT_LT);
}
#endif

/// Apply a parity correction. All values whose corresponding bits are zero will be negated.
static Is16vec16 ApplyParityCorrection(int16_t parity, Is16vec16 value)
{
  // Note that individual parity bits can be expanded into a complete set of 0's which forces
  // sign_epi16 to zero the result too. A single 1 is ORed into the parity bit to avoid this.
  const auto parityV = ExpandMsb(parity);
  const __m256i parityTieBreak = _mm256_or_si256(parityV, _mm256_set1_epi16(1));
  return _mm256_sign_epi16(value, parityTieBreak);
}

#ifdef _BBLIB_AVX512_
static Is16vec32 ApplyParityCorrection(int32_t parity, Is16vec32 value)
{
  return _mm512_mask_sub_epi16(value, parity, _mm512_setzero_si512(), value);
}
#endif

/// Perform an insertion of a new value into the best two values found so far, which are sorted in order.
static void InsertSort(Is16vec16& min1, Is16vec16& min2, Is16vec16& minPos, int newPos, Is16vec16 value)
{
  // If the new values are better than the current best, update the index.
  minPos = select_lt(value, min1, _mm256_set1_epi16(short(newPos)), minPos);

  // Sort-insertion network.
  const auto t = _mm256_max_epi16(min1, value);
  min2 = _mm256_min_epi16(t, min2);
  min1 = _mm256_min_epi16(min1, value);
}

#ifdef _BBLIB_AVX512_
static void InsertSort(Is16vec32& min1, Is16vec32& min2, Is16vec32& minPos, int newPos, Is16vec32 value)
{
  // Is the original min unchanged?
  const auto isOriginalStillTheMin = _mm512_cmp_epi16_mask(min1, value, _CMP_LE_OS);

  // If the new value is better than the original minimum, update the index.
  minPos = _mm512_mask_blend_epi16(isOriginalStillTheMin, _mm512_set1_epi16(short(newPos)), minPos);

  // Three-way minimum of the new value and the two original lowest values! If the new value is
  // smaller than the original minimum then the original minimum gets displaced to become the second
  // minimum, simply by passing through this mask instruction. If the original minimum wasn't
  // displaced, then choose between the minimum of the new value, and the so-far untested second
  // minimum.
  min2 = _mm512_mask_min_epi16(min1, isOriginalStillTheMin, value, min2);

  // This could be done using a min_epi16 instruction to allow it to be scheduled earlier, but that
  // min would take up the only execution port. Better to reuse the mask and allow the blend to use
  // any port to give higher throughput.
  min1 = _mm512_mask_blend_epi16(isOriginalStillTheMin, value, min1);
}
#endif

// The compiler's own implementation of this function is wrong (issue CMPLRLIBS-2780). Provide a
// corrected replacement until it has been fixed.
static Is16vec16 SelectEqWorkaround(Is16vec16 a, Is16vec16 b, Is16vec16 c, Is16vec16 d)
{
  return _mm256_blendv_epi8(d, c, _mm256_cmpeq_epi16(a, b));
}

#ifdef _BBLIB_AVX512_
static Is16vec32 SelectEqWorkaround(Is16vec32 a, Is16vec32 b, Is16vec32 c, Is16vec32 d)
{
  const auto mask = _mm512_cmp_epi16_mask(a, b, _MM_CMPINT_EQ);
  return _mm512_mask_blend_epi16(mask, d, c);
}
#endif


