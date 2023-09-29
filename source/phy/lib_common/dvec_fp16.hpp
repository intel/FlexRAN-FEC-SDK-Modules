// If no compiler support for FP16 is available, use an emulation library.
#include<immintrin.h>

#include "half.hpp"
#include <complex>
#include "dvec_com.hpp"

#pragma once

namespace W_SDK {
/*! \brief common function to define mask*/


template<typename HV>
class F16vec
{
public:

  static constexpr int k_numElements = sizeof(HV) / sizeof(float16);

  F16vec() = default;
  F16vec(HV v) : simd(v) { }

  void set_zero() { simd = HV(); }

  /// Indexing operator for constant values.
  float16 operator[](int index) const
  {
    assert(0 <= index && index < (int)k_numElements);
    const float16* elements = (const float16*)this;
    return elements[index];
  }

  /// Indexing operator for non-const values. This can be used for assigning elements.
  float16& operator[](int index)
  {
    assert(0 <= index && index < (int)k_numElements);
    float16* elements = (float16*)this;
    return elements[index];
  }

  HV simd;
};

class F16vec16 : public F16vec<__m256h>
{
public:
  F16vec16() = default;
  F16vec16(__m256h v) : F16vec(v) { }
  F16vec16(float16 value) : F16vec(_mm256_set1_ph(value)) { }

  operator __m256h() const { return simd; }

  friend F16vec16 operator+ (const F16vec16& lhs, const F16vec16& rhs) { return _mm256_add_ph(lhs, rhs); }
  friend F16vec16 operator- (const F16vec16& lhs, const F16vec16& rhs) { return _mm256_sub_ph(lhs, rhs); }
  friend F16vec16 operator* (const F16vec16& lhs, const F16vec16& rhs) { return _mm256_mul_ph(lhs, rhs); }
  friend F16vec16 operator/ (const F16vec16& lhs, const F16vec16& rhs) { return _mm256_div_ph(lhs, rhs); }

  F16vec16& operator+= (const F16vec16& rhs) { *this = *this + rhs; return *this; }
  F16vec16& operator-= (const F16vec16& rhs) { *this = *this - rhs; return *this; }
  F16vec16& operator*= (const F16vec16& rhs) { *this = *this * rhs; return *this; }
  F16vec16& operator/= (const F16vec16& rhs) { *this = *this / rhs; return *this; }

  F16vec16 operator- () const { return _mm256_sub_ph(__m256h(), simd); }

  friend F16vec16 abs(const F16vec16& v) { return _mm256_abs_ph(v); }
  friend F16vec16 rcp(const F16vec16& v) { return _mm256_rcp_ph(v); }
  friend F16vec16 rsqrt(const F16vec16& v) { return _mm256_rsqrt_ph(v); }
  friend F16vec16 sqrt(const F16vec16& v) { return _mm256_sqrt_ph(v); }
  friend F16vec16 exp(const F16vec16& v) { return _mm256_exp_ph(v); }
  friend F16vec16 log(const F16vec16& v) { return _mm256_log_ph(v); }
  friend F16vec16 min(const F16vec16& lhs, const F16vec16& rhs) { return _mm256_min_ph(lhs, rhs); }
  friend F16vec16 max(const F16vec16& lhs, const F16vec16& rhs) { return _mm256_max_ph(lhs, rhs); }
  friend F16vec16 ceil(const F16vec16& v) { return _mm256_roundscale_ph(v, _MM_FROUND_CEIL); }
  friend F16vec16 floor(const F16vec16& v) { return _mm256_roundscale_ph(v, _MM_FROUND_FLOOR); }
  friend F16vec16 trunc(const F16vec16& v) { return _mm256_roundscale_ph(v, _MM_FROUND_TO_ZERO); }
  // Note that roundscale is subtly different to std::round. On boundary conditions (e.g., 0.5, 1.5,
  // 2.5, 3.5) the hardware rounds towards the nearest even (i.e., 0, 2, 2, 4). Should round be
  // called that since it doesn't match std::round?  Maybe not, but since it's only the boundary
  // conditions which vary, and float16 of them do round to the expected number anyway, I (dtowner)
  // think that round's behaviour is close enough to what is expected to deserve to be called by this name.
  friend F16vec16 round(const F16vec16& v) { return _mm256_roundscale_ph(v, _MM_FROUND_TO_NEAREST_INT); }

  /// Clamp the input values to be contained within the range [low,high]. If a value is less than low
  /// it will return low. If a value is greater than high it will return high. Other values are
  /// unmodified.
  /// \param values The values to clamp.
  /// \param low The low value in the clamping range.
  /// \param high The high value in the clamping range.
  friend F16vec16 clamp(F16vec16 values, F16vec16 low, F16vec16 high) { return min(max(values, low), high); }

  /// Clamp the input values to the range [-high, high].
  /// \param values The values to clamp.
  /// \param high The magnitude to which to clamp the values. This must be a positive value.
  friend F16vec16 clamp(F16vec16 values, F16vec16 high) { return clamp(values, -high, high); }

  // Expose some intrinsics in a way which allows them to be easily called for the different data types.
  friend F16vec16 blend(int16_t select, F16vec16 src0, F16vec16 src1) { return _mm256_mask_blend_ph(select, src0, src1); }
  friend F16vec16 fmadd (F16vec16 a, F16vec16 b, F16vec16 c) { return _mm256_fmadd_ph(a, b, c); }
  friend F16vec16 fmaddsub (F16vec16 a, F16vec16 b, F16vec16 c) { return _mm256_fmaddsub_ph(a, b, c); }
  friend F16vec16 addsub (F16vec16 lhs, F16vec16 rhs) { return _mm256_fmaddsub_ph(lhs, _mm256_set1_ph(1.0f), rhs); }
  friend F16vec16 subadd (F16vec16 lhs, F16vec16 rhs) { return _mm256_fmsubadd_ph(lhs, _mm256_set1_ph(1.0f), rhs); }

  friend F16vec16 select_eq(F16vec16 a, F16vec16 b, F16vec16 c, F16vec16 d) {
    return blend(_mm256_cmp_ph_mask(a, b, _CMP_EQ_OS), d, c); }
  friend F16vec16 select_lt(F16vec16 a, F16vec16 b, F16vec16 c, F16vec16 d) {
    return blend(_mm256_cmp_ph_mask(a, b, _CMP_LT_OS), d, c); }
  friend F16vec16 select_le(F16vec16 a, F16vec16 b, F16vec16 c, F16vec16 d) {
    return blend(_mm256_cmp_ph_mask(a, b, _CMP_LE_OS), d, c); }
  friend F16vec16 select_gt(F16vec16 a, F16vec16 b, F16vec16 c, F16vec16 d) {
    return blend(_mm256_cmp_ph_mask(a, b, _CMP_GT_OS), d, c); }
  friend F16vec16 select_ge(F16vec16 a, F16vec16 b, F16vec16 c, F16vec16 d) {
    return blend(_mm256_cmp_ph_mask(a, b, _CMP_GE_OS), d, c); }
};

class F16vec32 : public F16vec<__m512h>
{
public:
  F16vec32() = default;
  F16vec32(__m512h v) : F16vec(v) { }
  F16vec32(float16 value) : F16vec(_mm512_set1_ph(value)) { }

  operator __m512h() const { return simd; }

  friend F16vec32 operator+ (const F16vec32& lhs, const F16vec32& rhs) { return _mm512_add_ph(lhs, rhs); }
  friend F16vec32 operator- (const F16vec32& lhs, const F16vec32& rhs) { return _mm512_sub_ph(lhs, rhs); }
  friend F16vec32 operator* (const F16vec32& lhs, const F16vec32& rhs) { return _mm512_mul_ph(lhs, rhs); }
  friend F16vec32 operator/ (const F16vec32& lhs, const F16vec32& rhs) { return _mm512_div_ph(lhs, rhs); }

  F16vec32& operator+= (const F16vec32& rhs) { *this = *this + rhs; return *this; }
  F16vec32& operator-= (const F16vec32& rhs) { *this = *this - rhs; return *this; }
  F16vec32& operator*= (const F16vec32& rhs) { *this = *this * rhs; return *this; }
  F16vec32& operator/= (const F16vec32& rhs) { *this = *this / rhs; return *this; }

  F16vec32 operator- () const { return _mm512_sub_ph(__m512h(), simd); }

  friend F16vec32 abs(const F16vec32& v) { return _mm512_abs_ph(v); }
  friend F16vec32 rcp(const F16vec32& v) { return _mm512_rcp_ph(v); }
  friend F16vec32 rsqrt(const F16vec32& v) { return _mm512_rsqrt_ph(v); }
  friend F16vec32 sqrt(const F16vec32& v) { return _mm512_sqrt_ph(v); }
  friend F16vec32 exp(const F16vec32& v) { return _mm512_exp_ph(v); }
  friend F16vec32 log(const F16vec32& v) { return _mm512_log_ph(v); }
  friend F16vec32 min(const F16vec32& lhs, const F16vec32& rhs) { return _mm512_min_ph(lhs, rhs); }
  friend F16vec32 max(const F16vec32& lhs, const F16vec32& rhs) { return _mm512_max_ph(lhs, rhs); }
  friend F16vec32 ceil(const F16vec32& v) { return _mm512_roundscale_ph(v, _MM_FROUND_CEIL); }
  friend F16vec32 floor(const F16vec32& v) { return _mm512_roundscale_ph(v, _MM_FROUND_FLOOR); }
  friend F16vec32 trunc(const F16vec32& v) { return _mm512_roundscale_ph(v, _MM_FROUND_TO_ZERO); }
  // Note that roundscale is subtly different to std::round. On boundary conditions (e.g., 0.5, 1.5,
  // 2.5, 3.5) the hardware rounds towards the nearest even (i.e., 0, 2, 2, 4). Should round be
  // called that since it doesn't match std::round?  Maybe not, but since it's only the boundary
  // conditions which vary, and float16 of them do round to the expected number anyway, I (dtowner)
  // think that round's behaviour is close enough to what is expected to deserve to be called by this name.
  friend F16vec32 round(const F16vec32& v) { return _mm512_roundscale_ph(v, _MM_FROUND_TO_NEAREST_INT); }

  /// Clamp the input values to be contained within the range [low,high]. If a value is less than low
  /// it will return low. If a value is greater than high it will return high. Other values are
  /// unmodified.
  /// \param values The values to clamp.
  /// \param low The low value in the clamping range.
  /// \param high The high value in the clamping range.
  friend F16vec32 clamp(F16vec32 values, F16vec32 low, F16vec32 high) { return min(max(values, low), high); }

  /// Clamp the input values to the range [-high, high].
  /// \param values The values to clamp.
  /// \param high The magnitude to which to clamp the values. This must be a positive value.
  friend F16vec32 clamp(F16vec32 values, F16vec32 high) { return clamp(values, -high, high); }

  // Expose some intrinsics in a way which allows them to be easily called for the different data types.
  friend F16vec32 blend(int32_t select, F16vec32 src0, F16vec32 src1) { return _mm512_mask_blend_ph(select, src0, src1); }
  friend F16vec32 fmadd (F16vec32 a, F16vec32 b, F16vec32 c) { return _mm512_fmadd_ph(a, b, c); }
  friend F16vec32 fmadd (F16vec32 a, F16vec32 b, F16vec32 c, Mask32 k) { return _mm512_mask3_fmadd_ph(a, b, c, k); }
  friend F16vec32 fmaddsub (F16vec32 a, F16vec32 b, F16vec32 c) { return _mm512_fmaddsub_ph(a, b, c); }
  friend F16vec32 addsub (F16vec32 lhs, F16vec32 rhs) { return _mm512_fmaddsub_ph(lhs, _mm512_set1_ph(1.0f), rhs); }
  friend F16vec32 subadd (F16vec32 lhs, F16vec32 rhs) { return _mm512_fmsubadd_ph(lhs, _mm512_set1_ph(1.0f), rhs); }

  friend F16vec32 select_eq(F16vec32 a, F16vec32 b, F16vec32 c, F16vec32 d) {
    return blend(_mm512_cmp_ph_mask(a, b, _CMP_EQ_OS), d, c); }
  friend F16vec32 select_lt(F16vec32 a, F16vec32 b, F16vec32 c, F16vec32 d) {
    return blend(_mm512_cmp_ph_mask(a, b, _CMP_LT_OS), d, c); }
  friend F16vec32 select_le(F16vec32 a, F16vec32 b, F16vec32 c, F16vec32 d) {
    return blend(_mm512_cmp_ph_mask(a, b, _CMP_LE_OS), d, c); }
  friend F16vec32 select_gt(F16vec32 a, F16vec32 b, F16vec32 c, F16vec32 d) {
    return blend(_mm512_cmp_ph_mask(a, b, _CMP_GT_OS), d, c); }
  friend F16vec32 select_ge(F16vec32 a, F16vec32 b, F16vec32 c, F16vec32 d) {
    return blend(_mm512_cmp_ph_mask(a, b, _CMP_GE_OS), d, c); }
  friend F16vec32 reduce_add_half(F16vec32 &a) {
    const I8vec64 xmmIdx = _mm512_set_epi8(
    13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2,
    13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2,
    13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2,
    13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2);
    a += _mm512_shuffle_f32x4(a, a, _MM_PERM_CDAB);
    a += _mm512_shuffle_ps(a, a, 0x4E);
    a += _mm512_shuffle_ps(a, a, 0x11);
    a += (F16vec32)(_mm512_shuffle_epi8((__m512i)a, xmmIdx));
    return a;
  }

};

// /* Reciprocal */
// inline F16vec32 rcp(const F16vec32 &a) { return _mm512_rcp_ph(a); }
// inline F16vec32 blend(int32_t select, const F16vec32 &a, const F16vec32 &b) { return _mm512_mask_blend_ph(select, a, b); }
// inline F16vec32 fmadd (const F16vec32 &a, const F16vec32 &b, const F16vec32 &c) { return _mm512_fmadd_ph(a, b, c); }
// inline F16vec32 fmaddsub (const F16vec32 &a, const F16vec32 &b, const F16vec32 &c) { return _mm512_fmaddsub_ph(a, b, c); }
// inline F16vec32 addsub (const F16vec32 &a, const F16vec32 &b) { return _mm512_fmaddsub_ph(a, _mm512_set1_ph(1.0f), b); }
// inline F16vec32 subadd (const F16vec32 &a, const F16vec32 &b) { return _mm512_fmsubadd_ph(a, _mm512_set1_ph(1.0f), b); }
// // complex mul a * b
// inline F16vec32 cfmul (const F16vec32 &a, const F16vec32 &b) { return _mm512_fmul_pch(a, b); }
// inline F16vec32 cfmadd (const F16vec32 &src, const F16vec32 &a, const F16vec32 &b) { return _mm512_fmadd_pch(src, a, b); }
// // complex conjugates mul a * conj(b)
// inline F16vec32 cfcmul (const F16vec32 &a, const F16vec32 &b) { return _mm512_fcmul_pch(a, b); }
// inline F16vec32 cfcmadd (const F16vec32 &src, const F16vec32 &a, const F16vec32 &b) { return _mm512_fcmadd_pch(src, a, b); }

class CF16vec16 {
public:
  CF16vec16() = default;

  CF16vec16(__m512h v) : vec(v) {}
  // CF16vec16() { vec = _mm512_set1_ph(0); }

  // TODO: Not 100% sure about it. It simplifies code when casting between FP16 and EPI16,
  // but also causes some ambiguity and requires additional casting to remove this ambiguity.
  // Something to think of.
  explicit CF16vec16(const __m512i& v) {
    vec = _mm512_castsi512_ph(v);
  }

  /// set a value for imag and real.
  CF16vec16(float16 a)
  {
    vec = _mm512_set1_ph(a);
  }
  CF16vec16(double a)
  {
    vec = _mm512_set1_ph(a);
  }
  CF16vec16(float a)
  {
    vec = _mm512_set1_ph(a);
  }
  CF16vec16(int32_t a)
  {
    vec = _mm512_castsi512_ph(_mm512_set1_epi32(a));
  }


  /// Type-punnning isn't the cleanest way to do this, but it consistently generates the fastest code.
  CF16vec16(const std::complex<float16>& e) {
    vec = _mm512_castsi512_ph(_mm512_set1_epi32(*((int*)&e)));
  }

  /// Allow a std::complex style initialisation from a pair of floats.
  CF16vec16(float16 r, float16 i)
  {
    vec = _mm512_mask_blend_ph(0xAAAAAAAA, _mm512_set1_ph(r), _mm512_set1_ph(i));
  }

  /// Conversion operator to underlying type.
  operator  __m512h() const { return vec; }

  /// Access the real part of the vector. Three forms are provided to match the behaviour of
  /// the std::complex type:
  ///  obj.real()      Return the real part of the number as a vector of float16 the size.
  ///  real(obj)       Return the real part of the number as a vector of float16 the size, using a
  ///                  free function.
  ///  obj.real(value) Set the real part of the vector to the given values.
  F16vec16 real() const {
    const auto asInt = _mm512_castph_si512(vec);
    const auto truncToReal = _mm512_cvtepi32_epi16(asInt);
    return _mm256_castsi256_ph(truncToReal);
  }

  friend F16vec16 real(CF16vec16 value) { return value.real(); }

  void real(F16vec16 value) {
    // Zero-extend each 16-bit value into 32-bits, and mask insert it into the original vec.
    const __m512h expandReal = _mm512_castsi512_ph(_mm512_cvtepi16_epi32(_mm256_castph_si256(value)));
    vec = _mm512_mask_blend_ph(0xAAAAAAAA, expandReal, vec);
  }

  /// Access the imag part of the vector. Three forms are provided to match the behaviour of
  /// the std::complex type:
  ///  obj.imag()      Return the imag part of the number as a vector of float16 the size.
  ///  imag(obj)       Return the imag part of the number as a vector of float16 the size, using a
  ///                  free function.
  ///  obj.imag(value) Set the imag part of the vector to the given values.
  F16vec16 imag() const {
    const auto asInt = _mm512_castph_si512(vec);
    const auto truncToImag = _mm512_cvtepi32_epi16(_mm512_srli_epi32(asInt, 16));
    return _mm256_castsi256_ph(truncToImag);
  }

  friend F16vec16 imag(CF16vec16 value) { return value.imag(); }

  void imag(F16vec16 value) {
    // Zero-extend each 16-bit value into 32-bits, shift it left to move it to the imaginary, and then mask insert
    // it into the original vec.
    const auto expandToImag = _mm512_castsi512_ph(_mm512_slli_epi32(_mm512_cvtepi16_epi32(_mm256_castph_si256(value)), 16));
    vec = _mm512_mask_blend_ph(0x55555555, expandToImag, vec);
  }

  /// Unary negation by flipping the sign bit.
  CF16vec16 operator-() const {
      const auto k_signBitMask = _mm512_set1_epi16(0x8000);
      return _mm512_castsi512_ph(_mm512_xor_si512(k_signBitMask, _mm512_castph_si512(vec)));
  }

  CF16vec16& operator*=(CF16vec16 rhs) { vec = _mm512_fmul_pch(vec, rhs.vec); return *this; }
  CF16vec16& operator*=(float16 rhs) { vec = _mm512_mul_ph(vec, _mm512_set1_ph(rhs)); return *this; }
  CF16vec16& operator+=(CF16vec16 rhs) { vec = _mm512_add_ph(vec, rhs.vec); return *this; }
  CF16vec16& operator-=(CF16vec16 rhs) { vec = _mm512_sub_ph(vec, rhs.vec); return *this; }

  friend CF16vec16 operator-(CF16vec16 lhs, CF16vec16 rhs) { return _mm512_sub_ph(lhs.vec, rhs.vec); }
  friend CF16vec16 operator+(CF16vec16 lhs, CF16vec16 rhs) { return _mm512_add_ph(lhs.vec, rhs.vec); }
  friend CF16vec16 operator*(CF16vec16 lhs, CF16vec16 rhs) { return _mm512_fmul_pch(lhs.vec, rhs.vec); }
  friend CF16vec16 operator*(float16 lhs, CF16vec16 rhs) { return _mm512_mul_ph(_mm512_set1_ph(lhs), rhs.vec); }
  friend CF16vec16 operator*(CF16vec16 lhs, float16 rhs) { return rhs * lhs; }

  /// Element-wise multiply. The lhs is likely to be a broadcast constant or the result of a real()
  /// or imag() call.
  friend CF16vec16 operator*(F16vec32 lhs, CF16vec16 rhs) {
    return _mm512_mul_ph(lhs, rhs.vec);
  }

  friend F16vec16 norm(CF16vec16 value)
  {
    // Computing the normal is equivalent to multiplying by the conjugate of itself, leaving a zero
    // imaginary. The alternative way to do it would be to do an element-wise multiply, followed by
    // the addition of the real/imag. That would take more cycles in the case of doing a single
    // norm. Note that if several norms are done, and then added together, the expanded form may be
    // slightly faster since it would reduce the number of FMA operations.
    CF16vec16 normInPlace = _mm512_fcmadd_pch(value.vec, value.vec, __m512h());
    return normInPlace.real();
  }

  /// Flip the sign bit of the imaginary. :TODO: If this were an intrinsic (e.g., _mm512_cfconj_ph),
  /// then it might be easier to create a peephole which converts conj(a) * b into a real conj
  /// instruction.
  friend CF16vec16 conj(CF16vec16 value) {
    const auto k_imagSignBitMask = _mm512_set1_epi32(0x80000000);
    return _mm512_castsi512_ph(_mm512_xor_si512(k_imagSignBitMask, _mm512_castph_si512(value.vec)));
  }

  /// FMA type operations
  friend CF16vec16 fmadd(const CF16vec16& valA, const CF16vec16& valB, const CF16vec16& acc)//[R23.03]Add const
  {
    return _mm512_fmadd_ph(valA.vec, valB.vec, acc.vec);
  }

  friend CF16vec16 fmadd(CF16vec16& valA, CF16vec16& valB, CF16vec16& acc, Mask32 mask)
  {
    return _mm512_mask3_fmadd_ph(valA.vec, valB.vec, acc.vec, mask);
  }
  friend CF16vec16 fmadd(Mask32 mask, CF16vec16& valA,  CF16vec16& valB, CF16vec16& acc)
  {
    return _mm512_maskz_fmadd_ph(mask, valA.vec, valB.vec, acc.vec);
  }
  friend CF16vec16 fma(CF16vec16& valA, CF16vec16& valB, CF16vec16& acc)
  {
    return _mm512_fmadd_pch(valA.vec, valB.vec, acc.vec);
  }

  friend CF16vec16 fmanorm(CF16vec16& val, CF16vec16& acc)
  {
    return _mm512_fcmadd_pch(val.vec, val.vec, acc.vec);
  }

  friend CF16vec16 fmaconj(CF16vec16& valA, CF16vec16& valB, CF16vec16& acc)
  {
    return _mm512_fcmadd_pch(valA.vec, valB.vec, acc.vec);
  }

  friend CF16vec16 fmaconj(Mask16& k, CF16vec16& valA, CF16vec16& valB, CF16vec16& acc)
  {
    return _mm512_maskz_fcmadd_pch(k, valA.vec, valB.vec, acc.vec);
  }

  // complex mul a * b
  friend CF16vec16 fmul (const CF16vec16 &a, const CF16vec16 &b) { return _mm512_fmul_pch(a.vec, b.vec); }

  // complex conjugates mul a * conj(b)
  friend CF16vec16 fmulconj (const CF16vec16 &a, const CF16vec16 &b) { return _mm512_fcmul_pch(a.vec, b.vec); }

  // complex conjugates mul a * conj(b) only for real part and the imag part is zero.
  // friend CF16vec16 dotCReal(CF16vec16 x[1]) {

  //   auto sum = _mm512_mul_ph(x[0], x[0]);

  //   return addRealImag(sum);
  // }

  /// Reciprocal functions
  friend CF16vec16 rcp(CF16vec16& v)
  {
    return CF16vec16(_mm512_mask_blend_ph(0xAAAAAAAA, _mm512_rcp_ph(v), __m512h()));
  }

  friend CF16vec16 rsqrt(CF16vec16& v)
  {
    // return _mm512_castsi512_ph(_mm512_mask_blend_epi16 (0x33333333, _mm512_set1_epi16(0), _mm512_cvtepi16_epi32 (_mm256_castph_si256(_mm512_cvtxps_ph(_mm512_rsqrt14_ps(_mm512_cvtxph_ps(v.real())))))));
    return CF16vec16(_mm512_mask_blend_ph(0xAAAAAAAA, _mm512_rsqrt_ph(v), __m512h()));
  }
  friend CF16vec16 sqrt(CF16vec16& v)
  {
    return CF16vec16(_mm512_mask_blend_ph(0xAAAAAAAA, _mm512_sqrt_ph(v), __m512h()));
  }

  friend CF16vec16 subs(CF16vec16& valA, CF16vec16& valB)
  {
    return _mm512_sub_ph(valA.vec, valB.vec);
  }
  friend CF16vec16 subs(Mask32 mask, CF16vec16& valA, CF16vec16& valB)
  {
    return _mm512_maskz_sub_ph(mask, valA.vec, valB.vec);
  }
  friend CF16vec16 adds(CF16vec16& valA, CF16vec16& valB)
  {
    return _mm512_add_ph(valA.vec, valB.vec);
  }

  friend float16 reduce_add(Mask32 mask, CF16vec16 a)
  {
    return _mm512_reduce_add_ph (_mm512_mask_blend_ph(mask, CF16vec16(), a));
  }

  friend float16 reduce_add(CF16vec16 a)
  {
    return _mm512_reduce_add_ph (a);
  }

  friend CF16vec16 reduce_add_half(CF16vec16 a) {
      a += _mm512_shuffle_f32x4(a, a, _MM_PERM_CDAB); // 11,10,9,8, 15,14,13,12, 3,2,1,0, 7,6,5,4
      a += _mm512_shuffle_ps(a, a, 0x4E); // 3,2,1,0 -> 1,0,3,2
      a += _mm512_shuffle_ps(a, a, 0x11); // 3,2,1,0 -> 2,3,0,1
      return a;
  }

  friend CF16vec16 blend(int16_t select, CF16vec16 src0, CF16vec16 src1) {
    return _mm512_castsi512_ph(_mm512_mask_blend_epi32(select, _mm512_castph_si512(src0), _mm512_castph_si512(src1))); }
  ///\brief Duplicates the real components of a complex vector
  /// into the imaginary components
  ///\param a Complex vector to have real components duplicated
  friend CF16vec16 duplicateReal(CF16vec16& a)
  {
    // Required prefixes to copy REAL components such that
    // input 'abcdef' produces 'aaccee'
    const auto k_order = _mm512_set_epi8(61, 60, 61, 60, 57, 56, 57, 56, 53, 52, 53, 52, 49, 48, 49,
     48, 45, 44, 45, 44, 41, 40, 41, 40, 37, 36, 37, 36, 33, 32, 33, 32, 29, 28, 29, 28, 25, 24, 25,
     24, 21, 20, 21, 20, 17, 16, 17, 16, 13, 12, 13, 12, 9, 8, 9, 8, 5, 4, 5, 4, 1, 0, 1, 0);

    auto temp = _mm512_shuffle_epi8(_mm512_castph_si512(a), k_order);

    return _mm512_castsi512_ph(temp);
  }

  /// Indexing operator for constant values.
  std::complex<float16> operator[](int index) const
  {
    assert(0 <= index && index < 16);
    std::complex<float16>* elements = (std::complex<float16>*)this;
    return elements[index];
  }

  // real negative imag
  inline friend CF16vec16 negImag(const CF16vec16 &a) {return _mm512_mask_sub_ph (a, 0xAAAAAAAA, CF16vec16(), a);}
  inline friend CF16vec16 max(const CF16vec16& lhs, const CF16vec16& rhs) { return _mm512_max_ph(lhs, rhs); }
  inline friend CF16vec16 min(const CF16vec16& lhs, const CF16vec16& rhs) { return _mm512_min_ph(lhs, rhs); }
  inline friend CF16vec16 clamp(CF16vec16 values, CF16vec16 low, CF16vec16 high) { return min(max(values, low), high); }

  __m512h vec;

};
// real + imag in vec
inline CF16vec16 addRealImag(__m512 a) {
  const auto k_order = _mm512_set_epi8(63, 62, 63, 62, 59, 58, 59, 58, 55, 54, 55, 54, 51, 50, 51,
     50, 47, 46, 47, 46, 43, 42, 43, 42, 39, 38, 39, 38, 35, 34, 35, 34, 31, 30, 31, 30, 27, 26, 27,
     26, 23, 22, 23, 22, 19, 18, 19, 18, 15, 14, 15, 14, 11, 10, 11, 10, 7, 6, 7, 6, 3, 2, 3, 2);
  auto temp = _mm512_shuffle_epi8(_mm512_castph_si512(a), k_order);
  auto re = _mm512_maskz_add_ph (0x55555555, a, _mm512_castsi512_ph(temp));
  return CF16vec16(re);
}
// real - imag in vec
inline CF16vec16 subRealImag(__m512 a) {
  const auto k_order = _mm512_set_epi8(63, 62, 63, 62, 59, 58, 59, 58, 55, 54, 55, 54, 51, 50, 51,
     50, 47, 46, 47, 46, 43, 42, 43, 42, 39, 38, 39, 38, 35, 34, 35, 34, 31, 30, 31, 30, 27, 26, 27,
     26, 23, 22, 23, 22, 19, 18, 19, 18, 15, 14, 15, 14, 11, 10, 11, 10, 7, 6, 7, 6, 3, 2, 3, 2);
  auto temp = _mm512_shuffle_epi8(_mm512_castph_si512(a), k_order);
  auto re = _mm512_maskz_sub_ph (0x55555555, a, _mm512_castsi512_ph(temp));
  return CF16vec16(re);
}

// real + imag in vec
inline F16vec32 addIQ(__m512 a) {
  const auto k_order = _mm512_set_epi8(63, 62, 63, 62, 59, 58, 59, 58, 55, 54, 55, 54, 51, 50, 51,
     50, 47, 46, 47, 46, 43, 42, 43, 42, 39, 38, 39, 38, 35, 34, 35, 34, 31, 30, 31, 30, 27, 26, 27,
     26, 23, 22, 23, 22, 19, 18, 19, 18, 15, 14, 15, 14, 11, 10, 11, 10, 7, 6, 7, 6, 3, 2, 3, 2);
  auto temp = _mm512_shuffle_epi8(_mm512_castph_si512(a), k_order);
  auto re = _mm512_maskz_add_ph (0x55555555, a, _mm512_castsi512_ph(temp));
  return F16vec32(re);
}
// real + imag in vec and shirnk in out size
inline F16vec32 addIQ_shrink(__m512h a) {
  const auto k_order = _mm512_set_epi8(63, 62, 63, 62, 59, 58, 59, 58, 55, 54, 55, 54, 51, 50, 51,
     50, 47, 46, 47, 46, 43, 42, 43, 42, 39, 38, 39, 38, 35, 34, 35, 34, 31, 30, 31, 30, 27, 26, 27,
     26, 23, 22, 23, 22, 19, 18, 19, 18, 15, 14, 15, 14, 11, 10, 11, 10, 7, 6, 7, 6, 3, 2, 3, 2);

  const auto pick = _mm512_set_epi16(31, 29, 27, 25, 23, 21, 19, 17,
                                    15, 13, 11, 9, 7, 5, 3, 1,
                                    30, 28, 26, 24, 22, 20, 18, 16,
                                    14, 12, 10, 8, 6, 4, 2, 0);

  auto temp = _mm512_shuffle_epi8(_mm512_castph_si512(a), k_order);

  auto re = _mm512_maskz_add_ph (0x55555555, a, _mm512_castsi512_ph(temp));

  auto permuted = _mm512_permutexvar_epi16(pick, _mm512_castph_si512(re));

  return F16vec32(permuted);
}
/*! \brief common function to mask load fp16 unalign to register*/
inline CF16vec16 loadu(CF16vec16 const *p)
{
    auto x = _mm512_loadu_si512(reinterpret_cast<void const*>(p));
    return _mm512_castsi512_ph(x);
}
/*! \brief common function to store to register */
inline void storeu(CF16vec16 *p, Mask16 &k, const CF16vec16 &src)
{
    const auto x = _mm512_castph_si512 (src);
    _mm512_mask_storeu_epi32(reinterpret_cast<void *>(p), k, x);}
/*! \brief common function to store to register */
inline void storeu(void *p, Mask16 &k, const CF16vec16 &src)
{
    const auto x = _mm512_castph_si512 (src);
    _mm512_mask_storeu_epi32(p, k, x);}
inline void storeu(CF16vec16 *p, const CF16vec16 &src)
{
    const auto x = _mm512_castph_si512 (src);
    _mm512_storeu_si512(reinterpret_cast<void *>(p), x);
}

inline FORCE_INLINE CF16vec16 loadu(const CF16vec16& src, Mask16 k, void const *p)
{
  auto x = _mm512_mask_loadu_epi32(_mm512_castph_si512(src), k, p);
  return _mm512_castsi512_ph(x);
}

//[R23.03] Add
inline FORCE_INLINE CF16vec16 loadu(Mask16 k, CF16vec16 const *p)
{
    return _mm512_castps_ph(_mm512_maskz_loadu_ps(k, reinterpret_cast<void const*>(p)));
}

inline FORCE_INLINE F16vec32 loadu(F16vec32 const *p){return _mm512_castsi512_ph(_mm512_loadu_si512(reinterpret_cast<void const*>(p)));}
inline FORCE_INLINE void storeu(void *p, const F16vec32 &src){_mm512_storeu_si512(p, _mm512_castph_si512 (src));}
inline FORCE_INLINE F16vec32 set(const float16 &a) {return _mm512_set1_ph (a);}
inline FORCE_INLINE float16 reduce_add(const F16vec32 &a) {return _mm512_reduce_add_ph(a);}
inline FORCE_INLINE F16vec32 mulhrs(const F16vec32 &a, const F16vec32 &b) {
    return _mm512_mul_ph(a, b);}

/*! \brief common function to permutexvar to register */
inline FORCE_INLINE CF16vec16 permutexvar(const I32vec16 &index, const CF16vec16 &a) {
    return _mm512_castsi512_ph(_mm512_permutexvar_epi32(index, _mm512_castph_si512(a))); }

/*! \brief common function to shuffle to register */
inline FORCE_INLINE CF16vec16 shuffle(const CF16vec16 &a, const I8vec64 &b) {
    return _mm512_castsi512_ph(_mm512_shuffle_epi8(_mm512_castph_si512(a), b)); }

/*! \brief common function to cmp to register */
inline FORCE_INLINE Mask32 cmpge(const F16vec32 &a, const F16vec32 &b){return _mm512_cmp_ph_mask(a, b, _CMP_GE_OS);}

inline FORCE_INLINE F16vec32 load(F16vec32 const *p) {return _mm512_castsi512_ph(_mm512_load_si512(reinterpret_cast<void const*>(p)));}   //buffer 64bytes align

inline FORCE_INLINE float16 reduce_add(const Mask32 &a,const F16vec32 &b) {return  _mm512_reduce_add_ph(_mm512_mask_blend_ph(a, F16vec32(), b));}

/*! \brief common function to mulhr to register */
inline FORCE_INLINE CF16vec16 mulhrs(const CF16vec16 &a, const CF16vec16 &b) {
    return _mm512_mul_ph(a, b);}

/*! \brief common function to adds to register */
inline FORCE_INLINE F16vec32 add(const F16vec32 &a, const F16vec32 &b) {return _mm512_add_ph(a, b);}
inline FORCE_INLINE F16vec32 add(const F16vec32 &a, const Mask32 &b, const F16vec32 &c, const F16vec32 &d){
  return _mm512_mask_add_ph(a, b, c, d);}
inline FORCE_INLINE CF16vec16 add(const CF16vec16 &a, const Mask32 &b, const CF16vec16 &c, const CF16vec16 &d){
  return _mm512_mask_add_ph(a.vec, b, c.vec, d.vec);}
/*! \brief common function to set a to register */
inline FORCE_INLINE F16vec32 set(const F16vec32 &a, const Mask32 &b, const F16vec32 c) {return blend(b, a, c); }//_mm512_mask_blend_ph

inline FORCE_INLINE void store(void *p, const F16vec32& a) {_mm512_store_si512(reinterpret_cast<void *>(p), _mm512_castph_si512(a));}
inline FORCE_INLINE void store(void *p, const F16vec16& a) {_mm256_store_si256(reinterpret_cast<__m256i *>(p), _mm256_castph_si256(a));}

template<typename SIMD_TYPE, typename ELEMENT_TYPE, int k_numElements>
void
DumpToStream(std::ostream& stream, SIMD_TYPE value)
{
  // Output the values using std::complex's own operator<< to keep things consistent.
  const auto elementPtr = (const std::complex<ELEMENT_TYPE>*)&value;

  for (int i=0; i<k_numElements; ++i)
  {
    // Insert an output separator where needed.
    if (i > 0)
      stream << ' ';

    // Note that the values are output in reverse to match the dvec::operator<<.
    const int revIndex = (k_numElements - 1) - i;
    // stream << '[' << revIndex << "]:" << "r = " << elementPtr[revIndex].real()
    // << " i = " << elementPtr[revIndex].imag();
    stream << '[' << revIndex << "]:" << elementPtr[revIndex].real()
    << " + " << elementPtr[revIndex].imag() << "j";
  }
}

inline std::ostream& operator<<(std::ostream& stream, const CF16vec16& value)
{
  DumpToStream<CF16vec16, half, 16>(stream, value);
  return stream;
}

template<typename HV>
inline std::ostream& operator<<(std::ostream& stream, const F16vec<HV>& value)
{
  // Note that the values are printed in reverse order.
  for (int i = 0; i < F16vec<HV>::k_numElements; ++i)
  {
    const int ri = (F16vec<HV>::k_numElements - 1) - i;
    if (i > 0) stream << ' ';
    stream << '[' << ri << "]:" << (float)value[ri];
  }

  return stream;
}

}
