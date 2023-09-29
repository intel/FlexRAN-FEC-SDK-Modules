//  Copyright (c) 2020 Intel Corporation.
//
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.

/*
 *  Definition of a C++ class interface to MMX(TM) instruction intrinsics.
 *
 */

#ifndef IVEC_H_INCLUDED
#define IVEC_H_INCLUDED

#if !defined __cplusplus
	#error ERROR: This file is only supported in C++ compilations!
#endif /* !__cplusplus */

#include <mmintrin.h>
#include <assert.h>

/* If using MSVC5.0, explicit keyword should be used */
#if (_MSC_VER >= 1100) || defined(__linux__) || defined(__unix__) || defined(__APPLE__)
        #define EXPLICIT explicit
#else
   #if (__INTEL_COMPILER)
        #define EXPLICIT explicit /* If MSVC4.x & Intel, use __explicit */
   #else
        #define EXPLICIT /* nothing */
        #pragma message( "explicit keyword not recognized")
   #endif
#endif

/* Figure out whether and how to define the output operators */
#if defined(_IOSTREAM_) || defined(_CPP_IOSTREAM) || defined(_GLIBCXX_IOSTREAM) || defined (_LIBCPP_IOSTREAM)
#define IVEC_DEFINE_OUTPUT_OPERATORS
#define IVEC_STD std::
#elif defined(_INC_IOSTREAM) || defined(_IOSTREAM_H_)
#define IVEC_DEFINE_OUTPUT_OPERATORS
#define IVEC_STD
#endif

/* Whether C++11 defaulted/deleted functions feature can be used.*/
#if (__cplusplus >= 201103L) && !defined(_DISALLOW_VEC_CPP11_DEFAULTED_FUNCTIONS)
/*
 * This specifically changes default and value initialization behavior:
 * void foo() {
 *      I8vec8 var1;            // default initialization
 *      const I8vec8 var2;      // const object default initialization
 *      I8vec8 var3 = I8vec8(); // value initialization
 *      ...
 * }
 * With use of C++11 defaulted functions the vec classes explicitly declare
 * to use compiler provided default ctor that performs no action (i.e. trivial).
 * As a consequence of that the value of variable "var1" in example above is
 * indeterminate. Otherwise (prior to C++11 or when disabled use of the feature)
 * user-provided default ctor is called (which explicitly performs
 * initialization to zero value). Another consequence of that is constant
 * object declaration ("var2") becomes ill-formed without user-provided
 * default ctor.
 *
 * That change may have noticeable performance impact when array of objects
 * is declared.
 * Although value initialization flow changes too (an object is first
 * zero initialized prior to default ctor) the final result
 * practically end up the same.
 */
#define IVEC_USE_CPP11_DEFAULTED_FUNCTIONS
#endif

class I8vec8;			/* 8 elements, each element a signed or unsigned char data type */
class Is8vec8;			/* 8 elements, each element a signed char data type */
class Iu8vec8;			/* 8 elements, each element an unsigned char data type */
class I16vec4;			/* 4 elements, each element a signed or unsigned short */
class Is16vec4;			/* 4 elements, each element a signed short */
class Iu16vec4;			/* 4 elements, each element an unsigned short */
class I32vec2;			/* 2 elements, each element a signed or unsigned long */
class Is32vec2;			/* 2 elements, each element a signed long */
class Iu32vec2;			/* 2 elements, each element a unsigned long */
class I64vec1;			/* 1 element, a __m64 data type - Base I64vec1 class  */

#define _MM_8UB(element,vector) (*((unsigned char*)&(vector) + (element)))
#define _MM_8B(element,vector) (*((signed char*)&(vector) + (element)))

#define _MM_4UW(element,vector) (*((unsigned short*)&(vector) + (element)))
#define _MM_4W(element,vector) (*((short*)&(vector) + (element)))

#define _MM_2UDW(element,vector) (*((unsigned int*)&(vector) + (element)))
#define _MM_2DW(element,vector) (*((int*)&(vector) + (element)))

#define _MM_QW (*((__int64*)&vec))

/* M64 Class:
 * 1 element, a __m64 data type
 * Contructors & Logical Operations
 */
class M64
{
protected:
    __m64 vec;

public:
#ifdef IVEC_USE_CPP11_DEFAULTED_FUNCTIONS
	M64() = default;
#else
	M64() { vec = _mm_setzero_si64(); }
#endif
	M64(__m64 mm)					{ vec = mm; }
	M64(__int64 mm)					{ vec = _mm_set_pi32((int)(mm >> 32), (int)mm); }
	M64(int i)						{ vec = _m_from_int(i); }

	operator __m64() const			{ return vec; }



};


/* I64vec1 Class:
 * 1 element, a __m64 data type
 * Contains Operations which can operate on any __m64 data type
 */

class I64vec1 : public M64
{
public:
#ifdef IVEC_USE_CPP11_DEFAULTED_FUNCTIONS
	I64vec1() = default;
#else
	I64vec1() : M64() { }
#endif
	I64vec1(__m64 mm) : M64(mm)				{ }
	EXPLICIT I64vec1(int i) : M64(i)		{ }
	EXPLICIT I64vec1(__int64 mm) : M64(mm)	{ }

};

/* I32vec2 Class:
 * 2 elements, each element either a signed or unsigned int
 */
class I32vec2 : public M64
{
public:
#ifdef IVEC_USE_CPP11_DEFAULTED_FUNCTIONS
	I32vec2() = default;
#else
	I32vec2() : M64() { }
#endif
	I32vec2(__m64 mm) : M64(mm) { }
	I32vec2(int i0, int i1) { vec = _mm_set_pi32(i0, i1); }
	EXPLICIT I32vec2(int i) : M64 (i) { }
	EXPLICIT I32vec2(__int64 i): M64(i) {}

};


/* Is32vec2 Class:
 * 2 elements, each element a signed int
 */
class Is32vec2 : public I32vec2
{
public:
#ifdef IVEC_USE_CPP11_DEFAULTED_FUNCTIONS
	Is32vec2() = default;
#else
	Is32vec2() : I32vec2() { }
#endif
	Is32vec2(__m64 mm) : I32vec2(mm) { }
	Is32vec2(signed int i0, signed int i1) : I32vec2(i0, i1) {}
	EXPLICIT Is32vec2(int i) : I32vec2 (i)		{}
	EXPLICIT Is32vec2(__int64 i): I32vec2(i)	{}

	/* Assignment Operator */
	Is32vec2& operator= (const M64 &a)		{ return *this = (Is32vec2) a; }

};


/* Iu32vec2 Class:
 * 2 elements, each element unsigned int
 */
class Iu32vec2 : public I32vec2
{
public:
#ifdef IVEC_USE_CPP11_DEFAULTED_FUNCTIONS
	Iu32vec2() = default;
#else
	Iu32vec2() : I32vec2() { }
#endif
	Iu32vec2(__m64 mm) : I32vec2(mm) { }
	Iu32vec2(unsigned int i0, unsigned int i1) : I32vec2(i0, i1) {}
	EXPLICIT Iu32vec2(int i) : I32vec2 (i)		{ }
	EXPLICIT Iu32vec2(__int64 i) : I32vec2 (i)	{ }

	/* Assignment Operator */
	Iu32vec2& operator= (const M64 &a)		{ return *this = (Iu32vec2) a; }

};


/* I16vec4 Class:
 * 4 elements, each element either a signed or unsigned short
 */
class I16vec4 : public M64
{
public:
#ifdef IVEC_USE_CPP11_DEFAULTED_FUNCTIONS
	I16vec4() = default;
#else
	I16vec4() : M64() { }
#endif
	I16vec4(__m64 mm) : M64(mm) { }
	I16vec4(short i0, short i1, short i2, short i3) {
        vec = _mm_set_pi16(i0, i1, i2, i3);
    }
	EXPLICIT I16vec4(__int64 i) : M64 (i) { }
	EXPLICIT I16vec4(int i) : M64 (i) { }

	/* Assignment Operator */
	I16vec4& operator= (const M64 &a)			{ return *this = (I16vec4) a; }


};


/* Is16vec4 Class:
 * 4 elements, each element signed short
 */
class Is16vec4 : public I16vec4
{
public:
#ifdef IVEC_USE_CPP11_DEFAULTED_FUNCTIONS
	Is16vec4() = default;
#else
	Is16vec4() : I16vec4() { }
#endif
	Is16vec4(__m64 mm) : I16vec4(mm) { }
	Is16vec4(short i0, short i1, short i2, short i3)
            : I16vec4(i0, i1, i2, i3) { }
	EXPLICIT Is16vec4(__int64 i) : I16vec4 (i)	{ }
	EXPLICIT Is16vec4(int i) : I16vec4 (i)		{ }

	/* Assignment Operator */
	Is16vec4& operator= (const M64 &a)		{ return *this = (Is16vec4) a; }

};
/* Iu16vec4 Class:
 * 4 elements, each element unsigned short
 */
class Iu16vec4 : public I16vec4
{
public:
#ifdef IVEC_USE_CPP11_DEFAULTED_FUNCTIONS
	Iu16vec4() = default;
#else
	Iu16vec4() : I16vec4() { }
#endif
	Iu16vec4(__m64 mm) : I16vec4(mm) { }
	Iu16vec4(unsigned short ui0, unsigned short ui1,
                 unsigned short ui2, unsigned short ui3)
            : I16vec4(ui0, ui1, ui2, ui3) { }
	EXPLICIT Iu16vec4(__int64 i) : I16vec4 (i) { }
	EXPLICIT Iu16vec4(int i) : I16vec4 (i) { }

	/* Assignment Operator */
	Iu16vec4& operator= (const M64 &a)		{ return *this = (Iu16vec4) a; }

};


/* I8vec8 Class:
 * 8 elements, each element either unsigned or signed char
 */
class I8vec8 : public M64
{
public:
#ifdef IVEC_USE_CPP11_DEFAULTED_FUNCTIONS
	I8vec8() = default;
#else
	I8vec8() : M64() { }
#endif
	I8vec8(__m64 mm) : M64(mm) { }
	I8vec8(char s0, char s1, char s2, char s3,
           char s4, char s5, char s6, char s7) {
        vec = _mm_set_pi8(s0, s1, s2, s3, s4, s5, s6, s7);
    }
	EXPLICIT I8vec8(__int64 i) : M64 (i) { }
	EXPLICIT I8vec8(int i) : M64 (i) { }

	/* Assignment Operator */
	I8vec8& operator= (const M64 &a)		{ return *this = (I8vec8) a; }


};


/* Is8vec8 Class:
 * 8 elements, each element signed char
 */
class Is8vec8 : public I8vec8
{
public:
#ifdef IVEC_USE_CPP11_DEFAULTED_FUNCTIONS
	Is8vec8() = default;
#else
	Is8vec8() : I8vec8() { }
#endif
	Is8vec8(__m64 mm) : I8vec8(mm) { }
	Is8vec8(signed char s0, signed char s1, signed char s2, signed char s3,
            signed char s4, signed char s5, signed char s6, signed char s7)
            : I8vec8(s0, s1, s2, s3, s4, s5, s6, s7) { }
	EXPLICIT Is8vec8(__int64 i) : I8vec8 (i) { }
	EXPLICIT Is8vec8(int i) : I8vec8 (i) { }

	/* Assignment Operator */
	Is8vec8& operator= (const M64 &a)		{ return *this = (Is8vec8) a; }

};

/* Iu8vec8 Class:
 * 8 elements, each element unsigned char
 */
class Iu8vec8 : public I8vec8
{
public:
#ifdef IVEC_USE_CPP11_DEFAULTED_FUNCTIONS
	Iu8vec8() = default;
#else
	Iu8vec8() : I8vec8() { }
#endif
	Iu8vec8(__m64 mm) : I8vec8(mm) { }
	Iu8vec8(unsigned char s0, unsigned char s1, unsigned char s2,
            unsigned char s3, unsigned char s4, unsigned char s5,
            unsigned char s6, unsigned char s7)
            : I8vec8(s0, s1, s2, s3, s4, s5, s6, s7) { }
	EXPLICIT Iu8vec8(__int64 i) : I8vec8 (i) { }
	EXPLICIT Iu8vec8(int i) : I8vec8 (i) { }

};


#endif // IVEC_H_INCLUDED


//  Copyright (c) 2020 Intel Corporation.
//
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.

/*
 *  Definition of a C++ class interface to Streaming SIMD Extension intrinsics.
 *
 *
 *	File name : fvec.h  Fvec class definitions
 *
 *	Concept: A C++ abstraction of Streaming SIMD Extensions designed to improve
 *
 *  programmer productivity.  Speed and accuracy are sacrificed for utility.
 *
 *	Facilitates an easy transition to compiler intrinsics
 *
 *	or assembly language.
 *
 *	F32vec4:	4 packed single precision
 *				32-bit floating point numbers
*/

#ifndef FVEC_H_INCLUDED
#define FVEC_H_INCLUDED

#if !defined __cplusplus
	#error ERROR: This file is only supported in C++ compilations!
#endif /* !__cplusplus */

#include <ia32intrin.h>
#include <assert.h>
#include <ivec.h>

#pragma pack(push,16) /* Must ensure class & union 16-B aligned */

/* If using MSVC5.0, explicit keyword should be used */
#if (_MSC_VER >= 1100) || defined(__linux__) || defined(__unix__) || defined(__APPLE__)
        #define EXPLICIT explicit
#else
   #if (__INTEL_COMPILER)
        #define EXPLICIT explicit /* If MSVC4.x & Intel, use __explicit */
   #else
        #define EXPLICIT /* nothing */
        #pragma message( "explicit keyword not recognized")
   #endif
#endif

/* Figure out whether and how to define the output operators */
#if defined(_IOSTREAM_) || defined(_CPP_IOSTREAM) || defined(_GLIBCXX_IOSTREAM) || defined (_LIBCPP_IOSTREAM)
#define FVEC_DEFINE_OUTPUT_OPERATORS
#define FVEC_STD std::
#elif defined(_INC_IOSTREAM) || defined(_IOSTREAM_H_)
#define FVEC_DEFINE_OUTPUT_OPERATORS
#define FVEC_STD
#endif

/* Whether C++11 defaulted/deleted functions feature can be used.*/
#if (__cplusplus >= 201103L) && !defined(_DISALLOW_VEC_CPP11_DEFAULTED_FUNCTIONS)
/*
 * This specifically changes default and value initialization behavior:
 * void foo() {
 *      I8vec8 var1;            // default initialization
 *      const I8vec8 var2;      // const object default initialization
 *      I8vec8 var3 = I8vec8(); // value initialization
 *      ...
 * }
 * With use of C++11 defaulted functions the vec classes explicitly declare
 * to use compiler provided default ctor that performs no action (i.e. trivial).
 * As a consequence of that the value of variable "var1" in example above is
 * indeterminate. Otherwise (prior to C++11 or when disabled use of the feature)
 * user-provided default ctor is called (which explicitly performs
 * initialization to zero value). Another consequence of that is constant
 * object declaration ("var2") becomes ill-formed without user-provided
 * default ctor.
 *
 * That change may have noticeable performance impact when array of objects
 * is declared.
 * Although value initialization flow changes too (an object is first
 * zero initialized prior to default ctor) the final result
 * practically end up the same.
 */
#define FVEC_USE_CPP11_DEFAULTED_FUNCTIONS
#endif

const union
{
    int i[4];
    __m128 m;
} __f32vec4_abs_mask_cheat = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};

class F32vec4
{
protected:
    __m128 vec;
public:

	/* Constructors: __m128, 4 floats, 1 float/double */

#ifdef FVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    F32vec4() = default;
#else
	F32vec4() { vec = _mm_setzero_ps(); }
#endif

	/* initialize 4 SP FP with __m128 data type */
	F32vec4(__m128 m)			{ vec = m;}

	/* initialize 4 SP FPs with 4 floats */
	F32vec4(float f3, float f2, float f1, float f0)	{ vec= _mm_set_ps(f3,f2,f1,f0); }

	/* Explicitly initialize each of 4 SP FPs with same float */
	EXPLICIT F32vec4(float f)	{ vec = _mm_set_ps1(f); }

	/* Explicitly initialize each of 4 SP FPs with same double */
	EXPLICIT F32vec4(double d)	{ vec = _mm_set_ps1((float) d); }

	/* Assignment operations */

	F32vec4& operator =(float f) { vec = _mm_set_ps1(f); return *this; }

	F32vec4& operator =(double d) {
        vec = _mm_set_ps1((float) d);
        return *this;
    }

};


class F32vec1
{
protected:
    __m128 vec;
public:

	/* Constructors: __m128, 1 float/double, 1 integer */

#ifdef FVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    F32vec1() = default;
#else
	F32vec1() { vec = _mm_setzero_ps(); }
#endif

	F32vec1(int i)		{ vec = _mm_cvt_si2ss(vec,i);};

	/* Initialize each of 4 SP FPs with same float */
	EXPLICIT F32vec1(float f)	{ vec = _mm_set_ss(f); }

	/* Initialize each of 4 SP FPs with same float */
	EXPLICIT F32vec1(double d)	{ vec = _mm_set_ss((float) d); }

	/* initialize with __m128 data type */
	F32vec1(__m128 m)	{ vec = m; }

	/* Conversion functions */
	operator  __m128() const	{ return vec; }		/* Convert to float */

};

						/* Conditional Selects:*/
/*(a OP b)? c : d; where OP is any compare operator
Macros expand to conditional selects which use all compare intrinsics.
Example:
friend F32vec1 select_eq(const F32vec1 &a, const F32vec1 &b, const F32vec1 &c, const F32vec1 &d)
{
	F32vec1 mask = _mm_cmpeq_ss(a,b);
	return( (mask & c) | F32vec1((_mm_andnot_ps(mask,d))));
}
*/
#pragma pack(pop) /* 16-B aligned */
#endif /* FVEC_H_INCLUDED */



//  Copyright (c) 2020 Intel Corporation.
//
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception.

/*
 *  Definition of C++ class interfaces to the Intel(R) Pentium(R) 4 processor
 *  SSE2 intrinsics, and Intel(R) Advanced Vector Extensions 512 intrinsics.
 *
 *	File name : dvec.h  class definitions
 *
 *	Concept: A C++ abstraction of simd intrinsics designed to
 *      improve programmer productivity.  Speed and accuracy are
 *      sacrificed for utility.  Facilitates an easy transition to
 *      compiler intrinsics or assembly language.
 *
 */

#ifndef DVEC_H_INCLUDED
#define DVEC_H_INCLUDED

#if !defined __cplusplus
	#error ERROR: This file is only supported in C++ compilations!
#endif /* !__cplusplus */

#include <ia32intrin.h>
#include <assert.h>
#include <fvec.h>

#pragma pack(push,16) /* Must ensure class & union 16-B aligned */


/* If using MSVC5.0, explicit keyword should be used */
#if (_MSC_VER >= 1100) || defined (__linux__) || defined(__unix__) || defined(__APPLE__)
        #define EXPLICIT explicit
#else
   #if (__INTEL_COMPILER)
        #define EXPLICIT explicit /* If MSVC4.x & Intel, use __explicit */
   #else
        #define EXPLICIT /* nothing */
        #pragma message( "explicit keyword not recognized")
   #endif
#endif

/* Figure out whether and how to define the output operators */
#if defined(_IOSTREAM_) || defined(_CPP_IOSTREAM) || defined(_GLIBCXX_IOSTREAM) || defined (_LIBCPP_IOSTREAM)
// #define DVEC_DEFINE_OUTPUT_OPERATORS
#define DVEC_STD std::
#elif defined(_INC_IOSTREAM) || defined(_IOSTREAM_H_)
// #define DVEC_DEFINE_OUTPUT_OPERATORS
#define DVEC_STD
#endif

/* Whether C++11 defaulted/deleted functions feature can be used.*/
#if (__cplusplus >= 201103L) && !defined(_DISALLOW_VEC_CPP11_DEFAULTED_FUNCTIONS)
/*
 * This specifically changes default and value initialization behavior:
 * void foo() {
 *      I8vec8 var1;            // default initialization
 *      const I8vec8 var2;      // const object default initialization
 *      I8vec8 var3 = I8vec8(); // value initialization
 *      ...
 * }
 * With use of C++11 defaulted functions the vec classes explicitly declare
 * to use compiler provided default ctor that performs no action (i.e. trivial).
 * As a consequence of that the value of variable "var1" in example above is
 * indeterminate. Otherwise (prior to C++11 or when disabled use of the feature)
 * user-provided default ctor is called (which explicitly performs
 * initialization to zero value). Another consequence of that is constant
 * object declaration ("var2") becomes ill-formed without user-provided
 * default ctor.
 *
 * That change may have noticeable performance impact when array of objects
 * is declared.
 * Although value initialization flow changes too (an object is first
 * zero initialized prior to default ctor) the final result
 * practically end up the same.
 */
#define DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
#endif

const union
{
    int i[4];
    __m128d m;
} __f64vec2_abs_mask_cheat = {0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff};

#define _f64vec2_abs_mask ((F64vec2)__f64vec2_abs_mask_cheat.m)

/* EMM Functionality Intrinsics */

class I8vec16;			/* 16 elements, each element a signed or unsigned char data type */
class Is8vec16;			/* 16 elements, each element a signed char data type */
class Iu8vec16;			/* 16 elements, each element an unsigned char data type */
class I16vec8;			/* 8 elements, each element a signed or unsigned short */
class Is16vec8;			/* 8 elements, each element a signed short */
class Iu16vec8;			/* 8 elements, each element an unsigned short */
class I32vec4;			/* 4 elements, each element a signed or unsigned long */
class Is32vec4;			/* 4 elements, each element a signed long */
class Iu32vec4;			/* 4 elements, each element an unsigned long */
class I64vec2;			/* 2 element, each a __m64 data type */
class I128vec1;			/* 1 element, a __m128i data type */

#define _MM_16UB(element,vector) (*((unsigned char*)&(vector) + (element)))
#define _MM_16B(element,vector) (*((signed char*)&(vector) + (element)))

#define _MM_8UW(element,vector) (*((unsigned short*)&(vector) + (element)))
#define _MM_8W(element,vector) (*((short*)&(vector) + (element)))

#define _MM_4UDW(element,vector) (*((unsigned int*)&(vector) + (element)))
#define _MM_4DW(element,vector) (*((int*)&(vector) + (element)))

#define _MM_2QW(element,vector) (*((__int64*)&(vector) + (element)))

/* We need a m128i constant, keeping performance in mind*/



/* M128 Class:
 * 1 element, a __m128i data type
 * Contructors & Logical Operations
 */

class M128
{
protected:
    __m128i vec;

public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    M128() = default;
#else
    M128() { vec = _mm_setzero_si128(); }
#endif
	M128(__m128i mm)					{ vec = mm; }

	operator __m128i() const			{ return vec; }

	/* Logical Operations */

};


/* I128vec1 Class:
 * 1 element, a __m128i data type
 * Contains Operations which can operate on any __m128i data type
 */

class I128vec1 : public M128
{
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    I128vec1() = default;
#else
    I128vec1() : M128() { }
#endif
	I128vec1(__m128i mm) : M128(mm)	{ }


};

/* I64vec2 Class:
 * 2 elements, each element signed or unsigned 64-bit integer
 */
class I64vec2 : public M128
{
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    I64vec2() = default;
#else
    I64vec2() : M128() { }
#endif
	I64vec2(__m128i mm) : M128(mm) { }

	I64vec2(__m64 q1, __m64 q0)
	{
		_MM_2QW(0,vec) = *(__int64*)&q0;
		_MM_2QW(1,vec) = *(__int64*)&q1;
	}

	/* Assignment Operator */
	I64vec2& operator= (const M128 &a) { return *this = (I64vec2) a; }

	/* Element Access for Debug, No data modified */
	const __int64& operator[](int i) const {
		assert(static_cast<unsigned int>(i) < 2);	/* Only 2 elements to access */
		return _MM_2QW(i,vec);
	}

	/* Element Access and Assignment for Debug */
	__int64& operator[](int i) {
		assert(static_cast<unsigned int>(i) < 2);	/* Only 2 elements to access */
		return _MM_2QW(i,vec);
	}
};

/* I32vec4 Class:
 * 4 elements, each element either a signed or unsigned int
 */
class I32vec4 : public M128
{
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    I32vec4() = default;
#else
    I32vec4() : M128() { }
#endif
	I32vec4(__m128i mm) : M128(mm) { }
	I32vec4(int i3, int i2, int i1, int i0) {
        vec = _mm_set_epi32(i3, i2, i1, i0);
    }

	/* Assignment Operator */
	I32vec4& operator= (const M128 &a)			{ return *this = (I32vec4) a; }


};

/* Is32vec4 Class:
 * 4 elements, each element signed integer
 */
class Is32vec4 : public I32vec4
{
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    Is32vec4() = default;
#else
    Is32vec4() : I32vec4() { }
#endif
	Is32vec4(__m128i mm) : I32vec4(mm) { }
	Is32vec4(int i3, int i2, int i1, int i0) : I32vec4(i3, i2, i1, i0) { }

	/* Assignment Operator */
	Is32vec4& operator= (const M128 &a)		{ return *this = (Is32vec4) a; }

};

/* Compares */


/* Iu32vec4 Class:
 * 4 elements, each element unsigned int
 */
class Iu32vec4 : public I32vec4
{
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    Iu32vec4() = default;
#else
    Iu32vec4() : I32vec4() { }
#endif
	Iu32vec4(__m128i mm) : I32vec4(mm) { }
	Iu32vec4(unsigned int ui3, unsigned int ui2,
             unsigned int ui1, unsigned int ui0)
            : I32vec4(ui3, ui2, ui1, ui0) { }

	/* Assignment Operator */
	Iu32vec4& operator= (const M128 &a)		{ return *this = (Iu32vec4) a; }

};

/* I16vec8 Class:
 * 8 elements, each element either unsigned or signed short
 */
class I16vec8 : public M128
{
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    I16vec8() = default;
#else
    I16vec8() : M128() { }
#endif
	I16vec8(__m128i mm) : M128(mm) { }
	I16vec8(short s7, short s6, short s5, short s4,
            short s3, short s2, short s1, short s0) {
        vec = _mm_set_epi16(s7, s6, s5, s4, s3, s2, s1, s0);
    }

	/* Assignment Operator */
	I16vec8& operator= (const M128 &a)		{ return *this = (I16vec8) a; }



};



/* Is16vec8 Class:
 * 8 elements, each element signed short
 */
class Is16vec8 : public I16vec8
{
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    Is16vec8() = default;
#else
    Is16vec8() : I16vec8() { }
#endif
	Is16vec8(__m128i mm) : I16vec8(mm) { }
	Is16vec8(signed short s7, signed short s6, signed short s5,
             signed short s4, signed short s3, signed short s2,
             signed short s1, signed short s0)
            : I16vec8(s7, s6, s5, s4, s3, s2, s1, s0) { }

	/* Assignment Operator */
	Is16vec8& operator= (const M128 &a)		{ return *this = (Is16vec8) a; }

};


/* Iu16vec8 Class:
 * 8 elements, each element unsigned short
 */
class Iu16vec8 : public I16vec8
{
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    Iu16vec8() = default;
#else
    Iu16vec8() : I16vec8() { }
#endif
	Iu16vec8(__m128i mm) : I16vec8(mm) { }
	Iu16vec8(unsigned short s7, unsigned short s6, unsigned short s5,
             unsigned short s4, unsigned short s3, unsigned short s2,
             unsigned short s1, unsigned short s0)
            : I16vec8(s7, s6, s5, s4, s3, s2, s1, s0) { }

	/* Assignment Operator */
	Iu16vec8& operator= (const M128 &a)		{ return *this = (Iu16vec8) a; }

};

/* I8vec16 Class:
 * 16 elements, each element either unsigned or signed char
 */
class I8vec16 : public M128
{
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    I8vec16() = default;
#else
    I8vec16() : M128() { }
#endif
	I8vec16(__m128i mm) : M128(mm) { }
    I8vec16(char s15, char s14, char s13, char s12, char s11, char s10,
            char s9, char s8, char s7, char s6, char s5, char s4,
            char s3, char s2, char s1, char s0) {
        vec = _mm_set_epi8(s15, s14, s13, s12, s11, s10, s9, s8,
                           s7, s6, s5, s4, s3, s2, s1, s0);
    }

	/* Assignment Operator */
	I8vec16& operator= (const M128 &a)		{ return *this = (I8vec16) a; }


};

/* Is8vec16 Class:
 * 16 elements, each element a signed char
 */
class Is8vec16 : public I8vec16
{
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    Is8vec16() = default;
#else
    Is8vec16() : I8vec16()  { }
#endif
	Is8vec16(__m128i mm) : I8vec16(mm) { }
        Is8vec16(char s15, char s14, char s13, char s12, char s11, char s10,
                 char s9, char s8, char s7, char s6, char s5, char s4,
                 char s3, char s2, char s1, char s0)
            : I8vec16(s15, s14, s13, s12, s11, s10, s9, s8,
                      s7, s6, s5, s4, s3, s2, s1, s0) { }

	/* Assignment Operator */
	Is8vec16& operator= (const M128 &a)		{ return *this = (Is8vec16) a; }

};


/* Iu8vec16 Class:
 * 16 elements, each element an unsigned char
 */
class Iu8vec16 : public I8vec16
{
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    Iu8vec16() = default;
#else
    Iu8vec16() : I8vec16() { }
#endif
	Iu8vec16(__m128i mm) : I8vec16(mm) { }
        Iu8vec16(unsigned char u15, unsigned char u14, unsigned char u13,
                 unsigned char u12, unsigned char u11, unsigned char u10,
                 unsigned char u9, unsigned char u8, unsigned char u7,
                 unsigned char u6, unsigned char u5, unsigned char u4,
                 unsigned char u3, unsigned char u2, unsigned char u1,
                 unsigned char u0)
            : I8vec16(u15, u14, u13, u12, u11, u10, u9, u8,
                      u7, u6, u5, u4, u3, u2, u1, u0) { }

	/* Assignment Operator */
	Iu8vec16& operator= (const M128 &a)		{ return *this = (Iu8vec16) a; }

};


class F64vec2
{
protected:
   	 __m128d vec;
public:

	/* Constructors: __m128d, 2 doubles */
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
     F64vec2() = default;
#else
     F64vec2() { vec = _mm_setzero_pd(); }
#endif

	/* initialize 2 DP FP with __m128d data type */
	F64vec2(__m128d m)					{ vec = m;}

	/* initialize 2 DP FPs with 2 doubles */
	F64vec2(double d1, double d0)		                { vec= _mm_set_pd(d1,d0); }

	/* Explicitly initialize each of 2 DP FPs with same double */
	EXPLICIT F64vec2(double d)	{ vec = _mm_set1_pd(d); }

	/* Conversion functions */
	operator  __m128d() const	{ return vec; }		/* Convert to __m128d */

	/* Element Access Only, no modifications to elements*/
	const double& operator[](int i) const {
		/* Assert enabled only during debug /DDEBUG */
		assert((0 <= i) && (i <= 1));			/* User should only access elements 0-1 */
		double *dp = (double*)&vec;
		return *(dp+i);
	}
	/* Element Access and Modification*/
	double& operator[](int i) {
		/* Assert enabled only during debug /DDEBUG */
		assert((0 <= i) && (i <= 1));			/* User should only access elements 0-1 */
		double *dp = (double*)&vec;
		return *(dp+i);
	}
};

						/* Miscellaneous */
#pragma pack(pop) /* 16-B aligned */

/******************************************************************************/
/************** Interface classes for Intel(R) AVX intrinsics *****************/
/******************************************************************************/

/* The Microsoft compiler (version VS2008 or older) cannot handle the #pragma pack(push,32) */
#if !defined(_MSC_VER) || (_MSC_VER >= 1600)
#pragma pack(push,32)
#endif

/*
 * class F32vec8
 *
 * Represents 256-bit vector composed of 8 single precision floating point elements.
 */
class F32vec8
{
protected:
    __m256 vec;

public:

    /* Constructors: __m256, 8 floats, 1 float */
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    F32vec8() = default;
#else
    F32vec8() { vec = _mm256_setzero_ps(); }
#endif

    /* initialize 8 SP FP with __m256 data type */
    F32vec8(__m256 m) { vec = m; }

    /* initialize 8 SP FPs with 8 floats */
    F32vec8(float f7, float f6, float f5, float f4, float f3, float f2, float f1, float f0)
    {
        vec = _mm256_set_ps(f7,f6,f5,f4,f3,f2,f1,f0);
    }

    /* Explicitly initialize each of 8 SP FPs with same float */
    EXPLICIT F32vec8(float f)   { vec = _mm256_set1_ps(f); }

    /* Explicitly initialize each of 8 SP FPs with same double */
    EXPLICIT F32vec8(double d)  { vec = _mm256_set1_ps((float) d); }

    /* Assignment operations */
    F32vec8& operator =(float f)
    {
        vec = _mm256_set1_ps(f);
        return *this;
    }

    F32vec8& operator =(double d)
    {
        vec = _mm256_set1_ps((float) d);
        return *this;
    }

    /* Conversion functions */
    operator  __m256() const { return vec; }


    /* Element Access Only, no modifications to elements*/
    const float& operator[](int i) const {
        /* Assert enabled only during debug /DDEBUG */
        assert((0 <= i) && (i <= 7));
        float *fp = (float*)&vec;
        return *(fp+i);
    }

    /* Element Access and Modification*/
    float& operator[](int i) {
        /* Assert enabled only during debug /DDEBUG */
        assert((0 <= i) && (i <= 7));
        float *fp = (float*)&vec;
        return *(fp+i);
    }
};

            /* Miscellaneous */
/*
 * class F64vec4
 *
 * Represents 256-bit vector composed of 4 double precision floating point elements.
 */
class F64vec4
{
protected:
    __m256d vec;

public:

    /* Constructors: __m256d, 4 doubles */
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    F64vec4() = default;
#else
    F64vec4() { vec = _mm256_setzero_pd(); }
#endif

    /* initialize 4 DP FP with __m256d data type */
    F64vec4(__m256d m) { vec = m; }

    /* initialize 4 DP FPs with 4 doubles */
    F64vec4(double d3, double d2, double d1, double d0)
    {
        vec = _mm256_set_pd(d3,d2,d1,d0);
    }

    /* Explicitly initialize each of 4 DP FPs with same double */
    EXPLICIT F64vec4(double d) { vec = _mm256_set1_pd(d); }

    /* Conversion functions */
    operator  __m256d() const { return vec; }


    /* Debug Features */
#if defined(DVEC_DEFINE_OUTPUT_OPERATORS)
    /* Output */
    friend DVEC_STD ostream & operator<<(DVEC_STD ostream &os,
                                         const F64vec4 &a) {
        /* To use: cout << "Elements of F64vec4 fvec are: " << fvec; */
        double *dp = (double*) &a;
        os <<  "[3]:" << *(dp+3)
           << " [2]:" << *(dp+2)
           << " [3]:" << *(dp+1)
           << " [0]:" << *dp;
        return os;
    }
#endif

    /* Element Access Only, no modifications to elements */
    const double& operator[](int i) const {
        /* Assert enabled only during debug /DDEBUG */
        assert((0 <= i) && (i <= 3));
        double *dp = (double*)&vec;
        return *(dp+i);
    }
    /* Element Access and Modification*/
    double& operator[](int i) {
        /* Assert enabled only during debug /DDEBUG */
        assert((0 <= i) && (i <= 3));
        double *dp = (double*)&vec;
        return *(dp+i);
    }
};

            /* Miscellaneous */

//
// Interface classes for 256 bit vectors with integer elements.
//

class M256;             // 1 element, a __m256i data type
class I64vec4;          // 4 element, each a __int64 data type
class Is64vec4;         // 4 element, each a signed __int64 data type
class Iu64vec4;         // 4 element, each a unsigned __int64 data type
class I32vec8;          // 8 elements, each element a signed or unsigned __int32
class Is32vec8;         // 8 elements, each element a signed __int32
class Iu32vec8;         // 8 elements, each element a unsigned __int32
class I16vec16;         // 16 elements, each element a signed or unsigned short
class Is16vec16;        // 16 elements, each element a signed short
class Iu16vec16;        // 16 elements, each element an unsigned short
class I8vec32;          // 32 elements, each element a signed or unsigned char
class Is8vec32;         // 32 elements, each element a signed char
class Iu8vec32;         // 32 elements, each element an unsigned char

#define _MM_nUQW(elem, vectr)   (*((unsigned __int64*)&(vectr) + (elem)))
#define _MM_nQW(elem,vectr)     (*((__int64*)&(vectr) + (elem)))
#define _MM_nUDW(elem,vectr)    (*((unsigned int*)&(vectr) + (elem)))
#define _MM_nDW(elem,vectr)     (*((int*)&(vectr) + (elem)))
#define _MM_nUW(elem,vectr)     (*((unsigned short*)&(vectr) + (elem)))
#define _MM_nW(elem,vectr)      (*((short*)&(vectr) + (elem)))
#define _MM_nUB(elem,vectr)     (*((unsigned char*)&(vectr) + (elem)))
#define _MM_nB(elem,vectr)      (*((signed char*)&(vectr) + (elem)))



//
// M256 Class:
// 1 element, a __m256i data type
// Contructors & Logical Operations
//
class M256 {
protected:
    __m256i vec;

public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    M256() = default;
#else
    M256() { vec = _mm256_setzero_si256(); }
#endif
    M256(__m256i mm) { vec = mm; }

    operator __m256i() const { return vec; }


};

//
// I64vec4 Class:
// 4 signed or unsigned 64-bit integer elements
//
class I64vec4 : public M256 {
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    I64vec4() = default;
#else
    I64vec4() : M256() {}
#endif
    I64vec4(__m256i mm) : M256(mm) {}

    EXPLICIT I64vec4(__int64 i) {
        vec = _mm256_set1_epi64x(i);
    }
    EXPLICIT I64vec4(int i) {
        vec = _mm256_set1_epi64x(i);
    }

    I64vec4(__int64 q3, __int64 q2, __int64 q1, __int64 q0) {
        vec = _mm256_set_epi64x(q3,q2,q1,q0);
    }

};

//
// Is64vec4 Class:
// 4 signed 64-bit integer elements
//
class Is64vec4 : public I64vec4 {
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    Is64vec4() = default;
#else
    Is64vec4() : I64vec4() {}
#endif
    Is64vec4(__m256i mm) : I64vec4(mm) {}
    EXPLICIT Is64vec4(int i) : I64vec4(i) {}
    EXPLICIT Is64vec4(__int64 i) : I64vec4(i) {}

    Is64vec4(__int64 q3, __int64 q2, __int64 q1, __int64 q0)
        : I64vec4(q3,q2,q1,q0) {}

    // copy ctr
    Is64vec4(const M256 &m) : I64vec4(m) {}


    // element access for debug, no data modified
    const __int64& operator[](int i) const {
        assert(static_cast<unsigned int>(i) < 4);
        return _MM_nQW(i,vec);
    }

    // element access and assignment for debug
    __int64& operator[](int i) {
        assert(static_cast<unsigned int>(i) < 4);
        return _MM_nQW(i,vec);
    }
};



//
// Iu64vec4 Class:
// 4 unsigned 64-bit integer elements
//
class Iu64vec4 : public I64vec4 {
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    Iu64vec4() = default;
#else
    Iu64vec4() : I64vec4() {}
#endif
    Iu64vec4(__m256i mm) : I64vec4(mm) {}

    EXPLICIT Iu64vec4(unsigned int ui)
        : I64vec4(static_cast<__int64>(ui)) {}
    EXPLICIT Iu64vec4(unsigned __int64 ui)
        : I64vec4(static_cast<__int64>(ui)) {}

    Iu64vec4(unsigned __int64 q3,unsigned __int64 q2,
             unsigned __int64 q1, unsigned __int64 q0)
        : I64vec4(q3,q2,q1,q0) {}



    // element access for debug, no data modified
    const unsigned __int64& operator[](int i) const {
        assert(static_cast<unsigned int>(i) < 4);
        return _MM_nUQW(i,vec);
    }

    // element access and assignment for debug
    unsigned __int64& operator[](int i) {
        assert(static_cast<unsigned int>(i) < 4);
        return _MM_nUQW(i,vec);
    }
};

//
// I32vec8 Class:
// 8 signed or unsigned 32-bit integer elements
//
class I32vec8 : public M256 {
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    I32vec8() = default;
#else
    I32vec8() : M256() {}
#endif
    I32vec8(__m256i mm) : M256(mm) {}
    EXPLICIT I32vec8(int i) {
        vec = _mm256_set1_epi32(i);
    }

    I32vec8(int i7, int i6, int i5, int i4,
            int i3, int i2, int i1, int i0) {
        vec = _mm256_set_epi32(i7, i6, i5, i4,
                               i3, i2, i1, i0);
    }

    // copy ctr
    I32vec8(const M256 &m) : M256(m) {}

};


//
// Is32vec8 Class:
// 8 signed 32-bit integer elements
//
class Is32vec8 : public I32vec8 {
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    Is32vec8() = default;
#else
    Is32vec8() : I32vec8() {}
#endif
    Is32vec8(__m256i mm) : I32vec8(mm) {}
    EXPLICIT Is32vec8(int i) : I32vec8(i) {}

    Is32vec8(int i7, int i6, int i5, int i4,
             int i3, int i2, int i1, int i0)
        : I32vec8(i7, i6, i5, i4, i3, i2, i1, i0) {}

    Is32vec8(const M256 &m) : I32vec8(m) {}

};

//
// Iu32vec8 Class:
// 8 unsigned 32-bit integer elements
//
class Iu32vec8 : public I32vec8 {
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    Iu32vec8() = default;
#else
    Iu32vec8() : I32vec8() {}
#endif
    Iu32vec8(__m256i mm) : I32vec8(mm) {}
    EXPLICIT Iu32vec8(unsigned int ui)
        : I32vec8(static_cast<int>(ui)) {}

    Iu32vec8(unsigned int ui7, unsigned int ui6,
             unsigned int ui5, unsigned int ui4,
             unsigned int ui3, unsigned int ui2,
             unsigned int ui1, unsigned int ui0)
        : I32vec8(ui7, ui6, ui5, ui4, ui3, ui2, ui1, ui0) {}

    // copy ctr
    Iu32vec8(const M256 &m) : I32vec8(m) {}

};



//
// I16vec16 Class:
// 16 unsigned or signed 16-bit integer elements
//
class I16vec16 : public M256 {
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    I16vec16() = default;
#else
    I16vec16() : M256() {}
#endif
    I16vec16(__m256i mm) : M256(mm) {}
    I16vec16(short s15, short s14, short s13, short s12,
             short s11, short s10, short s9, short s8,
             short s7, short s6, short s5, short s4,
             short s3, short s2, short s1, short s0) {
        vec = _mm256_set_epi16(s15, s14, s13, s12,
                               s11, s10, s9, s8,
                               s7, s6, s5, s4,
                               s3, s2, s1, s0);
    }
    I16vec16(const M256 &m) : M256(m) {}

};

//
// Is16vec16 Class:
// 16 signed 16-bit integer elements
//
class Is16vec16 : public I16vec16 {
public:
 #ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    Is16vec16() = default;
#else
    Is16vec16() : I16vec16() {}
#endif
    Is16vec16(__m256i mm) : I16vec16(mm) {}
    Is16vec16(signed short s15, signed short s14,
              signed short s13, signed short s12,
              signed short s11, signed short s10,
              signed short s9, signed short s8,
              signed short s7, signed short s6,
              signed short s5, signed short s4,
              signed short s3, signed short s2,
              signed short s1, signed short s0)
        : I16vec16(s15, s14, s13, s12, s11, s10, s9, s8,
                   s7, s6, s5, s4, s3, s2, s1, s0) {}

    Is16vec16(const M256 &m) : I16vec16(m) {}

};


//
// Iu16vec16 Class:
// 16 unsigned 16-bit integer elements
//
class Iu16vec16 : public I16vec16 {
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    Iu16vec16() = default;
#else
    Iu16vec16() : I16vec16() {}
#endif
    Iu16vec16(__m256i mm) : I16vec16(mm) {}
    Iu16vec16(unsigned short s15, unsigned short s14,
              unsigned short s13, unsigned short s12,
              unsigned short s11, unsigned short s10,
              unsigned short s9, unsigned short s8,
              unsigned short s7, unsigned short s6,
              unsigned short s5, unsigned short s4,
              unsigned short s3, unsigned short s2,
              unsigned short s1, unsigned short s0)
        : I16vec16(s15, s14, s13, s12, s11, s10, s9, s8,
                   s7, s6, s5, s4, s3, s2, s1, s0) {}

    Iu16vec16(const M256 &m) : I16vec16(m) {}

};


//
// I8vec32 Class:
// 32 unsigned or signed 8-bit integer elements
//
class I8vec32 : public M256 {
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    I8vec32() = default;
#else
    I8vec32() : M256() {}
#endif
    I8vec32(__m256i mm) : M256(mm) {}
    I8vec32(char s31, char s30, char s29, char s28,
            char s27, char s26, char s25, char s24,
            char s23, char s22, char s21, char s20,
            char s19, char s18, char s17, char s16,
            char s15, char s14, char s13, char s12,
            char s11, char s10, char s9, char s8,
            char s7, char s6, char s5, char s4,
            char s3, char s2, char s1, char s0) {
        vec = _mm256_set_epi8(s31, s30, s29, s28, s27, s26, s25, s24,
                              s23, s22, s21, s20, s19, s18, s17, s16,
                              s15, s14, s13, s12, s11, s10, s9, s8,
                              s7, s6, s5, s4, s3, s2, s1, s0);
    }

    I8vec32(const M256 &m) : M256(m) {}

    // add/sub operators

};



//
// Is8vec32 Class:
// 32 signed 8-bit integer elements
//
class Is8vec32 : public I8vec32 {
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    Is8vec32() = default;
#else
    Is8vec32() : I8vec32() {}
#endif
    Is8vec32(__m256i mm) : I8vec32(mm) {}
    Is8vec32(char s31, char s30, char s29, char s28,
             char s27, char s26, char s25, char s24,
             char s23, char s22, char s21, char s20,
             char s19, char s18, char s17, char s16,
             char s15, char s14, char s13, char s12,
             char s11, char s10, char s9, char s8,
             char s7, char s6, char s5, char s4,
             char s3, char s2, char s1, char s0)
        : I8vec32(s31, s30, s29, s28, s27, s26, s25, s24,
                  s23, s22, s21, s20, s19, s18, s17, s16,
                  s15, s14, s13, s12, s11, s10, s9, s8,
                  s7, s6, s5, s4, s3, s2, s1, s0) {}

    Is8vec32(const M256 &m) : I8vec32(m) {}

#if defined(DVEC_DEFINE_OUTPUT_OPERATORS)
    // output for debug
    friend DVEC_STD ostream& operator << (DVEC_STD ostream &os,
                                          const Is8vec32 &a) {
        os << " [31]:" << (int)a[31]
            << " [30]:" << (int)a[30]
            << " [29]:" << (int)a[29]
            << " [28]:" << (int)a[28]
            << " [27]:" << (int)a[27]
            << " [26]:" << (int)a[26]
            << " [25]:" << (int)a[25]
            << " [24]:" << (int)a[24]
            << " [23]:" << (int)a[23]
            << " [22]:" << (int)a[22]
            << " [21]:" << (int)a[21]
            << " [20]:" << (int)a[20]
            << " [19]:" << (int)a[19]
            << " [18]:" << (int)a[18]
            << " [17]:" << (int)a[17]
            << " [16]:" << (int)a[16]
            << " [15]:" << (int)a[15]
            << " [14]:" << (int)a[14]
            << " [13]:" << (int)a[13]
            << " [12]:" << (int)a[12]
            << " [11]:" << (int)a[11]
            << " [10]:" << (int)a[10]
            << " [9]:" << (int)a[9]
            << " [8]:" << (int)a[8]
            << " [7]:" << (int)a[7]
            << " [6]:" << (int)a[6]
            << " [5]:" << (int)a[5]
            << " [4]:" << (int)a[4]
            << " [3]:" << (int)a[3]
            << " [2]:" << (int)a[2]
            << " [1]:" << (int)a[1]
            << " [0]:" << (int)a[0];

        return os;
    }
#endif

};


//
// Iu8vec32 Class:
// 32 unsigned 8-bit integer elements
//
class Iu8vec32 : public I8vec32 {
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    Iu8vec32() = default;
#else
    Iu8vec32() : I8vec32() {}
#endif
    Iu8vec32(__m256i mm) : I8vec32(mm) {}
    Iu8vec32(unsigned char u31, unsigned char u30,
             unsigned char u29, unsigned char u28,
             unsigned char u27, unsigned char u26,
             unsigned char u25, unsigned char u24,
             unsigned char u23, unsigned char u22,
             unsigned char u21, unsigned char u20,
             unsigned char u19, unsigned char u18,
             unsigned char u17, unsigned char u16,
             unsigned char u15, unsigned char u14,
             unsigned char u13, unsigned char u12,
             unsigned char u11, unsigned char u10,
             unsigned char u9, unsigned char u8,
             unsigned char u7, unsigned char u6,
             unsigned char u5, unsigned char u4,
             unsigned char u3, unsigned char u2,
             unsigned char u1, unsigned char u0)
        : I8vec32(u31, u30, u29, u28, u27, u26, u25, u24,
                  u23, u22, u21, u20, u19, u18, u17, u16,
                  u15, u14, u13, u12, u11, u10, u9, u8,
                  u7, u6, u5, u4, u3, u2, u1, u0) {}

    Iu8vec32(const M256 &m) : I8vec32(m) {}

#if defined(DVEC_DEFINE_OUTPUT_OPERATORS)
    // output for debug
    friend DVEC_STD ostream& operator << (DVEC_STD ostream &os,
                                          const Iu8vec32 &a) {
        os << " [31]:" << (unsigned int)a[31]
           << " [30]:" << (unsigned int)a[30]
           << " [29]:" << (unsigned int)a[29]
           << " [28]:" << (unsigned int)a[28]
           << " [27]:" << (unsigned int)a[27]
           << " [26]:" << (unsigned int)a[26]
           << " [25]:" << (unsigned int)a[25]
           << " [24]:" << (unsigned int)a[24]
           << " [23]:" << (unsigned int)a[23]
           << " [22]:" << (unsigned int)a[22]
           << " [21]:" << (unsigned int)a[21]
           << " [20]:" << (unsigned int)a[20]
           << " [19]:" << (unsigned int)a[19]
           << " [18]:" << (unsigned int)a[18]
           << " [17]:" << (unsigned int)a[17]
           << " [16]:" << (unsigned int)a[16]
           << " [15]:" << (unsigned int)a[15]
           << " [14]:" << (unsigned int)a[14]
           << " [13]:" << (unsigned int)a[13]
           << " [12]:" << (unsigned int)a[12]
           << " [11]:" << (unsigned int)a[11]
           << " [10]:" << (unsigned int)a[10]
           << " [9]:" << (unsigned int)a[9]
           << " [8]:" << (unsigned int)a[8]
           << " [7]:" << (unsigned int)a[7]
           << " [6]:" << (unsigned int)a[6]
           << " [5]:" << (unsigned int)a[5]
           << " [4]:" << (unsigned int)a[4]
           << " [3]:" << (unsigned int)a[3]
           << " [2]:" << (unsigned int)a[2]
           << " [1]:" << (unsigned int)a[1]
           << " [0]:" << (unsigned int)a[0];
        return os;
    }
#endif


};

/************************************************************************/
/************** Interface classes for working with 512-bit intrinsics ***/
/************************************************************************/

#define __f32vec16_abs_mask ((F32vec16)_mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff)))
#define __f64vec8_abs_mask ((F64vec8)_mm512_castsi512_pd(_mm512_set1_epi64(0x7fffffffffffffffLL)))

/*
 * class F32vec16
 *
 * Represents 512-bit vector composed of 16 single precision
 * floating point elements.
 */

class F32vec16
{
protected:
    __m512 vec;
public:

    /* Constructors: __m512, 16 floats, 1 float */
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    F32vec16() = default;
#else
    F32vec16() { vec = _mm512_setzero(); }
#endif

    /* initialize 16 SP FP with __m512 data type */
    F32vec16(__m512 m) { vec = m; }

    /* initialize 16 SP FPs with 16 floats */
    F32vec16(float f15, float f14, float f13, float f12,
             float f11, float f10, float f9, float f8,
             float f7, float f6, float f5, float f4,
             float f3, float f2, float f1, float f0)
    {
        vec = _mm512_set_ps(f15,f14,f13,f12,f11,f10,f9,f8,
                            f7,f6,f5,f4,f3,f2,f1,f0);
    }

    /* Explicitly initialize each of 16 SP FPs with same float */
    EXPLICIT F32vec16(float f) { vec = _mm512_set1_ps(f); }

    /* Explicitly initialize each of 16 SP FPs with same double */
    EXPLICIT F32vec16(double d)	{ vec = _mm512_set1_ps((float) d); }

    /* Assignment operations */
    F32vec16& operator =(float f) {
        vec = _mm512_set1_ps(f);
        return *this;
    }

    F32vec16& operator =(double d)
    {
        vec = _mm512_set1_ps((float) d);
        return *this;
    }

    /* Conversion functions */
    operator  __m512() const { return vec; } /* Convert to __m512 */

    /* Logical Operators */
    friend F32vec16 operator &(const F32vec16 &a, const F32vec16 &b) {
        return _mm512_castsi512_ps(_mm512_and_si512(
                                   _mm512_castps_si512(a),
                                   _mm512_castps_si512(b)));
    }
    friend F32vec16 operator |(const F32vec16 &a, const F32vec16 &b) {
        return _mm512_castsi512_ps(_mm512_or_si512(
                                   _mm512_castps_si512(a),
                                   _mm512_castps_si512(b)));
    }
    friend F32vec16 operator ^(const F32vec16 &a, const F32vec16 &b) {
        return _mm512_castsi512_ps(_mm512_xor_si512(
                                   _mm512_castps_si512(a),
                                   _mm512_castps_si512(b)));
    }

    /* Arithmetic Operators */
    friend F32vec16 operator +(const F32vec16 &a, const F32vec16 &b) {
        return _mm512_add_ps(a, b);
    }
    friend F32vec16 operator -(const F32vec16 &a, const F32vec16 &b) {
        return _mm512_sub_ps(a, b);
    }
    friend F32vec16 operator *(const F32vec16 &a, const F32vec16 &b) {
        return _mm512_mul_ps(a, b);
    }
    friend F32vec16 operator /(const F32vec16 &a, const F32vec16 &b) {
        return _mm512_div_ps(a, b);
    }

    F32vec16& operator +=(const F32vec16 &a)
        { return *this = *this + a; }
    F32vec16& operator -=(const F32vec16 &a)
        { return *this = *this - a; }
    F32vec16& operator *=(const F32vec16 &a)
        { return *this = *this * a; }
    F32vec16& operator /=(const F32vec16 &a)
        { return *this = *this / a; }
    F32vec16& operator &=(const F32vec16 &a)
        { return *this = *this & a; }
    F32vec16& operator ^=(const F32vec16 &a)
        { return *this = *this ^ a; }
    F32vec16& operator |=(const F32vec16 &a)
        { return *this = *this | a; }

    F32vec16 operator - () const {
        return _mm512_castsi512_ps(_mm512_xor_si512(
                            _mm512_set1_epi32(0x80000000),
                            _mm512_castps_si512(*this)));
    }
    F32vec16& flip_sign () { return *this = - *this;}

    void set_zero() { vec = _mm512_setzero_ps(); }
    void init (float f0, float f1, float f2, float f3,
               float f4, float f5, float f6, float f7,
               float f8, float f9, float f10, float f11,
               float f12, float f13, float f14, float f15)
    {
        vec = _mm512_set_ps(f15,f14,f13,f12,f11,f10,f9,f8,
                            f7,f6,f5,f4,f3,f2,f1,f0);
    }
    /* Mixed vector-scalar operations */
    friend F32vec16 operator +(const F32vec16 &a, const float &f) {
        return _mm512_add_ps(a, _mm512_set1_ps(f));
    }
    friend F32vec16 operator -(const F32vec16 &a, const float &f) {
        return _mm512_sub_ps(a, _mm512_set1_ps(f));
    }
    friend F32vec16 operator /(const F32vec16 &a, const float &f) {
        return _mm512_div_ps(a, _mm512_set1_ps(f));
    }
    friend F32vec16 operator *(const F32vec16 &a, const float &f) {
        return _mm512_mul_ps(a, _mm512_set1_ps(f));
    }
    F32vec16& operator +=(const float &f) {
        return *this = *this + f;
    }
    F32vec16& operator -=(const float &f) {
        return *this = *this - f;
    }
    F32vec16& operator *=(const float &f) {
        return *this = *this * f;
    }
    F32vec16& operator /=(const float &f) {
        return *this = *this / f;
    }

    bool is_zero() const {
        __m512 a = _mm512_setzero_ps();
        __mmask16 k = _mm512_cmpeq_ps_mask(a, *this);
        return (k == 0xffff);
    }
    /* Dot product */
    void dot (float& p, const F32vec16& rhs) const {
        p = add_horizontal(*this * rhs);
    }
    float dot (const F32vec16& rhs) const {
        return (add_horizontal(*this * rhs));
    }
    /* Length */
    float length_sqr()  const { return dot(*this); }
    float length() const {
        float f = dot(*this);
        __m128 f2 = _mm_set_ss(f);
        __m128 f3 = _mm_sqrt_ss(f2);
        return _mm_cvtss_f32(f3);
    }

    /* Normalize */
    bool normalize() { float l = length(); *this /= l; return true; }

    /* Horizontal Add */
    friend float add_horizontal(const F32vec16 &a) {
        return _mm512_reduce_add_ps(a);
    }
    friend float mul_horizontal(const F32vec16 &a) {
        return _mm512_reduce_mul_ps(a);
    }
    friend F32vec16 rsqrt(const F32vec16 &a) {
        return _mm512_rsqrt14_ps(a);
    }

};

/* Reciprocal */
inline F32vec16 rcp(const F32vec16 &a) { return _mm512_rcp14_ps(a); }


/*
 * class F64vec8
 *
 * Represents 512-bit vector composed of 8 double precision
 * floating point elements.
 */
class F64vec8
{
protected:
    __m512d vec;

public:

    /* Constructors: __m512d, 8 doubles */
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    F64vec8() = default;
#else
    F64vec8() { vec = _mm512_setzero_pd(); }
#endif

    /* initialize 8 DP FP with __m512d data type */
    F64vec8(__m512d m) { vec = m; }

    /* initialize 8 DP FPs with 8 doubles */
    F64vec8(double d7, double d6, double d5, double d4,
            double d3, double d2, double d1, double d0)
    {
        vec = _mm512_set_pd(d7,d6,d5,d4,d3,d2,d1,d0);
    }

    /* Explicitly initialize each of 8 DP FPs with same double */
    EXPLICIT F64vec8(double d) { vec = _mm512_set1_pd(d); }

    /* Conversion functions */
    operator  __m512d() const { return vec; }

};


class I32vec16;  /* 16 elements, each element a signed or unsigned __int32 */
class Is32vec16; /* 16 elements, each element a signed __int32 */
class Iu32vec16; /* 16 elements, each element a unsigned __int32 */
class I64vec8;   /* 8 element, each a __int64 data type */
class Is64vec8;  /* 8 element, each a signed __int64 data type */
class Iu64vec8;  /* 8 element, each a unsigned __int64 data type */
class M512vec;   /* 1 element, a __m512i data type */

#define _MM_16UDW(element,vector) (*((unsigned int*)&(vector) + (element)))
#define _MM_16DW(element,vector) (*((int*)&(vector) + (element)))

#define _MM_8UQW(element,vector) (*((unsigned __int64*)&(vector) + (element)))
#define _MM_8QW(element,vector) (*((__int64*)&(vector) + (element)))

/* M512vec Class:
 * 1 element, a __m512i data type
 * Contructors & Logical Operations
 */

class M512vec
{
protected:
    __m512i vec;

public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    M512vec() = default;
#else
    M512vec() { vec = _mm512_setzero_si512(); }
#endif

    M512vec(__m512i mm) { vec = mm; }

    operator __m512i() const { return vec; }

    /* Logical Operations */
    friend M512vec operator&(const M512vec &a, const M512vec &b) {
        return _mm512_and_si512(a, b);
    }

    friend M512vec operator|(const M512vec &a, const M512vec &b) {
        return _mm512_or_si512(a, b);
    }
    M512vec& operator|=(const M512vec &a) {
        return *this = *this | a;
    }
    friend M512vec operator^(const M512vec &a, const M512vec &b) {
        return _mm512_xor_si512(a, b);
    }

};



/* I64vec8 Class:
 * 8 elements, each element signed or unsigned 64-bit integer
 */
class I64vec8 : public M512vec
{
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    I64vec8() = default;
#else
    I64vec8() : M512vec() { }
#endif
    I64vec8(__m512i mm) : M512vec(mm) { }
    EXPLICIT I64vec8(__int64 i) { vec = _mm512_set1_epi64(i);}
    EXPLICIT I64vec8(int i) { vec = _mm512_set1_epi64(i);}

    I64vec8(__int64 q7,__int64 q6,__int64 q5,__int64 q4,
            __int64 q3,__int64 q2,__int64 q1,__int64 q0) {
        vec = _mm512_set_epi64(q7,q6,q5,q4,q3,q2,q1,q0);
    }

    /* Copy Constructor */
    I64vec8(const M512vec &m) : M512vec(m) {}

};

/* Is64vec8 Class:
 * 8 elements, each element signed 64-bit integer
 */
class Is64vec8 : public I64vec8
{
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    Is64vec8() = default;
#else
    Is64vec8() : I64vec8() { }
#endif
    Is64vec8(__m512i mm) : I64vec8(mm) { }
    EXPLICIT Is64vec8(int i) : I64vec8((int)i) { }
    EXPLICIT Is64vec8(__int64 i) : I64vec8((__int64)i) { }

    Is64vec8(__int64 q7,__int64 q6,__int64 q5,__int64 q4,
             __int64 q3,__int64 q2,__int64 q1,__int64 q0) :
        I64vec8(q7,q6,q5,q4,q3,q2,q1,q0) {}

    /* Copy Constructor */
    Is64vec8(const M512vec &m) : I64vec8(m) {}

    /* Shift Arithmetiic Operators */

};
/* Iu64vec8 Class:
 * 8 elements, each element unsigned 64-bit integer
 */
class Iu64vec8 : public I64vec8
{
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    Iu64vec8() = default;
#else
    Iu64vec8() : I64vec8() { }
#endif
    Iu64vec8(__m512i mm) : I64vec8(mm) { }
    EXPLICIT Iu64vec8(unsigned int ui)
        : I64vec8(static_cast<__int64>(ui)) {}
    EXPLICIT Iu64vec8(unsigned __int64 ui)
        : I64vec8(static_cast<__int64>(ui)) {}

    Iu64vec8(unsigned __int64 q7, unsigned __int64 q6,unsigned __int64 q5,
             unsigned __int64 q4, unsigned __int64 q3,unsigned __int64 q2,
             unsigned __int64 q1, unsigned __int64 q0) :
        I64vec8(q7,q6,q5,q4,q3,q2,q1,q0) {}

    /* Copy Constructor */
    Iu64vec8(const M512vec &m) : I64vec8(m) {}

    /* Shift Logical Operators */


    Iu64vec8 operator>>(int count) const {
        return _mm512_srli_epi64(*this,count);
    }

};

/* I32vec16 Class:
 * 16 elements, each element either a signed or unsigned int
 */
class I32vec16 : public M512vec
{
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    I32vec16() = default;
#else
    I32vec16() : M512vec() { }
#endif
    I32vec16(__m512i mm) : M512vec(mm) { }
    EXPLICIT I32vec16(int i)     { vec = _mm512_set1_epi32(i); }
    EXPLICIT I32vec16(__int64 i) { vec = _mm512_set1_epi32((int)i); }

    I32vec16(int i15, int i14, int i13, int i12,
             int i11, int i10, int i9, int i8,
             int i7, int i6, int i5, int i4,
             int i3, int i2, int i1, int i0)
    {
        vec = _mm512_set_epi32(i15, i14, i13, i12,
                               i11, i10, i9, i8,
                               i7, i6, i5, i4,
                               i3, i2, i1, i0);
    }

    /* Copy Constructor */
    I32vec16(const M512vec &m) : M512vec(m) {}

    /* Addition & Subtraction Operators */
    I32vec16 operator +(const I32vec16 &a) const {
        return _mm512_add_epi32(*this, a);
    }

};

/* Is32vec16 Class:
 * 16 elements, each element signed integer
 */
class Is32vec16 : public I32vec16
{
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    Is32vec16() = default;
#else
    Is32vec16() : I32vec16() { }
#endif
    Is32vec16(__m512i mm) : I32vec16(mm) { }
    EXPLICIT Is32vec16(int i) : I32vec16(i) { }
    EXPLICIT Is32vec16(__int64 i) : I32vec16(i) { }

    Is32vec16(int i15, int i14, int i13, int i12,
              int i11, int i10, int i9, int i8,
              int i7, int i6, int i5, int i4,
              int i3, int i2, int i1, int i0)
        : I32vec16(i15, i14, i13, i12, i11, i10, i9, i8,
                   i7, i6, i5, i4, i3, i2, i1, i0) { }

    /* Constructor from M512vec */
    Is32vec16(const M512vec &m) : I32vec16(m) {}

    /* Shift Arithmetic Operations */
    Is32vec16 operator>>(const I32vec16 &a) const {
        return _mm512_srav_epi32(*this, a);
    }
    Is32vec16 operator>>(int count) const {
        return _mm512_srai_epi32(*this, count);
    }
    Is32vec16& operator>>=(const I32vec16 &a) {
        return *this = *this >> a;
    }
    Is32vec16& operator>>=(int count) {
        return *this = *this >> count;
    }

    /* Element Access for Debug, No data modified */
    const int& operator[](int i) const {
        /* Only 16 elements to access */
        assert(static_cast<unsigned int>(i) < 16);
        return _MM_16DW(i,vec);
    }

    /* Element Access for Debug */
    int& operator[](int i) {
        /* Only 16 elements to access */
        assert(static_cast<unsigned int>(i) < 16);
        return _MM_16DW(i,vec);
    }
    Is32vec16 operator *(const Is32vec16 &a) const {
        return _mm512_mullo_epi32(*this, a);
    }
    Is32vec16 operator *=(const Is32vec16 &a) {
        return *this = *this * a;
    }

    friend Is32vec16 operator+(const Is32vec16 &a, const Is32vec16 &b) {
        return (Is32vec16)_mm512_add_epi32(a, b);
    }

    Is32vec16& operator+= (const Is32vec16& rhs) { *this = *this + rhs; return *this; }
};

/* Iu32vec16 Class:
 * 16 elements, each element unsigned int
 */
class Iu32vec16 : public I32vec16
{
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    Iu32vec16() = default;
#else
    Iu32vec16() : I32vec16() { }
#endif
    Iu32vec16(__m512i mm) : I32vec16(mm) { }
    EXPLICIT Iu32vec16(unsigned int ui) : I32vec16((int)ui) { }
    EXPLICIT Iu32vec16(unsigned __int64 ui) : I32vec16((__int64)ui) { }

    Iu32vec16(unsigned int ui15, unsigned int ui14,
              unsigned int ui13, unsigned int ui12,
              unsigned int ui11, unsigned int ui10,
              unsigned int ui9, unsigned int ui8,
              unsigned int ui7, unsigned int ui6,
              unsigned int ui5, unsigned int ui4,
              unsigned int ui3, unsigned int ui2,
              unsigned int ui1, unsigned int ui0)
            : I32vec16(ui15, ui14, ui13, ui12, ui11, ui10, ui9, ui8,
                       ui7, ui6, ui5, ui4, ui3, ui2, ui1, ui0) { }

    /* Copy Constructor */
    Iu32vec16(const M512vec &m) : I32vec16(m) {}

};

/************************************************************************/
/************** Intel(R) AVX-512 Byte and Word Instructions *************/
/************************************************************************/
class I16vec32;  /* 32 elements, each element a signed or unsigned short */
class Is16vec32; /* 32 elements, each element a signed short */
class Iu16vec32; /* 32 elements, each element an unsigned short */
class I8vec64;   /* 64 elements, each element a signed or unsigned char */
class Is8vec64;  /* 64 elements, each element a signed char */
class Iu8vec64;  /* 64 elements, each element an unsigned char */

#define _MM_64UB(element,vector) (*((unsigned char*)&(vector) + (element)))
#define _MM_64B(element,vector) (*((signed char*)&(vector) + (element)))

#define _MM_32UW(element,vector) (*((unsigned short*)&(vector) + (element)))
#define _MM_32W(element,vector) (*((short*)&(vector) + (element)))

/* I16vec32 Class:
 * 32 elements, each element either unsigned or signed short
 */
class I16vec32 : public M512vec
{
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    I16vec32() = default;
#else
    I16vec32() : M512vec() { }
#endif
    I16vec32(__m512i mm) : M512vec(mm) { }
    I16vec32(short s31, short s30, short s29, short s28,
             short s27, short s26, short s25, short s24,
             short s23, short s22, short s21, short s20,
             short s19, short s18, short s17, short s16,
             short s15, short s14, short s13, short s12,
             short s11, short s10, short s9, short s8,
             short s7, short s6, short s5, short s4,
             short s3, short s2, short s1, short s0)
    {
        vec = _mm512_set_epi16(s31, s30, s29, s28, s27, s26, s25, s24,
                               s23, s22, s21, s20, s19, s18, s17, s16,
                               s15, s14, s13, s12, s11, s10, s9, s8,
                               s7, s6, s5, s4, s3, s2, s1, s0);
    }

    /* Constructor */
    I16vec32(const M512vec &m) : M512vec(m) {}

    /* Addition & Subtraction Assignment Operators */
    friend I16vec32 operator +(const I16vec32 &a, const I16vec32 &b) {
        return (I16vec32)_mm512_add_epi16(a, b);
    }

    I16vec32 operator<<(int count) const {
        return _mm512_slli_epi16(vec,count);
    }

};



/* Is16vec32 Class:
 * 32 elements, each element signed short
 */
class Is16vec32 : public I16vec32
{
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    Is16vec32() = default;
#else
    Is16vec32() : I16vec32() { }
#endif
    Is16vec32(__m512i mm) : I16vec32(mm) { }
    Is16vec32(signed short s31, signed short s30,
              signed short s29, signed short s28, signed short s27,
              signed short s26, signed short s25, signed short s24,
              signed short s23, signed short s22, signed short s21,
              signed short s20, signed short s19, signed short s18,
              signed short s17, signed short s16, signed short s15,
              signed short s14, signed short s13, signed short s12,
              signed short s11, signed short s10, signed short s9,
              signed short s8, signed short s7, signed short s6,
              signed short s5, signed short s4, signed short s3,
              signed short s2, signed short s1, signed short s0)
        : I16vec32(s31, s30, s29, s28, s27, s26, s25, s24,
                   s23, s22, s21, s20, s19, s18, s17, s16,
                   s15, s14, s13, s12, s11, s10, s9, s8,
                   s7, s6, s5, s4, s3, s2, s1, s0) { }

    /* Constructor */
    Is16vec32(const M512vec &m) : I16vec32(m) {}

    friend Is16vec32 operator-(Is16vec32 lhs, Is16vec32 rhs) { return _mm512_sub_epi16(lhs, rhs); }

};

inline Is16vec32 pack_sat(const Is32vec16 &a, const Is32vec16 &b) {
    return _mm512_packs_epi32(a,b);
}
/* Iu16vec32 Class:
 * 32 elements, each element unsigned short
 */
class Iu16vec32 : public I16vec32
{
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    Iu16vec32() = default;
#else
    Iu16vec32() : I16vec32() { }
#endif
    Iu16vec32(__m512i mm) : I16vec32(mm) { }
    Iu16vec32(unsigned short s31, unsigned short s30, unsigned short s29,
              unsigned short s28, unsigned short s27, unsigned short s26,
              unsigned short s25, unsigned short s24, unsigned short s23,
              unsigned short s22, unsigned short s21, unsigned short s20,
              unsigned short s19, unsigned short s18, unsigned short s17,
              unsigned short s16, unsigned short s15, unsigned short s14,
              unsigned short s13, unsigned short s12, unsigned short s11,
              unsigned short s10, unsigned short s9, unsigned short s8,
              unsigned short s7, unsigned short s6, unsigned short s5,
              unsigned short s4, unsigned short s3, unsigned short s2,
              unsigned short s1, unsigned short s0)
        : I16vec32(s31, s30, s29, s28, s27, s26, s25, s24,
                   s23, s22, s21, s20, s19, s18, s17, s16,
                   s15, s14, s13, s12, s11, s10, s9, s8,
                   s7, s6, s5, s4, s3, s2, s1, s0) { }

    /* Constructor */
    Iu16vec32(const M512vec &m) : I16vec32(m) {}

};

/* I8vec64 Class:
 * 64 elements, each element either unsigned or signed char
 */
class I8vec64 : public M512vec
{
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    I8vec64() = default;
#else
    I8vec64() : M512vec() { }
#endif
    I8vec64(__m512i mm) : M512vec(mm) { }
    I8vec64(char s63, char s62, char s61, char s60, char s59, char s58,
            char s57, char s56, char s55, char s54, char s53, char s52,
            char s51, char s50, char s49, char s48, char s47, char s46,
            char s45, char s44, char s43, char s42, char s41, char s40,
            char s39, char s38, char s37, char s36, char s35, char s34,
            char s33, char s32, char s31, char s30, char s29, char s28,
            char s27, char s26, char s25, char s24, char s23, char s22,
            char s21, char s20, char s19, char s18, char s17, char s16,
            char s15, char s14, char s13, char s12, char s11, char s10,
            char s9, char s8, char s7, char s6, char s5, char s4,
            char s3, char s2, char s1, char s0)
    {
        vec = _mm512_set_epi8(s63, s62, s61, s60, s59, s58, s57, s56,
                              s55, s54, s53, s52, s51, s50, s49, s48,
                              s47, s46, s45, s44, s43, s42, s41, s40,
                              s39, s38, s37, s36, s35, s34, s33, s32,
                              s31, s30, s29, s28, s27, s26, s25, s24,
                              s23, s22, s21, s20, s19, s18, s17, s16,
                              s15, s14, s13, s12, s11, s10, s9, s8,
                              s7, s6, s5, s4, s3, s2, s1, s0);
    }

    /* Constructor */
    I8vec64(const M512vec &m) : M512vec(m) {}

};


/* Is8vec64 Class:
 * 64 elements, each element a signed char
 */
class Is8vec64 : public I8vec64
{
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    Is8vec64() = default;
#else
    Is8vec64() : I8vec64() { }
#endif
    Is8vec64(__m512i mm) : I8vec64(mm) { }
    Is8vec64(char s63, char s62, char s61, char s60, char s59, char s58,
             char s57, char s56, char s55, char s54, char s53, char s52,
             char s51, char s50, char s49, char s48, char s47, char s46,
             char s45, char s44, char s43, char s42, char s41, char s40,
             char s39, char s38, char s37, char s36, char s35, char s34,
             char s33, char s32, char s31, char s30, char s29, char s28,
             char s27, char s26, char s25, char s24, char s23, char s22,
             char s21, char s20, char s19, char s18, char s17, char s16,
             char s15, char s14, char s13, char s12, char s11, char s10,
             char s9, char s8, char s7, char s6, char s5, char s4,
             char s3, char s2, char s1, char s0) :
        I8vec64 (s63, s62, s61, s60, s59, s58, s57, s56,
                 s55, s54, s53, s52, s51, s50, s49, s48,
                 s47, s46, s45, s44, s43, s42, s41, s40,
                 s39, s38, s37, s36, s35, s34, s33, s32,
                 s31, s30, s29, s28, s27, s26, s25, s24,
                 s23, s22, s21, s20, s19, s18, s17, s16,
                 s15, s14, s13, s12, s11, s10, s9, s8,
                 s7, s6, s5, s4, s3, s2, s1, s0) {}

    /* Constructor */
    Is8vec64(const M512vec &m) : I8vec64(m) {}

#if defined(DVEC_DEFINE_OUTPUT_OPERATORS)
    /* Output for Debug */
    friend DVEC_STD ostream& operator << (DVEC_STD ostream &os,
                                          const Is8vec64 &a) {
        os << " [63]:" << (int)a[63]
            << " [62]:" << (int)a[62]
            << " [61]:" << (int)a[61]
            << " [60]:" << (int)a[60]
            << " [59]:" << (int)a[59]
            << " [58]:" << (int)a[58]
            << " [57]:" << (int)a[57]
            << " [56]:" << (int)a[56]
            << " [55]:" << (int)a[55]
            << " [54]:" << (int)a[54]
            << " [53]:" << (int)a[53]
            << " [52]:" << (int)a[52]
            << " [51]:" << (int)a[51]
            << " [50]:" << (int)a[50]
            << " [49]:" << (int)a[49]
            << " [48]:" << (int)a[48]
            << " [47]:" << (int)a[47]
            << " [46]:" << (int)a[46]
            << " [45]:" << (int)a[45]
            << " [44]:" << (int)a[44]
            << " [43]:" << (int)a[43]
            << " [42]:" << (int)a[42]
            << " [41]:" << (int)a[41]
            << " [40]:" << (int)a[40]
            << " [39]:" << (int)a[39]
            << " [38]:" << (int)a[38]
            << " [37]:" << (int)a[37]
            << " [36]:" << (int)a[36]
            << " [35]:" << (int)a[35]
            << " [34]:" << (int)a[34]
            << " [33]:" << (int)a[33]
            << " [32]:" << (int)a[32]
            << " [31]:" << (int)a[31]
            << " [30]:" << (int)a[30]
            << " [29]:" << (int)a[29]
            << " [28]:" << (int)a[28]
            << " [27]:" << (int)a[27]
            << " [26]:" << (int)a[26]
            << " [25]:" << (int)a[25]
            << " [24]:" << (int)a[24]
            << " [23]:" << (int)a[23]
            << " [22]:" << (int)a[22]
            << " [21]:" << (int)a[21]
            << " [20]:" << (int)a[20]
            << " [19]:" << (int)a[19]
            << " [18]:" << (int)a[18]
            << " [17]:" << (int)a[17]
            << " [16]:" << (int)a[16]
            << " [15]:" << (int)a[15]
            << " [14]:" << (int)a[14]
            << " [13]:" << (int)a[13]
            << " [12]:" << (int)a[12]
            << " [11]:" << (int)a[11]
            << " [10]:" << (int)a[10]
            << " [9]:" << (int)a[9]
            << " [8]:" << (int)a[8]
            << " [7]:" << (int)a[7]
            << " [6]:" << (int)a[6]
            << " [5]:" << (int)a[5]
            << " [4]:" << (int)a[4]
            << " [3]:" << (int)a[3]
            << " [2]:" << (int)a[2]
            << " [1]:" << (int)a[1]
            << " [0]:" << (int)a[0];

        return os;
    }
#endif

};


/* Iu8vec64 Class:
 * 64 elements, each element a unsigned char
 */
class Iu8vec64 : public I8vec64
{
public:
#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
    Iu8vec64() = default;
#else
    Iu8vec64() : I8vec64() { }
#endif
    Iu8vec64(__m512i mm) : I8vec64(mm) { }
    Iu8vec64(unsigned char u63, unsigned char u62, unsigned char u61,
             unsigned char u60, unsigned char u59, unsigned char u58,
             unsigned char u57, unsigned char u56, unsigned char u55,
             unsigned char u54, unsigned char u53, unsigned char u52,
             unsigned char u51, unsigned char u50, unsigned char u49,
             unsigned char u48, unsigned char u47, unsigned char u46,
             unsigned char u45, unsigned char u44, unsigned char u43,
             unsigned char u42, unsigned char u41, unsigned char u40,
             unsigned char u39, unsigned char u38, unsigned char u37,
             unsigned char u36, unsigned char u35, unsigned char u34,
             unsigned char u33, unsigned char u32, unsigned char u31,
             unsigned char u30, unsigned char u29, unsigned char u28,
             unsigned char u27, unsigned char u26, unsigned char u25,
             unsigned char u24, unsigned char u23, unsigned char u22,
             unsigned char u21, unsigned char u20, unsigned char u19,
             unsigned char u18, unsigned char u17, unsigned char u16,
             unsigned char u15, unsigned char u14, unsigned char u13,
             unsigned char u12, unsigned char u11, unsigned char u10,
             unsigned char u9, unsigned char u8, unsigned char u7,
             unsigned char u6, unsigned char u5, unsigned char u4,
             unsigned char u3, unsigned char u2, unsigned char u1,
             unsigned char u0) :
        I8vec64 (u63, u62, u61, u60, u59, u58, u57, u56,
                 u55, u54, u53, u52, u51, u50, u49, u48,
                 u47, u46, u45, u44, u43, u42, u41, u40,
                 u39, u38, u37, u36, u35, u34, u33, u32,
                 u31, u30, u29, u28, u27, u26, u25, u24,
                 u23, u22, u21, u20, u19, u18, u17, u16,
                 u15, u14, u13, u12, u11, u10, u9, u8,
                 u7, u6, u5, u4, u3, u2, u1, u0) {}


    /* Constructor */
    Iu8vec64(const M512vec &m) : I8vec64(m) {}

#if defined(DVEC_DEFINE_OUTPUT_OPERATORS)
    /* Output for Debug */
    friend DVEC_STD ostream& operator << (DVEC_STD ostream &os,
                                          const Iu8vec64 &a) {
        os << " [63]:" << (unsigned int)a[63]
            << " [62]:" << (unsigned int)a[62]
            << " [61]:" << (unsigned int)a[61]
            << " [60]:" << (unsigned int)a[60]
            << " [59]:" << (unsigned int)a[59]
            << " [58]:" << (unsigned int)a[58]
            << " [57]:" << (unsigned int)a[57]
            << " [56]:" << (unsigned int)a[56]
            << " [55]:" << (unsigned int)a[55]
            << " [54]:" << (unsigned int)a[54]
            << " [53]:" << (unsigned int)a[53]
            << " [52]:" << (unsigned int)a[52]
            << " [51]:" << (unsigned int)a[51]
            << " [50]:" << (unsigned int)a[50]
            << " [49]:" << (unsigned int)a[49]
            << " [48]:" << (unsigned int)a[48]
            << " [47]:" << (unsigned int)a[47]
            << " [46]:" << (unsigned int)a[46]
            << " [45]:" << (unsigned int)a[45]
            << " [44]:" << (unsigned int)a[44]
            << " [43]:" << (unsigned int)a[43]
            << " [42]:" << (unsigned int)a[42]
            << " [41]:" << (unsigned int)a[41]
            << " [40]:" << (unsigned int)a[40]
            << " [39]:" << (unsigned int)a[39]
            << " [38]:" << (unsigned int)a[38]
            << " [37]:" << (unsigned int)a[37]
            << " [36]:" << (unsigned int)a[36]
            << " [35]:" << (unsigned int)a[35]
            << " [34]:" << (unsigned int)a[34]
            << " [33]:" << (unsigned int)a[33]
            << " [32]:" << (unsigned int)a[32]
            << " [31]:" << (unsigned int)a[31]
            << " [30]:" << (unsigned int)a[30]
            << " [29]:" << (unsigned int)a[29]
            << " [28]:" << (unsigned int)a[28]
            << " [27]:" << (unsigned int)a[27]
            << " [26]:" << (unsigned int)a[26]
            << " [25]:" << (unsigned int)a[25]
            << " [24]:" << (unsigned int)a[24]
            << " [23]:" << (unsigned int)a[23]
            << " [22]:" << (unsigned int)a[22]
            << " [21]:" << (unsigned int)a[21]
            << " [20]:" << (unsigned int)a[20]
            << " [19]:" << (unsigned int)a[19]
            << " [18]:" << (unsigned int)a[18]
            << " [17]:" << (unsigned int)a[17]
            << " [16]:" << (unsigned int)a[16]
            << " [15]:" << (unsigned int)a[15]
            << " [14]:" << (unsigned int)a[14]
            << " [13]:" << (unsigned int)a[13]
            << " [12]:" << (unsigned int)a[12]
            << " [11]:" << (unsigned int)a[11]
            << " [10]:" << (unsigned int)a[10]
            << " [9]:" << (unsigned int)a[9]
            << " [8]:" << (unsigned int)a[8]
            << " [7]:" << (unsigned int)a[7]
            << " [6]:" << (unsigned int)a[6]
            << " [5]:" << (unsigned int)a[5]
            << " [4]:" << (unsigned int)a[4]
            << " [3]:" << (unsigned int)a[3]
            << " [2]:" << (unsigned int)a[2]
            << " [1]:" << (unsigned int)a[1]
            << " [0]:" << (unsigned int)a[0];

        return os;
    }
#endif

    /* Element Access for Debug, No data modified */
    const unsigned char& operator[](int i) const {
        assert(static_cast<unsigned int>(i) < 64);
        /* Only 64 elements to access */
        return _MM_64UB(i,vec);
    }

    /* Element Access for Debug */
    unsigned char& operator[](int i) {
        assert(static_cast<unsigned int>(i) < 64);
        /* Only 64 elements to access */
        return _MM_64UB(i,vec);
    }
};


/* The Microsoft compiler (version VS2008 or older) cannot handle the #pragma pack(push,32) */
#if !defined(_MSC_VER) || (_MSC_VER >= 1600)
#pragma pack(pop) /* 32-B aligned */
#endif

#undef DVEC_DEFINE_OUTPUT_OPERATORS
#undef DVEC_STD

#ifdef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
#undef DVEC_USE_CPP11_DEFAULTED_FUNCTIONS
#endif

#endif // DVEC_H_INCLUDED
