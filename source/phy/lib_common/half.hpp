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

#include <complex>
#include <iostream>
#include <math.h>

/// Some code uses half-precision floating point. Try to use the compiler support where possible,
/// but otherwise fall back to an emulation library.

#ifdef __llvm__
using half = _Float16;
#else
using half = short float;
#endif

// Implement half rcp/rsqrt using float instead of half. This is very slightly more accurate
// (1ULP difference) but is so close that it is unnoticeable for most wireless kernels.
static inline half rcp(half value)
{
return half(1.0) / value;
}

static inline half rsqrt(half value)
{
return half(1.0) / half(sqrt(float(value)));
}


/*! \brief common to define data type
 *
 * data type declare
 */
#ifdef __llvm__
using float16 = half;
#endif
static inline std::ostream& operator<<(std::ostream& stream, half v)
{
  stream << (float)v;
  return stream;
}

static inline std::ostream& operator<<(std::ostream& stream, std::complex<half> v)
{
  // Force the use of the normal std::complex, so that it is consistent with other types.
  // stream << std::complex<float>(float(v.real()), float(v.imag()));
  stream << float(v.real()) << "+" << float(v.imag()) << "j";
  return stream;
}

// Overload some common operators.
static inline half abs(half a)
{
  union
  {
    half h;
    uint16_t ui;
  };

  h = a;
  ui &= 0x7FFF;

  return h;
}

static inline half sqrt(half in)
{
  return (half)sqrt((float)in);
}

static inline half min(half a, half b)
{
  if (a > b)
    return b;
  else
    return a;
}

static inline half max(half a, half b)
{
  if (a > b)
    return a;
  else
    return b;
}
