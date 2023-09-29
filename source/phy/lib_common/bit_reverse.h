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
/*! @file   bit_reverse.h
    @brief  Source code of conversion between float and int16, with agc gain.
*/

#ifndef _BIT_REVERSE_H_
#define _BIT_REVERSE_H_

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdint.h>
#include <immintrin.h>

#ifdef __cplusplus
extern "C" {
#endif

//! @{
/*! \brief Bit Reversion.
    \param [out] inout Input and Output buffer
    \param [in] num_data Number of data in bits for conversion.
    \return Return 0 for success, and -1 for error.
    \note Input and output is aligned with 512 bits.
*/
int16_t
bblib_bit_reverse(int8_t* inout, int32_t num_data);

void bblib_bit_reverse_avx2(int8_t* inout, int32_t num_data);
void bblib_bit_reverse_c(int8_t* inout, int32_t num_data);
void bblib_bit_reverse_avx512(int8_t* inout, int32_t num_data);
//! @}



#ifdef __cplusplus
}
#endif

#endif  /* _BIT_REVERSE_H_ */
