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

/*
 * @file   bit_reverse.cpp
 * @brief  Source code of bit reverse.
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>

#include "common_typedef_sdk.h"
#include "bit_reverse.h"


int16_t bblib_bit_reverse(int8_t* output, int32_t num_data)
{
#if defined(_BBLIB_AVX512_)
    bblib_bit_reverse_avx512(output, num_data);
    return 0;
#elif defined(_BBLIB_AVX2_)
    bblib_bit_reverse_avx2(output, num_data);
    return 0;
#else
    bblib_bit_reverse_c(output, num_data);
    return 0;
#endif
}

