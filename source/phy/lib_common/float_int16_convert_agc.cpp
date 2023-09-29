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
 * @file   float_int16_convert_agc.cpp
 * @brief  Source code of conversion between float and int16, with agc gain.
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>

#include "common_typedef_sdk.h"
#include "float_int16_convert_agc.h"


int16_t bblib_float_to_int16_agc( int16_t* output, float* input, int32_t num_data, float gain )
{
#if defined(_BBLIB_AVX512_)
    bblib_float_to_int16_agc_avx512( output, input, num_data, gain );
    return 0;
#elif defined(_BBLIB_AVX2_)
    bblib_float_to_int16_agc_avx2( output, input, num_data, gain );
    return 0;
#else
    bblib_float_to_int16_agc_c( output, input, num_data, gain );
    return 0;
#endif
}


int16_t bblib_float_to_int16_agc_threshold( int16_t* output, float* input, int32_t num_data,
        float gain, int16_t threshold )
{
#if defined(_BBLIB_AVX512_)
    return bblib_float_to_int16_agc_threshold_avx512( output, input, num_data, gain, threshold );
#elif defined(_BBLIB_AVX2_)
    return bblib_float_to_int16_agc_threshold_avx2( output, input, num_data, gain, threshold );
#else
    return bblib_float_to_int16_agc_threshold_c( output, input, num_data, gain, threshold );
#endif
}


int16_t bblib_int16_to_float_agc( float* output, int16_t* input, int32_t num_data, float gain )
{
#if defined(_BBLIB_AVX512_)
    bblib_int16_to_float_agc_avx512( output, input, num_data, gain );
    return 0;
#elif defined(_BBLIB_AVX2_)
    bblib_int16_to_float_agc_avx2( output, input, num_data, gain );
    return 0;
#else
    bblib_int16_to_float_agc_c( output, input, num_data, gain );
    return 0;
#endif
}


int16_t bblib_int16_to_int16_agc(int16_t* output, int16_t* input, int32_t num_data, float gain)
{
#if defined(_BBLIB_AVX512_)
    bblib_int16_to_int16_agc_avx512( output, input, num_data, gain );
    return 0;
#elif defined(_BBLIB_AVX2_)
    bblib_int16_to_int16_agc_avx2( output, input, num_data, gain );
    return 0;
#else
    bblib_int16_to_int16_agc_c( output, input, num_data, gain );
    return 0;
#endif
}

int16_t bblib_int16_to_int16_fxp_scale(int16_t *scaleOut, int16_t *scaleIn, int32_t num_samples, int16_t scale16)
{
#if defined(_BBLIB_AVX512_)
    bblib_int16_to_int16_fxp_scale_avx512( scaleOut, scaleIn, num_samples, scale16 );
    return 0;
#else
    bblib_int16_to_int16_fxp_scale_c( scaleOut, scaleIn, num_samples, scale16 );
    return 0;
#endif
}

