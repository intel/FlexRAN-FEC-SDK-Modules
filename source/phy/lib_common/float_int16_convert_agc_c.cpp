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
 * @file   float_int16_convert_agc_c.cpp
 * @brief  Source code of conversion between float and int16, with agc gain, with plain C code.
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdint.h>

#include "float_int16_convert_agc.h"

#define LEN_PACKET 256
#define LEN_FLOAT 32

/* number of float per instruction packet */
#define NFLOAT_IN_PACKET LEN_PACKET/LEN_FLOAT

/**
 * @brief conversion from float to int16, with float gain, with plain C code
 * @param[in] input input buffer for float
 * @param[in] num_data number of data for conversion
 * @param[in] gain gain for agc
 * @param[out] output output buffer for int16
 * @note input and output is aligned with 256 bits
 * @note (input data*gain) should be in the rage of -32768~32767, for the range of int16
 */
void bblib_float_to_int16_agc_c( int16_t* output, float* input, int32_t num_data, float gain )
{
    float temp_gain = gain;
    /* number of loops in instruction operation */
    int32_t num_loop = num_data/NFLOAT_IN_PACKET;

    /* temp for result after gain out of loop */
    float temp_mul = 0;

    /* deal with loop with 256 bits bus instructions */
#pragma ivdep
#pragma vector aligned
    for( int32_t index_loop = 0; index_loop<num_loop*NFLOAT_IN_PACKET; index_loop++ )
    {
        temp_mul = (input[index_loop]*temp_gain);
        if( temp_mul > 32767 )
            temp_mul = 32767;
        if( temp_mul < -32768 )
            temp_mul = -32768;
        output[index_loop] = (int16_t)temp_mul;
    }

    /* deal with data in tail, out of loop */
    for( int32_t index_tail = num_loop*NFLOAT_IN_PACKET; index_tail<num_data; index_tail++ )
    {
        temp_mul = input[index_tail]*gain;
        if( temp_mul > 32767 )
            temp_mul = 32767;
        if( temp_mul < -32768 )
            temp_mul = -32768;

        output[index_tail] = (int16_t)temp_mul;
    }
    return;
}

/**
 * @brief conversion from float to int16, with float gain and int16 threshold, with plain C code
 * @param[in] input input buffer for float
 * @param[in] num_data number of data for conversion
 * @param[in] gain gain for agc
 * @param[in] threshold threshold after agc, which should be >=0
 * @param[out] output output buffer for int16
 * @return 0 for success, and -1 for error
 * @note input and output is aligned with 256 bits
 */
int16_t bblib_float_to_int16_agc_threshold_c( int16_t* output, float* input, int32_t num_data, float gain, int16_t threshold )
{
    /* number of loops in instruction operation */
    int32_t num_loop = num_data/NFLOAT_IN_PACKET;

    /* temp for result after gain out of loop */
    float temp_mul = 0;

    /* negative threshold */
    int16_t neg_threshold = -1*threshold;

    /* deal with loop with 256 bits bus instructions */
#pragma ivdep
#pragma vector aligned
    for( int32_t index_loop = 0; index_loop<num_loop*NFLOAT_IN_PACKET; index_loop++ )
    {
        temp_mul = input[index_loop]*gain;
        if( temp_mul > threshold )
            temp_mul = threshold;
        if( temp_mul < neg_threshold )
            temp_mul = neg_threshold;

        output[index_loop] = (int16_t)temp_mul;
    }

    /* deal with data in tail, out of loop */
    for( int32_t index_tail = num_loop*NFLOAT_IN_PACKET; index_tail<num_data; index_tail++ )
    {
        temp_mul = input[index_tail]*gain;
        if( temp_mul > threshold )
            temp_mul = threshold;
        if( temp_mul < neg_threshold )
            temp_mul = neg_threshold;

        output[index_tail] = (int16_t)temp_mul;
    }

    return 0;
}

/**
 * @brief conversion from int16 to float, with float gain, with plain C code
 * @param[in] input input buffer for int16
 * @param[in] num_data number of data for conversion
 * @param[in] gain gain for agc
 * @param[out] output output buffer for float
 * @note input and output is aligned with 256 bits
 */
void bblib_int16_to_float_agc_c( float* output, int16_t* input, int32_t num_data, float gain )
{
    /* number of loops in instruction operation */
    int32_t num_loop = num_data/NFLOAT_IN_PACKET;

    float temp_gain = gain;

    /* deal with loop with 256 bits bus instructions */
#pragma ivdep
#pragma vector aligned
    for( int32_t index_loop = 0; index_loop<num_loop*NFLOAT_IN_PACKET; index_loop++ )
    {
        output[index_loop] = ((float)(input[index_loop]))*temp_gain;
    }

    /* deal with data in tail, out of loop */
    for( int32_t index_tail = num_loop*NFLOAT_IN_PACKET; index_tail<num_data; index_tail++ )
    {
        output[index_tail] = ((float)(input[index_tail]))*temp_gain;
    }

    return;
}

/**
* @brief conversion from int16 to int16, with float gain, with plain C code
* @param[in] input input buffer for int16
* @param[in] num_data number of data for conversion
* @param[in] gain gain for agc
* @param[out] output output buffer for int16
* @note input and output is aligned with 256 bits
* @note (input data*gain) should be in the rage of -32768~32767, for the range of int16
*/
void bblib_int16_to_int16_agc_c(int16_t* output, int16_t* input, int32_t num_data, float gain)
{

    /* number of loops in instruction operation */
    int32_t num_loop = num_data/NFLOAT_IN_PACKET;

    float temp_gain = gain;
    float temp_mul = 0;

    /* deal with data in tail, out of loop */
#pragma ivdep
#pragma vector aligned
    for (int32_t index_loop = 0; index_loop<num_loop*NFLOAT_IN_PACKET; index_loop++)
    {
        temp_mul = (float)(input[index_loop]) * temp_gain;
        if (temp_mul > 32767)
            temp_mul = 32767;
        if (temp_mul < -32768)
            temp_mul = -32768;

        output[index_loop] = (int16_t)temp_mul;
    }

    /* deal with data in tail, out of loop */
    for (int32_t index_tail = num_loop*NFLOAT_IN_PACKET; index_tail<num_data; index_tail++)
    {
        temp_mul = (float)(input[index_tail]) * temp_gain;
        if (temp_mul > 32767)
            temp_mul = 32767;
        if (temp_mul < -32768)
            temp_mul = -32768;

        output[index_tail] = (int16_t)temp_mul;
    }

    return;
}

/**
* @brief add fxp scale to int16
* @param[in] scaleIn scaling input
* @param[in] num_samples number of samples
* @param[in] scale16 scaling in int16
* @param[out] scaleOut scaling output
*/
void bblib_int16_to_int16_fxp_scale_c(int16_t *scaleOut, int16_t *scaleIn, int32_t num_samples, int16_t scale16)
{
    for(int32_t k=0; k< num_samples; k++)
    {
        int32_t temp = scaleIn[k] * scale16;
        scaleOut[k] = (int16_t)(temp >> 16);
    }
    return;
}

