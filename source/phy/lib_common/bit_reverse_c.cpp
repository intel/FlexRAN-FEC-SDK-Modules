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
 * @file   bit_revese_c.cpp
 * @brief  Source code of conversion between float and int16, with agc gain, with plain C code.
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <stdint.h>

#include "bit_reverse.h"


//! @{
/*! \brief Bit Reversion.
    \param [in] input Input buffer
    \param [in] num_data Number of data for conversion.
    \param [out] output Output buffer
    \return Return 0 for success, and -1 for error.
    \note Input and output is aligned with 512 bits.
*/
void bblib_bit_reverse_c(int8_t* pInOut, int32_t num_data)
{
    int8_t bit, tmpBuffer;
    int16_t byte;
    for (byte = 0; byte < num_data >> 3; byte++) {
        tmpBuffer = 0;
        for (bit = 0; bit < 8; bit++)
            if (((pInOut[byte] >> bit) & 1) == 1)
                tmpBuffer |= (1 << (7 - bit));
        pInOut[byte] = tmpBuffer;
    }
    return;
}

