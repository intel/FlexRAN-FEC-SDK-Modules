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

#ifndef _PSEUDO_RANDOM_SEQ_GEN_
#define _PSEUDO_RANDOM_SEQ_GEN_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
    \enum bblib_prbs_parameters.
    \brief Pseudo random sequence generation parameters
*/
enum bblib_prbs_parameters
{
    k_maxSymsSlot = 14,  /*!< maximum symbols per slot */

    k_maxSubCarr = 3300, /*!< maximum number of subcarriers */

    /*! Assume packed into 8-bit words */
    k_maxLengthPrs = ((k_maxSymsSlot * k_maxSubCarr)>>3) + 1 };

/*!
    \struct bblib_prbs_request.
    \brief Request structure for Pseudo random sequence generation
*/
struct bblib_prbs_request
{
    /*! cinit value- see TS38.211 5.2 for more details */
    uint32_t c_init;

    /*! Number of bits to output */
    uint16_t num_bits;

    /*! Gold code- Fixed for 4G and NR at 1600 */
    uint16_t gold_code_advance;
};

/*!
    \struct bblib_prbs_response.
    \brief Response structure for Pseudo random sequence generation
*/
struct bblib_prbs_response
{
    /*! Number of bits in the output sequence */
    uint16_t num_bits;

    /*! Output bit sequence of size num_bits (as defined in request),
        should be 64 byte aligned */
    uint8_t *bits;
};

/*!
    \brief Pseudo-random sequence generation as defined in TS38.211 section 5.2
    \param [in]     Request structure containing
    \param [out] Response structure containing
 */
void bblib_prbs_basic (const struct bblib_prbs_request* request, struct bblib_prbs_response* response);

#ifdef __cplusplus
}
#endif

#endif // _PSEUDO_RANDOM_SEQ_GEN_
