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
/*!
    \file   phy_ldpc_encoder_5gnr.h
    \brief  Source code of External API for 5GNR LDPC Encoder functions

    __Overview:__

    The bblib_ldpc_encoder_5gnr kernel is a 5G NR LDPC Encoder function.
    It is implemented as defined in TS38212 5.3.2

    __Requirements and Test Coverage:__

    BaseGraph 1(.2). Lifting Factor >176.

    __Algorithm Guidance:__

    The algorithm is implemented as defined in TS38212 5.3.2.
*/

#ifndef _PHY_LDPC_ENCODER_5GNR_H_
#define _PHY_LDPC_ENCODER_5GNR_H_

#include <stdint.h>

#include "common_typedef_sdk.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_CB_BLOCK (32)

/*!
    \struct bblib_ldpc_encoder_5gnr_request
    \brief Structure for input parameters in API of LDPC Encoder for 5GNR.
    \note ... \n
*/
struct bblib_ldpc_encoder_5gnr_request {

    uint16_t Zc; /*!< Lifting factor Zc as defined in TS38212-5.2.1. */

    int32_t baseGraph; /*!< LDPC Base graph, which can be 1 or 2  as defined in TS38212-5.2.1. */

    int32_t nRows;  /*!< Number Rows being used for the encoding native code rate - Minimum 4 */

    int8_t numberCodeblocks;  /*!<
        Used to run several code blocks in one operation notably for low expansion factor
        All code blocks must use the same parameters above.
        numberCodeblocks * Zc must not exceed 512 */

    int8_t *input[MAX_CB_BLOCK]; /*!< 
        Pointer to input stream related to each code block
        This corresponds to the bit sequence c_k as defined in TS38.212-5.3.2.
        This includes therefore the filler bits set as 0. */
};

/*!
    \struct bblib_ldpc_encoder_5gnr_response
    \brief structure for outputs of LDPC encoder for 5GNR.
    \note
 */
struct bblib_ldpc_encoder_5gnr_response {
    int8_t *output[MAX_CB_BLOCK]; /*!< 
        Output buffer for data stream after LDPC Encoding  for each  CodeBlocks
        This corresponds to the parity bit sequence w_k as defined in TS38.212-5.3.2
        The actual length is limited to the number of rows requested.  */
};

//! @{
/*! \brief Encoder for LDPC in 5GNR.
    \param [in] request Structure containing configuration information and input data.
    \param [out] response Structure containing kernel outputs.
    \note bblib_ldpc_encoder_5gnr provides the most appropriate version for the available ISA,
          the _avx512 etc. version allow direct access to specific ISA implementations.
    \return Success: return 0, else: return -1.
*/
int32_t bblib_ldpc_encoder_5gnr(struct bblib_ldpc_encoder_5gnr_request *request, struct bblib_ldpc_encoder_5gnr_response *response);
int32_t bblib_ldpc_encoder_5gnr_avx512( struct bblib_ldpc_encoder_5gnr_request *request, struct bblib_ldpc_encoder_5gnr_response *response);
//! @}

/*! \brief Report the version number for the encoder library.
 */
void bblib_print_ldpc_encoder_5gnr_version(void);

#ifdef __cplusplus
}
#endif

#endif
