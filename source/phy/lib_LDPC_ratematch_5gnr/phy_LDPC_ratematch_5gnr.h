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
    \file   phy_LDPC_ratematch_5gnr.h
    \brief  Source code of External API for 5GNR LDPC ratematch functions

    __Overview:__

    The bblib_LDPC_ratematch_5gnr kernel is a 5G NR LDPC ratematch function.
    It is implemeneted as defined in TS38212 5.4.2

    __Requirements and Test Coverage:__

    BaseGraph 1/2; Rvidx 0-3; Modulation Type BPSK/QPSK/16QAM/64QAM/256QAM

    __Algorithm Guidance:__

    The algorithm is implemented as defined in TS38212 5.4.2.
*/

#ifndef _PHY_LDPC_RATEMATCH_5GNR_H_
#define _PHY_LDPC_RATEMATCH_5GNR_H_

#include <stdint.h>

#include "common_typedef_sdk.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!
    \struct bblib_LDPC_ratematch_5gnr_request
    \brief Structure for input parameters in API of rate matching for 5GNR.
    \note input data alignment depends on modulation type.\n
          For BPSK, no alignment requirement.\n
          For QPSK, input should be aligned with 2 BITS.\n
          For 16QAM, input should be aligned with 4 BITS.\n
          For 64QAM, input should be aligned with 6 BITS.\n
          For 256QAM, input should be aligned with 1 Byte.\n
*/
struct bblib_LDPC_ratematch_5gnr_request {
    int32_t Ncb; /*!< Length of the circular buffer in bits. */

    int32_t Zc; /*!< Parameter defined in TS 38211-5.2.1. */

    int32_t E; /*!< Length of the output buffer in bits. Currently limited to 8448*8 bits */

    int32_t Qm; /*!< Modulation type, which can be 1/2/4/6/8. */

    int32_t rvidx; /*!< Redundancy version, which can be 0/1/2/3. */

    int32_t baseGraph; /*!< Base graph, which can be 1/2. */

    int32_t nullIndex; /*!< Position of starting null bits. -1 if no null bit */

    int32_t nLen; /*!< Length of null bits. 0 if no null bit */

    uint8_t *input; /*!< pointer to input stream. alignment depends on modulation type */
};

/*!
    \struct bblib_LDPC_ratematch_5gnr_response
    \brief structure for outputs of rate matching for 5GNR.
    \note output data alignment depends on modulation type.
          For BPSK, no alignment requirement
          For QPSK, input should be aligned with 2 BITS
          For 16QAM, input should be aligned with 4 BITS
          For 64QAM, input should be aligned with 6 BITS
          For 256QAM, input should be aligned with 1 Byte
 */
struct bblib_LDPC_ratematch_5gnr_response {
    uint8_t *output; /*!< Output buffer for data stream after rate matching. alignment depends on modulation type */
};

//! @{
/*! \brief rate matching for LDPC in 5GNR.
    \param [in] request Structure containing configuration information and input data.
    \param [out] response Structure containing kernel outputs.
    \note bblib_LDPC_ratematch_5gnr provides the most appropriate version for the available ISA,
          the _avx512 etc. version allow direct access to specific ISA implementations.
    \return Success: return 0, else: return -1.
*/
int32_t bblib_LDPC_ratematch_5gnr(const struct bblib_LDPC_ratematch_5gnr_request *request, struct bblib_LDPC_ratematch_5gnr_response *response);
int32_t bblib_LDPC_ratematch_5gnr_avx512(const struct bblib_LDPC_ratematch_5gnr_request *request, struct bblib_LDPC_ratematch_5gnr_response *response);
//! @}

/*! \brief Report the version number for the rate match library.
 */
void bblib_print_LDPC_ratematch_5gnr_version(void);

#ifdef __cplusplus
}
#endif

#endif
