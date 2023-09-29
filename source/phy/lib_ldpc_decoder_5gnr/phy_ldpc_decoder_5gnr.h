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
    \file   phy_ldpc_decoder_5gnr.h
    \brief  Source code of External API for 5GNR LDPC Encoder functions

    __Overview:__

    The bblib_ldpc_decoder_5gnr kernel is a 5G NR LDPC Encoder function.
    It is implemented as defined in TS38212 5.3.2

    __Requirements and Test Coverage:__

    BaseGraph 1(.2). Lifting Factor >176.

    __Algorithm Guidance:__

    The algorithm is implemented as defined in TS38212 5.3.2.
*/

#ifndef _PHY_LDPC_DECODER_5GNR_H_
#define _PHY_LDPC_DECODER_5GNR_H_

#include <stdint.h>

#include "common_typedef_sdk.h"

#ifdef __cplusplus
extern "C" {
#endif


/*!
    \struct bblib_ldpc_decoder_5gnr_request
    \brief Structure for input parameters in API of LDPC Decoder for 5GNR.
    \note ... \n
*/
struct bblib_ldpc_decoder_5gnr_request {

    uint16_t Zc; /*!< Lifting factor Zc as defined in TS38212-5.2.1. */

    int32_t baseGraph; /*!< LDPC Base graph, which can be 1 or 2 as defined in TS38212-5.2.1. */

    int32_t nRows;  /*!< Number Rows in the LDPC being used for the encoding native code rate - Minimum 4 */

    int8_t* varNodes;
    /*!<
        Pointer to the buffer used to store the code word 8-bit integer LLRs into the
        top-level of the decoder to each code block
        This corresponds to the bit sequence d_k as defined in TS38.212-5.3.2.
        This should be z*22 + z*nRows - z*2 - numFillerBits in length for BG1
        This should be z*10 + z*nRows - z*2 - numFillerBits in length for BG2
        The filler bits NULL are not included
     */

    int16_t numChannelLlrs; /*!<The number of post rate-matched output LLR values*/

    int16_t numFillerBits; /*!< The number of filler bits used in the encoding */

    int16_t maxIterations;
    /*!<
     The maximum number of iterations that the decoder will perform before
     it is forced to terminate
     */

    bool enableEarlyTermination;
    /*!<
    When true, the decoder is allowed to terminate before maxIterations
    if the parity-check equations all pass
     */
};

/*!
    \struct bblib_ldpc_decoder_5gnr_response
    \brief structure for outputs of LDPC decoder for 5GNR.
    \note
 */
struct bblib_ldpc_decoder_5gnr_response {

    int16_t* varNodes;
    /*!<
     Pointer to the buffer used to store the code word 16-bit LLR outputs
      Space allocation *must* be  >= z*22 + z*nRows for BG1
      Space allocation *must* be  >= z*10 + z*nRows for BG2
     */

    int numMsgBits;    /*!<
     Output message stored as individual bits in a pre-allocated buffer.
     Number of bytes allocation *must* be  >= ceil((z*22 - numFillerBits)/8) for BG1
     Number of bytes allocation *must* be  >= ceil((z*10 - numFillerBits)/8) for BG2
    */

    uint8_t* compactedMessageBytes; /*!< Decoded Message Data */

    int iterationAtTermination; /*!< The number of iterations executed before termination. */

    bool parityPassedAtTermination;/*!<
      True if the parity checks all had passed at termination  (Always true if
      response.iterationAtTermination  < request.maxIterations).
     */

};

//! @{
/*! \brief Encoder for LDPC in 5GNR.
    \param [in] request Structure containing configuration information and input data.
    \param [out] response Structure containing kernel outputs.
    \note bblib_ldpc_decoder_5gnr provides the most appropriate version for the available ISA,
          the _avx512 etc. version allow direct access to specific ISA implementations.
    \return Success: return 0, else: return -1.
*/
int32_t bblib_ldpc_decoder_5gnr(struct bblib_ldpc_decoder_5gnr_request *request, struct bblib_ldpc_decoder_5gnr_response *response);
int32_t bblib_ldpc_decoder_5gnr_avx2( struct bblib_ldpc_decoder_5gnr_request *request, struct bblib_ldpc_decoder_5gnr_response *response);
int32_t bblib_ldpc_decoder_5gnr_avx512( struct bblib_ldpc_decoder_5gnr_request *request, struct bblib_ldpc_decoder_5gnr_response *response);
//! @}

/*! \brief Report the version number for the decoder library.
 */
void bblib_print_ldpc_decoder_5gnr_version(void);

#ifdef __cplusplus
}
#endif

#endif
