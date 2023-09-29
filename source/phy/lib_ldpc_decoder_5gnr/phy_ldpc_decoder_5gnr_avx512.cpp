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
 *  @file   phy_ldpc_dencoder_5gnr_avx512.cpp
 *  @brief  AVX512 code for 5GNR LDPC Encoder functions.
 */

#include <stdlib.h>
#include <string.h>
#include <immintrin.h>  /* AVX512 */
#include "phy_ldpc_decoder_5gnr.h"
#include "phy_ldpc_decoder_5gnr_internal.h"
#include "LdpcDecoder.hpp"

#include "common_typedef_sdk.h"



//-------------------------------------------------------------------------------------------
/**
 *  @brief Decoding for LDPC in 5GNR.
 *  @param [in] request Structure containing configuration information and input data.
 *  @param [out] response Structure containing kernel outputs.
 *  @return Success: return 0, else: return -1.
**/
int32_t bblib_ldpc_decoder_5gnr_avx512(struct bblib_ldpc_decoder_5gnr_request *request, struct bblib_ldpc_decoder_5gnr_response *response)
{
	SimdLdpc::Request local_request;
	SimdLdpc::Response local_response;

	local_request.basegraph = request->baseGraph == 1 ?
			SimdLdpc::BaseGraph::BG1 : SimdLdpc::BaseGraph::BG2;
	local_request.enableEarlyTermination = request->enableEarlyTermination;
	local_request.maxIterations = request->maxIterations;
	local_request.nRows = request->nRows;
	local_request.numChannelLlrs = request->numChannelLlrs;
	local_request.numFillerBits = request->numFillerBits;
	local_request.varNodes = request->varNodes;
	local_request.z = request->Zc;
	local_response.compactedMessageBytes = response->compactedMessageBytes;
	local_response.varNodes = response->varNodes;

	SimdLdpc::DecodeAvx512(&local_request, &local_response);

	response->iterationAtTermination = local_response.iterationAtTermination;
	response->numMsgBits = local_response.numMsgBits;
	response->parityPassedAtTermination = local_response.parityPassedAtTermination;
	//FIXME : Workaround for now
	//Mask the last byte
	int bitsInLastByte = local_response.numMsgBits % 8;
	if (bitsInLastByte > 0) {
		int lastbyte  = (local_response.numMsgBits +7)/8 -1;
		response->compactedMessageBytes[lastbyte] &=
				(1 << bitsInLastByte) - 1;
	}

	return 0;
}
