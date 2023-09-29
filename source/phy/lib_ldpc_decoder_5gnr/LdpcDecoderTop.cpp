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

#include "InternalApi.hpp"

#include "Tables.hpp"
#include "LayerUtilities.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

// Not all functions from LayerUtils are used, and since they
// are static the compiler will complain
#pragma warning(disable:177)

// Compact the final messages into contiguous bytes (rather than a sequence of int16_t APP LLRs)
// These are "reversed" because bit#0 of byte#0 maps to the MSB of byte#0 (3GPP ordering)
template<typename SIMD>
static inline void CompactReverseMessages(const int16_t* varNodes, int numMessageBits, uint8_t* compactMessage)
{
  // MSG_TYPE will be the same as the return type of GetNegativeMask(SIMD)
  using MSG_TYPE = decltype(GetNegativeMask(std::declval<SIMD>()));

  const SIMD* appLLrs = (const SIMD*)varNodes;
  MSG_TYPE *compactMessagePtr = (MSG_TYPE*)compactMessage;

  // On each step of the conversion, a block of bits of type MSG_TYPE
  // is generated. The number of steps is rounded up to the nearest whole set.
  constexpr int k_numBitsPerStep = sizeof(MSG_TYPE) * 8;
  unsigned numSteps = RoundUpDiv(numMessageBits , k_numBitsPerStep);
  for (int n = 0; n != numSteps; ++n)
    compactMessagePtr[n] = GetNegativeMask(appLLrs[n]);
}

// Select the a-value for the basegraph: each Z-value = a*2^[0...N-1]
// If-else needs to be in this order as, for example, 30 % 15 = 0, but so is 30 % 5 = 0
static int ZvalueToIndex(int z)
{
  if ((z % 15) == 0)
    return (7);
  else if ((z % 13) == 0)
    return (6);
  else if ((z % 11) == 0)
    return (5);
  else if ((z % 9) == 0)
    return (4);
  else if ((z % 7) == 0)
    return (3);
  else if ((z % 5) == 0)
    return (2);
  else if ((z % 3) == 0)
    return (1);
  else if ((z % 2) == 0)
    return (0);
  else
    throw std::runtime_error(std::string("Z-value is not a product of 2, 3, 5, 7, 9, 11, 13 or 15"));
}

// Setup an internal request containg some extra information
static void LdpcSetupInternalRequest(SimdLdpc::DecoderParamsInt16 *decoderRequest,
                                     const SimdLdpc::Request* request)
{
  decoderRequest->varNodes = request->varNodes;

  decoderRequest->z = request->z;
  decoderRequest->numChannelLlrs = request->numChannelLlrs;
  decoderRequest->numFillerBits = request->numFillerBits;
  decoderRequest->nRows = request->nRows;

  //Beta = 8 --> LLR is 8s4 (beta is 0.5 in this format)
  decoderRequest->beta = 8;
  decoderRequest->maxIterations = request->maxIterations;
  decoderRequest->enableEarlyTermination = request->enableEarlyTermination;

  decoderRequest->basegraph = request->basegraph;

  int nCirculants;
  int nSystematicCols;

  if (request->basegraph == SimdLdpc::BaseGraph::BG1)
  {
    nSystematicCols = 22;
    nCirculants = int(k_bg1RowWeightsCumulative[request->nRows - 1]);

    //std::copy_n top change types and preserve const type declaration
    std::copy_n(k_bg1RowWeights, request->nRows, decoderRequest->rowWeights);
    std::copy_n(k_bg1ColumnPositions, nCirculants, decoderRequest->circulantsColPositions);

    constexpr int16_t const *k_circulantValuesLookup[8] =
    {k_bg1a2, k_bg1a3, k_bg1a5, k_bg1a7, k_bg1a9, k_bg1a11, k_bg1a13, k_bg1a15};

    int16_t const *circulantsTable = k_circulantValuesLookup[ZvalueToIndex(request->z)];
    std::copy_n(circulantsTable, nCirculants, decoderRequest->circulants);
  }
  else if (request->basegraph == SimdLdpc::BaseGraph::BG2)
  {
    nSystematicCols = 10;
    nCirculants = int(k_bg2RowWeightsCumulative[request->nRows - 1]);

    //std::copy_n top change types and preserve const type declaration
    std::copy_n(k_bg2RowWeights, request->nRows, decoderRequest->rowWeights);
    std::copy_n(k_bg2ColumnPositions, nCirculants, decoderRequest->circulantsColPositions);

    constexpr int16_t const *k_circulantValuesLookup[8] =
    {k_bg2a2, k_bg2a3, k_bg2a5, k_bg2a7, k_bg2a9, k_bg2a11, k_bg2a13, k_bg2a15};

    int16_t const *circulantsTable = k_circulantValuesLookup[ZvalueToIndex(request->z)];
    std::copy_n(circulantsTable, nCirculants, decoderRequest->circulants);
  }
  else
    throw std::runtime_error(std::string("DecoderRequest->basegraph value is neither 1 nor 2\n"));

  //Now g_circulants needs to be modulo-z
  for (int n = 0; n < nCirculants; ++n)
    decoderRequest->circulants[n] = int16_t(decoderRequest->circulants[n] % request->z);

  //Handle the filler bits
  decoderRequest->nCols = int16_t(nSystematicCols + request->nRows);
}

template<typename SIMD>
static void LdpcDecoderTop(const SimdLdpc::Request* request,
                           SimdLdpc::Response *response)
{
  SimdLdpc::DecoderParamsInt16 decoderRequest;
  LdpcSetupInternalRequest(&decoderRequest, request);

  SimdLdpc::DecoderResponseInt16 decoderResponse;
  decoderResponse.varNodes = response->varNodes;

  //Call the decoder
  SimdLdpc::LdpcLayeredDecoderAlignedInt16<SIMD>(decoderRequest, decoderResponse);

  //Outputs
  response->iterationAtTermination = decoderResponse.iter;
  response->numMsgBits = decoderResponse.numMsgBits;
  response->parityPassedAtTermination = (decoderResponse.parityErrorCount == 0);

  CompactReverseMessages<SIMD>(decoderResponse.varNodes, decoderResponse.numMsgBits,
                               response->compactedMessageBytes);
}

void SimdLdpc::DecodeAvx2(const SimdLdpc::Request* request, SimdLdpc::Response* response)
{
  LdpcDecoderTop<Is16vec16>(request, response);
}

void SimdLdpc::DecodeAvx512(const SimdLdpc::Request* request, SimdLdpc::Response *response)
{
  LdpcDecoderTop<Is16vec32>(request, response);
}
