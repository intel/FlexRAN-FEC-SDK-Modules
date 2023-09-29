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

#include "common.hpp"

#include "phy_ldpc_decoder_5gnr.h"

const std::string module_name = "ldpc_decoder_5gnr";

class LDPCDecoder5GNRPerf : public KernelTests {
protected:
    struct bblib_ldpc_decoder_5gnr_request ldpc_decoder_5gnr_request{};
    struct bblib_ldpc_decoder_5gnr_response ldpc_decoder_5gnr_response{};

    void SetUp() override {
        init_test("performance");

        const int buffer_len = 1024 * 1024;

        ldpc_decoder_5gnr_request.Zc = get_input_parameter<uint16_t>("Zc");
        ldpc_decoder_5gnr_request.baseGraph = get_input_parameter<int32_t>("baseGraph");
        ldpc_decoder_5gnr_request.nRows = get_input_parameter<int32_t>("nRows");

        ldpc_decoder_5gnr_request.numChannelLlrs = get_input_parameter<int16_t>("numChannelLlrs");
        ldpc_decoder_5gnr_request.numFillerBits = get_input_parameter<int16_t>("numFillerBits");
        ldpc_decoder_5gnr_request.maxIterations = get_input_parameter<int16_t>("maxIterations");
        ldpc_decoder_5gnr_request.enableEarlyTermination = get_input_parameter<bool>("enableEarlyTermination");

        ldpc_decoder_5gnr_request.varNodes = get_input_parameter<int8_t*>("input");
        ldpc_decoder_5gnr_response.varNodes = aligned_malloc<int16_t>(buffer_len, 64);
        ldpc_decoder_5gnr_response.compactedMessageBytes = aligned_malloc<uint8_t>(buffer_len, 64);
        memset(ldpc_decoder_5gnr_response.varNodes, 0, buffer_len);
        memset(ldpc_decoder_5gnr_response.compactedMessageBytes, 0, buffer_len);
    }

    void TearDown() override {
        aligned_free(ldpc_decoder_5gnr_request.varNodes);
        aligned_free(ldpc_decoder_5gnr_response.varNodes);
        aligned_free(ldpc_decoder_5gnr_response.compactedMessageBytes);
    }
};

#ifdef _BBLIB_AVX512_
TEST_P(LDPCDecoder5GNRPerf, AVX512_Perf)
{
    performance("AVX512", module_name, bblib_ldpc_decoder_5gnr_avx512, &ldpc_decoder_5gnr_request, &ldpc_decoder_5gnr_response);
}
#endif

#ifdef _BBLIB_AVX2_
TEST_P(LDPCDecoder5GNRPerf, AVX2_Perf)
{
    performance("AVX2", module_name, bblib_ldpc_decoder_5gnr_avx2, &ldpc_decoder_5gnr_request, &ldpc_decoder_5gnr_response);
}
#endif

INSTANTIATE_TEST_CASE_P(UnitTest, LDPCDecoder5GNRPerf,
                        testing::ValuesIn(get_sequence(LDPCDecoder5GNRPerf::get_number_of_cases("performance"))));