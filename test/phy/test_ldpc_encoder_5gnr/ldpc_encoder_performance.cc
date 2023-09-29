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

#include "phy_ldpc_encoder_5gnr.h"

const std::string module_name = "ldpc_encoder_5gnr";

class LDPCEncoder5GNRPerf : public KernelTests {
protected:
    struct bblib_ldpc_encoder_5gnr_request ldpc_encoder_5gnr_request{};
    struct bblib_ldpc_encoder_5gnr_response ldpc_encoder_5gnr_response{};

    void SetUp() override {
        init_test("performance");

        const int buffer_len = 1024 * 1024;

 		ldpc_encoder_5gnr_request.Zc = get_input_parameter<uint16_t>("Zc");
        ldpc_encoder_5gnr_request.baseGraph = get_input_parameter<int32_t>("baseGraph");
        ldpc_encoder_5gnr_request.nRows = get_input_parameter<int32_t>("nRows");
        ldpc_encoder_5gnr_request.numberCodeblocks = get_input_parameter<int32_t>("numberCodeblocks");
		for (int i=0; i<ldpc_encoder_5gnr_request.numberCodeblocks; i++) {
	        ldpc_encoder_5gnr_request.input[i] = get_input_parameter<int8_t*>("input");
	        ldpc_encoder_5gnr_response.output[i] = aligned_malloc<int8_t>(buffer_len, 64);
		    //must set 0, because for some cases, output is not byte aligned
	        memset(ldpc_encoder_5gnr_response.output[i], 0, buffer_len);
        }  
    }

    void TearDown() override {
		for (int i=0; i<ldpc_encoder_5gnr_request.numberCodeblocks; i++) {
	        aligned_free(ldpc_encoder_5gnr_request.input[i]);
	        aligned_free(ldpc_encoder_5gnr_response.output[i]);
        }
    }
};

#ifdef _BBLIB_AVX512_
TEST_P(LDPCEncoder5GNRPerf, AVX512_Perf)
{
    performance("AVX512", module_name, bblib_ldpc_encoder_5gnr_avx512, &ldpc_encoder_5gnr_request, &ldpc_encoder_5gnr_response);
}
#endif

INSTANTIATE_TEST_CASE_P(UnitTest, LDPCEncoder5GNRPerf,
                        testing::ValuesIn(get_sequence(LDPCEncoder5GNRPerf::get_number_of_cases("performance"))));