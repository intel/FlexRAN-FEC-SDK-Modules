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

#include "phy_LDPC_ratematch_5gnr.h"

const std::string module_name = "LDPC_ratematch_5gnr";

class LDPCRatematch5GNRCheck : public KernelTests {
protected:
    struct bblib_LDPC_ratematch_5gnr_request LDPC_ratematch_5gnr_request{};
    struct bblib_LDPC_ratematch_5gnr_response LDPC_ratematch_5gnr_response{};
    struct bblib_LDPC_ratematch_5gnr_response LDPC_ratematch_5gnr_reference{};

    void SetUp() override {
        init_test("functional");

        const int buffer_len = 1024 * 1024;

        LDPC_ratematch_5gnr_request.Ncb = get_input_parameter<int32_t>("Ncb");
        LDPC_ratematch_5gnr_request.Zc = get_input_parameter<int32_t>("Zc");
        LDPC_ratematch_5gnr_request.E = get_input_parameter<int32_t>("E");
        LDPC_ratematch_5gnr_request.Qm = get_input_parameter<int32_t>("Qm");
        LDPC_ratematch_5gnr_request.rvidx = get_input_parameter<int32_t>("rvidx");
        LDPC_ratematch_5gnr_request.baseGraph = get_input_parameter<int32_t>("baseGraph");
        LDPC_ratematch_5gnr_request.nullIndex = get_input_parameter<int32_t>("nullIndex");
        LDPC_ratematch_5gnr_request.nLen = get_input_parameter<int32_t>("nLen");
        LDPC_ratematch_5gnr_request.input = get_input_parameter<uint8_t*>("data_in");

        LDPC_ratematch_5gnr_response.output = aligned_malloc<uint8_t>(buffer_len, 64);
        //must set 0, because for some cases, output is not byte aligned
        memset(LDPC_ratematch_5gnr_response.output, 0, buffer_len);

        LDPC_ratematch_5gnr_reference.output = get_reference_parameter<uint8_t*>("data_out");
    }

    void TearDown() override {
        aligned_free(LDPC_ratematch_5gnr_request.input);
        aligned_free(LDPC_ratematch_5gnr_response.output);
        aligned_free(LDPC_ratematch_5gnr_reference.output);
    }

    template <typename F, typename ... Args>
    void functional(F function, const std::string isa, Args ... args)
    {
        function(args ...);

        ASSERT_ARRAY_EQ(LDPC_ratematch_5gnr_response.output,
                        LDPC_ratematch_5gnr_reference.output,
                        LDPC_ratematch_5gnr_request.E/8);

        print_test_description(isa, module_name);
    }
};

#ifdef _BBLIB_AVX512_
TEST_P(LDPCRatematch5GNRCheck, AVX512_Check)
{
    functional(bblib_LDPC_ratematch_5gnr_avx512, "AVX512", &LDPC_ratematch_5gnr_request, &LDPC_ratematch_5gnr_response);
}
#endif


TEST_P(LDPCRatematch5GNRCheck, Default_Check)
{
    functional(bblib_LDPC_ratematch_5gnr, "Default", &LDPC_ratematch_5gnr_request, &LDPC_ratematch_5gnr_response);
}

INSTANTIATE_TEST_CASE_P(UnitTest, LDPCRatematch5GNRCheck,
                        testing::ValuesIn(get_sequence(LDPCRatematch5GNRCheck::get_number_of_cases("functional"))));
