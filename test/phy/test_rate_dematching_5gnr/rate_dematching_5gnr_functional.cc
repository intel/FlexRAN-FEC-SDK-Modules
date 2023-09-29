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

#include "phy_rate_dematching_5gnr.h"

const std::string module_name = "rate_dematching_5gnr";

class RateDematching5GNRCheck : public KernelTests {
protected:
    struct bblib_rate_dematching_5gnr_request request{};
    struct bblib_rate_dematching_5gnr_request request_ref{};
    struct bblib_rate_dematching_5gnr_response response{};
    struct bblib_rate_dematching_5gnr_response reference{};

    void SetUp() override {
        init_test("functional");

        /* Maximum test vector length is <128k */
        const unsigned harq_buffer = 26*1024;

        request.ncb = get_input_parameter<int32_t>("ncb");
        request.start_null_index = get_input_parameter<int32_t>("start_null");
        request.num_of_null = get_input_parameter<int32_t>("n_null");
        request.e = get_input_parameter<int32_t>("e");
        request.rvid = get_input_parameter<int32_t>("rv_id");
        request.zc = get_input_parameter<int32_t>("z_c");
        request.modulation_order = get_input_parameter<bblib_modulation_order>("mod_q");
        request.base_graph = get_input_parameter<int32_t>("flag_of_bg");
        request.isretx = get_input_parameter<int32_t>("is_retx");

#ifndef WIN32
        request.p_in = generate_random_int_numbers<int8_t>(request.e, 64, -127, 127);
#else
        request.p_in = (int8_t*)generate_random_int_numbers<int16_t>(request.e, 64, -127, 127);
#endif

        request.p_harq = aligned_malloc<int8_t>(harq_buffer, 64);
        std::memset(request.p_harq, 0, harq_buffer);
        request_ref = request;
        request_ref.p_harq = aligned_malloc<int8_t>(harq_buffer, 64);
        std::memset(request_ref.p_harq, 0, harq_buffer);
        /* Generate reference using C code */
        bblib_rate_dematching_5gnr_c(&request_ref, &reference);

    }

    void TearDown() override {
        aligned_free(request.p_in);
        aligned_free(request.p_harq);
        aligned_free(request_ref.p_harq);
    }

    template <typename F>
    void functional(F function, const std::string isa)
    {
        function(&request, &response);

        ASSERT_ARRAY_EQ(request.p_harq, request_ref.p_harq, request.ncb);
        std::memset(request.p_harq, 0,  request.ncb);
        std::memset(request_ref.p_harq, 0,  request.ncb);
        print_test_description(isa, module_name);
    }
};

#ifdef _BBLIB_AVX512_
TEST_P(RateDematching5GNRCheck, AVX512_Check)
{
        functional(bblib_rate_dematching_5gnr_avx512, "AVX512");
}
#endif

TEST_P(RateDematching5GNRCheck, C_Check)
{
        functional(bblib_rate_dematching_5gnr_c, "C");
}

TEST_P(RateDematching5GNRCheck, default_Check)
{
        functional(bblib_rate_dematching_5gnr, "Default");
}


INSTANTIATE_TEST_CASE_P(UnitTest, RateDematching5GNRCheck,
                        testing::ValuesIn(get_sequence(RateDematching5GNRCheck::get_number_of_cases("functional"))));
