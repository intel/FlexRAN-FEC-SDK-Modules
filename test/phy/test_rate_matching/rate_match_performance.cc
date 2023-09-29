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

#include "phy_rate_match.h"

const std::string module_name = "rate_matching";

enum class TestType {
    DL = 0,
    UL = 1
};

class RateMatchingPerf : public KernelTests {
protected:
    struct bblib_rate_match_dl_request dl_request{};
    struct bblib_rate_match_dl_response dl_response{};

    struct bblib_rate_match_ul_request ul_request{};
    struct bblib_rate_match_ul_response ul_response{};

    /* There are 2 functions to test, 2 types of tests are implemented:
       bblib_rate_match_dl (TestType::DL),
       bblib_rate_match_ul (TestType::UL) */
    TestType test_type;

    void SetUp() override {
        init_test("performance");

        const int buffer_len = 1024 * 128 * 1200;

        test_type = TestType(get_input_parameter<int>("test_type"));

        switch (test_type) {
            case TestType::DL :
                dl_request.r = get_input_parameter<int32_t>("r");
                dl_request.C = get_input_parameter<int32_t>("C");
                dl_request.direction = get_input_parameter<int8_t>("direction");
                dl_request.Nsoft = get_input_parameter<int32_t>("Nsoft");
                dl_request.KMIMO = get_input_parameter<int32_t>("KMIMO");
                dl_request.MDL_HARQ = get_input_parameter<int32_t>("MDL_HARQ");
                dl_request.G = get_input_parameter<int32_t>("G");
                dl_request.NL = get_input_parameter<int32_t>("NL");
                dl_request.Qm = get_input_parameter<int32_t>("Qm");
                dl_request.rvidx = get_input_parameter<int32_t>("rvidx");
                dl_request.bypass_rvidx = get_input_parameter<int8_t>("bypass_rvidx");
                dl_request.Kidx = get_input_parameter<int32_t>("Kidx");
                dl_request.nLen = get_input_parameter<int32_t>("nLen");

                dl_request.tin0 = get_input_parameter<uint8_t*>("tin0");
                dl_request.tin1 = get_input_parameter<uint8_t*>("tin1");
                dl_request.tin2 = get_input_parameter<uint8_t*>("tin2");

                dl_response.output = aligned_malloc<uint8_t>(buffer_len, 64);
                break;

            case TestType::UL :
                ul_request.k0withoutnull = get_input_parameter<int32_t>("k0withoutnull");
                ul_request.ncb = get_input_parameter<int32_t>("ncb");
                ul_request.e = get_input_parameter<int32_t>("e");
                ul_request.isretx = get_input_parameter<int32_t>("isretx");
                ul_request.isinverted = get_input_parameter<int32_t>("isinverted");

                ul_request.pdmout = get_input_parameter<uint8_t*>("pdmout");

                ul_response.pharqbuffer = aligned_malloc<uint8_t>(buffer_len, 64);
                ul_response.pinteleavebuffer = aligned_malloc<uint8_t>(buffer_len, 64);
                ul_response.pharqout = aligned_malloc<uint8_t>(buffer_len, 64);
                break;
        }
    }

    void TearDown() override {
        switch (test_type) {
            case TestType::DL :
                aligned_free(dl_request.tin0);
                aligned_free(dl_request.tin1);
                aligned_free(dl_request.tin2);
                aligned_free(dl_response.output);
                break;
            case TestType::UL :
                aligned_free(ul_request.pdmout);
                aligned_free(ul_response.pharqbuffer);
                aligned_free(ul_response.pinteleavebuffer);
                aligned_free(ul_response.pharqout);
                break;
        }
    }
};

#if defined(_BBLIB_SSE4_2_)
TEST_P(RateMatchingPerf, SSE4_2_Perf)
{
    switch (test_type) {
        case TestType::DL :
            performance("SSE4_2", module_name, bblib_rate_match_dl_sse, &dl_request, &dl_response);
            break;
        case TestType::UL :
            std::cout << "[----------] No function defined for UL SSE4_2." << std::endl;
            break;
    }
}
#endif

#ifdef _BBLIB_AVX2_
TEST_P(RateMatchingPerf, AVX2_Perf)
{
    switch (test_type) {
        case TestType::DL :
            performance("AVX2", module_name, bblib_rate_match_dl_avx2, &dl_request, &dl_response);
            break;
        case TestType::UL :
            performance("AVX2", module_name, bblib_rate_match_ul_avx2, &ul_request, &ul_response);
            break;
    }
}
#endif

#ifdef _BBLIB_AVX512_
TEST_P(RateMatchingPerf, AVX512_Perf)
{
    switch (test_type) {
        case TestType::DL :
            std::cout << "[----------] No function defined for DL AVX512." << std::endl;
            break;
        case TestType::UL :
            performance("AVX512", module_name, bblib_rate_match_ul_avx512, &ul_request, &ul_response);
            break;
    }
}
#endif

INSTANTIATE_TEST_CASE_P(UnitTest, RateMatchingPerf,
                        testing::ValuesIn(get_sequence(RateMatchingPerf::get_number_of_cases("performance"))));