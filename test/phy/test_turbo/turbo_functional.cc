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

#include "phy_turbo.h"
#include <algorithm>

const std::string module_name = "turbo";

enum class TestType {
    DEC = 0,
    ENC = 1
};

class TurboCheck : public KernelTests {
protected:
    struct bblib_turbo_decoder_request dec_request{};
    struct bblib_turbo_decoder_response dec_response{};
    struct bblib_turbo_decoder_response dec_reference{};

    struct bblib_turbo_encoder_request enc_request{};
    struct bblib_turbo_encoder_response enc_response{};
    struct bblib_turbo_encoder_response enc_reference{};

    /* There are 2 functions to test, 2 types of tests are implemented:
       bblib_turbo_decoder (TestType::DEC),
       bblib_turbo_encoder (TestType::ENC) */
    TestType test_type;

    int dec_output_len;
    int dec_force_error;
    int enc_output_len;

    void SetUp() override {
        init_test("functional");

        const int dec_ag_buf_len = 6528 * 16;

        test_type = TestType(get_input_parameter<int>("test_type"));

        switch (test_type) {
#if defined(_BBLIB_AVX2_) || defined(_BBLIB_AVX512_)
            case TestType::DEC :
                dec_request.c = get_input_parameter<int32_t>("c");
                dec_request.k = get_input_parameter<int32_t>("k");
                dec_request.k_idx = get_input_parameter<int32_t>("k_idx");
                dec_request.max_iter_num = get_input_parameter<int32_t>("max_iter_num");
                dec_request.early_term_disable = get_input_parameter<int32_t>("early_term_disable");
                dec_request.input = get_input_parameter<int8_t*>("input");

                dec_force_error = get_input_parameter<int>("force_error");
                dec_output_len = get_input_parameter<int>("output_len");
                dec_response.output = aligned_malloc<uint8_t>(dec_output_len + 1, 64);
                dec_response.ag_buf = aligned_malloc<int8_t>(dec_ag_buf_len, 64);
                dec_response.cb_buf = aligned_malloc<uint16_t>((dec_request.k / 8), 64);
                memset(dec_response.output, 0, dec_output_len);

                dec_reference.output = get_reference_parameter<uint8_t*>("output");

                break;
#endif
            case TestType::ENC :
                enc_request.length = get_input_parameter<uint32_t>("turbo_len");
                enc_request.case_id = get_input_parameter<uint8_t>("case_id");
                enc_request.input_win = get_input_parameter<uint8_t*>("input_win");

                enc_output_len = get_input_parameter<int>("output_len");
                enc_response.output_win_0 = aligned_malloc<uint8_t>(enc_output_len, 64);
                enc_response.output_win_1 = aligned_malloc<uint8_t>(enc_output_len, 64);
                enc_response.output_win_2 = aligned_malloc<uint8_t>(enc_output_len, 64);

                enc_reference.output_win_0 = get_reference_parameter<uint8_t*>("output_win_0");
                enc_reference.output_win_1 = get_reference_parameter<uint8_t*>("output_win_1");
                enc_reference.output_win_2 = get_reference_parameter<uint8_t*>("output_win_2");
                break;
        }
   }

    void TearDown() override {
        switch (test_type) {
#if defined(_BBLIB_AVX2_) || defined(_BBLIB_AVX512_)
            case TestType::DEC :
                aligned_free(dec_request.input);
                aligned_free(dec_response.output);
                aligned_free(dec_response.ag_buf);
                aligned_free(dec_response.cb_buf);
                aligned_free(dec_reference.output);
                break;
#endif
            case TestType::ENC :
                aligned_free(enc_request.input_win);
                aligned_free(enc_response.output_win_0);
                aligned_free(enc_response.output_win_1);
                aligned_free(enc_response.output_win_2);
                aligned_free(enc_reference.output_win_0);
                aligned_free(enc_reference.output_win_1);
                aligned_free(enc_reference.output_win_2);
                break;
        }
    }

    template <typename F, typename ... Args>
    void functional(F function, const std::string isa, Args ... args)
    {
        int ans;
        ans = function(args ...);

        switch (test_type) {
        case TestType::DEC :
            if ( (dec_force_error == 1)||((ans == -1) && std::all_of(dec_response.output, dec_response.output+dec_output_len-1, [](int i){return i==0;})) )
            {
                break;
            }
            ASSERT_ARRAY_EQ(dec_reference.output,
                            dec_response.output,
                            dec_output_len);
            break;
        case TestType::ENC :
            if (enc_request.case_id == 0)
            {
                break;
            }
            ASSERT_ARRAY_EQ(enc_reference.output_win_0,
                            enc_response.output_win_0,
                            enc_output_len);
            ASSERT_ARRAY_EQ(enc_reference.output_win_1,
                            enc_response.output_win_1,
                            enc_output_len);
            ASSERT_ARRAY_EQ(enc_reference.output_win_2,
                            enc_response.output_win_2,
                            enc_output_len);
            break;
        }
        print_test_description(isa, module_name);
    }
};

/**
 * Test cases are not extracted as there are no test vectors to test windows smaller
 * than 32, hence ISA specific functions cannot be tested. This problem should be
 * addressed in the future (SCSY-1380).
 */

#ifdef _BBLIB_SSE4_2_
TEST_P(TurboCheck, SSE_Check)
{
    switch (test_type) {
        case TestType::DEC :
            if (0 == dec_request.max_iter_num)
            {
                functional(bblib_lte_turbo_decoder_16windows_sse, "SSE", &dec_request, &dec_response);
            }
            else
            {
                functional(bblib_lte_turbo_decoder_16windows_3iteration_sse, "SSE", &dec_request, &dec_response);
            }
            break;
        case TestType::ENC :
            functional(bblib_lte_turbo_encoder_sse, "SSE", &enc_request, &enc_response);
            break;
    }
}
#endif

#ifdef _BBLIB_AVX2_
TEST_P(TurboCheck, AVX2_Check)
{
    switch (test_type) {
        case TestType::DEC :
            functional(bblib_lte_turbo_decoder_32windows_avx2, "AVX2", &dec_request, &dec_response);
            break;
        case TestType::ENC :
            functional(bblib_lte_turbo_encoder_avx2, "AVX2", &enc_request, &enc_response);
            break;
    }
}
#endif

#ifdef _BBLIB_AVX512_
TEST_P(TurboCheck, AVX512_Check)
{
    switch (test_type) {
        case TestType::DEC :
            functional(bblib_lte_turbo_decoder_64windows_avx512, "AVX512", &dec_request, &dec_response);
            break;
        case TestType::ENC :
        // AVX512 has problems (see d73f51264838d41656679123733764ee3f80cb0c)
            if (enc_request.case_id >= 60)
            {
                functional(bblib_lte_turbo_encoder_avx512, "AVX512", &enc_request, &enc_response);
            }
            break;
    }
}
#endif

TEST_P(TurboCheck,Check)
{
    switch (test_type) {
        case TestType::DEC :
            functional(bblib_turbo_decoder, "NoAssign", &dec_request, &dec_response);
            break;
        case TestType::ENC :
            functional(bblib_turbo_encoder, "NoAssign", &enc_request, &enc_response);
            break;
    }
}

INSTANTIATE_TEST_CASE_P(UnitTest, TurboCheck,
                        testing::ValuesIn(get_sequence(TurboCheck::get_number_of_cases("functional"))));
