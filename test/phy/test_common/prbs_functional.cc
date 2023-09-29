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

#include "pseudo_random_seq_gen.h"

const std::string module_name = "prbs";

class PRBSCheck : public KernelTests {
protected:
    struct bblib_prbs_request request {};
    struct bblib_prbs_response response {};
    struct bblib_prbs_response reference {};

    int case_num;
    uint32_t *input_vars;
    uint32_t *ref_output_num;
    uint8_t *ref_output_data;

    void SetUp() override {
        init_test("prbs_functional");

        /* Test vecotrs conain data for number of cases = case_num */
        case_num = get_input_parameter<int>("case_num");
        input_vars = get_input_parameter<uint32_t*>("input_vars");
        ref_output_num = get_reference_parameter<uint32_t*>("output_num");
        ref_output_data = get_reference_parameter<uint8_t*>("output_data");
    }

    void SetUpCase(uint16_t c_init, uint16_t gold_code_advance, uint16_t num_bits,
                       uint16_t ref_num_bits, uint8_t *ref_bits) {
        request.c_init = c_init;
        request.gold_code_advance = gold_code_advance;
        request.num_bits = num_bits;

        response.bits = aligned_malloc<uint8_t>(request.num_bits, 64);

        reference.bits = ref_bits;
        reference.num_bits = ref_num_bits;
    }

    void TearDown() override {
        aligned_free(input_vars);
        aligned_free(ref_output_num);
        aligned_free(ref_output_data);
    }

    void TearDownCase() {
        aligned_free(response.bits);
    }

    template <typename F>
    void functional(F function)
    {
        uint8_t *ref_bits = ref_output_data;

        /* Instead of loops separate test cases should be defined in conf.json
           Loop lkept after refactoring due to common test vectors (for all test cases)
           and large number of cases.*/
        for (int test_case = 0; test_case < case_num; test_case++) {
            /* input_vars is a table of 3 parameters: c_init, gold_code_advance, num_bits */
            SetUpCase((uint16_t)input_vars[test_case * 3],
                      (uint16_t)input_vars[test_case * 3 + 1],
                      (uint16_t)input_vars[test_case * 3 + 2],
                      (uint16_t)ref_output_num[test_case],
                      ref_bits);
            function(&request, &response);

            ASSERT_EQ(response.num_bits, reference.num_bits);

            unsigned output_len_bytes = response.num_bits / 8;
            if (response.num_bits % 8) //ceiling operator
                output_len_bytes++;

            ASSERT_ARRAY_EQ(reference.bits, response.bits, output_len_bytes);

            ref_bits += output_len_bytes;
            TearDownCase();
        }
    }
};

TEST_P(PRBSCheck, C_Check)
{
    functional(bblib_prbs_basic);
}

INSTANTIATE_TEST_CASE_P(UnitTest, PRBSCheck,
                        testing::ValuesIn(get_sequence(PRBSCheck::get_number_of_cases("prbs_functional"))));