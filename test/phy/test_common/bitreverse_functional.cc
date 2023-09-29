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

#include "bit_reverse.h"

const std::string module_name = "bit_reverse";

class BitReverseCheck : public KernelTests {
protected:
    int8_t* pInOut;
    int8_t* pRef;
    int32_t num_data;

    void SetUp() override {
        init_test("bitreverse_functional");
        num_data = get_input_parameter<int32_t>("num_data");
        pInOut = get_input_parameter<int8_t*>("input");
        pRef = get_reference_parameter<int8_t*>("output");
    }

    void TearDown() override {
        aligned_free(pInOut);
        aligned_free(pRef);
    }

    template <typename F, typename ... Args>
    void functional(F function, const std::string isa, Args ... args)
    {
        function(args ...);
        ASSERT_ARRAY_EQ(pInOut,
                        pRef,
                        num_data/8);
        print_test_description(isa, module_name);
    }
};

#ifdef _BBLIB_AVX512_
TEST_P(BitReverseCheck, AVX512_Check)
{
    functional(bblib_bit_reverse_avx512, "AVX512", pInOut, num_data);
}
#endif

#ifdef _BBLIB_AVX2_
TEST_P(BitReverseCheck, AVX2_Check)
{
    functional(bblib_bit_reverse_avx2, "AVX2", pInOut, num_data);
}
#endif

TEST_P(BitReverseCheck, C_Check)
{
    functional(bblib_bit_reverse_c, "C", pInOut, num_data);
}

TEST_P(BitReverseCheck, Default_Check)
{
    functional(bblib_bit_reverse, "Default", pInOut, num_data);
}

INSTANTIATE_TEST_CASE_P(UnitTest, BitReverseCheck,
                        testing::ValuesIn(get_sequence(BitReverseCheck::get_number_of_cases("bitreverse_functional"))));
