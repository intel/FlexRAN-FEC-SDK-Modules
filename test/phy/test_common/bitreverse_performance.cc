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

class BitReversePerf : public KernelTests {
protected:
    int8_t* pInOut;
    int32_t num_data;

    void SetUp() override {
        init_test("bitreverse_performance");
        num_data = get_input_parameter<int32_t>("num_data");
        pInOut = get_input_parameter<int8_t*>("input");
    }

    void TearDown() override {
        aligned_free(pInOut);
    }
};

#ifdef _BBLIB_AVX512_
TEST_P(BitReversePerf, AVX512_Perf)
{
    performance("AVX512", module_name, bblib_bit_reverse_avx512, pInOut, num_data);
}
#endif

#ifdef _BBLIB_AVX2_
TEST_P(BitReversePerf, AVX2_Perf)
{
    performance("AVX2", module_name, bblib_bit_reverse_avx2, pInOut, num_data);
}
#endif

TEST_P(BitReversePerf, C_Perf)
{
    performance("C", module_name, bblib_bit_reverse_c, pInOut, num_data);
}

INSTANTIATE_TEST_CASE_P(UnitTest, BitReversePerf,
                        testing::ValuesIn(get_sequence(BitReversePerf::get_number_of_cases("bitreverse_performance"))));