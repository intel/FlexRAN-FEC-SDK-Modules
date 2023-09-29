#######################################################################
#
# 
#  Copyright [2019 - 2023] [Intel Corporation]
#  
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  
#  You may obtain a copy of the License at
#  
#      http://www.apache.org/licenses/LICENSE-2.0 
#  
#  Unless required by applicable law or agreed to in writing, software 
#  distributed under the License is distributed on an "AS IS" BASIS, 
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
#  See the License for the specific language governing permissions and 
#  limitations under the License. 
#  
#  SPDX-License-Identifier: Apache-2.0 
#  
# 
#
#######################################################################

# Options sepcific to unitttests
# Call add_compile_options' for EACH option
if(WIN32)
  # windows options
  add_compile_options("")
else()
  # linux options
  add_compile_options("")
endif()

# Define kernel being tested, eg lib_modulation
set(kernel
    lib_ldpc_decoder_5gnr
)

# Define all test files
set(test_files
    ldpc_decoder_functional.cc
    ldpc_decoder_performance.cc
)

# Call macro to create test binary
ADD_TEST_SUITE("${kernel}" "${test_files}" "unittests")


