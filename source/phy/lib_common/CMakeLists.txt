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

# Kernel specific flags
# Call add_compile_options' for EACH option
if(WIN32)
  # windows options
  add_compile_options("")
else()
  # linux options
    add_compile_options("-fPIC")
endif()

# Kernel sources
set (KernelSrcs
  divide.cpp
  float_int16_convert_agc_avx2.cpp
  float_int16_convert_agc_avx512.cpp
  float_int16_convert_agc_c.cpp
  float_int16_convert_agc.cpp
  bit_reverse_avx2.cpp
  bit_reverse_avx512.cpp
  bit_reverse_c.cpp
  bit_reverse.cpp
  pseudo_random_seq_gen.cpp
  sdk_version.cpp
  simd_utils.cpp
  phy_matrix_inv_cholesky.cpp
  bblib_common.cpp
  phy_tafo_table_gen.cpp
)
# Kernel includes (public only)
set (KernelIncs
  common_typedef_sdk.h
  divide.h
  float_int16_convert_agc.h
  pseudo_random_seq_gen.h
  bit_reverse.h
  mkl_utils.h
  sdk_version.h
  bblib_common_const.h
  phy_matrix_inv_cholesky.h
  phy_tafo_table_gen.h
)

# NOTE: Enclosing variables in quotes forms a single argument parameter to allow
# for more than one file to be present.
ADD_KERNEL("${KernelSrcs}" "${KernelIncs}")
