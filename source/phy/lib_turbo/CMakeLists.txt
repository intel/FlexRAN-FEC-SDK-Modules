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
  add_compile_options("/w")
else()
  # linux options
  add_compile_options("-w")
  add_compile_options("-fPIC")
endif()

include_directories(../lib_crc)

# Kernel sources
set (KernelSrcs
  phy_turbo.cpp
  phy_turbo_encoder_avx2.cpp
  phy_turbo_encoder_avx512.cpp
  phy_turbo_decoder_16windows_sse.cpp
  phy_turbo_decoder_32windows_avx2.cpp
  phy_turbo_decoder_64windows_avx512.cpp
  phy_turbo_encoder_sse.cpp
  phy_turbo_decoder_8windows_sse.cpp
  phy_turbo_fast_interleave_sse.cpp
  phy_turbo_decoder_MakeTable.cpp
)

# Kernel includes (public only)
set (KernelIncs
  phy_turbo.h
)

# NOTE: Enclosing variables in quotes forms a single argument parameter to allow
# for more than one file to be present.
ADD_KERNEL("${KernelSrcs}" "${KernelIncs}")

