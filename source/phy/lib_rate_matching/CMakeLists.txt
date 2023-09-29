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
  add_compile_options("")
  add_compile_options("-fPIC")
endif()

# Kernel sources
set (KernelSrcs
  phy_rate_match.cpp
  phy_rate_match_avx2.cpp
  phy_rate_match_sse.cpp
  phy_rate_match_sse_short.cpp
  phy_rate_match_sse_k6144.cpp
  phy_de_rate_match_avx2.cpp
  phy_de_rate_match_avx512.cpp
)

# Kernel includes (public only)
set (KernelIncs
  phy_rate_match.h
)


# NOTE: Enclosing variables in quotes forms a single argument parameter to allow
# for more than one file to be present.
ADD_KERNEL("${KernelSrcs}" "${KernelIncs}")
