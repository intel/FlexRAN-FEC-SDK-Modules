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

# This file contains the GCC compiler options for both Windows and Linux

# Settings are global to WIRELESS_SDK project

# TBD

#
# Linux
#
# Set CMAKE_BUILD_TYPE specific c++ compile flags (overrides CMake defaults)
set(CMAKE_CXX_FLAGS_DEBUG  "-O1 -g")
set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "-O2 -g -DNDEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL  "-Os -DNDEBUG")
set(CMAKE_CXX_FLAGS_VTUNE  "-O3 -g")
set(CMAKE_CXX_FLAGS_DYNAMIC  "-O3 -DNDEBUG")

# Compile flags for all ISA and build types

add_compile_options("-Wall")

# More extensive error checking disabled until SDK updates
#add_compile_options("-Wcheck")
#add_compile_options("-Wremarks ")
#add_compile_options("-Werror")

add_compile_options("-std=c++17")
#add_compile_options("-restrict")
#add_compile_options("-diag-enable=all")
#add_compile_options("-diag-disable=${INTEL_DIAG_DISABLE}")
#add_compile_options("-qopt-report=4")
#add_compile_options("-qopt-report-phase=all")
#add_compile_options("-ipo")

# Compile to dynamic library in such flag
if(${CMAKE_BUILD_TYPE} MATCHES "Dynamic")
add_compile_options("-fPIC")
endif()
# Set ISA specific compile flags (do not get passed to linker)
if(${ISA_AVX2})
# Compile flags / defintions for AVX2 (Linux)
add_compile_options("-march=broadwell")

elseif(${ISA_AVX512})
# Compile flags / defintions for AVX512 (Linux)
add_compile_options("-march=skylake-avx512")
elseif(${ISA_SNC})
# Compile flags / defintions for SNC (Sunny-cove) (Linux)
add_compile_options("-march=icelake-server")
elseif(${ISA_SPR})
# Compile flags / defintions for SNC (Sunny-cove) (Linux)
add_compile_options("-march=sapphirerapids")
endif()

# linux linker flags for unittests executable
set(CMAKE_EXE_LINKER_FLAGS "-lpthread")
