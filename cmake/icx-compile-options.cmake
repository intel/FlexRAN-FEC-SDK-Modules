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

# This file contains the Intel compiler options for both Windows and Linux

# Settings are global to WIRELESS_SDK project

# TODO - cannot enable Werror /WX until warnings fixed


# diag-disable
#   :10397 Some error details are only displayed if optimisation reports are enabled, so they are
#          enabled. However this generates a load of
#          "icc: remark #10397: optimization reports are generated in *.optrpt files in the output location"
#          comments. 10397 disables this output from icc, and instead is replaced by a single cmake
#          message below
message("icc: remark #10397: optimization reports are generated in *.optrpt files in the output location")
# - 10382, cpu-dispatch Suppresses remarks about automatic cpu dispatch when xHost option is used
# - 13000 is suppressing a warning about locale settings for Windows.
# - 981 'operator evaluation unordered' - Intel recommend turning off for C++.
# - 9 'nested comments are not allowed' - this happens in IPP header files.
# - 869 Parameter unused. Commenting out the parameter everywhere is clunky.
# - 383 Reference to temporary - very common with STL so disable.
# - 2547 Include paths for MKL are set twice to same location. One is ignored anyway.
# - 11074
# - 11075
# - 11076
set (INTEL_DIAG_DISABLE "9,10397,10382,13000,981,869,383,2547,11074,11075,11076,cpu-dispatch")

if (WIN32)
  #
  # Windows
  #
  # Set CMAKE_BUILD_TYPE specific c++ compile flags (overrides CMake defaults)
  set(CMAKE_CXX_FLAGS_DEBUG  "/g")
  set(CMAKE_CXX_FLAGS_RELEASE  "/O3 /DNDEBUG")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "/O2 /g /DNDEBUG")
  set(CMAKE_CXX_FLAGS_MINSIZEREL  "/Os /DNDEBUG")
  set(CMAKE_CXX_FLAGS_VTUNE  "/O3 /g")

  # Compile flags for all ISA and build types (do not get passed to linker)

  add_compile_options("/W5")

  # More extensive error checking disabled until SDK updates
  #add_compile_options("/Wcheck")
  #add_compile_options("/WX")

  add_compile_options("/EHsc")
  add_compile_options("/Qstd=c++11")
  add_compile_options("/Qrestrict")
  add_compile_options("/Qdiag-disable:${INTEL_DIAG_DISABLE}")
  add_compile_options("/Qmkl:sequential")

  # Set ISA specific compile flags (do not get passed to linker)
  if(${ISA_SSE4_2})
    # Compile flags / defintions for SSE4_2 (Windows)
    add_compile_options("/QxSSE4.2")
  elseif(${ISA_AVX2})
    # Compile flags / defintions for AVX2 (Windows)
    add_compile_options("/QxCORE-AVX2")
  elseif(${ISA_AVX512})
    # Compile flags / defintions for AVX512 (Windows)
    add_compile_options("/QxCORE-AVX512")
    add_compile_options("/Qopt-zmm-usage=high")
    # Compile flags / defintions for SNC (Sunny-cove) (Linux)
  elseif(${ISA_SNC})
    add_compile_options("/Qxicelake-server")
  endif()

else()
  #
  # Linux
  #
  # Set CMAKE_BUILD_TYPE specific c++ compile flags (overrides CMake defaults)
  
  set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g -std=c++17")
  set(CMAKE_CXX_FLAGS_RELEASE  "-O3 -DNDEBUG -g -std=c++17")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "-O2 -g -DNDEBUG -std=c++17")
  set(CMAKE_CXX_FLAGS_MINSIZEREL  "-Os -DNDEBUG -std=c++17")
  set(CMAKE_CXX_FLAGS_VTUNE  "-O3 -g -std=c++17")
  set(CMAKE_CXX_FLAGS_DYNAMIC  "-O3 -DNDEBUG -std=c++17")


  if (${CODE_COVERAGE} MATCHES "1")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -fprofile-instr-generate -fcoverage-mapping")
    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-instr-generate -fcoverage-mapping")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fprofile-instr-generate -fcoverage-mapping")
    set(CMAKE_CXX_FLAGS_MINSIZEREL  "${CMAKE_CXX_FLAGS_MINSIZEREL} -fprofile-instr-generate -fcoverage-mapping")
    set(CMAKE_CXX_FLAGS_VTUNE  "${CMAKE_CXX_FLAGS_VTUNE} -fprofile-instr-generate -fcoverage-mapping")
    set(CMAKE_CXX_FLAGS_DYNAMIC  "${CMAKE_CXX_FLAGS_DYNAMIC} -fprofile-instr-generate -fcoverage-mapping")
  endif()
  
  message(STATUS,"CXXFLAGS: ${CMAKE_CXX_FLAGS_DEBUG}")

  # Compile flags for all ISA and build types (do not get passed to linker)

  add_compile_options("-Wno-c++11-narrowing")
  add_compile_options("-Wall")

  # More extensive error checking disabled until SDK updates
  #add_compile_options("-Wcheck")
  #add_compile_options("-Wremarks ")
  #add_compile_options("-Werror")

  #add_compile_options("-std=c++17")
  #add_compile_options("-restrict")
  add_compile_options("-diag-enable=all")
  add_compile_options("-diag-disable=${INTEL_DIAG_DISABLE}")
  #  add_compile_options("-qopt-report=4")
  # add_compile_options("-qopt-report-phase=all")
  # add_compile_options("-ipo")

  # Compile to dynamic library in such flag
  if(${CMAKE_BUILD_TYPE} MATCHES "Dynamic")
    add_compile_options("-fPIC")
  endif()
  # Set ISA specific compile flags (do not get passed to linker)
  if(${ISA_SSE4_2})
    # Compile flags / defintions for SSE4_2 (Linux)
    add_compile_options("-xSSE4.2")
  elseif(${ISA_AVX2})
    # Compile flags / defintions for AVX2 (Linux)
   add_compile_options("-xCORE-AVX2")
   #add_compile_options("-mavx512f")
   #add_compile_options("-mavx512vl")
   #add_compile_options("-mavx512bw")
   #add_compile_options("-mavx512dq")

  elseif(${ISA_AVX512})
    # Compile flags / defintions for AVX512 (Linux)
    add_compile_options("-xCORE-AVX512")
  elseif(${ISA_SPR})
    # Compile flags / defintions for sapphirerapids (Linux)
    message("icx: -march=sapphirerapids")
    add_compile_options("-march=sapphirerapids")
    add_compile_options("-mllvm")
    add_compile_options("-x86-cfma-min=8")
  elseif(${ISA_SNC})
    # Compile flags / defintions for SNC (Sunny-cove) (Linux)
    add_compile_options("-xicelake-server")
  endif()

  add_compile_options("-mcmodel=large")
  add_compile_options("-fPIC")
  add_compile_options("-mintrinsic-promote")

  # linux linker flags for unittests executable
  set(CMAKE_EXE_LINKER_FLAGS "-lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5 -lpthread -lipps")
  SET (CMAKE_RANLIB  "llvm-ranlib")
endif()
