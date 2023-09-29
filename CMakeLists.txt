#######################################################################
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

# Cmake 2.8.12 needed for "add_compile_options" command
cmake_minimum_required(VERSION 2.8.12)

#
# Top level project label
#
project (WIRELESS_SDK CXX C)

#
# Compiler options
#
# Include compile options file for chosen toolchain
# Load compiler specific kernel lists
message("WIRELESS_SDK_TOOLCHAIN is $ENV{WIRELESS_SDK_TOOLCHAIN}")
message("WIRELESS_SDK_SOURCE_DIR is ${WIRELESS_SDK_SOURCE_DIR}")
if($ENV{WIRELESS_SDK_TOOLCHAIN} MATCHES "icc")
  include (${WIRELESS_SDK_SOURCE_DIR}/cmake/intel-compile-options.cmake)
  include (${WIRELESS_SDK_SOURCE_DIR}/kernel-list-icc.cmake)
elseif($ENV{WIRELESS_SDK_TOOLCHAIN} MATCHES "gcc")
  include (${WIRELESS_SDK_SOURCE_DIR}/cmake/gcc-compile-options.cmake)
  include (${WIRELESS_SDK_SOURCE_DIR}/kernel-list-gcc.cmake)
elseif($ENV{WIRELESS_SDK_TOOLCHAIN} MATCHES "icx")
  include (${WIRELESS_SDK_SOURCE_DIR}/cmake/icx-compile-options.cmake)
  include (${WIRELESS_SDK_SOURCE_DIR}/kernel-list-icx.cmake)
endif()

#
# Wireless SDK options
#
# Set Wireless SDK preprocessor definitions for target ISA
#
if(WIN32)
  if(${ISA_SSE4_2})
    add_definitions("/D_BBLIB_SSE4_2_")
  elseif(${ISA_AVX2})
    add_definitions("/D_BBLIB_SSE4_2_ /D_BBLIB_AVX2_")
  elseif(${ISA_AVX512})
    add_definitions("/D_BBLIB_SSE4_2_ /D_BBLIB_AVX2_ /D_BBLIB_AVX512_")
  elseif(${ISA_SNC})
    add_definitions("/D_BBLIB_SSE4_2_ /D_BBLIB_AVX2_ /D_BBLIB_AVX512_ -D_BBLIB_SNC_")
  endif()
else()
  if(${ISA_SSE4_2})
    add_definitions("-D_BBLIB_SSE4_2_")
  elseif(${ISA_AVX2})
    add_definitions("-D_BBLIB_SSE4_2_ -D_BBLIB_AVX2_")
  elseif(${ISA_AVX512})
    add_definitions("-D_BBLIB_SSE4_2_ -D_BBLIB_AVX2_ -D_BBLIB_AVX512_")
  elseif(${ISA_SNC})
    add_definitions("-D_BBLIB_SSE4_2_ -D_BBLIB_AVX2_ -D_BBLIB_AVX512_ -D_BBLIB_SNC_")
  endif()
endif()


#
#  Cmake build options
#
# Tell Cmake not to prefix library names with "lib"
set(CMAKE_STATIC_LIBRARY_PREFIX "")

# The 'deliverable' output of the builds are collected together via 'make install'.
# Kernel libraries are installed in directories specified by CMAKE_INSTALL_PREFIX.
set(CMAKE_INSTALL_PREFIX install)
IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX install CACHE PATH "Cmake install prefix" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# Display compiler flags set through CMAKE_BUILD_TYPE
if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
  message("CMAKE_CXX_FLAGS_DEBUG is ${CMAKE_CXX_FLAGS_DEBUG}")
elseif(${CMAKE_BUILD_TYPE} MATCHES "Release")
  message("CMAKE_CXX_FLAGS_RELEASE is ${CMAKE_CXX_FLAGS_RELEASE}")
elseif(${CMAKE_BUILD_TYPE} MATCHES "RelWithDebInfo")
  message("CMAKE_CXX_FLAGS_RELWITHDEBINFO is ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
elseif(${CMAKE_BUILD_TYPE} MATCHES "MinSizeRel")
  message("CMAKE_CXX_FLAGS_MINSIZEREL is ${CMAKE_CXX_FLAGS_MINSIZEREL}")
elseif(${CMAKE_BUILD_TYPE} MATCHES "Vtune")
  message("CMAKE_CXX_FLAGS_VTUNE is ${CMAKE_CXX_FLAGS_VTUNE}")
elseif(${CMAKE_BUILD_TYPE} MATCHES "Dynamic")
  message("CMAKE_CXX_FLAGS_DYNAMIC is ${CMAKE_CXX_FLAGS_DYNAMIC}")
endif()

#
# Don't require modules to be built before installation
#
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY true)

#
# The kernel code
#
add_subdirectory(source/phy)

#
# c++ unit testing
#
if(EXISTS "$ENV{GTEST_ROOT}")
  enable_testing()
  find_package(GTest REQUIRED)
  add_subdirectory(test/phy)
endif()

#
# kfw
#
if(DEFINED ENV{ENABLE_KFW})
  message("ENABLING KERNEL FRAMEWORK")
  add_subdirectory(kfw)
endif()


