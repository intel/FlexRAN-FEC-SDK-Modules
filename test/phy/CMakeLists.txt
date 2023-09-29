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

project(PhyTests)

# Set CPU ID used for performance tests
set(PERF_CPU_ID 4)

# Options sepcific to all tests
# Call add_compile_options' for EACH option
if(WIN32)
  # windows options
  add_compile_options("")
  add_definitions("/DCPU_ID=${PERF_CPU_ID}")
else()
  # linux options
  add_compile_options("")
  add_definitions("-DCPU_ID=${PERF_CPU_ID}")
endif()

# Macro used in test directories
#
# KernelName = Name of kernel under test (eg lib_modulation)
# TestFiles  = One or more gtest test files (eg modulation_functional.cc)
# BinaryName = Output binary name (eg unittests)
#
macro(ADD_TEST_SUITE KernelName TestFiles BinaryName)

  # derive kernel library name
  string(REPLACE "lib_" "lib" KernelLib "${KernelName}")
  # derive kernel test dir name
  string(REPLACE "lib_" "test_" KernelTest "${KernelName}")


  # Define include directories
  include_directories(/opt/intel/mkl/include
                      ${GTEST_INCLUDE_DIRS}
                      ${CMAKE_SOURCE_DIR}/test/phy/common/
                      ${CMAKE_SOURCE_DIR}/source/phy/lib_common/
                      ${CMAKE_SOURCE_DIR}/source/phy/${KernelName}/)

  # Add unittessts executable target
  add_executable (${KernelTest}
                  ${TestFiles}
                  ../common/bblib_unit_test_main.cc
                 )

  # Libraries to use when linking unittests
  target_link_libraries (${KernelTest}
                        ${GTEST_BOTH_LIBRARIES}
                        ${CMAKE_BINARY_DIR}/source/phy/${KernelName}/${KernelLib}.a
                        ${CMAKE_BINARY_DIR}/test/phy/common/utility.a
                        ${CMAKE_BINARY_DIR}/source/phy/lib_common/libcommon.a)

  # Add test config, vectors and tables (needed at runtime)
  add_custom_command(TARGET ${KernelTest} POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy
                     ${CMAKE_SOURCE_DIR}/test/phy/${KernelTest}/*conf*
                     $<TARGET_FILE_DIR:${KernelTest}>/
                    )
  find_file(NAMES ${CMAKE_SOURCE_DIR}/test/phy/${KernelTest}/test_tables
            ${CMAKE_SOURCE_DIR}/test/phy/${KernelTest}/test_vectors
            )
  if(EXISTS ${CMAKE_SOURCE_DIR}/test/phy/${KernelTest}/test_tables)
  add_custom_command(TARGET ${KernelTest} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                    ${CMAKE_SOURCE_DIR}/test/phy/${KernelTest}/test_tables
                    $<TARGET_FILE_DIR:${KernelTest}>/test_tables
                    )
  endif()
  if(EXISTS ${CMAKE_SOURCE_DIR}/test/phy/${KernelTest}/test_vectors)
  add_custom_command(TARGET ${KernelTest} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_directory
                    ${CMAKE_SOURCE_DIR}/test/phy/${KernelTest}/test_vectors
                    $<TARGET_FILE_DIR:${KernelTest}>/test_vectors
                    )
  endif()


  set_target_properties(${KernelTest} PROPERTIES OUTPUT_NAME ${BinaryName})

  add_dependencies(${KernelTest} libcommon utility ${KernelLib})

endmacro()

# Macro used in test directories
#
# KernelName = Name of kernel under test (eg lib_modulation)
# LibraryPath  = Path to the kernel (the *.a file) that should be linked against the test binary
# DependencyName = Name of the kernel library (eg libcrc)
#
macro(ADD_DEPENDENCY KernelName LibraryPath DependencyName)

  # derive kernel library name
  string(REPLACE "lib_" "lib" KernelLib "${KernelName}")
  # derive kernel test dir name
  string(REPLACE "lib_" "test_" KernelTest "${KernelName}")

  # link the library
  target_link_libraries(${KernelTest} ${LibraryPath})

  # add dependency to allow parallel build
  add_dependencies(${KernelTest} ${DependencyName})

endmacro()

#
# Add commmon code
#
add_subdirectory(common)

# Add the test directories.
#
macro(ADD_TEST_DIRS KernelNames)
  foreach(kernel ${KernelNames})
    # check naming convention - source folder should exist named lib_kernelname
    if(EXISTS ${CMAKE_SOURCE_DIR}/source/phy/lib_${kernel})
      add_subdirectory ("test_${kernel}")
    else()
      message(FATAL_ERROR "test_${kernel} - invalid test directory name, source directory lib_${kernel} not found")
    endif()
  endforeach()
endmacro()

if($ENV{WIRELESS_SDK_STANDARD} MATCHES "lte")
  ADD_TEST_DIRS("${COMMON_KERNELS}")
  ADD_TEST_DIRS("${LTE_KERNELS}")
elseif($ENV{WIRELESS_SDK_STANDARD} MATCHES "5gnr")
  ADD_TEST_DIRS("${COMMON_KERNELS}")
  ADD_TEST_DIRS("${5GNR_KERNELS}")
elseif($ENV{WIRELESS_SDK_STANDARD} MATCHES "common")
  ADD_TEST_DIRS("${COMMON_KERNELS}")
else()
  ADD_TEST_DIRS("${COMMON_KERNELS}")
  ADD_TEST_DIRS("${LTE_KERNELS}")
  ADD_TEST_DIRS("${5GNR_KERNELS}")
  ADD_TEST_DIRS("${OTHER_KERNELS}")
endif()
