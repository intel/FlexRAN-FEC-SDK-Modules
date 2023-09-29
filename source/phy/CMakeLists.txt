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

project (Kernels CXX C)

# Common files
include_directories(lib_common)

# Macro helper function to remove compile option
macro(REMOVE_COMPILE_OPTION opt)
  get_directory_property( comp_opts DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMPILE_OPTIONS )
  list(REMOVE_ITEM comp_opts "${opt}")
  set_directory_properties(PROPERTIES COMPILE_OPTIONS "${comp_opts}")
endmacro()

# Macro helper function to create a kernel library.
SET(PKGCFG_DIR "${CMAKE_BINARY_DIR}/pkgcfg")
file(MAKE_DIRECTORY ${PKGCFG_DIR})
macro(ADD_KERNEL KernelSrcs Headers)

  foreach(src ${KernelSrcs})
    list(APPEND KERNEL_SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/${src})
  endforeach()

  # Derive library names from directory name.
  file(RELATIVE_PATH kernel_dir ${CMAKE_SOURCE_DIR}/source/phy ${CMAKE_CURRENT_SOURCE_DIR})

  # Remove underscore
  string(REPLACE "lib_" "lib" kernellib "${kernel_dir}")

  # Add libray
  if($ENV{CMAKE_BUILD_TYPE} MATCHES "Dynamic")
    add_library(${kernellib} SHARED ${KERNEL_SRC_LIST} ${COMMON_SRC_LIST})
  else()
    add_library(${kernellib} ${KERNEL_SRC_LIST} ${COMMON_SRC_LIST})
  endif()
  # Copy files needed at runtime to source folder (tables etc)
  file(GLOB BIN_FILES *.dat *.bin)
  file(COPY ${BIN_FILES} DESTINATION ${CMAKE_BINARY_DIR}/source/phy/${kernel_dir})
  
  string(REPLACE "lib_" "" SDK_LINK "${kernel_dir}")
  SET(KERNEL_PCIN "${kernel_dir}.pc.in")
  SET(KERNEL_PC "flexran_sdk_${SDK_LINK}.pc")
  SET(DEST_DIR "${CMAKE_BINARY_DIR}/install/${kernel_dir}")
  SET(SDK_LIB "flexran_sdk_${SDK_LINK}")
  SET(SDK_DESC "FlexRAN SDK ${kernel_dir}")
  SET(SDK_VER "FEC_SDK_21.11")
  CONFIGURE_FILE("${KERNEL_PCIN}" "${KERNEL_PC}" @ONLY)
  file(COPY ${CMAKE_BINARY_DIR}/source/phy/${kernel_dir}/${KERNEL_PC} DESTINATION ${PKGCFG_DIR})

  # install
  install(FILES ${Headers} DESTINATION ${kernel_dir})
  install(TARGETS ${kernellib} DESTINATION ${kernel_dir} OPTIONAL)
  install(FILES ${BIN_FILES} DESTINATION ${kernel_dir} OPTIONAL)

endmacro()

# Pull in all the kernel directories.
macro(ADD_LIB_DIRS KernelNames)
  foreach(kernel ${KernelNames})
    # prefix lib_ onto kernel name
    add_subdirectory("lib_${kernel}")
  endforeach()
endmacro()

if($ENV{WIRELESS_SDK_STANDARD} MATCHES "lte")
  ADD_LIB_DIRS("${COMMON_KERNELS}")
  ADD_LIB_DIRS("${LTE_KERNELS}")
elseif($ENV{WIRELESS_SDK_STANDARD} MATCHES "5gnr")
  ADD_LIB_DIRS("${COMMON_KERNELS}")
  ADD_LIB_DIRS("${5GNR_KERNELS}")
else()
  ADD_LIB_DIRS("${COMMON_KERNELS}")
  ADD_LIB_DIRS("${LTE_KERNELS}")
  ADD_LIB_DIRS("${5GNR_KERNELS}")
  ADD_LIB_DIRS("${OTHER_KERNELS}")
endif()

