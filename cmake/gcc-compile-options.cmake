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

if (WIN32)
  #
  # Windows
  #
  # Set CMAKE_BUILD_TYPE specific c++ compile flags (overrides CMake defaults)

  # Compile flags for all ISA and build types

  # Set ISA specific compile flags

else()
  #
  # Linux
  #
  # Set CMAKE_BUILD_TYPE specific c++ compile flags (overrides CMake defaults)

  # Compile flags for all ISA and build types

  # Set ISA specific compile flags

endif()
