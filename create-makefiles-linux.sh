#!/bin/bash

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

# This script creates the Makefiles needed to build the kernels.
# The makefiles are created in $DIR_WIRELESS_SDK_BUILD

# Set DIR_WIRELESS_SDK
export DIR_WIRELESS_SDK="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Checks
case "$GTEST_ROOT" in
"")
    echo "INFO:  Environment variable GTEST_ROOT not set, testing disabled"
    echo "       Expecting GTEST_ROOT=/your_path_to_gtest/gtest-1.7.0/"
    ;;
*)
    echo "INFO:  Environment variable GTEST_ROOT=$GTEST_ROOT"
    ;;
esac

case "$CMAKE_BUILD_TYPE" in
"Debug" | "Release" | "RelWithDebInfo" | "MinSizeRel" | "Vtune" | "Dynamic")
    echo "INFO:  Environment variable CMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE"
    ;;
"")
    export CMAKE_BUILD_TYPE=Release
    echo "INFO:  Environment variable CMAKE_BUILD_TYPE not found, defaulting to $CMAKE_BUILD_TYPE"
    ;;
*)
    echo "ERROR: Environment variable CMAKE_BUILD_TYPE set to invalid value"
    echo "       Valid settings: Debug, Release, RelWithDebInfo, MinSizeRel, Vtune"
    exit 1
    ;;
esac

case "$WIRELESS_SDK_TOOLCHAIN" in
"icc" | "gcc" | "icx")
    echo "INFO:  Environment variable WIRELESS_SDK_TOOLCHAIN=$WIRELESS_SDK_TOOLCHAIN"
    ;;
"")
    export WIRELESS_SDK_TOOLCHAIN=icc
    echo "INFO:  Environment variable WIRELESS_SDK_TOOLCHAIN not found, defaulting to $WIRELESS_SDK_TOOLCHAIN"
    ;;
*)
    echo "ERROR: Environment variable WIRELESS_SDK_TOOLCHAIN set to invalid value"
    echo "       Valid settings: icc, gcc"
    exit 1
    ;;
esac

case "$WIRELESS_SDK_TARGET_ISA" in
"sse4_2" | "avx2" | "avx512" | "snc")
    echo "INFO:  Environment variable WIRELESS_SDK_TARGET_ISA=$WIRELESS_SDK_TARGET_ISA"
    ;;
"")
    # Auto detect CPU features
    CPU_FEATURES_DETECT_SNC=`cat /proc/cpuinfo | grep avx512vbmi | grep avx512_vbmi2 | grep avx512ifma | grep avx512_bitalg | grep avx512_vpopcntdq | wc -l`
    CPU_FEATURES_DETECT_AVX512=`cat /proc/cpuinfo | grep avx512 | wc -l`
    CPU_FEATURES_DETECT_AVX2=`cat /proc/cpuinfo | grep avx2 | grep f16c | grep fma | grep bmi | wc -l`

    if [ $CPU_FEATURES_DETECT_SNC -ne 0 ]
    then
        export WIRELESS_SDK_TARGET_ISA=snc
    elif [ $CPU_FEATURES_DETECT_AVX512 -ne 0 ]
    then
        export WIRELESS_SDK_TARGET_ISA=avx512
    elif [ $CPU_FEATURES_DETECT_AVX2 -ne 0 ]
    then
        export WIRELESS_SDK_TARGET_ISA=avx2
    fi

    echo "INFO:  Environment variable WIRELESS_SDK_TARGET_ISA not found, auto-detecting $WIRELESS_SDK_TARGET_ISA "
    ;;

*)
    echo "ERROR: Environment variable WIRELESS_SDK_TARGET_ISA not set correctly"
    echo "       Valid settings: avx2, avx512, snc"
    exit 1
    ;;
esac

case "$WIRELESS_SDK_STANDARD" in
"lte" | "5gnr" | "all")
    echo "INFO:  Environment variable WIRELESS_SDK_STANDARD=$WIRELESS_SDK_STANDARD"
    ;;
"")
    export WIRELESS_SDK_STANDARD=all
    echo "INFO:  Environment variable WIRELESS_SDK_STANDARD not found, defaulting to $WIRELESS_SDK_STANDARD"
    ;;

*)
    echo "ERROR: Environment variable WIRELESS_SDK_STANDARD not set correctly"
    echo "       Valid settings: lte, 5gnr, all"
    exit 1
    ;;
esac

# Do not support DESTDIR
case "$DESTDIR" in
"")
    ;;
*)
    echo "ERROR: Environment variable DESTDIR=$DESTDIR detected"
    echo "       Use of DESTDIR not supported, please delete from environment"
    exit 1
    ;;
esac

# Checks OK - select ISA and TOOLCHAIN
export DIR_WIRELESS_SDK_BUILD=build-$WIRELESS_SDK_TARGET_ISA-$WIRELESS_SDK_TOOLCHAIN

# define toolchain file based on $WIRELESS_SDK_TOOLCHAIN
if [ $WIRELESS_SDK_TOOLCHAIN == "icc" ]
then
    TOOLCHAIN_FILE=$DIR_WIRELESS_SDK/cmake/toolchain-intel-linux.cmake
elif [ $WIRELESS_SDK_TOOLCHAIN == "gcc" ]
then
    TOOLCHAIN_FILE=$DIR_WIRELESS_SDK/cmake/toolchain-gcc-linux.cmake
elif [ $WIRELESS_SDK_TOOLCHAIN == "icx" ]
then
    TOOLCHAIN_FILE=$DIR_WIRELESS_SDK/cmake/toolchain-icx-linux.cmake
fi

# define ISA switches based on $WIRELESS_SDK_TARGET_ISA
if [ $WIRELESS_SDK_TARGET_ISA == "avx2" ]
then
    ISA_SELECT="-DISA_AVX2=1"
elif [ $WIRELESS_SDK_TARGET_ISA == "avx512" ]
then
    ISA_SELECT="-DISA_AVX512=1"
elif [ $WIRELESS_SDK_TARGET_ISA == "snc" ]
then
    ISA_SELECT="-DISA_SNC=1"
fi

# Create clean build directory
cd $DIR_WIRELESS_SDK
rm -rf $DIR_WIRELESS_SDK_BUILD
mkdir $DIR_WIRELESS_SDK_BUILD

# Generate makefiles
cd $DIR_WIRELESS_SDK_BUILD
cmake -G "Unix Makefiles" $ISA_SELECT -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_FILE .. || exit 1
cd ..
