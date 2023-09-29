#!/bin/bash

#######################################################################
#
# <COPYRIGHT_TAG>
#
#######################################################################

localPath="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


showHelp ()
{
  echo ""
  echo "export_settings.sh"
  echo "=============="
  echo "Note: This script is designed to be *sourced* rather than executed"
  echo "      It sets the environment variables used to build intel_flexran_collab"
  echo "      and DPDK libraries and then to run them with the ORAN FAPI"
  echo ""
  echo "-h --help               Display this help and exit"
  echo "-o --oneapi             Compile using the Intel OneAPI icx compiler if not present use icc or gcc"
  echo "-i --icc                Compile using the Intel System Studio 2019 if -o not present, if both -o and -i are not present default to gcc"
  echo "-s --snc                Target snc architecture, if not present default to avx512"
  echo "-avx2                   Target avx2 architecture"
  echo "-spr                    Target spr architecture"
  echo "-avx512                 Target avx512 architecture,default if no architecture params present"
}

oneapi=0
sunnycove=0
intelvs=0
continue_run=1
avx512sel=0
avx2sel=0
sprsel=0

# Check that the script is being sourced, not executed
[[ $0 != "$BASH_SOURCE" ]] || showHelp
[[ $0 != "$BASH_SOURCE" ]] || exit

# Parse script parameters
while :
do
  case $1 in
    -h | --help)
      showHelp
      continue_run=0
      shift 1
      break
      ;;
    -o | --oneapi)
      oneapi=1
      shift 1
      ;;
    -i | --icc)
      intelvs=1
      shift 1
      ;;
    -s | --snc)
     sunnycove=1
     shift 1
     ;;
    -avx2)
     avx2sel=1
     shift 1
     ;;
    -spr)
     sprsel=1
     shift 1
     ;;
    -avx512)
     avx512sel=1
     shift 1
     ;;
    *)
      if [ -z $1 ]; then
        # End of options. Move to start of execution
        break
      fi
      # Unknown options. Display help and exit
      echo "unknown option $1"
      showHelp
      continue_run=0
      break
      ;;
  esac
done
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

# Set up shell variables and toolchain
if [ $continue_run -eq 1 ]; then
   export CMAKE_BUILD_TYPE=Release
   export WIRELESS_SDK_STANDARD=all
   export GTEST_ROOT=/opt/gtest/gtest-1.7.0
   export WIRELESS_SDK_TARGET_ISA=avx512
   export WIRELESS_SDK_TOOLCHAIN=icx
   export DIR_WIRELESS_SDK=$localPath/build-${WIRELESS_SDK_TARGET_ISA}-${WIRELESS_SDK_TOOLCHAIN}
   
   if [ $sunnycove -eq 1 ]; then
        export WIRELESS_SDK_TARGET_ISA=snc
   elif [ $avx2sel -eq 1 ]; then
        export WIRELESS_SDK_TARGET_ISA=avx2
   elif [ $sprsel -eq 1 ]; then
        export WIRELESS_SDK_TARGET_ISA=spr
   else
        export WIRELESS_SDK_TARGET_ISA=avx512
   fi
   
   echo "WIRELESS_SDK_TARGET_ISA=${WIRELESS_SDK_TARGET_ISA}"  

   if [ $oneapi -eq 1 ]; then
        export WIRELESS_SDK_TOOLCHAIN=icx
        export RTE_TARGET=x86_64-native-linuxapp-${WIRELESS_SDK_TOOLCHAIN}
        export DIR_WIRELESS_SDK=$localPath/build-${WIRELESS_SDK_TARGET_ISA}-${WIRELESS_SDK_TOOLCHAIN}
        export SDK_BUILD=build-${WIRELESS_SDK_TARGET_ISA}-${WIRELESS_SDK_TOOLCHAIN}
        source /opt/intel/oneapi/setvars.sh --force
	export PATH=$PATH:/opt/intel/oneapi/compiler/latest/linux/bin-llvm/
        if [ -n "$(uname -a | grep Ubuntu)" ]; then
		echo "Ubuntu OS"
        else
   	        echo "Changing the toolchain to GCC 8.3.1 20190311 (Red Hat 8.3.1-3)"
                source /opt/rh/devtoolset-8/enable
                export LD_LIBRARY_PATH=/opt/rh/devtoolset-8/root/lib/gcc/x86_64-rdhat-linux/8/:$LD_LIBRARY_PATH
        fi   
   elif [ $intelvs -eq 1 ]; then
        export WIRELESS_SDK_TOOLCHAIN=icc
        export RTE_TARGET=x86_64-native-linuxapp-${WIRELESS_SDK_TOOLCHAIN}
        export DIR_WIRELESS_SDK=$localPath/build-${WIRELESS_SDK_TARGET_ISA}-${WIRELESS_SDK_TOOLCHAIN}
        export SDK_BUILD=build-${WIRELESS_SDK_TARGET_ISA}-${WIRELESS_SDK_TOOLCHAIN}
        source /opt/intel_2019/system_studio_2019/bin/iccvars.sh intel64 -platform linux
   else
        echo "gcc compiler"
        export WIRELESS_SDK_TOOLCHAIN=gcc
        export RTE_TARGET=x86_64-native-linuxapp-${WIRELESS_SDK_TOOLCHAIN}
        export DIR_WIRELESS_SDK=$localPath/build-${WIRELESS_SDK_TARGET_ISA}-${WIRELESS_SDK_TOOLCHAIN}
        export SDK_BUILD=build-${WIRELESS_SDK_TARGET_ISA}-${WIRELESS_SDK_TOOLCHAIN}
   fi  
 
   export SDK_BUILD_DIR=${SDK_BUILD}
   export PKG_CONFIG_PATH=$DIR_WIRELESS_SDK/pkgcfg:$PKG_CONFIG_PATH
fi
