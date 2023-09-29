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
# To add a new kernel insert an entry in the appropriate list below.
# Please keep lists in alphabetical order and obey naming conventions

# Commom kernels (lte || 5gnr || all)
set(COMMON_KERNELS
  common
  crc
  turbo
  ldpc_encoder_5gnr
  ldpc_decoder_5gnr
  rate_dematching_5gnr
  LDPC_ratematch_5gnr
  rate_matching
)
# LTE kernels (lte || all)
set(LTE_KERNELS
)
# 5GNR kernels (5gnr || all)
set(5GNR_KERNELS
)
# Other kernels (all)
set(OTHER_KERNELS
)
