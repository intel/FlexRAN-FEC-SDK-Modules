/**********************************************************************
 *
*
*  Copyright [2019 - 2023] [Intel Corporation]
* 
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  
*  You may obtain a copy of the License at
*  
*     http://www.apache.org/licenses/LICENSE-2.0 
*  
*  Unless required by applicable law or agreed to in writing, software 
*  distributed under the License is distributed on an "AS IS" BASIS, 
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and 
*  limitations under the License. 
*  
*  SPDX-License-Identifier: Apache-2.0 
*  
* 
 *
 **********************************************************************/
/*!
    \file   phy_ldpc_decoder_5gnr_internal.h
    \brief  Source code of External API for 5GNR LDPC Decoder functions

    __Overview:__

    The bblib_ldpc_decoder_5gnr kernel is a 5G NR LDPC Decoder function.
    It is implemented as defined in TS38212 5.3.2

\*/

#ifndef _PHY_LDPC_DECODER_5GNR_INTERNAL_H_
#define _PHY_LDPC_DECODER_5GNR_INTERNAL_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h> // AVX
#include <stdint.h>

#include "common_typedef_sdk.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PROC_BYTES 64

#ifdef __cplusplus
}
#endif

#endif
