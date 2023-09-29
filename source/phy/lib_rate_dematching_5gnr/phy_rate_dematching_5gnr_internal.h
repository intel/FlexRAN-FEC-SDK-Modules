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
    \file phy_rate_dematching_5gnr_internal.h
    \brief  Internal header file 5G rate dematching
 */

#ifndef _PHY_RATE_DEMATCHING_5GNR_INTERNAL_H_
#define _PHY_RATE_DEMATCHING_5GNR_INTERNAL_H_

#include <immintrin.h> // AVX

#include "phy_rate_dematching_5gnr.h"
#include "sdk_version.h"
#include "divide.h"

#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define MAX(a,b) (((a) > (b)) ? (a) : (b))

#ifdef __cplusplus
extern "C" {
#endif

void get_k0(struct bblib_rate_dematching_5gnr_request *pRM);

#ifdef __cplusplus
}
#endif

#endif
