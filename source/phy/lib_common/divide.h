/*******************************************************************************
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
******************************************************************************/

/*******************************************************************************
*  @file divide.h
*  @brief This file performs ceil and floor
******************************************************************************/

/*******************************************************************************
 * Include private header files
 ******************************************************************************/
#ifndef _COMMON_DIVIDE_H_
#define _COMMON_DIVIDE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t ceili(int32_t a, int32_t b);
int32_t floori(int32_t a, int32_t b);

#ifdef __cplusplus
}
#endif

#endif
