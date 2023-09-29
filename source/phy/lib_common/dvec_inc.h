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
*******************************************************************************/

/**
 * @file dvec_inc.h
 * @brief This header file used to differentiate between different compilers and include appropriate dvec header file.
 * used global.
 */

#ifndef _DVEC_INC_H_
#define _DVEC_INC_H_

#include <immintrin.h>

#ifndef _WIN32
    #if defined (__ICC)
        #include <dvec.h>
    #elif defined(__GNUC__) && !defined(__clang__)
        #include "dvec_fvec_ivec_gcc.h"
    #else
        #include <dvec.h>
        #ifdef _BBLIB_SPR_
        #include "dvec_fp16.hpp"
        #endif
    #endif
#else
    #include <dvec.h>
#endif

#endif /* #ifndef _DVEC_INC_H_*/

