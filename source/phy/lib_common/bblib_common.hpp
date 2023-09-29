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
/*! \file bblib_common.hpp
    \brief  Common header file containing helper functions used throughout the
     BBLIB SDK
*/

#ifndef _BBLIB_COMMON_HPP_
#define _BBLIB_COMMON_HPP_


#include <cmath>
#include <fstream>
#include <numeric>
#include <vector>
#include <stdexcept>
#include <cstdlib>

#ifndef _WIN64
#include <unistd.h>
#include <sys/syscall.h>
#include <stdint.h>
#else
#include <Windows.h>
#endif


/* Common helper functions */

/*!
    \brief Reads binary input data from file within the SDK ROOT DIR

    \param [in] filename string containing the path and file name of the file
                to read. The path should start from the SDK ROOT DIR
    \param [in] output_buffer buffer with enough memory allocation to store the
                data read from filename.
    \return 0 on success
*/
int bblib_common_read_binary_data(const std::string filename, char * output_buffer, uint32_t buf_size);

/* This function is to fix or avoid Klocwork SPECTRE.VARIANT1
   issues by restricting the variable value range (x >= 0).
   Klocwork worries that the variable may be used to read
   arbitrary data beyond boundary of an array as a index. So we
   firstly use if condition to check if the variable has a valid
   value. But it is not enough because Klocwork doesn't care
   about if condition at all, we must additionally use abs()
   to double ensure the variable value is within the range.
*/
#define KLOCWORK_SPECTRE_VARIANT1_ISSUE_AVOID_SIGNED(x)                     \
{                                                                           \
    if (unlikely(0 > (x)))                                                  \
    {                                                                       \
        printf("%s(): %s(%d) negative error!\n", __FUNCTION__, (#x), (x));  \
        exit(-1);                                                           \
    }                                                                       \
    (x) = abs((x));                                                         \
};

// #define SUBMODULE_TICK     //Enable submodule statistic in response
// #define SUBMODULE_PERF_UT // Enable print API of submodule profiling statistic in SDK UT test

#ifdef SUBMODULE_TICK
#define LOG_TICK_INIT(nSm)\
{\
    for(int32_t iSm = 0;iSm < nSm; iSm++)\
    {\
        nSubModuleTick[iSm][2] = 0;\
    }\
}
#define LOG_TICK_START(a) nSubModuleTick[a][0] = __rdtsc();
#define LOG_TICK_END(a)\
{\
    nSubModuleTick[a][1] = __rdtsc();\
    nSubModuleTick[a][2] += nSubModuleTick[a][1] - nSubModuleTick[a][0];\
}
#define LOG_TICK_INSTANCE(p) p->n_cnt++;
#ifdef SUBMODULE_PERF_UT
#define LOG_TICK_REPORT(p, nSm)\
{\
    for(int32_t iSm = 0;iSm < nSm; iSm++)\
    {\
        p->n_tick[iSm] += nSubModuleTick[iSm][2];\
    }\
}

#else
#define LOG_TICK_REPORT(p, nSm)\
{\
    for(int32_t iSm = 0;iSm < nSm; iSm++)\
    {\
        p->n_tick[iSm] = nSubModuleTick[iSm][2];\
    }\
}

#endif

#else
#define LOG_TICK_INIT(nSm)
#define LOG_TICK_START(a)
#define LOG_TICK_END(a)
#define LOG_TICK_INSTANCE(p)
#define LOG_TICK_REPORT(p, nSm)
#endif

#ifdef SUBMODULE_PERF_UT

#define LOG_PERF_RESET(p, nSm)\
{\
    p->n_cnt = 0;\
    for(int32_t iSm=0;iSm<nSm;iSm++)\
    {\
        p->n_tick[iSm] = 0;\
    }\
}
#define LOG_PERF_PRINT(p, nSm)\
{\
    int64_t nTick;\
    printf("--------------------------------------\n");\
    printf("Submodule performance: iteration %u\n",p->n_cnt);\
    for(int32_t iSm = 0 ;iSm < nSm; iSm++)\
    {\
        nTick = 0;\
        nTick += p->n_tick[iSm];\
        printf("%f \tcycles\n", (float)(nTick) / (float)p->n_cnt);\
    }\
    printf("--------------------------------------\n");\
}
#else
#define LOG_PERF_RESET(p, nSm)
#define LOG_PERF_PRINT(p, nSm)
#endif
#endif /* #ifndef _BBLIB_COMMON_HPP_ */

