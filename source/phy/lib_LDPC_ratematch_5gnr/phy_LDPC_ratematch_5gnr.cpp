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

/*
 * @file   phy_LDPC_ratematch_5gnr.cpp
 * @brief  Source code of External API for scrambling and descrambling functions
*/

#include <cstdio>
#include <cstdint>
#include <functional>

#include "phy_LDPC_ratematch_5gnr.h"
#include "sdk_version.h"

typedef int32_t (*LDPC_ratematch_5gnr_function)(const bblib_LDPC_ratematch_5gnr_request *request,
    bblib_LDPC_ratematch_5gnr_response *response);

struct bblib_LDPC_ratematch_5gnr_init
{
    bblib_LDPC_ratematch_5gnr_init()
    {
#if !defined(_BBLIB_AVX512_)
        printf("__func__ bblib_LDPC_ratematch_5gnr_init() cannot run with this CPU type, needs AVX512\n");
#endif
        bblib_print_LDPC_ratematch_5gnr_version();
    }
};

bblib_LDPC_ratematch_5gnr_init do_constructor_LDPC_ratematch;

int16_t
bblib_LDPC_ratematch_5gnr_version(char *version, int buffer_size)
{
    /* The version string will be updated before the build process starts  by the
     *       jobs building the library and/or preparing the release packages.
     *       Do not edit the version string manually */
    const char *msg = "FlexRAN SDK bblib_lte_LDPC_ratematch version FEC_SDK_21.11";

    return(bblib_sdk_version(&version, &msg, buffer_size));
}

void
bblib_print_LDPC_ratematch_5gnr_version()
{
    static bool was_executed = false;
    if(!was_executed) {
        was_executed = true;
        char version[BBLIB_SDK_VERSION_STRING_MAX_LEN] = { };
        bblib_LDPC_ratematch_5gnr_version(version, sizeof(version));
        printf("%s\n", version);
    }
}

static LDPC_ratematch_5gnr_function
bblib_LDPC_ratematch_5gnr_select_on_isa() {
#ifdef _BBLIB_AVX512_
    return bblib_LDPC_ratematch_5gnr_avx512;
#else
    printf("LDPC rate matching support AVX512 only currently\n");
    exit(-1);
#endif
}

static LDPC_ratematch_5gnr_function default_LDPC_ratematch_5gnr = bblib_LDPC_ratematch_5gnr_select_on_isa();

int32_t
bblib_LDPC_ratematch_5gnr(const struct bblib_LDPC_ratematch_5gnr_request *request, struct bblib_LDPC_ratematch_5gnr_response *response)
{
    return default_LDPC_ratematch_5gnr(request, response);
}

