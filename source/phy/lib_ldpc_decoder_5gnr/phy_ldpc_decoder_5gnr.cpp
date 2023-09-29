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
 * @file   phy_ldpc_decoder_5gnr.cpp
 * @brief  Source code of External API for LDPC encoder functions
*/

#include <cstdio>
#include <cstdint>
#include <functional>

#include "phy_ldpc_decoder_5gnr.h"
#include "phy_ldpc_decoder_5gnr_internal.h"
#include "sdk_version.h"

typedef int32_t (*ldpc_decoder_5gnr_function)(bblib_ldpc_decoder_5gnr_request *request,
    bblib_ldpc_decoder_5gnr_response *response);

struct bblib_ldpc_decoder_5gnr_init
{
    bblib_ldpc_decoder_5gnr_init()
    {
#if !defined(_BBLIB_AVX2_) && !defined(_BBLIB_AVX512_)
        printf("__func__ bblib_ldpc_decoder_5gnr_init() cannot run with this CPU type, needs AVX2 or AVX512\n");
        exit(-1);
#endif
        bblib_print_ldpc_decoder_5gnr_version();
    }
};

bblib_ldpc_decoder_5gnr_init do_constructor_ldpc_decoder;

int16_t
bblib_ldpc_decoder_5gnr_version(char *version, int buffer_size)
{
    /* The version string will be updated before the build process starts  by the
     *       jobs building the library and/or preparing the release packages.
     *       Do not edit the version string manually */
    const char *msg = "FlexRAN SDK bblib_lte_ldpc_decoder version FEC_SDK_21.11";

    return(bblib_sdk_version(&version, &msg, buffer_size));
}

void
bblib_print_ldpc_decoder_5gnr_version()
{
    static bool was_executed = false;
    if(!was_executed) {
        was_executed = true;
        char version[BBLIB_SDK_VERSION_STRING_MAX_LEN] = { };
        bblib_ldpc_decoder_5gnr_version(version, sizeof(version));
        printf("%s\n", version);
    }
}

static ldpc_decoder_5gnr_function
bblib_ldpc_decoder_5gnr_select_on_isa() {
#ifdef _BBLIB_AVX512_
    return bblib_ldpc_decoder_5gnr_avx512;
#elif defined _BBLIB_AVX2_
    return bblib_ldpc_decoder_5gnr_avx2;
#else
    printf("LDPC support AVX2/512 only currently\n");
    exit(-1);
#endif
}

static ldpc_decoder_5gnr_function default_ldpc_decoder_5gnr = bblib_ldpc_decoder_5gnr_select_on_isa();

int32_t
bblib_ldpc_decoder_5gnr(struct bblib_ldpc_decoder_5gnr_request *request, struct bblib_ldpc_decoder_5gnr_response *response)
{
    return default_ldpc_decoder_5gnr(request, response);
}

