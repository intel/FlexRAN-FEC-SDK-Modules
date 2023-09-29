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
 * @file   phy_turbo.cpp
 * @brief  External API for LTE turbo coder/decoder
*/

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string.h>

#include "phy_turbo.h"
#include "phy_turbo_internal.h"

#include "sdk_version.h"

_TurboInterleaver g_TurboInterleaver;

/* Turbo table */
extern __align(64) uint16_t g_TurboBufAddr[1068000];
extern __align(64) int32_t g_TurboBufAddrOffset[188];

/* TS36.212 Table 5.1.3-3, Turbo code internal interleaver parameters with XXXX */
extern int32_t g_Kidx_K_Nmaxrep_shuf_sdk[188][5];

typedef int32_t (*encode_function)(const bblib_turbo_encoder_request *request,
    bblib_turbo_encoder_response *response);

struct bblib_lte_turbo_init
{
    bblib_lte_turbo_init() {
#if !defined(_BBLIB_AVX2_) && !defined(_BBLIB_SSE4_2_) && !defined(_BBLIB_AVX512_)
        printf("__func__ bblib_scramble_init() cannot run with this CPU type, needs "
                       "AVX2 or SSE\n");
        exit(-1);
#endif
        /* Get SDK root folder from environment variable */
        char *pTablePath = getenv("DIR_WIRELESS_SDK");
        char *pCheckPath1 = NULL, *pCheckPath2 = NULL, *pCheckPath3 = NULL;
        uint32_t path_len, str_offset;
        if (pTablePath == NULL) {
            printf("Need to setup environment variable 'DIR_WIRELESS_SDK' for turbo,"
                           " please set it up to folder where SDK is stored, with 'export"
                           " DIR_WIRELESS_SDK=...'\n");
            exit(-1);
        }

        path_len = strlen(pTablePath);
        //check to see if string has necessary characters
#ifdef WIN32
        pCheckPath1 = strstr(pTablePath, "sdk");
        
        if ((pCheckPath1 == NULL) && (pCheckPath2 == NULL))
        {
            printf("%s is not valid!!!  DIR_WIRELESS_SDK must include\n", pTablePath);
            printf("sdk somewhere in path\n");
            exit(1);
        }
#else
        pCheckPath1 = strstr(pTablePath, "sdk/build-avx");
        pCheckPath2 = strstr(pTablePath, "sdk//build-avx");
        pCheckPath3 = strstr(pTablePath, "sdk/build-snc");

        if ((pCheckPath1 == NULL) && (pCheckPath2 == NULL) && (pCheckPath3 == NULL))
        {
            printf("%s is not valid!!!  DIR_WIRELESS_SDK must include\n", pTablePath);
            printf("sdk/build-avx or sdk//build-avx or sdk/build-snc somewhere in path\n");
            exit(-1);
        }
#endif

        //check to see if string has characters trying to access
        //files in other directories
#ifdef WIN32
        pCheckPath1 = strstr(pTablePath, "../");
#else
        pCheckPath1 = strstr(pTablePath, "..\\");
#endif
        if (pCheckPath1 != NULL)
        {
            printf("Trying to access illegal path, %s\n", pTablePath);
            printf("cannot have ../ in path!\n");
            exit(-1);
        }
    
        //check to see if string has NULL characters in the middle of the
        //string trying to widen access to other files in current directory
        //NULL character should only be at the end of the string
        pCheckPath1 = strchr(pTablePath, 0);
        str_offset = (uint64_t)(pCheckPath1) - (uint64_t)(pTablePath);
        //if NULL not at end, return fail
        if (str_offset - path_len)
        {
            printf("NULL character found in path, %s\n", pTablePath);
            printf("cannot have NULL characters in the middle!\n");
            exit(-1);
        }

        bblib_lte_turbo_interleaver_initTable(pTablePath);
        init_turbo_decoder_interleaver_table(pTablePath, &g_TurboInterleaver);
        init_common_tables(pTablePath);
        get_turbo_buf_addr_table_new(g_TurboBufAddr, g_TurboBufAddrOffset,
                                     g_Kidx_K_Nmaxrep_shuf_sdk);
    }
};

bblib_lte_turbo_init do_constructor_turbo;

void
bblib_print_turbo_version()
{
    static bool was_executed = false;
    if(!was_executed) {
        was_executed = true;
        char version[BBLIB_SDK_VERSION_STRING_MAX_LEN] = { };
        bblib_lte_turbo_version(version, sizeof(version));
        printf("%s\n", version);
    }
}

int16_t bblib_lte_turbo_version(char *version, int buffer_size) {
    /* The version string will be updated before the build process starts by the
     *       jobs building the library and/or preparing the release packages.
     *       Do not edit the version string manually */
    const char *msg = "FlexRAN SDK bblib_lte_turbo version FEC_SDK_21.11";

    return(bblib_sdk_version(&version, &msg, buffer_size));
}

static encode_function
bblib_encoder_select_on_isa() {
#ifdef _BBLIB_AVX512_
    // AVX512 has problems (see d73f51264838d41656679123733764ee3f80cb0c)
    return bblib_lte_turbo_encoder_avx2;
#elif defined _BBLIB_AVX2_
    return bblib_lte_turbo_encoder_avx2;
#else
    return bblib_lte_turbo_encoder_sse;
#endif
}

static encode_function default_encode = bblib_encoder_select_on_isa();

int32_t bblib_turbo_encoder(const struct bblib_turbo_encoder_request *request,
    struct bblib_turbo_encoder_response *response) {

    return default_encode(request, response);
}

int32_t bblib_turbo_decoder(const struct bblib_turbo_decoder_request *request,
    struct bblib_turbo_decoder_response *response) {
#ifdef _BBLIB_AVX512_
    if ((request->k) % 16 != 0) {
        return bblib_lte_turbo_decoder_8windows_sse(request, response);
    }
    else if(request->k % 32 != 0) {
        if (0 == request->max_iter_num)
            return bblib_lte_turbo_decoder_16windows_3iteration_sse(request, response);
        else
            return bblib_lte_turbo_decoder_16windows_sse(request, response);
    }
    else if(request->k % 64 != 0) {
        return bblib_lte_turbo_decoder_32windows_avx2(request, response);
    }
    else {
        return bblib_lte_turbo_decoder_64windows_avx512(request, response);
    }
#elif defined _BBLIB_AVX2_
    if ((request->k) % 16 != 0) {
        return bblib_lte_turbo_decoder_8windows_sse(request, response);
    }
    else if(request->k % 32 != 0) {
        if (0 == request->max_iter_num)
            return bblib_lte_turbo_decoder_16windows_3iteration_sse(request, response);
        else
            return bblib_lte_turbo_decoder_16windows_sse(request, response);
    }
    else {
        return bblib_lte_turbo_decoder_32windows_avx2(request, response);
    }
#else
    if ((request->k) % 16 != 0) {
        return bblib_lte_turbo_decoder_8windows_sse(request, response);
    }
    else {
        if (0 == request->max_iter_num)
            return bblib_lte_turbo_decoder_16windows_3iteration_sse(request, response);
        else
            return bblib_lte_turbo_decoder_16windows_sse(request, response);
    }
#endif
}
