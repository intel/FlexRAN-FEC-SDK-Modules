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
 * @file
 * @brief  Source code of External API for LTE Rate Matching,
 * Dematching ( HARQ & deinterleaver ) functions in LTE
*/

#include <functional>
#include <cstdio>
#include <cstdint>

#include "phy_rate_match.h"
#include "phy_rate_match_internal.h"
#include "sdk_version.h"

typedef int32_t (*rate_match_dl_func)(const struct bblib_rate_match_dl_request *request,
    struct bblib_rate_match_dl_response *response);

typedef int32_t (*rate_match_ul_func)(const struct bblib_rate_match_ul_request *request,
    struct bblib_rate_match_ul_response *response);

typedef int32_t (*harq_combine_ul_func)(const struct bblib_harq_combine_ul_request *request,
    struct bblib_harq_combine_ul_response *response);

typedef int32_t (*deinterleave_ul_func)(const struct bblib_deinterleave_ul_request *request,
    struct bblib_deinterleave_ul_response *response);

typedef int32_t (*turbo_adapter_ul_func)(const struct bblib_turbo_adapter_ul_request *request,
    struct bblib_turbo_adapter_ul_response *response);

struct bblib_rate_match_init
{
    bblib_rate_match_init()
    {
#if !defined(_BBLIB_AVX2_) && !defined(_BBLIB_AVX512_)
        printf("__func__ rate_match cannot run with this CPU type, needs AVX2 or greater.\n");
        exit(-1);
#endif
        bblib_print_rate_match_version();
    }
};

bblib_rate_match_init do_constructor_rate_matching;

int16_t bblib_rate_match_version(char *version, int buffer_size) {
    /* The version string will be updated before the build process starts  by the
     *       jobs building the library and/or preparing the release packages.
     *       Do not edit the version string manually */
    const char *msg = "FlexRAN SDK bblib_lte_rate_matching version FEC_SDK_23.07";

    return (bblib_sdk_version(&version, &msg, buffer_size));
}

/** Return a pointer-to-function for a specific implementation */
static rate_match_dl_func bblib_rate_match_dl_select_on_isa() {
    /* Although there are both SSE and AVX2 version of the DL code
     * there is only AVX2 support for the UL code.  Therefore the default
     * for DL will also be AVX2.
     */
    return bblib_rate_match_dl_avx2;
}

/** Return a pointer-to-function for a specific implementation */
static rate_match_ul_func bblib_rate_match_ul_select_on_isa() {
#ifdef _BBLIB_AVX512_
    return bblib_rate_match_ul_avx512;
#else
    return bblib_rate_match_ul_avx2;
#endif
}

/** Return a pointer-to-function for a specific implementation */
static harq_combine_ul_func bblib_harq_combine_ul_select_on_isa() {
#ifdef _BBLIB_AVX512_
    //return bblib_harq_combine_ul_avx512;
    return bblib_harq_combine_ul_avx2;
#else
    return bblib_harq_combine_ul_avx2;
#endif
}

/** Return a pointer-to-function for a specific implementation */
static deinterleave_ul_func bblib_deinterleave_ul_select_on_isa() {
#ifdef _BBLIB_AVX512_
    return bblib_deinterleave_ul_avx512;
#else
    return bblib_deinterleave_ul_avx2;
#endif
}

/** Return a pointer-to-function for a specific implementation */
static turbo_adapter_ul_func bblib_turbo_adapter_ul_select_on_isa() {
#ifdef _BBLIB_AVX512_
    return bblib_turbo_adapter_ul_avx512;
#else
    return bblib_turbo_adapter_ul_avx2;
#endif
}

static rate_match_dl_func default_dl_func = bblib_rate_match_dl_select_on_isa();
static rate_match_ul_func default_ul_func = bblib_rate_match_ul_select_on_isa();
static harq_combine_ul_func default_harq_func = bblib_harq_combine_ul_select_on_isa();
static deinterleave_ul_func default_deinterleave_func = bblib_deinterleave_ul_select_on_isa();
static turbo_adapter_ul_func default_turbo_adapter_func = bblib_turbo_adapter_ul_select_on_isa();

int32_t bblib_rate_match_dl(const struct bblib_rate_match_dl_request *request,
        struct bblib_rate_match_dl_response *response) {
    return default_dl_func(request, response);
}

int32_t bblib_rate_match_ul(const struct bblib_rate_match_ul_request *request,
        struct bblib_rate_match_ul_response *response) {
    return default_ul_func(request, response);
}

int32_t bblib_harq_combine_ul(const struct bblib_harq_combine_ul_request *request,
        struct bblib_harq_combine_ul_response *response) {
    return default_harq_func(request, response);
}

int32_t bblib_deinterleave_ul(const struct bblib_deinterleave_ul_request *request,
        struct bblib_deinterleave_ul_response *response) {
    return default_deinterleave_func(request, response);
}

int32_t bblib_turbo_adapter_ul(const struct bblib_turbo_adapter_ul_request *request,
        struct bblib_turbo_adapter_ul_response *response) {
    return default_turbo_adapter_func(request, response);
}

void bblib_print_rate_match_version() {
    static bool was_executed = false;
    if (!was_executed) {
        was_executed = true;
        char version[BBLIB_SDK_VERSION_STRING_MAX_LEN] = { };
        bblib_rate_match_version(version, sizeof(version));
        printf("%s\n", version);
    }
}
