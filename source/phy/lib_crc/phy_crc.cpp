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
 * @file   phy_crc.cpp
 * @brief  Source code of External API for LTE CRC24A/CRC24B and the corresponding
 *         crc check functions
*/

#include <stdio.h>
#include <stdint.h>
#include <cstdlib>
#include <functional>

#include "phy_crc.h"
#include "phy_crc_internal.h"
#include "sdk_version.h"

typedef void (*crc_function)(bblib_crc_request *request, bblib_crc_response *response);

struct bblib_lte_crc_init
{
    bblib_lte_crc_init()
    {
#if !defined(_BBLIB_SSE4_2_)
        printf("__func__ bblib_init_crc cannot run with this CPU type, needs SSE\n");
        exit(-1);
#endif
    }
};

bblib_lte_crc_init do_constructor_crc;

void
bblib_print_crc_version()
{
    static bool was_executed = false;
    if(!was_executed) {
        was_executed = true;
        char version[BBLIB_SDK_VERSION_STRING_MAX_LEN] = { };
        bblib_lte_crc_version(version, sizeof(version));
        printf("%s\n", version);
    }
}

int16_t
bblib_lte_crc_version(char *version, int buffer_size)
{
    /* The version string will be updated before the build process starts  by the
     *       jobs building the library and/or preparing the release packages.
     *       Do not edit the version string manually */
    const char *msg = "FlexRAN SDK bblib_lte_crc version FEC_SDK_21.11";

    return(bblib_sdk_version(&version, &msg, buffer_size));
}


static crc_function
bblib_crc24a_gen_select_on_isa() {
#ifdef _BBLIB_SNC_
    return bblib_lte_crc24a_gen_snc;
#else
    #ifdef _BBLIB_AVX512_
        return bblib_lte_crc24a_gen_avx512;
    #else
        return bblib_lte_crc24a_gen_sse;
    #endif
#endif
}

static crc_function default_crc24a_gen = bblib_crc24a_gen_select_on_isa();

void bblib_lte_crc24a_gen(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    default_crc24a_gen(request, response);
}


static crc_function
bblib_crc24a_check_select_on_isa() {
#ifdef _BBLIB_SNC_
    return bblib_lte_crc24a_check_snc;
#else
    #ifdef _BBLIB_AVX512_
        return bblib_lte_crc24a_check_avx512;
    #else
        return bblib_lte_crc24a_check_sse;
    #endif
#endif
}

static crc_function default_crc24a_check = bblib_crc24a_check_select_on_isa();

void bblib_lte_crc24a_check(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    default_crc24a_check(request, response);
}


static crc_function
bblib_crc24b_gen_select_on_isa() {
#ifdef _BBLIB_SNC_
    return bblib_lte_crc24b_gen_snc;
#else
    #ifdef _BBLIB_AVX512_
        return bblib_lte_crc24b_gen_avx512;
    #else
        return bblib_lte_crc24b_gen_sse;
    #endif
#endif
}

static crc_function default_crc24b_gen = bblib_crc24b_gen_select_on_isa();

void bblib_lte_crc24b_gen(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    default_crc24b_gen(request, response);
}


static crc_function
bblib_crc24b_check_select_on_isa() {
#ifdef _BBLIB_SNC_
    return bblib_lte_crc24b_check_snc;
#else
    #ifdef _BBLIB_AVX512_
        return bblib_lte_crc24b_check_avx512;
    #else
        return bblib_lte_crc24b_check_sse;
    #endif
#endif
}

static crc_function default_crc24b_check = bblib_crc24b_check_select_on_isa();

void bblib_lte_crc24b_check(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    default_crc24b_check(request, response);
}


static crc_function
bblib_crc24c_gen_select_on_isa() {
#ifdef _BBLIB_SNC_
    return bblib_lte_crc24c_gen_snc;
#else
    return bblib_lte_crc24c_gen_avx512;
#endif
}

static crc_function default_crc24c_gen = bblib_crc24c_gen_select_on_isa();

void bblib_lte_crc24c_gen(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    default_crc24c_gen(request, response);
}


static crc_function
bblib_crc24c_check_select_on_isa() {
#ifdef _BBLIB_SNC_
    return bblib_lte_crc24c_check_snc;
#else
    return bblib_lte_crc24c_check_avx512;
#endif
}

static crc_function default_crc24c_check = bblib_crc24c_check_select_on_isa();

void bblib_lte_crc24c_check(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    default_crc24c_check(request, response);
}


static crc_function
bblib_crc24c_1_gen_select_on_isa() {
    return bblib_lte_crc24c_1_gen_avx512;
}

static crc_function default_crc24c_1_gen = bblib_crc24c_1_gen_select_on_isa();

void bblib_lte_crc24c_1_gen(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    default_crc24c_1_gen(request, response);
}


static crc_function
bblib_crc24c_1_check_select_on_isa() {
    return bblib_lte_crc24c_1_check_avx512;
}

static crc_function default_crc24c_1_check = bblib_crc24c_1_check_select_on_isa();

void bblib_lte_crc24c_1_check(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    default_crc24c_1_check(request, response);
}


static crc_function
bblib_crc16_gen_select_on_isa() {
#ifdef _BBLIB_SNC_
    return bblib_lte_crc16_gen_snc;
#else
    #ifdef _BBLIB_AVX512_
        return bblib_lte_crc16_gen_avx512;
    #else
        return bblib_lte_crc16_gen_sse;
    #endif
#endif
}

static crc_function default_crc16_gen = bblib_crc16_gen_select_on_isa();

void bblib_lte_crc16_gen(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    default_crc16_gen(request, response);
}


static crc_function
bblib_crc16_check_select_on_isa() {
#ifdef _BBLIB_SNC_
    return bblib_lte_crc16_check_snc;
#else
    #ifdef _BBLIB_AVX512_
        return bblib_lte_crc16_check_avx512;
    #else
        return bblib_lte_crc16_check_sse;
    #endif
#endif
}

static crc_function default_crc16_check = bblib_crc16_check_select_on_isa();

void bblib_lte_crc16_check(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    default_crc16_check(request, response);
}


static crc_function
bblib_crc11_gen_select_on_isa() {
#ifdef _BBLIB_SNC_
    return bblib_lte_crc11_gen_snc;
#else
    #ifdef _BBLIB_AVX512_
        return bblib_lte_crc11_gen_avx512;
    #else
        return bblib_lte_crc11_gen_sse;
    #endif
#endif
}

static crc_function default_crc11_gen = bblib_crc11_gen_select_on_isa();

void bblib_lte_crc11_gen(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    default_crc11_gen(request, response);
}


static crc_function
bblib_crc11_check_select_on_isa() {
#ifdef _BBLIB_SNC_
    return bblib_lte_crc11_check_snc;
#else
    #ifdef _BBLIB_AVX512_
        return bblib_lte_crc11_check_avx512;
    #else
        return bblib_lte_crc11_check_sse;
    #endif
#endif
}

static crc_function default_crc11_check = bblib_crc11_check_select_on_isa();

void bblib_lte_crc11_check(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    default_crc11_check(request, response);
}


static crc_function
bblib_crc6_gen_select_on_isa() {
#ifdef _BBLIB_SNC_
    return bblib_lte_crc6_gen_snc;
#else
    #ifdef _BBLIB_AVX512_
        return bblib_lte_crc6_gen_avx512;
    #else
        return bblib_lte_crc6_gen_sse;
    #endif
#endif
}

static crc_function default_crc6_gen = bblib_crc6_gen_select_on_isa();

void bblib_lte_crc6_gen(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    default_crc6_gen(request, response);
}


static crc_function
bblib_crc6_check_select_on_isa() {
#ifdef _BBLIB_SNC_
    return bblib_lte_crc6_check_snc;
#else
    #ifdef _BBLIB_AVX512_
        return bblib_lte_crc6_check_avx512;
    #else
        return bblib_lte_crc6_check_sse;
    #endif
#endif
}

static crc_function default_crc6_check = bblib_crc6_check_select_on_isa();

void bblib_lte_crc6_check(struct bblib_crc_request *request, struct bblib_crc_response *response)
{
    default_crc6_check(request, response);
}
