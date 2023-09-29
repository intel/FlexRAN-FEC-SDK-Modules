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
 * @brief  Source code of External API for rate dematching functions
 */

#include <functional>
#include <cstdio>

#include "phy_rate_dematching_5gnr.h"
#include "sdk_version.h"

typedef void (*rate_dematching_5gnr_function)(struct bblib_rate_dematching_5gnr_request *request,
                                            struct bblib_rate_dematching_5gnr_response *response);


void bblib_print_rate_dematching_5gnr_version() {
    static bool was_executed = false;
    if (!was_executed) {
        was_executed = true;
        char version[BBLIB_SDK_VERSION_STRING_MAX_LEN] = { };
        bblib_rate_dematching_5gnr_version(version, sizeof(version));
        printf("%s\n", version);
    }
}

struct bblib_rate_dematching_5gnr_init
{
    bblib_rate_dematching_5gnr_init()
    {
        bblib_print_rate_dematching_5gnr_version();
    }
};

bblib_rate_dematching_5gnr_init do_constructor_rate_dematching_5gnr;

/** Return a pointer-to-function for a specific implementation */
static rate_dematching_5gnr_function bblib_rate_dematching_5gnr_select_on_isa() {

#ifdef _BBLIB_AVX512_
    return bblib_rate_dematching_5gnr_avx512;
#else
    return bblib_rate_dematching_5gnr_c;
#endif

}

static rate_dematching_5gnr_function default_rate_dematching_5gnr = bblib_rate_dematching_5gnr_select_on_isa();

void bblib_rate_dematching_5gnr(struct bblib_rate_dematching_5gnr_request *request,
        struct bblib_rate_dematching_5gnr_response *response) {
    return default_rate_dematching_5gnr(request, response);
}

int16_t bblib_rate_dematching_5gnr_version(char *version, int buffer_size) {
    /* The version string will be updated before the build process starts  by the
     *       jobs building the library and/or preparing the release packages.
     *       Do not edit the version string manually */
    const char *msg = "FlexRAN SDK bblib_lte_rate_dematching_5gnr version FEC_SDK_23.07";

    return (bblib_sdk_version(&version, &msg, buffer_size));
}

