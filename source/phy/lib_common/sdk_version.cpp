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
 * @file   sdk_version.cpp
 * @brief  Report versions of all SDK modules
*/

#include <stdint.h>
#include <string.h>
#include "sdk_version.h"

int16_t
bblib_sdk_version(char **buffer, const char **version, int buffer_size)
{
    /* Check that the version string is set and that the buffer is
       sufficiently large */
    if (buffer_size < 1)
        return -1;

    if (!*version || (buffer_size <= strlen(*version))) {
        strncpy(*buffer, "", buffer_size-1);
        return -1;
    }

    strncpy(*buffer, *version, buffer_size-1);
    return 0;
}


struct bblib_common_init
{
    bblib_common_init()
    {
        bblib_print_common_version();
    }
};

bblib_common_init do_constructor_common;




int16_t
bblib_common_version(char *version, int buffer_size)
{
    /* The version string will be updated before the build process starts by the
     *       jobs building the library and/or preparing the release packages.
     *       Do not edit the version string manually */
    const char *msg = "FlexRAN SDK bblib_common version FEC_SDK_21.11";

    return(bblib_sdk_version(&version, &msg, buffer_size));
}

void
bblib_print_common_version()
{
    static bool was_executed = false;
    if(!was_executed) {
        was_executed = true;
        char version[BBLIB_SDK_VERSION_STRING_MAX_LEN] = { };
        bblib_common_version(version, sizeof(version));
        printf("%s\n", version);
    }
}
