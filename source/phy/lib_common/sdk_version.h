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
/*! \file   sdk_version.h
    \brief  This file stores the SDK version number reported by the libraries.
*/

#ifndef _SDK_VERSION_H_
#define _SDK_VERSION_H_

#include <stdint.h>

#include "common_typedef_sdk.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BBLIB_SDK_VERSION_STRING_MAX_LEN 150

/*! \brief Fill in the buffer_size long string array pointed by buff with the version string
           pointed by version.
    \param buffer Output buffer.
    \param version Version string.
    \param buffer_size Size of the buffer.
    \return 0 if success, else -1.
*/
int16_t
bblib_sdk_version(char **buffer, const char **version, int buffer_size);

/*! \brief Report the version number for the bblib_common library.
    \param [in] version Pointer to a char buffer where the version string
    should be copied.
    \param [in] buffer_size The length of the string buffer, length
    BBLIB_SDK_VERSION_STRING_MAX_LEN characters.
    \return 0 if the version string was populated, otherwise -1.
*/
int16_t
bblib_common_version(char *version, int buffer_size);

/*! \brief Print the version string of the SDK common library */
void bblib_print_common_version();

#ifdef __cplusplus
}
#endif

#endif /* #ifndef _SDK_VERSION_H_ */

