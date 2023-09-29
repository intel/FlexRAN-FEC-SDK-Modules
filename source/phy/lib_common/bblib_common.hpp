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
int bblib_common_read_binary_data(const std::string filename, char * output_buffer);




#endif /* #ifndef _BBLIB_COMMON_HPP_ */

