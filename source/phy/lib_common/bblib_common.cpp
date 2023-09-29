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
/*! \file bblib_common.cpp
    \brief  Common file containing helper functions used throughout the
     BBLIB SDK
*/



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



struct reading_input_file_exception : public std::exception
{
    const char * what () const throw () override {
        return "Input file cannot be read!";
    }
};

int bblib_common_read_binary_data(const std::string filename, char  * output_buffer) {


    //std::string sdk_dir = getenv("DIR_WIRELESS_SDK");

    char *sdk_dir = getenv("DIR_WIRELESS_SDK");
    if(sdk_dir == NULL) {
        printf("Failed to get environment variable DIR_WIRELESS_SDK!");
        return -1;
    }

    std::string inputfile = std::string(sdk_dir) + filename;

    std::ifstream input_stream(inputfile, std::ios::binary);
    std::vector<char> buffer((std::istreambuf_iterator<char>(input_stream)),
                                  std::istreambuf_iterator<char>());
    if(buffer.size() == 0)
        throw reading_input_file_exception();

    if(buffer.size() < sizeof(output_buffer)) {
        printf("Input file error");
        return -1;
    }

    std::copy(buffer.begin(), buffer.end(), output_buffer);


    return 0;
}



