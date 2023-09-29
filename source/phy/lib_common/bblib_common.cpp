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
#include <iostream>

#ifndef _WIN64
#include <unistd.h>
#include <sys/syscall.h>
#include <stdint.h>
#else
#include <Windows.h>
#endif



struct reading_input_file_exception : public std::exception
{
    const char * what () const throw () override {
        return "Input file cannot be read!";
    }
};

int bblib_common_read_binary_data(const std::string filename, char  * output_buffer, uint32_t buf_size) {


    //std::string sdk_dir = getenv("DIR_WIRELESS_SDK");

    char *sdk_dir = getenv("DIR_WIRELESS_SDK");
#ifdef _WIN32
    char *cFileName = (char *)(&filename);
#endif
    if(sdk_dir == NULL) {
        std::cout<<"Failed to get environment variable DIR_WIRELESS_SDK!"<<std::endl;
        return -1;
    }

    std::string inputfile = std::string(sdk_dir) + filename;

    std::ifstream input_stream(inputfile, std::ios::binary);
    std::vector<char> buffer((std::istreambuf_iterator<char>(input_stream)),
                                  std::istreambuf_iterator<char>());
    if(buffer.size() == 0)
        throw reading_input_file_exception();

    if(buffer.size() != buf_size) {
#ifndef _WIN32
        std::cout<<"Input file ("<<filename<<") size error, expected: "<<buf_size<<" actual: "<<buffer.size()<<std::endl;
#else
        std::cout << "Input file (" << cFileName << ") size error, expected: " << buf_size << " actual: " << buffer.size() << std::endl;
#endif
        return -1;
    }

    std::copy(buffer.begin(), buffer.end(), output_buffer);


    return 0;
}



