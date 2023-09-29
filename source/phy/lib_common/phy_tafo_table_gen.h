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
/*! \file   phy_tafo_table_gen.h
    \brief  This file will generate ta/fo tables.
*/
#ifndef _TAFO_TABLE_GEN
#define _TAFO_TABLE_GEN


#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

/*!
    \struct bblib_ta_request.
    \brief Request structure for ta table
*/
struct bblib_ta_request
{
    int16_t n_fft_size; /*!< FFT size */                                 
    int32_t n_fullband_sc; /*!< Number of subcarriers infull band */                               
    int32_t n_cp; /*!< Number of CPs */
};

/*!
    \struct bblib_ta_response.
    \brief Response structure for ta table
*/
struct bblib_ta_response
{
    int16_t *pCeTaFftShiftScCp;/*!< TA table */
};


/*!
    \struct bblib_fo_request.
    \brief Request structure for fo table
*/
struct bblib_fo_request
{    
    int16_t n_fft_size; /*!< FFT size */                                          
};


/*!
    \struct bblib_fo_response.
    \brief Response structure for fo table
*/
struct bblib_fo_response
{
    int16_t *pFoCompScCp; /*!< FO table */
};

//! @{
/*! \brief ta/fo table generate procedures.
    \param [in] request Structure containing the input data which need to be 64 bytes alignment.
    \param [out] response Structure containing the compensated output data which need 64 byte alignment.
    \warning
    \b EXPERIMENTAL: Further optimization is possible, API may change in future release without prior notice.
*/
void bblib_init_common_time_offset_tables(const struct bblib_ta_request *request, struct bblib_ta_response *response);
void bblib_init_common_frequency_compensation_tables(const struct bblib_fo_request *request, struct bblib_fo_response *response);
//! @}

#ifdef __cplusplus
}
#endif

#endif