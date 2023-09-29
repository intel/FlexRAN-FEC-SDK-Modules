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
/*!
    \file   phy_crc.h
    \brief  External API for lib_crc, which comprises of CRC generate and CRC validate functions
     for the following CRC algorithms as specified in 3GPP TS 38.212 v15.1.1:
     CRC24A, CRC24B, CRC24C, CRC16, CRC11 & CRC6, all initialised with zeros, and CRC24C
     initialised with ones as specified in 3GPP TS 38.212 section 7.3.2.

     The CRC generate function (bblib_lte_<algorithm>_gen) is used to calculate the CRC value
     based on the sequence of input data and data length, in bits passed to the function in the
     request structure. The CRC value is then appended to the end of the input data sequence and
     available in the response structure. Due to the nature of the algorithms being byte based,
     maximum performance is obtained when input data length is in multiples of 8 bits (ie. bytes),
     since padding of the data isn't required.

     The CRC validate function (bblib_lte_<algorithm>_check) is used to validate input data that
     already contains a CRC appended to the end. It calculates a CRC value and then compares that
     value to the one at the end of the data. The result (pass or fail) is indicated in the response
     structure.

     Testing:
     Each CRC algorithm's generate & validate function is tested over a range of test vectors from 1 to
     65536 bits, with both multiples and non-multiples of 8 bits.
     A series of performance tests have also been defined generally based on 2344 & 2340 bit test vectors.
*/

#ifndef _PHY_CRC_H_
#define _PHY_CRC_H_

#include <stdint.h>
#include <stdbool.h>

#include "common_typedef_sdk.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!
    \struct bblib_crc_request
    \brief Request structure containing pointer to input data sequence and its length (in bits).
*/
struct bblib_crc_request {
    uint8_t *data; /*!< Pointer to input data sequence for the CRC generator or CRC validate functions.
                        Input data should be byte aligned in memory*/


    uint32_t len;  /*!< The length of input data, in bits.*/
};

/*!
    \struct bblib_crc_response
    \brief Response structure containing pointer to input data with appended CRC, new data length, CRC value
     and result of a CRC validate function.
*/
struct bblib_crc_response {
    uint8_t *data;      /*!< Pointer to output data comprising of the input data with appended CRC value.*/

    uint32_t len;       /*!< The length of output data including the CRC value, in bits.*/

    uint32_t crc_value; /*!< The calculated CRC value rounded to whole bytes,by appending zeros.
                             Example: CRC11 bit binary value of 10010110001 (0x962) would be rounded to
                                      the nearest byte therefore becomes 010010110001 (0x4b1)*/

    bool check_passed;  /*!< Result of CRC Check Function, where true = passed, false = failed. */

};



/*! \brief Report the version number for the bblib_lte_crc library
    \param [in] version Pointer to a char buffer where the version string should be copied.
    \param [in] buffer_size The length of the string buffer, must be at least
           BBLIB_SDK_VERSION_STRING_MAX_LEN characters.
    \return 0 if the version string was populated, otherwise -1.
*/
int16_t
bblib_lte_crc_version(char *version, int buffer_size);

//! @{
/*! \brief Performs CRC24A generate, calculating the CRC value and appending to the data.
    \param [in] request structure containing pointer to input data and data length.
    \param [out] response structure containing calculated CRC value, CRC appended data and new length.
    \return void
    \note  Memory for both request.data & response.data structures should be allocated.
     Response.data can point to request.data if sufficient space is available at end of
     request.data structure for the appended CRC.
    \note  CRC polynomial is "D24 + D23 + D18 + D17 + D14 + D11 + D10 + D7 + D6 + D5
           + D4 + D3 + D + 1", refer to 3GPP TS 38.212, section 5.1.
*/
void bblib_lte_crc24a_gen(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc24a_gen_avx512(struct bblib_crc_request *request, struct bblib_crc_response *response);
void bblib_lte_crc24a_gen_snc(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc24a_gen_sse(struct bblib_crc_request *request, struct bblib_crc_response *response);
//! @}

//! @{
/*!
    \brief Performs CRC24A validate, indicating if the input data sequence has a valid CRC value.
    \param [in] request structure containing pointer to input data and data length.
    \note  Length should be for the data part only, since the CRC algorithm determines the CRC length.
    \param [out] response structure with indication if CRC validation has passed.
    \return void
    \note  Memory for both request.data & response.data structures should be allocated.
     Response.data can point to request.data if sufficient space is available at end of
     request.data structure for the appended CRC.
    \note CRC polynomial is "D24 + D23 + D18 + D17 + D14 + D11 + D10 + D7 + D6 + D5 + D4 + D3
          + D + 1", refer to 3GPP TS 38.212, section 5.1.
*/
void bblib_lte_crc24a_check(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc24a_check_avx512(struct bblib_crc_request *request, struct bblib_crc_response *response);
void bblib_lte_crc24a_check_snc(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc24a_check_sse(struct bblib_crc_request *request, struct bblib_crc_response *response);
//! @}

//! @{
/*!
    \brief Performs CRC24B generate, calculating the CRC value and appending to the data.
    \param [in] request structure containing pointer to input data and data length.
    \param [out] response structure containing calculated CRC value, CRC appended data and new length.
    \return void.
    \note  Memory for both request.data & response.data structures should be allocated.
     Response.data can point to request.data if sufficient space is available at end of
     request.data structure for the appended CRC.
    \note CRC polynomial is "D24 + D23 + D6 + D5 + D + 1",  refer to 3GPP TS 38.212, section 5.1.

*/
void bblib_lte_crc24b_gen(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc24b_gen_avx512(struct bblib_crc_request *request, struct bblib_crc_response *response);
void bblib_lte_crc24b_gen_snc(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc24b_gen_sse(struct bblib_crc_request *request, struct bblib_crc_response *response);
//! @}

//! @{
/*!
    \brief Performs CRC24B validate, indicating if the input data sequence has a valid CRC value.
    \param [in] request structure containing pointer to input data and data length.
    \note  Length should be for the data part only, since the CRC algorithm determines the CRC length.
    \param [out] response structure with indication if CRC validation has passed.
    \return void
    \note  Memory for both request.data & response.data structures should be allocated.
     Response.data can point to request.data if sufficient space is available at end of
     request.data structure for the appended CRC.
    \note CRC polynomial is "D24 + D23 + D6 + D5 + D + 1",  refer to 3GPP TS 38.212, section 5.1.
*/
void bblib_lte_crc24b_check(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc24b_check_avx512(struct bblib_crc_request *request, struct bblib_crc_response *response);
void bblib_lte_crc24b_check_snc(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc24b_check_sse(struct bblib_crc_request *request, struct bblib_crc_response *response);
//! @}

//! @{
/*! \brief Performs CRC24C generate, calculating the CRC value and appending to the data.
    \param [in] request structure containing pointer to input data and data length.
    \param [out] response structure containing calculated CRC value, CRC appended data and new length.
    \return void.
    \note  Memory for both request.data & response.data structures should be allocated.
     Response.data can point to request.data if sufficient space is available at end of
     request.data structure for the appended CRC.
    \note  CRC polynomial is "D24 + D23 + D21 + D20 + D17 + D15 + D13 + D12 + D8 + D4
           + D2 + D + 1", refer to 3GPP TS 38.212, section 5.1.
*/
void bblib_lte_crc24c_gen(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc24c_gen_avx512(struct bblib_crc_request *request, struct bblib_crc_response *response);
void bblib_lte_crc24c_gen_snc(struct bblib_crc_request *request, struct bblib_crc_response *response);

//! @}

//! @{
/*!
    \brief Performs CRC24C validate, indicating if the input data sequence has a valid CRC value.
    \param [in] request structure containing pointer to input data and data length.
    \note  Length should be for the data part only, since the CRC algorithm determines the CRC length.
    \param [out] response structure with indication if CRC validation has passed.
    \return void
    \note  Memory for both request.data & response.data structures should be allocated.
     Response.data can point to request.data if sufficient space is available at end of
     request.data structure for the appended CRC.
    \note  CRC polynomial is "D24 + D23 + D21 + D20 + D17 + D15 + D13 + D12 + D8 + D4
           + D2 + D + 1", refer to 3GPP TS 38.212, section 5.1.
*/
void bblib_lte_crc24c_check(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc24c_check_avx512(struct bblib_crc_request *request, struct bblib_crc_response *response);
void bblib_lte_crc24c_check_snc(struct bblib_crc_request *request, struct bblib_crc_response *response);

//! @}

//! @{
/*! \brief Performs CRC24C initialised with 1s, generate, calculating the CRC value and appending to the data.
    \param [in] request structure containing pointer to input data and data length.
    \param [out] response structure containing calculated CRC value, CRC appended data and new length.
    \return void.
    \note  Memory for both request.data & response.data structures should be allocated.
     Response.data can point to request.data if sufficient space is available at end of
     request.data structure for the appended CRC.
    \note  CRC polynomial is "D24 + D23 + D21 + D20 + D17 + D15 + D13 + D12 + D8 + D4
           + D2 + D + 1", refer to 3GPP TS 38.212, section 5.1.
*/
void bblib_lte_crc24c_1_gen(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc24c_1_gen_avx512(struct bblib_crc_request *request, struct bblib_crc_response *response);

//! @}

//! @{
/*!
    \brief Performs CRC24C initialised with 1s, validate, indicating if the input data sequence has a valid CRC value.
    \param [in] request structure containing pointer to input data and data length.
    \note  Length should be for the data part only, since the CRC algorithm determines the CRC length.
    \param [out] response structure with indication if CRC validation has passed.
    \return void
    \note  Memory for both request.data & response.data structures should be allocated.
     Response.data can point to request.data if sufficient space is available at end of
     request.data structure for the appended CRC.
    \note  CRC polynomial is "D24 + D23 + D21 + D20 + D17 + D15 + D13 + D12 + D8 + D4
           + D2 + D + 1", refer to 3GPP TS 38.212, section 5.1.
*/
void bblib_lte_crc24c_1_check(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc24c_1_check_avx512(struct bblib_crc_request *request, struct bblib_crc_response *response);

//! @}

//! @{
/*!
    \brief Performs CRC16 generate, calculating the CRC value and appending to the data.
    \param [in] request structure containing pointer to input data and data length.
    \param [out] response structure containing calculated CRC value, CRC appended data and new length.
    \return void.
    \note  Memory for both request.data & response.data structures should be allocated.
     Response.data can point to request.data if sufficient space is available at end of
     request.data structure for the appended CRC.
    \note CRC polynomial is "D16 + D12 + D5 + 1",  refer to 3GPP TS 38.212, section 5.1.
*/
void bblib_lte_crc16_gen(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc16_gen_avx512(struct bblib_crc_request *request, struct bblib_crc_response *response);
void bblib_lte_crc16_gen_snc(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc16_gen_sse(struct bblib_crc_request *request, struct bblib_crc_response *response);
//! @}

//! @{
/*!
    \brief Performs CRC16 validate, indicating if the input data sequence has a valid CRC value.
    \param [in] request structure containing pointer to input data and data length.
    \note  Length should be for the data part only, since the CRC algorithm determines the CRC length.
    \param [out] response structure with indication if CRC validation has passed.
    \return void.
    \note  Memory for both request.data & response.data structures should be allocated.
     Response.data can point to request.data if sufficient space is available at end of
     request.data structure for the appended CRC.
    \note CRC polynomial is "D16 + D12 + D5 + 1",  refer to 3GPP TS 38.212, section 5.1.
*/
void bblib_lte_crc16_check(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc16_check_avx512(struct bblib_crc_request *request, struct bblib_crc_response *response);
void bblib_lte_crc16_check_snc(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc16_check_sse(struct bblib_crc_request *request, struct bblib_crc_response *response);
//! @}


//! @{
/*!
    \brief Performs CRC11 generate, calculating the CRC value and appending to the data.
    \param [in] request structure containing pointer to input data and data length.
    \param [out] response structure containing calculated CRC value, CRC appended data and new length.
    \return void
    \note  Memory for both request.data & response.data structures should be allocated.
     Response.data can point to request.data if sufficient space is available at end of
     request.data structure for the appended CRC.
    \note CRC polynomial is "D11 + D10 + D9 + D5 + 1",  refer to 3GPP TS 38.212, section 5.1.
*/
void bblib_lte_crc11_gen(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc11_gen_avx512(struct bblib_crc_request *request, struct bblib_crc_response *response);
void bblib_lte_crc11_gen_snc(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc11_gen_sse(struct bblib_crc_request *request, struct bblib_crc_response *response);
//! @}

//! @{
/*!
    \brief Performs CRC11 validate, indicating if the input data sequence has a valid CRC value.
    \param [in] request structure containing pointer to input data and data length.
    \note  Length should be for the data part only, since the CRC algorithm determines the CRC length.
    \param [out] response structure with indication if CRC validation has passed.
    \return void.
    \note  Memory for both request.data & response.data structures should be allocated.
     Response.data can point to request.data if sufficient space is available at end of
     request.data structure for the appended CRC.
    \note CRC polynomial is "D11 + D10 + D9 + D5 + 1",  refer to 3GPP TS 38.212, section 5.1.

*/
void bblib_lte_crc11_check(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc11_check_avx512(struct bblib_crc_request *request, struct bblib_crc_response *response);
void bblib_lte_crc11_check_snc(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc11_check_sse(struct bblib_crc_request *request, struct bblib_crc_response *response);
//! @}


//! @{
/*!
    \brief Performs CRC6 generate, calculating the CRC value and appending to the data.
    \param [in] request structure containing pointer to input data and data length.
    \param [out] response structure containing calculated CRC value, CRC appended data and new length.
    \return void
    \note  Memory for both request.data & response.data structures should be allocated.
     Response.data can point to request.data if sufficient space is available at end of
     request.data structure for the appended CRC.
    \note CRC polynomial is "D6 + D5 + 1",  refer to 3GPP TS 38.212.
*/
void bblib_lte_crc6_gen(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc6_gen_avx512(struct bblib_crc_request *request, struct bblib_crc_response *response);
void bblib_lte_crc6_gen_snc(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc6_gen_sse(struct bblib_crc_request *request, struct bblib_crc_response *response);
//! @}

//! @{
/*!
    \brief Performs CRC6 validate, indicating if the input data sequence has a valid CRC value.
    \param [in] request structure containing pointer to input data and data length.
    \note  Length should be for the data part only, since the CRC algorithm determines the CRC length.
    \param [out] response structure with indication if CRC validation has passed.
    \return void.
    \note  Memory for both request.data & response.data structures should be allocated.
     Response.data can point to request.data if sufficient space is available at end of
     request.data structure for the appended CRC.
    \note CRC polynomial is "D6 + D5 + 1",  refer to 3GPP TS 38.212.
*/
void bblib_lte_crc6_check(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc6_check_avx512(struct bblib_crc_request *request, struct bblib_crc_response *response);
void bblib_lte_crc6_check_snc(struct bblib_crc_request *request, struct bblib_crc_response *response);

void bblib_lte_crc6_check_sse(struct bblib_crc_request *request, struct bblib_crc_response *response);
//! @}



#ifdef __cplusplus
}
#endif

#endif
