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
 * @file phy_rate_dematching_5gnr_avx512.cpp
 * @brief  Implementation for rate dematching functions
 */

#include "phy_rate_dematching_5gnr_internal.h"
#include <string.h>


/* Bit interleaving as per 3GPP 38.212 5.4.2.2 */
void deInterleave(int8_t *pCbIn, int8_t *pDeInterleave, bblib_rate_dematching_5gnr_request *pRM) {
    int32_t byte, mod, intl_size = pRM->e / pRM->modulation_order;
    for (byte = 0; byte < intl_size; byte++)
        for (mod =0; mod < pRM->modulation_order; mod++)
            pDeInterleave[byte + mod * intl_size] =
                    pCbIn[pRM->modulation_order * byte + mod];
}

/* Adds and saturate to MAX_LLR the 2 LLR streams */
void combine(int8_t *pHarq, int8_t *p_in, int16_t length) {
    int16_t byte, temp;
    for (byte = 0; byte < length; byte++) {
        temp = ((int16_t) pHarq[byte]) + ((int16_t) p_in[byte]);
        temp = MIN(MAX_LLR, MAX(temp, MIN_LLR));
        pHarq[byte] = (int8_t) temp;
    }
}

void harq_combine(struct bblib_rate_dematching_5gnr_request *request,
    struct bblib_rate_dematching_5gnr_response *response, int8_t *pDeInterleave)
{
    if (request->isretx == 0)
        memset(request->p_harq,0x00,request->ncb);
    get_k0(request);
    int32_t ncb_, length, offset_e=0, offset_ncb=request->k0;
    ncb_ = request->ncb - request->num_of_null;
    if (offset_ncb > request->start_null_index)
        offset_ncb -= request->num_of_null;

    while (offset_e < request->e) {
        length = MIN(request->e - offset_e, ncb_ - offset_ncb);
        combine((int8_t *) request->p_harq + offset_ncb, (int8_t *) pDeInterleave + offset_e, length);
        offset_ncb = length + offset_ncb;
        if (offset_ncb == ncb_)
            offset_ncb = 0;
        offset_e += length;
    }
}

/**
 * @brief Implements rate dematching with AVX512
 * @param [in] request Structure containing the configuration, input data
 * @param [out] response Structure containing the output data.
**/
void bblib_rate_dematching_5gnr_c(struct bblib_rate_dematching_5gnr_request *req,
struct bblib_rate_dematching_5gnr_response *resp)
{
    __align(64) int8_t internalBuffer[128 * 1024];
    deInterleave(req->p_in, internalBuffer, req);
    harq_combine(req,resp,internalBuffer);
}
