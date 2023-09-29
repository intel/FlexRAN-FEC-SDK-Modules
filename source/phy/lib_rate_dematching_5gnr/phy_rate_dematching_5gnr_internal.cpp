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
 * @file phy_rate_dematching_5gnr_internal.cpp
 * @brief  Implementation for rate dematching functions
 */

#include "phy_rate_dematching_5gnr_internal.h"


/**
 * @brief This function implements k0 calculation
 * @param[in] the pointer of rate dematching struct
 *
 */
void get_k0(struct bblib_rate_dematching_5gnr_request *pRM)
{
    int32_t k0;
    int32_t Ncb = pRM->ncb;
    int32_t Zc = pRM->zc;
    int32_t flagOfBG = pRM->base_graph;

    /* calculating k0 according to spec. */
    switch (pRM->rvid) {
        case 1:
            if(flagOfBG == 1)
                k0 = Zc * ((17*Ncb)/(66*Zc));
            else
                k0 = Zc * ((13*Ncb)/(50*Zc));
            break;
        case 2:
            if(flagOfBG == 1)
                k0 = Zc * ((33*Ncb)/(66*Zc));
            else
                k0 = Zc * ((25*Ncb)/(50*Zc));
            break;
        case 3:
            if(flagOfBG == 1)
                k0 = Zc * ((56*Ncb)/(66*Zc));
            else
                k0 = Zc * ((43*Ncb)/(50*Zc));
            break;
        default:
            k0 = 0;
    }
    pRM->k0 = k0;
}
