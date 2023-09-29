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

#pragma once

#include "InternalApi.hpp"

extern const uint8_t k_bg1ColumnPositions[SimdLdpc::k_maxCirculants];
extern const uint8_t k_bg1RowWeights[SimdLdpc::k_maxRows];
extern const int16_t k_bg1RowWeightsCumulative[SimdLdpc::k_maxRows];
extern const int16_t k_bg1a2[SimdLdpc::k_maxCirculants];
extern const int16_t k_bg1a3[SimdLdpc::k_maxCirculants];
extern const int16_t k_bg1a5[SimdLdpc::k_maxCirculants];
extern const int16_t k_bg1a7[SimdLdpc::k_maxCirculants];
extern const int16_t k_bg1a9[SimdLdpc::k_maxCirculants];
extern const int16_t k_bg1a11[SimdLdpc::k_maxCirculants];
extern const int16_t k_bg1a13[SimdLdpc::k_maxCirculants];
extern const int16_t k_bg1a15[SimdLdpc::k_maxCirculants];
extern const uint8_t k_bg2ColumnPositions[SimdLdpc::k_bg2Circulants];
extern const uint8_t k_bg2RowWeights[42];
extern const int16_t k_bg2RowWeightsCumulative[42];
extern const int16_t k_bg2a2[SimdLdpc::k_bg2Circulants];
extern const int16_t k_bg2a3[SimdLdpc::k_bg2Circulants];
extern const int16_t k_bg2a5[SimdLdpc::k_bg2Circulants];
extern const int16_t k_bg2a7[SimdLdpc::k_bg2Circulants];
extern const int16_t k_bg2a9[SimdLdpc::k_bg2Circulants];
extern const int16_t k_bg2a11[SimdLdpc::k_bg2Circulants];
extern const int16_t k_bg2a13[SimdLdpc::k_bg2Circulants];
extern const int16_t k_bg2a15[SimdLdpc::k_bg2Circulants];
