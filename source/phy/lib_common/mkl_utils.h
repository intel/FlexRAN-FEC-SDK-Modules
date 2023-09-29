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
******************************************************************************/
/**
 * @brief This file consists of utilities using MKL (Math Kernel Libraries)
 * @file lte_bs_mkl_utils.h
 * @ingroup group_lte_source_phy_utils
 * @author Intel Corporation]
 *
 **/


#ifndef _MKL_UTILS_H_
#define _MKL_UTILS_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <mkl.h>
#include <mkl_dfti.h>

#define MKL_UTILS_NUM_IDFT_SIZES            ( 34 )
#define MKL_UTILS_NUM_FFT_SIZES             ( 2 )
#define MKL_UTILS_NUM_IDX_1024              ( 0 )
#define MKL_UTILS_NUM_IDX_2048              ( 1 )

#define MKL_FFT                             ( 0 )
#define MKL_IFFT                            ( 1 )
#define MKL_IDFT                            ( 2 )
#define MKL_PRACH                           ( 3 )

#define MKL_FORWARD                         ( 0 )
#define MKL_BACKWARD                        ( 1 )

__int32 mkl_utils_init_fft_ifft(__int32 Nfft);
__int32 mkl_utils_destroy_fft_ifft(void);
MKL_LONG mkl_utils_run_fft_ifft(__int32 Nfft, __int32 type, __int32 direction, void *pIn, void *pOut);

__int32 mkl_utils_init_idft(void);
__int32 mkl_utils_destroy_idft(void);
MKL_LONG mkl_utils_run_idft(__int32 idx, __int32 type, __int32 direction, void *pIn, void *pOut);

#endif /* #ifndef _LTE_BS_MKL_UTILS_H_ */

