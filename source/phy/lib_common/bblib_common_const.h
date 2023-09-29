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

/**
 * @file bblib_common_const.h
 * @brief This header file defines common global constants uses throughout the
 * bblib libraries.
 */

#ifndef _BBLIB_COMMON_CONST_
#define _BBLIB_COMMON_CONST_

#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

#ifndef RUP512B
#define RUP512B(x) (((x)+511)&(~511))
#endif
#ifndef RUP256B
#define RUP256B(x) (((x)+255)&(~255))
#endif
#ifndef RUP128B
#define RUP128B(x) (((x)+127)&(~127))
#endif
#ifndef RUP64B
#define RUP64B(x) (((x)+63)&(~63))
#endif
#ifndef RUP32B
#define RUP32B(x) (((x)+31)&(~31))
#endif
#ifndef RUP16B
#define RUP16B(x) (((x)+15)&(~15))
#endif
#ifndef RUP8B
#define RUP8B(x)  (((x)+7)&(~7))
#endif
#ifndef RUP4B
#define RUP4B(x)  (((x)+3)&(~3))
#endif
#ifndef RUP2B
#define RUP2B(x)  (((x)+1)&(~1))
#endif

#ifndef PI
#define PI ((float) 3.14159265358979323846)
#endif

#ifndef PI_double
#define PI_double ((double) 3.14159265358979323846)
#endif

#ifndef MAX_MU_NUM
#define MAX_MU_NUM (5)
#endif

#ifndef N_FFT_SIZE_MU0_5MHZ
#define N_FFT_SIZE_MU0_5MHZ (512)
#endif

#ifndef N_FFT_SIZE_MU0_10MHZ
#define N_FFT_SIZE_MU0_10MHZ (1024)
#endif

#ifndef N_FFT_SIZE_MU0_15MHZ
#define N_FFT_SIZE_MU0_15MHZ (1536)
#endif

#ifndef N_FFT_SIZE_MU0_20MHZ
#define N_FFT_SIZE_MU0_20MHZ (2048)
#endif

#ifndef N_FFT_SIZE_MU0_40MHZ
#define N_FFT_SIZE_MU0_40MHZ (4096)
#endif

#ifndef N_FFT_SIZE_MU1_10MHZ
#define N_FFT_SIZE_MU1_10MHZ (512)
#endif

#ifndef N_FFT_SIZE_MU1_20MHZ
#define N_FFT_SIZE_MU1_20MHZ (1024)
#endif

#ifndef N_FFT_SIZE_MU1_40MHZ
#define N_FFT_SIZE_MU1_40MHZ (2048)
#endif

#ifndef N_FFT_SIZE_MU1_50MHZ
#define N_FFT_SIZE_MU1_50MHZ (2048)
#endif

#ifndef N_FFT_SIZE_MU1_60MHZ
#define N_FFT_SIZE_MU1_60MHZ (3072)
#endif

#ifndef N_FFT_SIZE_MU1_100MHZ
#define N_FFT_SIZE_MU1_100MHZ (4096)
#endif

#ifndef N_FFT_SIZE_MU3
#define N_FFT_SIZE_MU3 (1024)
#endif

#ifndef N_MAX_CP_MU0_5MHZ
#define N_MAX_CP_MU0_5MHZ (40)
#endif

#ifndef N_MAX_CP_MU0_10MHZ
#define N_MAX_CP_MU0_10MHZ (80)
#endif

#ifndef N_MAX_CP_MU0_15MHZ
#define N_MAX_CP_MU0_15MHZ (120)
#endif

#ifndef N_MAX_CP_MU0_20MHZ
#define N_MAX_CP_MU0_20MHZ (160)
#endif

#ifndef N_MAX_CP_MU0_40MHZ
#define N_MAX_CP_MU0_40MHZ (320)
#endif

#ifndef N_MAX_CP_MU1_10MHZ
#define N_MAX_CP_MU1_10MHZ (44)
#endif

#ifndef N_MAX_CP_MU1_20MHZ
#define N_MAX_CP_MU1_20MHZ (88)
#endif

#ifndef N_MAX_CP_MU1_40MHZ
#define N_MAX_CP_MU1_40MHZ (176)
#endif

#ifndef N_MAX_CP_MU1_50MHZ
#define N_MAX_CP_MU1_50MHZ (176)
#endif

#ifndef N_MAX_CP_MU1_60MHZ
#define N_MAX_CP_MU1_60MHZ (264)
#endif

#ifndef N_MAX_CP_MU1_100MHZ
#define N_MAX_CP_MU1_100MHZ (352)
#endif

#ifndef N_MAX_CP_MU3
#define N_MAX_CP_MU3 (136)
#endif

#ifndef N_MIN_CP_MU0_5MHZ
#define N_MIN_CP_MU0_5MHZ (36)
#endif

#ifndef N_MIN_CP_MU0_10MHZ
#define N_MIN_CP_MU0_10MHZ (72)
#endif

#ifndef N_MIN_CP_MU0_15MHZ
#define N_MIN_CP_MU0_15MHZ (108)
#endif

#ifndef N_MIN_CP_MU0_20MHZ
#define N_MIN_CP_MU0_20MHZ (144)
#endif

#ifndef N_MIN_CP_MU0_40MHZ
#define N_MIN_CP_MU0_40MHZ (288)
#endif

#ifndef N_MIN_CP_MU1_10MHZ
#define N_MIN_CP_MU1_10MHZ (30)
#endif

#ifndef N_MIN_CP_MU1_20MHZ
#define N_MIN_CP_MU1_20MHZ (72)
#endif

#ifndef N_MIN_CP_MU1_40MHZ
#define N_MIN_CP_MU1_40MHZ (144)
#endif

#ifndef N_MIN_CP_MU1_50MHZ
#define N_MIN_CP_MU1_50MHZ (144)
#endif

#ifndef N_MIN_CP_MU1_60MHZ
#define N_MIN_CP_MU1_60MHZ (216)
#endif

#ifndef N_MIN_CP_MU1_100MHZ
#define N_MIN_CP_MU1_100MHZ (288)
#endif

#ifndef N_MIN_CP_MU3
#define N_MIN_CP_MU3 (72)
#endif

#ifndef N_FULLBAND_SC_MU0_5MHZ
#define N_FULLBAND_SC_MU0_5MHZ (300)
#endif

#ifndef N_FULLBAND_SC_MU0_10MHZ
#define N_FULLBAND_SC_MU0_10MHZ (624)
#endif

#ifndef N_FULLBAND_SC_MU0_20MHZ
#define N_FULLBAND_SC_MU0_20MHZ (1272)
#endif

#ifndef N_FULLBAND_SC_MU0_40MHZ
#define N_FULLBAND_SC_MU0_40MHZ (2592)
#endif

#ifndef N_FULLBAND_SC_MU1_10MHZ
#define N_FULLBAND_SC_MU1_10MHZ (288)
#endif

#ifndef N_FULLBAND_SC_MU1_20MHZ
#define N_FULLBAND_SC_MU1_20MHZ (612)
#endif

#ifndef N_FULLBAND_SC_MU1_40MHZ
#define N_FULLBAND_SC_MU1_40MHZ (1272)
#endif

#ifndef N_FULLBAND_SC_MU1_50MHZ
#define N_FULLBAND_SC_MU1_50MHZ (1596)
#endif

#ifndef N_FULLBAND_SC_MU1_60MHZ
#define N_FULLBAND_SC_MU1_60MHZ (1944)
#endif

#ifndef N_FULLBAND_SC_MU1_100MHZ
#define N_FULLBAND_SC_MU1_100MHZ (3276)
#endif

#ifndef N_FULLBAND_SC_MU3
#define N_FULLBAND_SC_MU3 (792)
#endif

#ifndef N_SAMPLE_RATE_MU0_40MHZ
#define N_SAMPLE_RATE_MU0_40MHZ (61.44*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU0_20MHZ
#define N_SAMPLE_RATE_MU0_20MHZ (30.72*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU0_15MHZ
#define N_SAMPLE_RATE_MU0_15MHZ (23.04*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU0_10MHZ
#define N_SAMPLE_RATE_MU0_10MHZ (15.36*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU0_5MHZ
#define N_SAMPLE_RATE_MU0_5MHZ (7.68*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU1_10MHZ
#define N_SAMPLE_RATE_MU1_10MHZ (15.36*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU1_20MHZ
#define N_SAMPLE_RATE_MU1_20MHZ (30.72*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU1_40MHZ
#define N_SAMPLE_RATE_MU1_40MHZ (61.44*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU1_50MHZ
#define N_SAMPLE_RATE_MU1_50MHZ (61.44*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU1_60MHZ
#define N_SAMPLE_RATE_MU1_60MHZ (91.16*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU1_100MHZ
#define N_SAMPLE_RATE_MU1_100MHZ (122.88*1000*1000)
#endif

#ifndef N_SAMPLE_RATE_MU3
#define N_SAMPLE_RATE_MU3 (122.88*1000*1000)
#endif

#ifndef N_DMRS_TYPE1_SC_PER_RB
#define N_DMRS_TYPE1_SC_PER_RB (6)
#endif

#ifndef N_DMRS_TYPE2_SC_PER_RB
#define N_DMRS_TYPE2_SC_PER_RB (4)
#endif

#ifndef N_DMRS_TYPE1_DELTA
#define N_DMRS_TYPE1_DELTA (2)
#endif

#ifndef N_DMRS_TYPE2_DELTA
#define N_DMRS_TYPE2_DELTA (3)
#endif

#ifndef MAX_NUM_OF_DELTA
#define MAX_NUM_OF_DELTA (3)
#endif

#ifndef DMRS_TYPE1_MAX_PORT_NUM
#define DMRS_TYPE1_MAX_PORT_NUM (8)
#endif

#ifndef DMRS_TYPE2_MAX_PORT_NUM
#define DMRS_TYPE2_MAX_PORT_NUM (12)
#endif

#ifndef DMRS_TYPE1_SINGLE_DMRS_MAX_PORT_NUM
#define DMRS_TYPE1_SINGLE_DMRS_MAX_PORT_NUM (4)
#endif

#ifndef DMRS_TYPE2_SINGLE_DMRS_MAX_PORT_NUM
#define DMRS_TYPE2_SINGLE_DMRS_MAX_PORT_NUM (6)
#endif

#ifndef DMRS_MAX_PORT_NUM_PER_CDM
#define DMRS_MAX_PORT_NUM_PER_CDM (4)
#endif

#endif /* _BBLIB_COMMON_CONST_ */



