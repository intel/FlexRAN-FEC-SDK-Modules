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

#ifndef _PHY_TURBO_INTERNAL_H_
#define _PHY_TURBO_INTERNAL_H_

#include <stdint.h>
#include "common_typedef_sdk.h"
#include "bblib_common_const.h"

#define MAX_DATA_LEN_INTERLEAVE (8192)

typedef struct {
    int8_t pattern[448][16];
    int32_t offset[188];
    int32_t inter_row_out_addr_for_interleaver[21693];
    int32_t inter_row_out_addr_for_deinterleaver[21693];
    int32_t intra_row_perm_pattern_for_interleaver[21693];
    int32_t intra_row_perm_pattern_for_deinterleaver[21693];
} _TurboInterleaver;

extern _TurboInterleaver g_TurboInterleaver;

/** @fn lte_turbo_interleaver_8windows_sse
 *  @brief  This function implements Turbo internal interleaver defined in TS36.212 section
       5.1.3.2.1. Used in transmitter.
 *  @param [in] caseId is i defined in TS36.212 Table 5.1.3-3
 *  @param [in] pInData buffer stores information bits and CRC bits. It should be 16-byte aligned so that SSE load is fesible.
 *  @param [in] pOutData buffer stores interleaved bit stream. It should be 16-byte aligned so that SSE load is fesible.
 *  @return int32_t, 1 is successful. otherwire is fail.
 */
int32_t
bblib_lte_turbo_interleaver_8windows_sse(uint8_t caseId, uint8_t *pInData, uint8_t* pOutData);

/** @fn lte_turbo_decoder_64windows_avx512
 *  @brief This function implements Turbo decoder when CW size is multiple of 64.
 *  @param[in] p is pointer of TurboDecoder_para
 *  @return 0 successful, -1 is fail.
*/
int32_t lte_turbo_decoder_64windows_avx512(void *p);

/** @fn lte_turbo_interleaver_initTable
 *  @brief  inintalize interleaver table function
 *  @param [in] pTabelPath the table path.
 *  @return void
 */
void
bblib_lte_turbo_interleaver_initTable(char* pTabelPath);

void
bblib_print_turbo_version();

void
init_turbo_decoder_interleaver_table(char *Table_Path, _TurboInterleaver *p);

void
get_turbo_buf_addr_table_new(uint16_t *p_table_TurboBufAddr,
                             int32_t *p_table_TurboBufAddr_Offset,
                             int32_t (*p_table_Kidx_K_Nmaxrep_shuf)[5]);

int32_t
init_common_tables(char *pTabelPath);

// for softbit mapping
#define _LOG_P1_P0

#endif
