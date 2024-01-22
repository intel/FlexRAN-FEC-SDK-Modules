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
 * @file
 * @brief  Implementation LTE rate matching in TS 136.212 table 5.1.3-3, with SSE instructions
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <smmintrin.h> /* SSE 4 for media */

#include "bblib_common_const.h"
#include "phy_rate_match.h"
#include "phy_rate_match_internal.h"
#if defined(_BBLIB_SSE4_2_) || defined(_BBLIB_AVX2_) || defined(_BBLIB_AVX512_)
int32_t g_nNum_NULL[188];
int32_t g_nIndex_NULL[188][84];
/* rate matching table */
int32_t g_ratetable[188][18444];

/* Rate matching bit to byte table */
__align(64) uint8_t g_BitToByteTABLE[8192];


struct bblib_rate_match_init_sse
{
    bblib_rate_match_init_sse() {
        static bool was_executed = false;
#if !defined(_BBLIB_SSE_4_2) && !defined(_BBLIB_AVX2_) && !defined(_BBLIB_AVX512_)
        printf("__func__ rate_match cannot run with this CPU type, needs SSE or greater.\n");
        exit(-1);
#endif
        bblib_print_rate_match_version();
        if (!was_executed) {
            was_executed = true;
            (void) init_rate_matching_lte_sse();
        }
    }
};

bblib_rate_match_init_sse do_constructor_rate_matching_sse;

/**
 * @brief Bit to byte table generation
 * @param[out] pTable (Lookup table for bit to byte conversion)
 * @return void
 */
void MakeBitToByteTable(uint8_t* pTable)
{
    __m128i seq_byte, vconst;

    seq_byte = _mm_setzero_si128();
    vconst = _mm_set1_epi8(0x80);

    for(int32_t Idx = 0; Idx < 256; Idx++)
    {
        seq_byte = _mm_insert_epi8(seq_byte, (Idx << 7), 7);
        seq_byte = _mm_insert_epi8(seq_byte, (Idx << 6), 6);
        seq_byte = _mm_insert_epi8(seq_byte, (Idx << 5), 5);
        seq_byte = _mm_insert_epi8(seq_byte, (Idx << 4), 4);
        seq_byte = _mm_insert_epi8(seq_byte, (Idx << 3), 3);
        seq_byte = _mm_insert_epi8(seq_byte, (Idx << 2), 2);
        seq_byte = _mm_insert_epi8(seq_byte, (Idx << 1), 1);
        seq_byte = _mm_insert_epi8(seq_byte, Idx, 0);

        seq_byte = _mm_insert_epi8(seq_byte, (Idx << 7), 15);
        seq_byte = _mm_insert_epi8(seq_byte, (Idx << 6), 14);
        seq_byte = _mm_insert_epi8(seq_byte, (Idx << 5), 13);
        seq_byte = _mm_insert_epi8(seq_byte, (Idx << 4), 12);
        seq_byte = _mm_insert_epi8(seq_byte, (Idx << 3), 11);
        seq_byte = _mm_insert_epi8(seq_byte, (Idx << 2), 10);
        seq_byte = _mm_insert_epi8(seq_byte, (Idx << 1), 9);
        seq_byte = _mm_insert_epi8(seq_byte, Idx, 8);

        seq_byte = _mm_and_si128(seq_byte, vconst);
        _mm_store_si128 ((__m128i *)pTable, seq_byte);
        pTable+=16;

    }
}

/**
 * @brief Initialize LTE rate matching with SSE instructions, read some files into global tables.
 * @return 0: init success, -1: init error.
 */
int32_t init_rate_matching_lte_sse()
{
    /* Buffer for whole file name, including root folder of SDK */
    char file_name[1024] = {0};
    /* File name, under SDK folder */
    char file_nNum_NULL[100] = "/source/phy/lib_rate_matching/phy_rate_match_nnum_null.bin";
    char file_nIndex_NULL[100] = "/source/phy/lib_rate_matching/phy_rate_match_nindex_null.bin";
    char file_ratetable[100] = "/source/phy/lib_rate_matching/phy_rate_match_ratetable.bin";
    char *pCheckPath1 = NULL, *pCheckPath2 = NULL, *pCheckPath3 = NULL;

    uint8_t *pBitToByte = &g_BitToByteTABLE[0];
    FILE* stream;
    uint32_t len_read, path_len, str_offset;

    /* Get SDK root folder from environment variable */
    char* dir_sdk = getenv( "DIR_WIRELESS_SDK" );
    if( dir_sdk == NULL )
    {
        printf("Need to setup environment variable 'DIR_WIRELESS_SDK' for rate matching, pls set it up to folder where SDK is stored, with 'export DIR_WIRELESS_SDK=......'\n");
        exit(-1);
    }

    path_len = strlen(dir_sdk);

    //Remove to overcome https://github.com/intel/FlexRAN-FEC-SDK-Modules/issues/4
    //check to see if string has necessary characters
    //#ifdef WIN32
    //    pCheckPath1 = strstr(dir_sdk, "sdk");
    //
    //    if ((pCheckPath1 == NULL) && (pCheckPath2 == NULL))
    //    {
    //        printf("%s is not valid!!!  DIR_WIRELESS_SDK must include\n", dir_sdk);
    //        printf("sdk somewhere in path\n");
    //        exit(1);
    //    }
    //#else
    //    pCheckPath1 = strstr(dir_sdk, "sdk/build-avx");
    //    pCheckPath2 = strstr(dir_sdk, "sdk/build-snc");
    //    pCheckPath3 = strstr(dir_sdk, "sdk/build-spr");
    //
    //    if ((pCheckPath1 == NULL) && (pCheckPath2 == NULL) && (pCheckPath3 == NULL))
    //    {
    //        printf("%s is not valid!!!  DIR_WIRELESS_SDK must include\n", dir_sdk);
    //        printf("sdk/build-avx or sdk/build-snc or sdk/build-spr somewhere in path\n");
    //        exit(-1);
    //    }
    //#endif

    //check to see if string has characters trying to access
    //files in other directories
#ifdef WIN32
    pCheckPath1 = strstr(dir_sdk, "../");
#else
    pCheckPath1 = strstr(dir_sdk, "..\\");
#endif
    if (pCheckPath1 != NULL)
    {
        printf("Trying to access illegal path, %s\n", dir_sdk);
        printf("cannot have ../ in path!\n");
        exit(-1);
    }

    //check to see if string has NULL characters in the middle of the
    //string trying to widen access to other files in current directory
    //NULL character should only be at the end of the string
    pCheckPath1 = strchr(dir_sdk, 0);
    str_offset = (uint64_t)(pCheckPath1) - (uint64_t)(dir_sdk);
    //if NULL not at end, return fail
    if (str_offset - path_len)
    {
        printf("NULL character found in path, %s\n", dir_sdk);
        printf("cannot have NULL characters in the middle!\n");
        exit(-1);
    }

    /* Read from file "nnum_null.bin", into global buffer g_nNum_NULL[188] */
    if (path_len > (sizeof(file_name) - sizeof(file_nNum_NULL) - 32))
    {
        printf("DIR_WIRELESS_SDK path is too long!!   %s\n", dir_sdk);
        exit(-1);
    }

    memset(file_name, 0, sizeof(file_name));
    strncpy( file_name, dir_sdk, (sizeof(file_name) - 32));
    file_name[path_len] = '\0';
    strncat( file_name, file_nNum_NULL, (sizeof(file_nNum_NULL) + 1));

    if((stream=fopen(file_name,"rb"))!=NULL)
    {
        len_read = fread( g_nNum_NULL, sizeof(int32_t), sizeof(g_nNum_NULL)/4, stream );
        fclose(stream);
    }
    else
    {
        len_read = 0;
        printf( "File %s could not be opened in rate matching, size  %d\n", file_name, len_read);
        exit(-1);
    }

    /* Read from file "nindex_null.bin", into global buffer g_nIndex_NULL[188][84] */
    memset(file_name, 0, sizeof(file_name));
    strncpy( file_name, dir_sdk, (sizeof(file_name) - 32));
    file_name[path_len] = '\0';
    strncat( file_name, file_nIndex_NULL, (sizeof(file_nIndex_NULL) + 1));

    if((stream=fopen( file_name,"rb"))!=NULL)
    {
        len_read = fread( g_nIndex_NULL, sizeof(int32_t), sizeof(g_nIndex_NULL)/4, stream );
        fclose(stream);
    }
    else
    {
        len_read = 0;
        printf("File %s could not be opened in rate matching, size %d\n", file_name, len_read);
        exit(-1);
    }

    /* Read from file "ratetable.bin", into global buffer g_ratetable[188][18444] */
    memset(file_name, 0, sizeof(file_name));
    strncpy( file_name, dir_sdk, (sizeof(file_name) - 32));
    file_name[path_len] = '\0';
    strncat( file_name, file_ratetable, (sizeof(file_ratetable) + 1));

    if((stream=fopen( file_name,"rb"))!=NULL)
    {
        len_read = fread( g_ratetable, sizeof(int32_t), sizeof(g_ratetable)/4, stream );
        fclose(stream);
    }
    else
    {
        len_read = 0;
        printf("File %s could not be opened in rate matching, size  %d\n", file_name, len_read);
        exit(-1);
    }

    /* init bit to byte conversion table */
    MakeBitToByteTable( pBitToByte );

    return 0;
}

/**
 * @brief Downlink rate matching for LTE
 * @param[in] request structure containing configuration information and input data
 * @param[out] response structure containing kernel outputs
 * @return success: return 0, else: return -1
 */
int32_t bblib_rate_match_dl_sse(const struct bblib_rate_match_dl_request *request,
        struct bblib_rate_match_dl_response *response)
{
    int32_t E_table[MAX_CODE_BLOCK_IN_ONE_TB];

    uint32_t temp_outputlen = 0;
    /* short code block case */
    if( request->Kidx<91 )
    {
        temp_outputlen = rate_matching_turbo_lte_short_sse(request->r, request->C, request->direction,
                request->Nsoft, request->KMIMO, request->MDL_HARQ, request->G, request->NL,
                request->Qm, request->rvidx, request->bypass_rvidx, request->Kidx, request->nLen, request->tin0,
                request->tin1, request->tin2, response->output, response->OutputLen, &g_BitToByteTABLE[0] );
    }
    /* long code block case */
    else
    {
        temp_outputlen = lte_rate_matching_lte_k6144_sse(request->r, request->C, request->direction,
                request->Nsoft, request->KMIMO, request->MDL_HARQ, request->G, request->NL,
                request->Qm, request->rvidx, request->bypass_rvidx, request->Kidx, request->nLen, request->tin0,
                request->tin1, request->tin2, response->output, response->OutputLen, &E_table[0] );
    }
    return (temp_outputlen);
}
#else
int32_t bblib_rate_match_dl_sse(const struct bblib_rate_match_dl_request *request,
        struct bblib_rate_match_dl_response *response)
{
    printf("bblib_rate_matching requires at least SSE4.2 ISA support to run\n");
    return(-1);
}
#endif
