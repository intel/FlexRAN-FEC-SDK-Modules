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
/*******************************************************************************
* @file test_inversion.cpp
* @brief test function for matrix inversion.
*******************************************************************************/

/*******************************************************************************
* Include private header files
*******************************************************************************/

#include <string.h>

#include "phy_matrix_inv_cholesky.h"
#include "common.hpp"

#ifndef WIN32
#include <pthread.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sched.h>
#include <utmpx.h>
#include <time.h>
#include <sys/time.h>
#include <signal.h>
#include <semaphore.h>
#include <sys/mman.h>
#endif


#define iAssert(p) if(!(p)){fprintf(stderr,\
    "Assertion failed: %s, file %s, line %d\n",\
#p, __FILE__, __LINE__);exit(-1);}

#ifdef _BBLIB_AVX512_
TEST(MatrixInv2x2Check, Avx512)
{
    __m512 matARe[NUM_MEM][2][2], matAIm[NUM_MEM][2][2];
    __m512 invARe[NUM_MEM][2][2], invAIm[NUM_MEM][2][2];
    int32_t nDim = 2;
    int32_t i1, i2, i3, i4;
    int32_t numOperation = 16;
    int32_t nLoadLoop = 1;//NUM_MEM;

    FILE *fpARe, *fpAIm;
    char fileARe[1024], fileAIm[1024];

    snprintf(fileARe, 1024, "./test_vectors/Hermit_Input_Real_%dx%dx16.bin", nDim, nDim);
    fpARe = fopen(fileARe, "rb");
    iAssert(fpARe != NULL);

    snprintf(fileAIm, 1024, "./test_vectors/Hermit_Input_Imag_%dx%dx16.bin", nDim, nDim);
    fpAIm = fopen(fileAIm, "rb");
    iAssert(fpAIm != NULL);

    int32_t nLoadNum = numOperation;

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpAIm);
            }
        }
    }
    fclose(fpARe);
    fclose(fpAIm);

    // start testing
    int32_t run_time = 10000;
    int32_t loop_time = 1;

    __int64 beg, end, timeDuration;
    __int64 interval;
    interval = 0;
    timeDuration = 0;

    for (int32_t k = 0; k < run_time; k++)
    {
        for (int32_t iLoop = 0; iLoop < loop_time; iLoop++)
        {
            beg = __rdtsc();

            matrix_inv_cholesky_2x2(matARe[iLoop], matAIm[iLoop], invARe[iLoop], invAIm[iLoop]);

            end = __rdtsc();
            interval = end - beg;

            if (k > 0)
            {
                timeDuration += interval;
            }
        }
    }

    FILE *fpinvARe, *fpinvAIm;
    char fileinvARe[1024], fileinvAIm[1024]; 

    snprintf(fileinvARe, 1024, "./test_vectors/Hermit_Output_Real_%dx%dx16.bin", nDim, nDim);
    fpinvARe = fopen(fileinvARe, "rb");
    iAssert(fpinvARe != NULL);

    snprintf(fileinvAIm, 1024, "./test_vectors/Hermit_Output_Imag_%dx%dx16.bin", nDim, nDim);
    fpinvAIm = fopen(fileinvAIm, "rb");
    iAssert(fpinvAIm != NULL);

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpinvARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpinvAIm);
            }
        }
    }

    fclose(fpinvARe);
    fclose(fpinvAIm);

    float nGoldRe, nGoldIm, nEstRe, nEstIm;
    float nErrorRe, nErrorIm, MSE;
    double avgMSEdB = 0.0;
    double totaldB = 0.0;
    for (i1 = 0; i1 < loop_time; i1++)
    {
        for (i4 = 0; i4 < nLoadNum; i4++)
        {
            avgMSEdB = 0;
            for (i2 = 0; i2 < nDim; i2++)
            {
                for (i3 = 0; i3 < nDim; i3++)
                {
                    nGoldRe = *((float*)&matARe[i1][i2][i3] + i4);
                    nGoldIm = *((float*)&matAIm[i1][i2][i3] + i4);
                    nEstRe = *((float*)&invARe[i1][i2][i3] + i4);
                    nEstIm = *((float*)&invAIm[i1][i2][i3] + i4);
                    nErrorRe = (float)nEstRe - (float)nGoldRe;
                    nErrorIm = (float)nEstIm - (float)nGoldIm;
                    MSE = (nErrorRe*nErrorRe + nErrorIm*nErrorIm)/((float)nGoldRe*nGoldRe + (float)nGoldIm*nGoldIm);
                    avgMSEdB += (double)MSE;
                }
            }

            avgMSEdB /= (nDim*nDim);
            totaldB += avgMSEdB;
            if (i1 < 1)
                printf("i4 = %d, MSEdB = %f \n", i4, (float)10 * log10((float)avgMSEdB));
        }
    }
    totaldB /= (loop_time*nLoadNum);

    ASSERT_GT(-45, (float)10 * log10((float)totaldB));

    printf("%dx%d Matrix inversion: The average MSEdB = %lf dB\n\n", nDim, nDim, (float)10 * log10((float)totaldB));
    printf("\n%dx%d matrix inversion Average Time: %8.1f cycles\n\n",
        nDim, nDim, timeDuration / loop_time / (float)(run_time-1)/numOperation);
}

TEST(MatrixInv3x3Check, Avx512)
{
    __m512 matARe[NUM_MEM][3][3], matAIm[NUM_MEM][3][3];
    __m512 invARe[NUM_MEM][3][3], invAIm[NUM_MEM][3][3];
    int32_t nDim = 3;
    int32_t i1, i2, i3, i4;
    int32_t numOperation = 16;
    int32_t nLoadLoop = 1;//NUM_MEM;

    FILE *fpARe, *fpAIm;
    char fileARe[1024], fileAIm[1024];

    snprintf(fileARe, 1024, "./test_vectors/Hermit_Input_Real_%dx%dx16.bin", nDim, nDim);
    fpARe = fopen(fileARe, "rb");
    iAssert(fpARe != NULL);

    snprintf(fileAIm, 1024, "./test_vectors/Hermit_Input_Imag_%dx%dx16.bin", nDim, nDim);
    fpAIm = fopen(fileAIm, "rb");
    iAssert(fpAIm != NULL);

    int32_t nLoadNum = numOperation;

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpAIm);
            }
        }
    }
    fclose(fpARe);
    fclose(fpAIm);

    // start testing
    int32_t run_time = 10000;
    int32_t loop_time = 1;

    __int64 beg, end, timeDuration;
    __int64 interval;
    interval = 0;
    timeDuration = 0;

    for (int32_t k = 0; k < run_time; k++)
    {
        for (int32_t iLoop = 0; iLoop < loop_time; iLoop++)
        {
            beg = __rdtsc();

            matrix_inv_cholesky_3x3(matARe[iLoop], matAIm[iLoop], invARe[iLoop], invAIm[iLoop]);

            end = __rdtsc();
            interval = end - beg;

            if (k > 0)
            {
                timeDuration += interval;
            }
        }
    }

    FILE *fpinvARe, *fpinvAIm;
    char fileinvARe[1024], fileinvAIm[1024]; 

    snprintf(fileinvARe, 1024, "./test_vectors/Hermit_Output_Real_%dx%dx16.bin", nDim, nDim);
    fpinvARe = fopen(fileinvARe, "rb");
    iAssert(fpinvARe != NULL);

    snprintf(fileinvAIm, 1024, "./test_vectors/Hermit_Output_Imag_%dx%dx16.bin", nDim, nDim);
    fpinvAIm = fopen(fileinvAIm, "rb");
    iAssert(fpinvAIm != NULL);

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpinvARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpinvAIm);
            }
        }
    }

    fclose(fpinvARe);
    fclose(fpinvAIm);

    float nGoldRe, nGoldIm, nEstRe, nEstIm;
    float nErrorRe, nErrorIm, MSE;
    double avgMSEdB = 0.0;
    double totaldB = 0.0;
    for (i1 = 0; i1 < loop_time; i1++)
    {
        for (i4 = 0; i4 < nLoadNum; i4++)
        {
            avgMSEdB = 0;
            for (i2 = 0; i2 < nDim; i2++)
            {
                for (i3 = 0; i3 < nDim; i3++)
                {
                    nGoldRe = *((float*)&matARe[i1][i2][i3] + i4);
                    nGoldIm = *((float*)&matAIm[i1][i2][i3] + i4);
                    nEstRe = *((float*)&invARe[i1][i2][i3] + i4);
                    nEstIm = *((float*)&invAIm[i1][i2][i3] + i4);
                    nErrorRe = (float)nEstRe - (float)nGoldRe;
                    nErrorIm = (float)nEstIm - (float)nGoldIm;
                    MSE = (nErrorRe*nErrorRe + nErrorIm*nErrorIm)/((float)nGoldRe*nGoldRe + (float)nGoldIm*nGoldIm);
                    avgMSEdB += (double)MSE;
                }
            }

            avgMSEdB /= (nDim*nDim);
            totaldB += avgMSEdB;
            if (i1 < 1)
                printf("i4 = %d, MSEdB = %f \n", i4, (float)10 * log10((float)avgMSEdB));
        }
    }
    totaldB /= (loop_time*nLoadNum);

    ASSERT_GT(-45, (float)10 * log10((float)totaldB));

    printf("%dx%d Matrix inversion: The average MSEdB = %lf dB\n\n", nDim, nDim, (float)10 * log10((float)totaldB));
    printf("\n%dx%d matrix inversion Average Time: %8.1f cycles\n\n",
        nDim, nDim, timeDuration / loop_time / (float)(run_time-1)/numOperation);
}

TEST(MatrixInv4x4Check, Avx512)
{
    __m512 matARe[NUM_MEM][4][4], matAIm[NUM_MEM][4][4];
    __m512 invARe[NUM_MEM][4][4], invAIm[NUM_MEM][4][4];
    int32_t nDim = 4;
    int32_t i1, i2, i3, i4;
    int32_t numOperation = 16;
    int32_t nLoadLoop = 1;//NUM_MEM;

    FILE *fpARe, *fpAIm;
    char fileARe[1024], fileAIm[1024];

    snprintf(fileARe, 1024, "./test_vectors/Hermit_Input_Real_%dx%dx16.bin", nDim, nDim);
    fpARe = fopen(fileARe, "rb");
    iAssert(fpARe != NULL);

    snprintf(fileAIm, 1024, "./test_vectors/Hermit_Input_Imag_%dx%dx16.bin", nDim, nDim);
    fpAIm = fopen(fileAIm, "rb");
    iAssert(fpAIm != NULL);

    int32_t nLoadNum = numOperation;

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpAIm);
            }
        }
    }
    fclose(fpARe);
    fclose(fpAIm);

    // start testing
    int32_t run_time = 10000;
    int32_t loop_time = 1;

    __int64 beg, end, timeDuration;
    __int64 interval;
    interval = 0;
    timeDuration = 0;

    for (int32_t k = 0; k < run_time; k++)
    {
        for (int32_t iLoop = 0; iLoop < loop_time; iLoop++)
        {
            beg = __rdtsc();

            matrix_inv_cholesky_4x4(matARe[iLoop], matAIm[iLoop], invARe[iLoop], invAIm[iLoop]);

            end = __rdtsc();
            interval = end - beg;

            if (k > 0)
            {
                timeDuration += interval;
            }
        }
    }

    FILE *fpinvARe, *fpinvAIm;
    char fileinvARe[1024], fileinvAIm[1024]; 

    snprintf(fileinvARe, 1024, "./test_vectors/Hermit_Output_Real_%dx%dx16.bin", nDim, nDim);
    fpinvARe = fopen(fileinvARe, "rb");
    iAssert(fpinvARe != NULL);

    snprintf(fileinvAIm, 1024, "./test_vectors/Hermit_Output_Imag_%dx%dx16.bin", nDim, nDim);
    fpinvAIm = fopen(fileinvAIm, "rb");
    iAssert(fpinvAIm != NULL);

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpinvARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpinvAIm);
            }
        }
    }

    fclose(fpinvARe);
    fclose(fpinvAIm);

    float nGoldRe, nGoldIm, nEstRe, nEstIm;
    float nErrorRe, nErrorIm, MSE;
    double avgMSEdB = 0.0;
    double totaldB = 0.0;
    for (i1 = 0; i1 < loop_time; i1++)
    {
        for (i4 = 0; i4 < nLoadNum; i4++)
        {
            avgMSEdB = 0;
            for (i2 = 0; i2 < nDim; i2++)
            {
                for (i3 = 0; i3 < nDim; i3++)
                {
                    nGoldRe = *((float*)&matARe[i1][i2][i3] + i4);
                    nGoldIm = *((float*)&matAIm[i1][i2][i3] + i4);
                    nEstRe = *((float*)&invARe[i1][i2][i3] + i4);
                    nEstIm = *((float*)&invAIm[i1][i2][i3] + i4);
                    nErrorRe = (float)nEstRe - (float)nGoldRe;
                    nErrorIm = (float)nEstIm - (float)nGoldIm;
                    MSE = (nErrorRe*nErrorRe + nErrorIm*nErrorIm)/((float)nGoldRe*nGoldRe + (float)nGoldIm*nGoldIm);
                    avgMSEdB += (double)MSE;
                }
            }

            avgMSEdB /= (nDim*nDim);
            totaldB += avgMSEdB;
            if (i1 < 1)
                printf("i4 = %d, MSEdB = %f \n", i4, (float)10 * log10((float)avgMSEdB));
        }
    }
    totaldB /= (loop_time*nLoadNum);

    ASSERT_GT(-45, (float)10 * log10((float)totaldB));

    printf("%dx%d Matrix inversion: The average MSEdB = %lf dB\n\n", nDim, nDim, (float)10 * log10((float)totaldB));
    printf("\n%dx%d matrix inversion Average Time: %8.1f cycles\n\n",
        nDim, nDim, timeDuration / loop_time / (float)(run_time-1)/numOperation);
}

TEST(MatrixInv5x5Check, Avx512)
{
    __m512 matARe[NUM_MEM][5][5], matAIm[NUM_MEM][5][5];
    __m512 invARe[NUM_MEM][5][5], invAIm[NUM_MEM][5][5];
    int32_t nDim = 5;
    int32_t i1, i2, i3, i4;
    int32_t numOperation = 16;
    int32_t nLoadLoop = 1;//NUM_MEM;

    FILE *fpARe, *fpAIm;
    char fileARe[1024], fileAIm[1024];

    snprintf(fileARe, 1024, "./test_vectors/Hermit_Input_Real_%dx%dx16.bin", nDim, nDim);
    fpARe = fopen(fileARe, "rb");
    iAssert(fpARe != NULL);

    snprintf(fileAIm, 1024, "./test_vectors/Hermit_Input_Imag_%dx%dx16.bin", nDim, nDim);
    fpAIm = fopen(fileAIm, "rb");
    iAssert(fpAIm != NULL);

    int32_t nLoadNum = numOperation;

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpAIm);
            }
        }
    }
    fclose(fpARe);
    fclose(fpAIm);

    // start testing
    int32_t run_time = 10000;
    int32_t loop_time = 1;

    __int64 beg, end, timeDuration;
    __int64 interval;
    interval = 0;
    timeDuration = 0;

    for (int32_t k = 0; k < run_time; k++)
    {
        for (int32_t iLoop = 0; iLoop < loop_time; iLoop++)
        {
            beg = __rdtsc();

            matrix_inv_cholesky_5x5(matARe[iLoop], matAIm[iLoop], invARe[iLoop], invAIm[iLoop]);

            end = __rdtsc();
            interval = end - beg;

            if (k > 0)
            {
                timeDuration += interval;
            }
        }
    }

    FILE *fpinvARe, *fpinvAIm;
    char fileinvARe[1024], fileinvAIm[1024]; 

    snprintf(fileinvARe, 1024, "./test_vectors/Hermit_Output_Real_%dx%dx16.bin", nDim, nDim);
    fpinvARe = fopen(fileinvARe, "rb");
    iAssert(fpinvARe != NULL);

    snprintf(fileinvAIm, 1024, "./test_vectors/Hermit_Output_Imag_%dx%dx16.bin", nDim, nDim);
    fpinvAIm = fopen(fileinvAIm, "rb");
    iAssert(fpinvAIm != NULL);

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpinvARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpinvAIm);
            }
        }
    }

    fclose(fpinvARe);
    fclose(fpinvAIm);

    float nGoldRe, nGoldIm, nEstRe, nEstIm;
    float nErrorRe, nErrorIm, MSE;
    double avgMSEdB = 0.0;
    double totaldB = 0.0;
    for (i1 = 0; i1 < loop_time; i1++)
    {
        for (i4 = 0; i4 < nLoadNum; i4++)
        {
            avgMSEdB = 0;
            for (i2 = 0; i2 < nDim; i2++)
            {
                for (i3 = 0; i3 < nDim; i3++)
                {
                    nGoldRe = *((float*)&matARe[i1][i2][i3] + i4);
                    nGoldIm = *((float*)&matAIm[i1][i2][i3] + i4);
                    nEstRe = *((float*)&invARe[i1][i2][i3] + i4);
                    nEstIm = *((float*)&invAIm[i1][i2][i3] + i4);
                    nErrorRe = (float)nEstRe - (float)nGoldRe;
                    nErrorIm = (float)nEstIm - (float)nGoldIm;
                    MSE = (nErrorRe*nErrorRe + nErrorIm*nErrorIm)/((float)nGoldRe*nGoldRe + (float)nGoldIm*nGoldIm);
                    avgMSEdB += (double)MSE;
                }
            }

            avgMSEdB /= (nDim*nDim);
            totaldB += avgMSEdB;
            if (i1 < 1)
                printf("i4 = %d, MSEdB = %f \n", i4, (float)10 * log10((float)avgMSEdB));
        }
    }
    totaldB /= (loop_time*nLoadNum);

    ASSERT_GT(-45, (float)10 * log10((float)totaldB));

    printf("%dx%d Matrix inversion: The average MSEdB = %lf dB\n\n", nDim, nDim, (float)10 * log10((float)totaldB));
    printf("\n%dx%d matrix inversion Average Time: %8.1f cycles\n\n",
        nDim, nDim, timeDuration / loop_time / (float)(run_time-1)/numOperation);
}

TEST(MatrixInv6x6Check, Avx512)
{
    __m512 matARe[NUM_MEM][6][6], matAIm[NUM_MEM][6][6];
    __m512 invARe[NUM_MEM][6][6], invAIm[NUM_MEM][6][6];
    int32_t nDim = 6;
    int32_t i1, i2, i3, i4;
    int32_t numOperation = 16;
    int32_t nLoadLoop = 1;//NUM_MEM;

    FILE *fpARe, *fpAIm;
    char fileARe[1024], fileAIm[1024];

    snprintf(fileARe, 1024, "./test_vectors/Hermit_Input_Real_%dx%dx16.bin", nDim, nDim);
    fpARe = fopen(fileARe, "rb");
    iAssert(fpARe != NULL);

    snprintf(fileAIm, 1024, "./test_vectors/Hermit_Input_Imag_%dx%dx16.bin", nDim, nDim);
    fpAIm = fopen(fileAIm, "rb");
    iAssert(fpAIm != NULL);

    int32_t nLoadNum = numOperation;

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpAIm);
            }
        }
    }
    fclose(fpARe);
    fclose(fpAIm);

    // start testing
    int32_t run_time = 10000;
    int32_t loop_time = 1;

    __int64 beg, end, timeDuration;
    __int64 interval;
    interval = 0;
    timeDuration = 0;

    for (int32_t k = 0; k < run_time; k++)
    {
        for (int32_t iLoop = 0; iLoop < loop_time; iLoop++)
        {
            beg = __rdtsc();

            matrix_inv_cholesky_6x6(matARe[iLoop], matAIm[iLoop], invARe[iLoop], invAIm[iLoop]);

            end = __rdtsc();
            interval = end - beg;

            if (k > 0)
            {
                timeDuration += interval;
            }
        }
    }

    FILE *fpinvARe, *fpinvAIm;
    char fileinvARe[1024], fileinvAIm[1024]; 

    snprintf(fileinvARe, 1024, "./test_vectors/Hermit_Output_Real_%dx%dx16.bin", nDim, nDim);
    fpinvARe = fopen(fileinvARe, "rb");
    iAssert(fpinvARe != NULL);

    snprintf(fileinvAIm, 1024, "./test_vectors/Hermit_Output_Imag_%dx%dx16.bin", nDim, nDim);
    fpinvAIm = fopen(fileinvAIm, "rb");
    iAssert(fpinvAIm != NULL);

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpinvARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpinvAIm);
            }
        }
    }

    fclose(fpinvARe);
    fclose(fpinvAIm);

    float nGoldRe, nGoldIm, nEstRe, nEstIm;
    float nErrorRe, nErrorIm, MSE;
    double avgMSEdB = 0.0;
    double totaldB = 0.0;
    for (i1 = 0; i1 < loop_time; i1++)
    {
        for (i4 = 0; i4 < nLoadNum; i4++)
        {
            avgMSEdB = 0;
            for (i2 = 0; i2 < nDim; i2++)
            {
                for (i3 = 0; i3 < nDim; i3++)
                {
                    nGoldRe = *((float*)&matARe[i1][i2][i3] + i4);
                    nGoldIm = *((float*)&matAIm[i1][i2][i3] + i4);
                    nEstRe = *((float*)&invARe[i1][i2][i3] + i4);
                    nEstIm = *((float*)&invAIm[i1][i2][i3] + i4);
                    nErrorRe = (float)nEstRe - (float)nGoldRe;
                    nErrorIm = (float)nEstIm - (float)nGoldIm;
                    MSE = (nErrorRe*nErrorRe + nErrorIm*nErrorIm)/((float)nGoldRe*nGoldRe + (float)nGoldIm*nGoldIm);
                    avgMSEdB += (double)MSE;
                }
            }

            avgMSEdB /= (nDim*nDim);
            totaldB += avgMSEdB;
            if (i1 < 1)
                printf("i4 = %d, MSEdB = %f \n", i4, (float)10 * log10((float)avgMSEdB));
        }
    }
    totaldB /= (loop_time*nLoadNum);

    ASSERT_GT(-45, (float)10 * log10((float)totaldB));

    printf("%dx%d Matrix inversion: The average MSEdB = %lf dB\n\n", nDim, nDim, (float)10 * log10((float)totaldB));
    printf("\n%dx%d matrix inversion Average Time: %8.1f cycles\n\n",
        nDim, nDim, timeDuration / loop_time / (float)(run_time-1)/numOperation);
}

TEST(MatrixInv7x7Check, Avx512)
{
    __m512 matARe[NUM_MEM][7][7], matAIm[NUM_MEM][7][7];
    __m512 invARe[NUM_MEM][7][7], invAIm[NUM_MEM][7][7];
    int32_t nDim = 7;
    int32_t i1, i2, i3, i4;
    int32_t numOperation = 16;
    int32_t nLoadLoop = 1;//NUM_MEM;

    FILE *fpARe, *fpAIm;
    char fileARe[1024], fileAIm[1024];

    snprintf(fileARe, 1024, "./test_vectors/Hermit_Input_Real_%dx%dx16.bin", nDim, nDim);
    fpARe = fopen(fileARe, "rb");
    iAssert(fpARe != NULL);

    snprintf(fileAIm, 1024, "./test_vectors/Hermit_Input_Imag_%dx%dx16.bin", nDim, nDim);
    fpAIm = fopen(fileAIm, "rb");
    iAssert(fpAIm != NULL);

    int32_t nLoadNum = numOperation;

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpAIm);
            }
        }
    }
    fclose(fpARe);
    fclose(fpAIm);

    // start testing
    int32_t run_time = 10000;
    int32_t loop_time = 1;

    __int64 beg, end, timeDuration;
    __int64 interval;
    interval = 0;
    timeDuration = 0;

    for (int32_t k = 0; k < run_time; k++)
    {
        for (int32_t iLoop = 0; iLoop < loop_time; iLoop++)
        {
            beg = __rdtsc();

            matrix_inv_cholesky_7x7(matARe[iLoop], matAIm[iLoop], invARe[iLoop], invAIm[iLoop]);

            end = __rdtsc();
            interval = end - beg;

            if (k > 0)
            {
                timeDuration += interval;
            }
        }
    }

    FILE *fpinvARe, *fpinvAIm;
    char fileinvARe[1024], fileinvAIm[1024]; 

    snprintf(fileinvARe, 1024, "./test_vectors/Hermit_Output_Real_%dx%dx16.bin", nDim, nDim);
    fpinvARe = fopen(fileinvARe, "rb");
    iAssert(fpinvARe != NULL);

    snprintf(fileinvAIm, 1024, "./test_vectors/Hermit_Output_Imag_%dx%dx16.bin", nDim, nDim);
    fpinvAIm = fopen(fileinvAIm, "rb");
    iAssert(fpinvAIm != NULL);

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpinvARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpinvAIm);
            }
        }
    }

    fclose(fpinvARe);
    fclose(fpinvAIm);

    float nGoldRe, nGoldIm, nEstRe, nEstIm;
    float nErrorRe, nErrorIm, MSE;
    double avgMSEdB = 0.0;
    double totaldB = 0.0;
    for (i1 = 0; i1 < loop_time; i1++)
    {
        for (i4 = 0; i4 < nLoadNum; i4++)
        {
            avgMSEdB = 0;
            for (i2 = 0; i2 < nDim; i2++)
            {
                for (i3 = 0; i3 < nDim; i3++)
                {
                    nGoldRe = *((float*)&matARe[i1][i2][i3] + i4);
                    nGoldIm = *((float*)&matAIm[i1][i2][i3] + i4);
                    nEstRe = *((float*)&invARe[i1][i2][i3] + i4);
                    nEstIm = *((float*)&invAIm[i1][i2][i3] + i4);
                    nErrorRe = (float)nEstRe - (float)nGoldRe;
                    nErrorIm = (float)nEstIm - (float)nGoldIm;
                    MSE = (nErrorRe*nErrorRe + nErrorIm*nErrorIm)/((float)nGoldRe*nGoldRe + (float)nGoldIm*nGoldIm);
                    avgMSEdB += (double)MSE;
                }
            }

            avgMSEdB /= (nDim*nDim);
            totaldB += avgMSEdB;
            if (i1 < 1)
                printf("i4 = %d, MSEdB = %f \n", i4, (float)10 * log10((float)avgMSEdB));
        }
    }
    totaldB /= (loop_time*nLoadNum);

    ASSERT_GT(-45, (float)10 * log10((float)totaldB));

    printf("%dx%d Matrix inversion: The average MSEdB = %lf dB\n\n", nDim, nDim, (float)10 * log10((float)totaldB));
    printf("\n%dx%d matrix inversion Average Time: %8.1f cycles\n\n",
        nDim, nDim, timeDuration / loop_time / (float)(run_time-1)/numOperation);
}

TEST(MatrixInv8x8Check, Avx512)
{
    __m512 matARe[NUM_MEM][8][8], matAIm[NUM_MEM][8][8];
    __m512 invARe[NUM_MEM][8][8], invAIm[NUM_MEM][8][8];
    int32_t nDim = 8;
    int32_t i1, i2, i3, i4;
    int32_t numOperation = 16;
    int32_t nLoadLoop = 1;//NUM_MEM;

    FILE *fpARe, *fpAIm;
    char fileARe[1024], fileAIm[1024];

    snprintf(fileARe, 1024, "./test_vectors/Hermit_Input_Real_%dx%dx16.bin", nDim, nDim);
    fpARe = fopen(fileARe, "rb");
    iAssert(fpARe != NULL);

    snprintf(fileAIm, 1024, "./test_vectors/Hermit_Input_Imag_%dx%dx16.bin", nDim, nDim);
    fpAIm = fopen(fileAIm, "rb");
    iAssert(fpAIm != NULL);

    int32_t nLoadNum = numOperation;

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpAIm);
            }
        }
    }
    fclose(fpARe);
    fclose(fpAIm);

    // start testing
    int32_t run_time = 10000;
    int32_t loop_time = 1;

    __int64 beg, end, timeDuration;
    __int64 interval;
    interval = 0;
    timeDuration = 0;

    for (int32_t k = 0; k < run_time; k++)
    {
        for (int32_t iLoop = 0; iLoop < loop_time; iLoop++)
        {
            beg = __rdtsc();

            matrix_inv_cholesky_8x8(matARe[iLoop], matAIm[iLoop], invARe[iLoop], invAIm[iLoop]);

            end = __rdtsc();
            interval = end - beg;

            if (k > 0)
            {
                timeDuration += interval;
            }
        }
    }

    FILE *fpinvARe, *fpinvAIm;
    char fileinvARe[1024], fileinvAIm[1024]; 

    snprintf(fileinvARe, 1024, "./test_vectors/Hermit_Output_Real_%dx%dx16.bin", nDim, nDim);
    fpinvARe = fopen(fileinvARe, "rb");
    iAssert(fpinvARe != NULL);

    snprintf(fileinvAIm, 1024, "./test_vectors/Hermit_Output_Imag_%dx%dx16.bin", nDim, nDim);
    fpinvAIm = fopen(fileinvAIm, "rb");
    iAssert(fpinvAIm != NULL);

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpinvARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpinvAIm);
            }
        }
    }

    fclose(fpinvARe);
    fclose(fpinvAIm);

    float nGoldRe, nGoldIm, nEstRe, nEstIm;
    float nErrorRe, nErrorIm, MSE;
    double avgMSEdB = 0.0;
    double totaldB = 0.0;
    for (i1 = 0; i1 < loop_time; i1++)
    {
        for (i4 = 0; i4 < nLoadNum; i4++)
        {
            avgMSEdB = 0;
            for (i2 = 0; i2 < nDim; i2++)
            {
                for (i3 = 0; i3 < nDim; i3++)
                {
                    nGoldRe = *((float*)&matARe[i1][i2][i3] + i4);
                    nGoldIm = *((float*)&matAIm[i1][i2][i3] + i4);
                    nEstRe = *((float*)&invARe[i1][i2][i3] + i4);
                    nEstIm = *((float*)&invAIm[i1][i2][i3] + i4);
                    nErrorRe = (float)nEstRe - (float)nGoldRe;
                    nErrorIm = (float)nEstIm - (float)nGoldIm;
                    MSE = (nErrorRe*nErrorRe + nErrorIm*nErrorIm)/((float)nGoldRe*nGoldRe + (float)nGoldIm*nGoldIm);
                    avgMSEdB += (double)MSE;
                }
            }

            avgMSEdB /= (nDim*nDim);
            totaldB += avgMSEdB;
            if (i1 < 1)
                printf("i4 = %d, MSEdB = %f \n", i4, (float)10 * log10((float)avgMSEdB));
        }
    }
    totaldB /= (loop_time*nLoadNum);

    ASSERT_GT(-45, (float)10 * log10((float)totaldB));

    printf("%dx%d Matrix inversion: The average MSEdB = %lf dB\n\n", nDim, nDim, (float)10 * log10((float)totaldB));
    printf("\n%dx%d matrix inversion Average Time: %8.1f cycles\n\n",
        nDim, nDim, timeDuration / loop_time / (float)(run_time-1)/numOperation);
}

TEST(MatrixInv9x9Check, Avx512)
{
    __m512 matARe[NUM_MEM][9][9], matAIm[NUM_MEM][9][9];
    __m512 invARe[NUM_MEM][9][9], invAIm[NUM_MEM][9][9];
    int32_t nDim = 9;
    int32_t i1, i2, i3, i4;
    int32_t numOperation = 16;
    int32_t nLoadLoop = 1;//NUM_MEM;

    FILE *fpARe, *fpAIm;
    char fileARe[1024], fileAIm[1024];

    snprintf(fileARe, 1024, "./test_vectors/Hermit_Input_Real_%dx%dx16.bin", nDim, nDim);
    fpARe = fopen(fileARe, "rb");
    iAssert(fpARe != NULL);

    snprintf(fileAIm, 1024, "./test_vectors/Hermit_Input_Imag_%dx%dx16.bin", nDim, nDim);
    fpAIm = fopen(fileAIm, "rb");
    iAssert(fpAIm != NULL);

    int32_t nLoadNum = numOperation;

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpAIm);
            }
        }
    }
    fclose(fpARe);
    fclose(fpAIm);

    // start testing
    int32_t run_time = 10000;
    int32_t loop_time = 1;

    __int64 beg, end, timeDuration;
    __int64 interval;
    interval = 0;
    timeDuration = 0;

    for (int32_t k = 0; k < run_time; k++)
    {
        for (int32_t iLoop = 0; iLoop < loop_time; iLoop++)
        {
            beg = __rdtsc();

            matrix_inv_cholesky_9x9(matARe[iLoop], matAIm[iLoop], invARe[iLoop], invAIm[iLoop]);

            end = __rdtsc();
            interval = end - beg;

            if (k > 0)
            {
                timeDuration += interval;
            }
        }
    }

    FILE *fpinvARe, *fpinvAIm;
    char fileinvARe[1024], fileinvAIm[1024]; 

    snprintf(fileinvARe, 1024, "./test_vectors/Hermit_Output_Real_%dx%dx16.bin", nDim, nDim);
    fpinvARe = fopen(fileinvARe, "rb");
    iAssert(fpinvARe != NULL);

    snprintf(fileinvAIm, 1024, "./test_vectors/Hermit_Output_Imag_%dx%dx16.bin", nDim, nDim);
    fpinvAIm = fopen(fileinvAIm, "rb");
    iAssert(fpinvAIm != NULL);

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpinvARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpinvAIm);
            }
        }
    }

    fclose(fpinvARe);
    fclose(fpinvAIm);

    float nGoldRe, nGoldIm, nEstRe, nEstIm;
    float nErrorRe, nErrorIm, MSE;
    double avgMSEdB = 0.0;
    double totaldB = 0.0;
    for (i1 = 0; i1 < loop_time; i1++)
    {
        for (i4 = 0; i4 < nLoadNum; i4++)
        {
            avgMSEdB = 0;
            for (i2 = 0; i2 < nDim; i2++)
            {
                for (i3 = 0; i3 < nDim; i3++)
                {
                    nGoldRe = *((float*)&matARe[i1][i2][i3] + i4);
                    nGoldIm = *((float*)&matAIm[i1][i2][i3] + i4);
                    nEstRe = *((float*)&invARe[i1][i2][i3] + i4);
                    nEstIm = *((float*)&invAIm[i1][i2][i3] + i4);
                    nErrorRe = (float)nEstRe - (float)nGoldRe;
                    nErrorIm = (float)nEstIm - (float)nGoldIm;
                    MSE = (nErrorRe*nErrorRe + nErrorIm*nErrorIm)/((float)nGoldRe*nGoldRe + (float)nGoldIm*nGoldIm);
                    avgMSEdB += (double)MSE;
                }
            }

            avgMSEdB /= (nDim*nDim);
            totaldB += avgMSEdB;
            if (i1 < 1)
                printf("i4 = %d, MSEdB = %f \n", i4, (float)10 * log10((float)avgMSEdB));
        }
    }
    totaldB /= (loop_time*nLoadNum);

    ASSERT_GT(-45, (float)10 * log10((float)totaldB));

    printf("%dx%d Matrix inversion: The average MSEdB = %lf dB\n\n", nDim, nDim, (float)10 * log10((float)totaldB));
    printf("\n%dx%d matrix inversion Average Time: %8.1f cycles\n\n",
        nDim, nDim, timeDuration / loop_time / (float)(run_time-1)/numOperation);
}

TEST(MatrixInv10x10Check, Avx512)
{
    __m512 matARe[NUM_MEM][10][10], matAIm[NUM_MEM][10][10];
    __m512 invARe[NUM_MEM][10][10], invAIm[NUM_MEM][10][10];
    int32_t nDim = 10;
    int32_t i1, i2, i3, i4;
    int32_t numOperation = 16;
    int32_t nLoadLoop = 1;//NUM_MEM;

    FILE *fpARe, *fpAIm;
    char fileARe[1024], fileAIm[1024];

    snprintf(fileARe, 1024, "./test_vectors/Hermit_Input_Real_%dx%dx16.bin", nDim, nDim);
    fpARe = fopen(fileARe, "rb");
    iAssert(fpARe != NULL);

    snprintf(fileAIm, 1024, "./test_vectors/Hermit_Input_Imag_%dx%dx16.bin", nDim, nDim);
    fpAIm = fopen(fileAIm, "rb");
    iAssert(fpAIm != NULL);

    int32_t nLoadNum = numOperation;

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpAIm);
            }
        }
    }
    fclose(fpARe);
    fclose(fpAIm);

    // start testing
    int32_t run_time = 10000;
    int32_t loop_time = 1;

    __int64 beg, end, timeDuration;
    __int64 interval;
    interval = 0;
    timeDuration = 0;

    for (int32_t k = 0; k < run_time; k++)
    {
        for (int32_t iLoop = 0; iLoop < loop_time; iLoop++)
        {
            beg = __rdtsc();

            matrix_inv_cholesky_10x10(matARe[iLoop], matAIm[iLoop], invARe[iLoop], invAIm[iLoop]);

            end = __rdtsc();
            interval = end - beg;

            if (k > 0)
            {
                timeDuration += interval;
            }
        }
    }

    FILE *fpinvARe, *fpinvAIm;
    char fileinvARe[1024], fileinvAIm[1024]; 

    snprintf(fileinvARe, 1024, "./test_vectors/Hermit_Output_Real_%dx%dx16.bin", nDim, nDim);
    fpinvARe = fopen(fileinvARe, "rb");
    iAssert(fpinvARe != NULL);

    snprintf(fileinvAIm, 1024, "./test_vectors/Hermit_Output_Imag_%dx%dx16.bin", nDim, nDim);
    fpinvAIm = fopen(fileinvAIm, "rb");
    iAssert(fpinvAIm != NULL);

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpinvARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpinvAIm);
            }
        }
    }

    fclose(fpinvARe);
    fclose(fpinvAIm);

    float nGoldRe, nGoldIm, nEstRe, nEstIm;
    float nErrorRe, nErrorIm, MSE;
    double avgMSEdB = 0.0;
    double totaldB = 0.0;
    for (i1 = 0; i1 < loop_time; i1++)
    {
        for (i4 = 0; i4 < nLoadNum; i4++)
        {
            avgMSEdB = 0;
            for (i2 = 0; i2 < nDim; i2++)
            {
                for (i3 = 0; i3 < nDim; i3++)
                {
                    nGoldRe = *((float*)&matARe[i1][i2][i3] + i4);
                    nGoldIm = *((float*)&matAIm[i1][i2][i3] + i4);
                    nEstRe = *((float*)&invARe[i1][i2][i3] + i4);
                    nEstIm = *((float*)&invAIm[i1][i2][i3] + i4);
                    nErrorRe = (float)nEstRe - (float)nGoldRe;
                    nErrorIm = (float)nEstIm - (float)nGoldIm;
                    MSE = (nErrorRe*nErrorRe + nErrorIm*nErrorIm)/((float)nGoldRe*nGoldRe + (float)nGoldIm*nGoldIm);
                    avgMSEdB += (double)MSE;
                }
            }

            avgMSEdB /= (nDim*nDim);
            totaldB += avgMSEdB;
            if (i1 < 1)
                printf("i4 = %d, MSEdB = %f \n", i4, (float)10 * log10((float)avgMSEdB));
        }
    }
    totaldB /= (loop_time*nLoadNum);

    ASSERT_GT(-45, (float)10 * log10((float)totaldB));

    printf("%dx%d Matrix inversion: The average MSEdB = %lf dB\n\n", nDim, nDim, (float)10 * log10((float)totaldB));
    printf("\n%dx%d matrix inversion Average Time: %8.1f cycles\n\n",
        nDim, nDim, timeDuration / loop_time / (float)(run_time-1)/numOperation);
}
TEST(MatrixInv11x11Check, Avx512)
{
    __m512 matARe[NUM_MEM][11][11], matAIm[NUM_MEM][11][11];
    __m512 invARe[NUM_MEM][11][11], invAIm[NUM_MEM][11][11];
    int32_t nDim = 11;
    int32_t i1, i2, i3, i4;
    int32_t numOperation = 16;
    int32_t nLoadLoop = 1;//NUM_MEM;

    FILE *fpARe, *fpAIm;
    char fileARe[1024], fileAIm[1024];

    snprintf(fileARe, 1024, "./test_vectors/Hermit_Input_Real_%dx%dx16.bin", nDim, nDim);
    fpARe = fopen(fileARe, "rb");
    iAssert(fpARe != NULL);

    snprintf(fileAIm, 1024, "./test_vectors/Hermit_Input_Imag_%dx%dx16.bin", nDim, nDim);
    fpAIm = fopen(fileAIm, "rb");
    iAssert(fpAIm != NULL);

    int32_t nLoadNum = numOperation;

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpAIm);
            }
        }
    }
    fclose(fpARe);
    fclose(fpAIm);

    // start testing
    int32_t run_time = 10000;
    int32_t loop_time = 1;

    __int64 beg, end, timeDuration;
    __int64 interval;
    interval = 0;
    timeDuration = 0;

    for (int32_t k = 0; k < run_time; k++)
    {
        for (int32_t iLoop = 0; iLoop < loop_time; iLoop++)
        {
            beg = __rdtsc();

            matrix_inv_cholesky_11x11(matARe[iLoop], matAIm[iLoop], invARe[iLoop], invAIm[iLoop]);

            end = __rdtsc();
            interval = end - beg;

            if (k > 0)
            {
                timeDuration += interval;
            }
        }
    }

    FILE *fpinvARe, *fpinvAIm;
    char fileinvARe[1024], fileinvAIm[1024]; 

    snprintf(fileinvARe, 1024, "./test_vectors/Hermit_Output_Real_%dx%dx16.bin", nDim, nDim);
    fpinvARe = fopen(fileinvARe, "rb");
    iAssert(fpinvARe != NULL);

    snprintf(fileinvAIm, 1024, "./test_vectors/Hermit_Output_Imag_%dx%dx16.bin", nDim, nDim);
    fpinvAIm = fopen(fileinvAIm, "rb");
    iAssert(fpinvAIm != NULL);

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpinvARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpinvAIm);
            }
        }
    }

    fclose(fpinvARe);
    fclose(fpinvAIm);

    float nGoldRe, nGoldIm, nEstRe, nEstIm;
    float nErrorRe, nErrorIm, MSE;
    double avgMSEdB = 0.0;
    double totaldB = 0.0;
    for (i1 = 0; i1 < loop_time; i1++)
    {
        for (i4 = 0; i4 < nLoadNum; i4++)
        {
            avgMSEdB = 0;
            for (i2 = 0; i2 < nDim; i2++)
            {
                for (i3 = 0; i3 < nDim; i3++)
                {
                    nGoldRe = *((float*)&matARe[i1][i2][i3] + i4);
                    nGoldIm = *((float*)&matAIm[i1][i2][i3] + i4);
                    nEstRe = *((float*)&invARe[i1][i2][i3] + i4);
                    nEstIm = *((float*)&invAIm[i1][i2][i3] + i4);
                    nErrorRe = (float)nEstRe - (float)nGoldRe;
                    nErrorIm = (float)nEstIm - (float)nGoldIm;
                    MSE = (nErrorRe*nErrorRe + nErrorIm*nErrorIm)/((float)nGoldRe*nGoldRe + (float)nGoldIm*nGoldIm);
                    avgMSEdB += (double)MSE;
                }
            }

            avgMSEdB /= (nDim*nDim);
            totaldB += avgMSEdB;
            if (i1 < 1)
               printf("i4 = %d, MSEdB = %f \n", i4, (float)10 * log10((float)avgMSEdB));
        }
    }
    totaldB /= (loop_time*nLoadNum);

    ASSERT_GT(-45, (float)10 * log10((float)totaldB));

    printf("%dx%d Matrix inversion: The average MSEdB = %lf dB\n\n", nDim, nDim, (float)10 * log10((float)totaldB));
    printf("\n%dx%d matrix inversion Average Time: %8.1f cycles\n\n",
        nDim, nDim, timeDuration / loop_time / (float)(run_time-1)/numOperation);
}
TEST(MatrixInv12x12Check, Avx512)
{
    __m512 matARe[NUM_MEM][12][12], matAIm[NUM_MEM][12][12];
    __m512 invARe[NUM_MEM][12][12], invAIm[NUM_MEM][12][12];
    int32_t nDim = 12;
    int32_t i1, i2, i3, i4;
    int32_t numOperation = 16;
    int32_t nLoadLoop = 1;//NUM_MEM;

    FILE *fpARe, *fpAIm;
    char fileARe[1024], fileAIm[1024];

    snprintf(fileARe, 1024, "./test_vectors/Hermit_Input_Real_%dx%dx16.bin", nDim, nDim);
    fpARe = fopen(fileARe, "rb");
    iAssert(fpARe != NULL);

    snprintf(fileAIm, 1024, "./test_vectors/Hermit_Input_Imag_%dx%dx16.bin", nDim, nDim);
    fpAIm = fopen(fileAIm, "rb");
    iAssert(fpAIm != NULL);

    int32_t nLoadNum = numOperation;

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpAIm);
            }
        }
    }
    fclose(fpARe);
    fclose(fpAIm);

    // start testing
    int32_t run_time = 10000;
    int32_t loop_time = 1;

    __int64 beg, end, timeDuration;
    __int64 interval;
    interval = 0;
    timeDuration = 0;

    for (int32_t k = 0; k < run_time; k++)
    {
        for (int32_t iLoop = 0; iLoop < loop_time; iLoop++)
        {
            beg = __rdtsc();

            matrix_inv_cholesky_12x12(matARe[iLoop], matAIm[iLoop], invARe[iLoop], invAIm[iLoop]);

            end = __rdtsc();
            interval = end - beg;

            if (k > 0)
            {
                timeDuration += interval;
            }
        }
    }

    FILE *fpinvARe, *fpinvAIm;
    char fileinvARe[1024], fileinvAIm[1024]; 

    snprintf(fileinvARe, 1024, "./test_vectors/Hermit_Output_Real_%dx%dx16.bin", nDim, nDim);
    fpinvARe = fopen(fileinvARe, "rb");
    iAssert(fpinvARe != NULL);

    snprintf(fileinvAIm, 1024, "./test_vectors/Hermit_Output_Imag_%dx%dx16.bin", nDim, nDim);
    fpinvAIm = fopen(fileinvAIm, "rb");
    iAssert(fpinvAIm != NULL);

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpinvARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpinvAIm);
            }
        }
    }

    fclose(fpinvARe);
    fclose(fpinvAIm);

    float nGoldRe, nGoldIm, nEstRe, nEstIm;
    float nErrorRe, nErrorIm, MSE;
    double avgMSEdB = 0.0;
    double totaldB = 0.0;
    for (i1 = 0; i1 < loop_time; i1++)
    {
        for (i4 = 0; i4 < nLoadNum; i4++)
        {
            avgMSEdB = 0;
            for (i2 = 0; i2 < nDim; i2++)
            {
                for (i3 = 0; i3 < nDim; i3++)
                {
                    nGoldRe = *((float*)&matARe[i1][i2][i3] + i4);
                    nGoldIm = *((float*)&matAIm[i1][i2][i3] + i4);
                    nEstRe = *((float*)&invARe[i1][i2][i3] + i4);
                    nEstIm = *((float*)&invAIm[i1][i2][i3] + i4);
                    nErrorRe = (float)nEstRe - (float)nGoldRe;
                    nErrorIm = (float)nEstIm - (float)nGoldIm;
                    MSE = (nErrorRe*nErrorRe + nErrorIm*nErrorIm)/((float)nGoldRe*nGoldRe + (float)nGoldIm*nGoldIm);
                    avgMSEdB += (double)MSE;
                }
            }

            avgMSEdB /= (nDim*nDim);
            totaldB += avgMSEdB;
            if (i1 < 1)
               printf("i4 = %d, MSEdB = %f \n", i4, (float)10 * log10((float)avgMSEdB));
        }
    }
    totaldB /= (loop_time*nLoadNum);

    ASSERT_GT(-45, (float)10 * log10((float)totaldB));

    printf("%dx%d Matrix inversion: The average MSEdB = %lf dB\n\n", nDim, nDim, (float)10 * log10((float)totaldB));
    printf("\n%dx%d matrix inversion Average Time: %8.1f cycles\n\n",
        nDim, nDim, timeDuration / loop_time / (float)(run_time-1)/numOperation);
}
TEST(MatrixInv13x13Check, Avx512)
{
    __m512 matARe[NUM_MEM][13][13], matAIm[NUM_MEM][13][13];
    __m512 invARe[NUM_MEM][13][13], invAIm[NUM_MEM][13][13];
    int32_t nDim = 13;
    int32_t i1, i2, i3, i4;
    int32_t numOperation = 16;
    int32_t nLoadLoop = 1;//NUM_MEM;

    FILE *fpARe, *fpAIm;
    char fileARe[1024], fileAIm[1024];

    snprintf(fileARe, 1024, "./test_vectors/Hermit_Input_Real_%dx%dx16.bin", nDim, nDim);
    fpARe = fopen(fileARe, "rb");
    iAssert(fpARe != NULL);

    snprintf(fileAIm, 1024, "./test_vectors/Hermit_Input_Imag_%dx%dx16.bin", nDim, nDim);
    fpAIm = fopen(fileAIm, "rb");
    iAssert(fpAIm != NULL);

    int32_t nLoadNum = numOperation;

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpAIm);
            }
        }
    }
    fclose(fpARe);
    fclose(fpAIm);

    // start testing
    int32_t run_time = 10000;
    int32_t loop_time = 1;

    __int64 beg, end, timeDuration;
    __int64 interval;
    interval = 0;
    timeDuration = 0;

    for (int32_t k = 0; k < run_time; k++)
    {
        for (int32_t iLoop = 0; iLoop < loop_time; iLoop++)
        {
            beg = __rdtsc();

            matrix_inv_cholesky_13x13(matARe[iLoop], matAIm[iLoop], invARe[iLoop], invAIm[iLoop]);

            end = __rdtsc();
            interval = end - beg;

            if (k > 0)
            {
                timeDuration += interval;
            }
        }
    }

    FILE *fpinvARe, *fpinvAIm;
    char fileinvARe[1024], fileinvAIm[1024]; 

    snprintf(fileinvARe, 1024, "./test_vectors/Hermit_Output_Real_%dx%dx16.bin", nDim, nDim);
    fpinvARe = fopen(fileinvARe, "rb");
    iAssert(fpinvARe != NULL);

    snprintf(fileinvAIm, 1024, "./test_vectors/Hermit_Output_Imag_%dx%dx16.bin", nDim, nDim);
    fpinvAIm = fopen(fileinvAIm, "rb");
    iAssert(fpinvAIm != NULL);

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpinvARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpinvAIm);
            }
        }
    }

    fclose(fpinvARe);
    fclose(fpinvAIm);

    float nGoldRe, nGoldIm, nEstRe, nEstIm;
    float nErrorRe, nErrorIm, MSE;
    double avgMSEdB = 0.0;
    double totaldB = 0.0;
    for (i1 = 0; i1 < loop_time; i1++)
    {
        for (i4 = 0; i4 < nLoadNum; i4++)
        {
            avgMSEdB = 0;
            for (i2 = 0; i2 < nDim; i2++)
            {
                for (i3 = 0; i3 < nDim; i3++)
                {
                    nGoldRe = *((float*)&matARe[i1][i2][i3] + i4);
                    nGoldIm = *((float*)&matAIm[i1][i2][i3] + i4);
                    nEstRe = *((float*)&invARe[i1][i2][i3] + i4);
                    nEstIm = *((float*)&invAIm[i1][i2][i3] + i4);
                    nErrorRe = (float)nEstRe - (float)nGoldRe;
                    nErrorIm = (float)nEstIm - (float)nGoldIm;
                    MSE = (nErrorRe*nErrorRe + nErrorIm*nErrorIm)/((float)nGoldRe*nGoldRe + (float)nGoldIm*nGoldIm);
                    avgMSEdB += (double)MSE;
                }
            }

            avgMSEdB /= (nDim*nDim);
            totaldB += avgMSEdB;
            if (i1 < 1)
               printf("i4 = %d, MSEdB = %f \n", i4, (float)10 * log10((float)avgMSEdB));
        }
    }
    totaldB /= (loop_time*nLoadNum);

    ASSERT_GT(-45, (float)10 * log10((float)totaldB));

    printf("%dx%d Matrix inversion: The average MSEdB = %lf dB\n\n", nDim, nDim, (float)10 * log10((float)totaldB));
    printf("\n%dx%d matrix inversion Average Time: %8.1f cycles\n\n",
        nDim, nDim, timeDuration / loop_time / (float)(run_time-1)/numOperation);
}
TEST(MatrixInv14x14Check, Avx512)
{
    __m512 matARe[NUM_MEM][14][14], matAIm[NUM_MEM][14][14];
    __m512 invARe[NUM_MEM][14][14], invAIm[NUM_MEM][14][14];
    int32_t nDim = 14;
    int32_t i1, i2, i3, i4;
    int32_t numOperation = 16;
    int32_t nLoadLoop = 1;//NUM_MEM;

    FILE *fpARe, *fpAIm;
    char fileARe[1024], fileAIm[1024];

    snprintf(fileARe, 1024, "./test_vectors/Hermit_Input_Real_%dx%dx16.bin", nDim, nDim);
    fpARe = fopen(fileARe, "rb");
    iAssert(fpARe != NULL);

    snprintf(fileAIm, 1024, "./test_vectors/Hermit_Input_Imag_%dx%dx16.bin", nDim, nDim);
    fpAIm = fopen(fileAIm, "rb");
    iAssert(fpAIm != NULL);

    int32_t nLoadNum = numOperation;

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpAIm);
            }
        }
    }
    fclose(fpARe);
    fclose(fpAIm);

    // start testing
    int32_t run_time = 10000;
    int32_t loop_time = 1;

    __int64 beg, end, timeDuration;
    __int64 interval;
    interval = 0;
    timeDuration = 0;

    for (int32_t k = 0; k < run_time; k++)
    {
        for (int32_t iLoop = 0; iLoop < loop_time; iLoop++)
        {
            beg = __rdtsc();

            matrix_inv_cholesky_14x14(matARe[iLoop], matAIm[iLoop], invARe[iLoop], invAIm[iLoop]);

            end = __rdtsc();
            interval = end - beg;

            if (k > 0)
            {
                timeDuration += interval;
            }
        }
    }

    FILE *fpinvARe, *fpinvAIm;
    char fileinvARe[1024], fileinvAIm[1024]; 

    snprintf(fileinvARe, 1024, "./test_vectors/Hermit_Output_Real_%dx%dx16.bin", nDim, nDim);
    fpinvARe = fopen(fileinvARe, "rb");
    iAssert(fpinvARe != NULL);

    snprintf(fileinvAIm, 1024, "./test_vectors/Hermit_Output_Imag_%dx%dx16.bin", nDim, nDim);
    fpinvAIm = fopen(fileinvAIm, "rb");
    iAssert(fpinvAIm != NULL);

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpinvARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpinvAIm);
            }
        }
    }

    fclose(fpinvARe);
    fclose(fpinvAIm);

    float nGoldRe, nGoldIm, nEstRe, nEstIm;
    float nErrorRe, nErrorIm, MSE;
    double avgMSEdB = 0.0;
    double totaldB = 0.0;
    for (i1 = 0; i1 < loop_time; i1++)
    {
        for (i4 = 0; i4 < nLoadNum; i4++)
        {
            avgMSEdB = 0;
            for (i2 = 0; i2 < nDim; i2++)
            {
                for (i3 = 0; i3 < nDim; i3++)
                {
                    nGoldRe = *((float*)&matARe[i1][i2][i3] + i4);
                    nGoldIm = *((float*)&matAIm[i1][i2][i3] + i4);
                    nEstRe = *((float*)&invARe[i1][i2][i3] + i4);
                    nEstIm = *((float*)&invAIm[i1][i2][i3] + i4);
                    nErrorRe = (float)nEstRe - (float)nGoldRe;
                    nErrorIm = (float)nEstIm - (float)nGoldIm;
                    MSE = (nErrorRe*nErrorRe + nErrorIm*nErrorIm)/((float)nGoldRe*nGoldRe + (float)nGoldIm*nGoldIm);
                    avgMSEdB += (double)MSE;
                }
            }

            avgMSEdB /= (nDim*nDim);
            totaldB += avgMSEdB;
            if (i1 < 1)
               printf("i4 = %d, MSEdB = %f \n", i4, (float)10 * log10((float)avgMSEdB));
        }
    }
    totaldB /= (loop_time*nLoadNum);

    ASSERT_GT(-45, (float)10 * log10((float)totaldB));

    printf("%dx%d Matrix inversion: The average MSEdB = %lf dB\n\n", nDim, nDim, (float)10 * log10((float)totaldB));
    printf("\n%dx%d matrix inversion Average Time: %8.1f cycles\n\n",
        nDim, nDim, timeDuration / loop_time / (float)(run_time-1)/numOperation);
}
TEST(MatrixInv15x15Check, Avx512)
{
    __m512 matARe[NUM_MEM][15][15], matAIm[NUM_MEM][15][15];
    __m512 invARe[NUM_MEM][15][15], invAIm[NUM_MEM][15][15];
    int32_t nDim = 15;
    int32_t i1, i2, i3, i4;
    int32_t numOperation = 16;
    int32_t nLoadLoop = 1;//NUM_MEM;

    FILE *fpARe, *fpAIm;
    char fileARe[1024], fileAIm[1024];

    snprintf(fileARe, 1024, "./test_vectors/Hermit_Input_Real_%dx%dx16.bin", nDim, nDim);
    fpARe = fopen(fileARe, "rb");
    iAssert(fpARe != NULL);

    snprintf(fileAIm, 1024, "./test_vectors/Hermit_Input_Imag_%dx%dx16.bin", nDim, nDim);
    fpAIm = fopen(fileAIm, "rb");
    iAssert(fpAIm != NULL);

    int32_t nLoadNum = numOperation;

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpAIm);
            }
        }
    }
    fclose(fpARe);
    fclose(fpAIm);

    // start testing
    int32_t run_time = 10000;
    int32_t loop_time = 1;

    __int64 beg, end, timeDuration;
    __int64 interval;
    interval = 0;
    timeDuration = 0;

    for (int32_t k = 0; k < run_time; k++)
    {
        for (int32_t iLoop = 0; iLoop < loop_time; iLoop++)
        {
            beg = __rdtsc();

            matrix_inv_cholesky_15x15(matARe[iLoop], matAIm[iLoop], invARe[iLoop], invAIm[iLoop]);

            end = __rdtsc();
            interval = end - beg;

            if (k > 0)
            {
                timeDuration += interval;
            }
        }
    }

    FILE *fpinvARe, *fpinvAIm;
    char fileinvARe[1024], fileinvAIm[1024]; 

    snprintf(fileinvARe, 1024, "./test_vectors/Hermit_Output_Real_%dx%dx16.bin", nDim, nDim);
    fpinvARe = fopen(fileinvARe, "rb");
    iAssert(fpinvARe != NULL);

    snprintf(fileinvAIm, 1024, "./test_vectors/Hermit_Output_Imag_%dx%dx16.bin", nDim, nDim);
    fpinvAIm = fopen(fileinvAIm, "rb");
    iAssert(fpinvAIm != NULL);

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpinvARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpinvAIm);
            }
        }
    }

    fclose(fpinvARe);
    fclose(fpinvAIm);

    float nGoldRe, nGoldIm, nEstRe, nEstIm;
    float nErrorRe, nErrorIm, MSE;
    double avgMSEdB = 0.0;
    double totaldB = 0.0;
    for (i1 = 0; i1 < loop_time; i1++)
    {
        for (i4 = 0; i4 < nLoadNum; i4++)
        {
            avgMSEdB = 0;
            for (i2 = 0; i2 < nDim; i2++)
            {
                for (i3 = 0; i3 < nDim; i3++)
                {
                    nGoldRe = *((float*)&matARe[i1][i2][i3] + i4);
                    nGoldIm = *((float*)&matAIm[i1][i2][i3] + i4);
                    nEstRe = *((float*)&invARe[i1][i2][i3] + i4);
                    nEstIm = *((float*)&invAIm[i1][i2][i3] + i4);
                    nErrorRe = (float)nEstRe - (float)nGoldRe;
                    nErrorIm = (float)nEstIm - (float)nGoldIm;
                    MSE = (nErrorRe*nErrorRe + nErrorIm*nErrorIm)/((float)nGoldRe*nGoldRe + (float)nGoldIm*nGoldIm);
                    avgMSEdB += (double)MSE;
                }
            }

            avgMSEdB /= (nDim*nDim);
            totaldB += avgMSEdB;
            if (i1 < 1)
              printf("i4 = %d, MSEdB = %f \n", i4, (float)10 * log10((float)avgMSEdB));
        }
    }
    totaldB /= (loop_time*nLoadNum);

    ASSERT_GT(-45, (float)10 * log10((float)totaldB));

    printf("%dx%d Matrix inversion: The average MSEdB = %lf dB\n\n", nDim, nDim, (float)10 * log10((float)totaldB));
    printf("\n%dx%d matrix inversion Average Time: %8.1f cycles\n\n",
        nDim, nDim, timeDuration / loop_time / (float)(run_time-1)/numOperation);
}
TEST(MatrixInv16x16Check, Avx512)
{
    __m512 matARe[NUM_MEM][16][16], matAIm[NUM_MEM][16][16];
    __m512 invARe[NUM_MEM][16][16], invAIm[NUM_MEM][16][16];
    int32_t nDim = 16;
    int32_t i1, i2, i3, i4;
    int32_t numOperation = 16;
    int32_t nLoadLoop = 1;//NUM_MEM;

    FILE *fpARe, *fpAIm;
    char fileARe[1024], fileAIm[1024];

    snprintf(fileARe, 1024, "./test_vectors/Hermit_Input_Real_%dx%dx16.bin", nDim, nDim);
    fpARe = fopen(fileARe, "rb");
    iAssert(fpARe != NULL);

    snprintf(fileAIm, 1024, "./test_vectors/Hermit_Input_Imag_%dx%dx16.bin", nDim, nDim);
    fpAIm = fopen(fileAIm, "rb");
    iAssert(fpAIm != NULL);

    int32_t nLoadNum = numOperation;

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpAIm);
            }
        }
    }
    fclose(fpARe);
    fclose(fpAIm);

    // start testing
    int32_t run_time = 10000;
    int32_t loop_time = 1;

    __int64 beg, end, timeDuration;
    __int64 interval;
    interval = 0;
    timeDuration = 0;

    for (int32_t k = 0; k < run_time; k++)
    {
        for (int32_t iLoop = 0; iLoop < loop_time; iLoop++)
        {
            beg = __rdtsc();

            matrix_inv_cholesky_16x16(matARe[iLoop], matAIm[iLoop], invARe[iLoop], invAIm[iLoop]);

            end = __rdtsc();
            interval = end - beg;

            if (k > 0)
            {
                timeDuration += interval;
            }
        }
    }

    FILE *fpinvARe, *fpinvAIm;
    char fileinvARe[1024], fileinvAIm[1024]; 

    snprintf(fileinvARe, 1024, "./test_vectors/Hermit_Output_Real_%dx%dx16.bin", nDim, nDim);
    fpinvARe = fopen(fileinvARe, "rb");
    iAssert(fpinvARe != NULL);

    snprintf(fileinvAIm, 1024, "./test_vectors/Hermit_Output_Imag_%dx%dx16.bin", nDim, nDim);
    fpinvAIm = fopen(fileinvAIm, "rb");
    iAssert(fpinvAIm != NULL);

    for (i1 = 0; i1 < nLoadLoop; i1++)
    {
        for (i2 = 0; i2 < nDim; i2++)
        {
            for (i3 = 0; i3 < nDim; i3++)
            {
                fread((float*) &(matARe[i1][i2][i3]), sizeof(float), nLoadNum, fpinvARe);
                fread((float*) &(matAIm[i1][i2][i3]), sizeof(float), nLoadNum, fpinvAIm);
            }
        }
    }

    fclose(fpinvARe);
    fclose(fpinvAIm);

    float nGoldRe, nGoldIm, nEstRe, nEstIm;
    float nErrorRe, nErrorIm, MSE;
    double avgMSEdB = 0.0;
    double totaldB = 0.0;
    for (i1 = 0; i1 < loop_time; i1++)
    {
        for (i4 = 0; i4 < nLoadNum; i4++)
        {
            avgMSEdB = 0;
            for (i2 = 0; i2 < nDim; i2++)
            {
                for (i3 = 0; i3 < nDim; i3++)
                {
                    nGoldRe = *((float*)&matARe[i1][i2][i3] + i4);
                    nGoldIm = *((float*)&matAIm[i1][i2][i3] + i4);
                    nEstRe = *((float*)&invARe[i1][i2][i3] + i4);
                    nEstIm = *((float*)&invAIm[i1][i2][i3] + i4);
                    nErrorRe = (float)nEstRe - (float)nGoldRe;
                    nErrorIm = (float)nEstIm - (float)nGoldIm;
                    MSE = (nErrorRe*nErrorRe + nErrorIm*nErrorIm)/((float)nGoldRe*nGoldRe + (float)nGoldIm*nGoldIm);
                    avgMSEdB += (double)MSE;
                }
            }

            avgMSEdB /= (nDim*nDim);
            totaldB += avgMSEdB;
            if (i1 < 1)
                printf("i4 = %d, MSEdB = %f \n", i4, (float)10 * log10((float)avgMSEdB));
        }
    }
    totaldB /= (loop_time*nLoadNum);

    ASSERT_GT(-45, (float)10 * log10((float)totaldB));

    printf("%dx%d Matrix inversion: The average MSEdB = %lf dB\n\n", nDim, nDim, (float)10 * log10((float)totaldB));
    printf("\n%dx%d matrix inversion Average Time: %8.1f cycles\n\n",
        nDim, nDim, timeDuration / loop_time / (float)(run_time-1)/numOperation);
}

#endif
