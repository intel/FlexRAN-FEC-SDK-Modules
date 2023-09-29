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
    \file phy_rate_match.h
    \brief  External API for LTE Rate Matching, Dematching (HARQ & deinterleaver) functions in LTE.
 */

#ifndef _PHY_RATE_MATCH_H_
#define _PHY_RATE_MATCH_H_

#include <stdint.h>

#include "common_typedef_sdk.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
    \struct bblib_rate_match_dl_request
    \brief Structure for input parameters in API of rate matching for LTE.
    \note tin0, tin1, tin2, and output need to be aligned with 128bits.
*/
struct bblib_rate_match_dl_request {
    int32_t r; /*!< index of current code block in all code blocks. */
    int32_t C; /*!< Total number of code blocks. */
    int8_t direction; /*!< flag of DL or UL, 1 for DL and 0 for UL. */
    int32_t Nsoft; /*!< Total number of soft bits according to UE categories. */
    int32_t KMIMO; /*!< 2, which is related to MIMO type. */
    int32_t MDL_HARQ; /*!< Maximum number of DL HARQ. */
    int32_t G; /*!< length of bits before modulation for 1 UE in 1 subframe. */
    int32_t NL; /*!< Number of layer. */
    int32_t Qm; /*!< Modulation type, which can be 2/4/6. */
    int32_t rvidx; /*!< Redundancy version, which can be 0/1/2/3. */
    int8_t bypass_rvidx; /*!< If set rvidx is ignored and k0 set to 0 */
    int32_t Kidx; /*!< Position in turbo code internal interleave table,
                       Kidx=i-1 in TS 136.212 table 5.1.3-3. */
    int32_t nLen; /*!< Length of input data from tin0/tin1/tin2 in bits,
                       nLen=K(Kidx+1)+4 in TS 136.212 table 5.1.3-3. */
    uint8_t *tin0; /*!< pointer to input stream 0 from turbo encoder. */
    uint8_t *tin1; /*!< tin1 pointer to input stream 1 from turbo encoder. */
    uint8_t *tin2; /*!< tin2 pointer to input stream 2 from turbo encoder. */
};

/**
    \struct bblib_rate_match_dl_response
    \brief structure for outputs of rate matching for LTE.
 */
struct bblib_rate_match_dl_response {
    uint8_t *output; /*!< Output buffer for data stream after rate matching. */
    uint32_t OutputLen; /*!< outputLen Accumulated output length in bytes in bytes of rate
                             matching before this code block. */
};

/*!
    \struct bblib_rate_match_ul_request
    \brief Structure for parameters in API of HARQ, deinterleaver (rate dematching) for LTE.
    \note pdmout, pharqbuffer, pinteleavebuffer and pharqout need to be aligned with 256 bits.
 */
struct bblib_rate_match_ul_request {
    uint8_t *pdmout; /*!< Demodulation output, and input of HARQ. */
    int32_t k0withoutnull; /*!< K0 without NULL based on RV, position of this input
                                HARQ sequence in ring buffer. */
    int32_t ncb; /*!< Dyclic buffer length.  */
    int32_t e; /*!< HARQ combine input length in bytes. */
    int32_t isretx; /*!< Dlag of retransmission, 0: no retransmission, 1: retransmission. */
    int32_t isinverted; /**< input soft decisions are inverted - set to 1 if '1' bit is represented by a positive value */
};

/*!
    \struct bblib_rate_match_ul_response
    \brief Response structure for rate macthing UL.
 */
struct bblib_rate_match_ul_response {
    uint8_t *pharqbuffer; /**< Output of HARQ combine, and input of sub-block deinterleaver. */
    uint8_t *pinteleavebuffer; /**< Output of sub-block deinterleaver, and input of turbo
                                    decoder adapter. */
    uint8_t *pharqout; /**< Final output of this function, and input for turbo decoder. */
};

/*!
    \struct bblib_harq_combine_ul_request
    \brief Request structure for parameters in API of HARQ for LTE
    \note pdmout and pharqbuffer need to be aligned with 256 bits
 */
struct bblib_harq_combine_ul_request {
    uint8_t *pdmout; /**< demodulation output, and input of HARQ */
    int32_t k0withoutnull; /**< K0 without NULL based on RV, position of this input HARQ sequence in ring buffer */
    int32_t ncb; /**< cyclic buffer length  */
    int32_t e; /**< HARQ combine input length in bytes */
    int32_t isretx; /**< flag of retransmission, 0: no retransmission, 1: retransmission */
};

/*!
    \struct bblib_harq_combine_ul_response
    \brief Response structure for parameters in API of HARQ for LTE
 */
struct bblib_harq_combine_ul_response {
    uint8_t *pharqbuffer; /**< output of HARQ combine, and input of sub-block deinterleaver */
};

/*!
    \enum circular_buffer_format
    \brief circular buffer format; defines format of circular buffer given as input
*/
enum circular_buffer_format{
    BBLIB_CIRCULAR_BUFFER_WITHOUT_PADDING = 0, /**< Cicular buffer without dummy padding bits. */
    BBLIB_FULL_CIRCULAR_BUFFER = 1 /**< Full circular buffer, i.e. with dummy padding bits
                                     as discribed in 3GPP 36.212 subclause 5.1.4.1.1. */
};

/*!
    \struct bblib_deinterleave_ul_request
    \brief Request structure for parameters in API of deinterleaver (rate dematching) for LTE
    \note pharqbuffer and pinteleavebuffer need to be aligned with 256 bits
 */
struct bblib_deinterleave_ul_request {
    uint8_t *pharqbuffer; /**< output of HARQ combine, and input of sub-block deinterleaver */
    int32_t ncb; /**< cyclic buffer length  */
    enum circular_buffer_format circ_buffer; /**< Defines input of sub-block deinterleaver format:
                                             circular buffer without padding or full circular buffer
                                             with dummy padding bits. */
};

/*!
    \struct bblib_deinterleave_ul_response
    \brief Response structure for parameters in API of deinterleaver (rate dematching) for LTE
 */
struct bblib_deinterleave_ul_response {
    uint8_t *pinteleavebuffer; /**< output of sub-block deinterleaver, and input of turbo decoder adapter */
};

/*!
    \struct bblib_turbo_adapter_ul_request
    \brief Request structure for parameters in API of Turbo Adapter for LTE
    \note pinteleavebuffer and pharqout need to be aligned with 256 bits
 */
struct bblib_turbo_adapter_ul_request {
    uint8_t *pinteleavebuffer; /**< output of sub-block deinterleaver, and input of turbo decoder adapter */
    int32_t ncb; /**< cyclic buffer length  */
    int32_t isinverted; /**< input soft decisions are inverted - set to 1 if '1' bit is represented by a positive value */
};

/*!
    \struct bblib_turbo_adapter_ul_response
    \brief Response structure for parameters in API of Turbo Adapter for LTE
 */
struct bblib_turbo_adapter_ul_response {
    uint8_t *pharqout; /**< final output of this function, and input for turbo decoder */
};


//! @{
/*! \brief Downlink rate matching for LTE.
    \param [in] request Structure containing configuration information and input data.
    \param [out] response Structure containing kernel outputs.
    \note bblib_rate_match_dl provides the most appropriate version for the available ISA,
          the _avx2 etc. version allow direct access to specific ISA implementations.
    \return Success: return 0, else: return -1.
*/
int32_t
bblib_rate_match_dl(const struct bblib_rate_match_dl_request *request,
    struct bblib_rate_match_dl_response *response);

int32_t bblib_rate_match_dl_sse(const struct bblib_rate_match_dl_request *request,
    struct bblib_rate_match_dl_response *response);
int32_t bblib_rate_match_dl_avx2(const struct bblib_rate_match_dl_request *request,
    struct bblib_rate_match_dl_response *response);
//! @}

//! @{
/*! \brief Uplink rate matching for LTE.

    Includes HARQ combining, subblock deinterleaving and data formatting for turbo decoding.

    \param [in] request Structure containing configuration information and input data.
    \param [out] response Structure containing kernel outputs.
    \return Success: return 0, else: return -1.
 */
int32_t
bblib_rate_match_ul(const struct bblib_rate_match_ul_request *request,
    struct bblib_rate_match_ul_response *response);

int32_t bblib_rate_match_ul_avx2(const struct bblib_rate_match_ul_request *request,
    struct bblib_rate_match_ul_response *response);

int32_t bblib_rate_match_ul_avx512(const struct bblib_rate_match_ul_request *request,
    struct bblib_rate_match_ul_response *response);
//! @}

//! @{
/** \brief HARQ combining for LTE
    \param[in] request structure containing configuration information and input data
    \param[out] response structure containing kernel outputs
    \return Success: return 0, else: return -1
 */
int32_t
bblib_harq_combine_ul(const struct bblib_harq_combine_ul_request *request,
        struct bblib_harq_combine_ul_response *response);

int32_t bblib_harq_combine_ul_avx2(const struct bblib_harq_combine_ul_request *request,
        struct bblib_harq_combine_ul_response *response);
int32_t bblib_harq_combine_ul_avx512(const struct bblib_harq_combine_ul_request *request,
    struct bblib_harq_combine_ul_response *response);
//! @}

//! @{
/** \brief Subblock deinterleaving for LTE.

    Use circular_buffer_format in request struct to indicate type of input buffer:
    - BBLIB_CIRCULAR_BUFFER_WITHOUT_PADDING - for circular buffer without dummy bits;
    - BBLIB_FULL_CIRCULAR_BUFFER for full circular buffer (with dummy padding bits).

    \param[in] request structure containing configuration information and input data
    \param[out] response structure containing kernel outputs
    \return Success: return 0, else: return -1
 */
int32_t
bblib_deinterleave_ul(const struct bblib_deinterleave_ul_request *request,
        struct bblib_deinterleave_ul_response *response);

int32_t bblib_deinterleave_ul_avx2(const struct bblib_deinterleave_ul_request *request,
        struct bblib_deinterleave_ul_response *response);
int32_t bblib_deinterleave_ul_avx512(const struct bblib_deinterleave_ul_request *request,
    struct bblib_deinterleave_ul_response *response);
//! @}

//! @{
/** \brief Data formatting for turbo decoding for LTE
    \param[in] request structure containing configuration information and input data
    \param[out] response structure containing kernel outputs
    \return Success: return 0, else: return -1
 */
int32_t
bblib_turbo_adapter_ul(const struct bblib_turbo_adapter_ul_request *request,
        struct bblib_turbo_adapter_ul_response *response);

int32_t bblib_turbo_adapter_ul_avx2(const struct bblib_turbo_adapter_ul_request *request,
        struct bblib_turbo_adapter_ul_response *response);
int32_t bblib_turbo_adapter_ul_avx512(const struct bblib_turbo_adapter_ul_request *request,
    struct bblib_turbo_adapter_ul_response *response);

//! @}

/*! \brief Report the version number for the rate match library.
    \param [in] version Pointer to a char buffer where the version string should be copied.
    \param [in] buffer_size The length of the string buffer, typically no more than
           BBLIB_SDK_VERSION_STRING_MAX_LEN characters.
    \return 0 if the version string was populated, otherwise -1.
 */
int16_t bblib_rate_match_version(char *version, int buffer_size);

#ifdef __cplusplus
}
#endif

#endif
