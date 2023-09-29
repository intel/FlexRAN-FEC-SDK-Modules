#include "ldpc_encoder_cycshift.h"

#ifdef _BBLIB_AVX512_
/* Cycle Shift AVX512 implementation when Zc Size > 288 and is multiple of 32bits */
inline __m512i cycle_bit_left_shift_from288to384(__m512i data, int16_t cycLeftShift, int16_t zcSize, int8_t zcIndex, __m512i swapIdx0)
{
    __m512i x1,x2,x3,x4;
    int16_t cycleLeftShift1 = cycLeftShift >> 5;
    int32_t cycleLeftShift2 = cycLeftShift & 0x1f;
    __m512i swapIdx1 = _mm512_loadu_si512 ((void const*)(permuteTableFrom288to384[zcIndex] + cycleLeftShift1));
    //left shift cycleLeftShift1
    x1 = _mm512_permutex2var_epi32 (data, swapIdx1, data);
    x2 = _mm512_permutex2var_epi32 (x1, swapIdx0, x1);
    x3 = _mm512_srli_epi32 (x1, cycleLeftShift2);
    x4 = _mm512_slli_epi32 (x2, 32 - cycleLeftShift2);
    x1 = _mm512_or_epi64 (x3, x4);
    return x1;
}

/* Cycle Shift AVX512 implementation when Zc Size is multiple of 16bits */
inline __m512i cycle_bit_left_shift_from144to256(__m512i data, int16_t cycLeftShift, int16_t zcSize, int8_t zcIndex, __m512i swapIdx0)
{
    __m512i x1,x2,x3,x4;
    int16_t cycleLeftShift1;
    int32_t cycleLeftShift2;
    __m256i swapIdx11;
    __m512i swapIdx1;

    // Reduce the circular shift from H_BG(I_LS) based on actual Lifting factor
    while (cycLeftShift > zcSize)
        cycLeftShift -= zcSize; // cycLeftShift % zcSize
    cycleLeftShift1 = cycLeftShift >> 4;
    cycleLeftShift2 = cycLeftShift & 0xf;

    if (zcSize > 128)
        swapIdx11 = _mm256_loadu_si256 ((__m256i const*)(permuteTableFrom144to256[zcIndex] + cycleLeftShift1));
    else
        swapIdx11 = _mm256_loadu_si256 ((__m256i const*)(permuteTabUpto128[zcIndex] + cycleLeftShift1));
    swapIdx1 = _mm512_broadcast_i32x8 (swapIdx11);
    swapIdx1 = _mm512_mask_add_epi16 (swapIdx1, 0xffff0000, swapIdx1, _mm512_set1_epi16(16));
    //left shift cycleLeftShift1
    x1 = _mm512_permutex2var_epi16 (data, swapIdx1, data);
    x2 = _mm512_permutex2var_epi16 (x1, swapIdx0, x1);
    x3 = _mm512_srli_epi16 (x1, cycleLeftShift2);
    x4 = _mm512_slli_epi16 (x2, 16 - cycleLeftShift2);
    x1 = _mm512_or_epi32 (x3, x4);
    return x1;
}

// Cycle shift in AVX512 when Zc size <=64 to be able to use directly _mm512_srli_epi64
inline __m512i cycle_bit_left_shift_less_than_64(__m512i data, int16_t cycLeftShift, int16_t zcSize, int8_t zcIndex_, __m512i swapIdx0_)
{
    __m512i x1,x2,bitMask;
    // Reduce the circular shift from H_BG(I_LS) based on actual Lifting factor
    cycLeftShift = cycLeftShift % zcSize;
    int e0,e1,e2;
    
    if (zcSize >= 64) {
        e0 = 0xFFFFFFFF;
        e1 = 0xFFFFFFFF;
        e2 = (1 << (zcSize - 64)) - 1;
    } else if (zcSize >= 32) {
        e0 = 0xFFFFFFFF;
        e1 = (1 << (zcSize - 32)) - 1;
        e2 = 0;
    } else {      
        e0 = (1 << zcSize) - 1;
        e1 = 0;
        e2 = 0;
    }
    bitMask = _mm512_set_epi32 (0,0,0,0,0,0,0,0,0,0,0,0,0,e2,e1,e0);
    data = _mm512_and_si512 (data, bitMask);
    //left shift cycleLeftShift1
    x1 = _mm512_srli_epi64 (data, cycLeftShift);
    x2 = _mm512_slli_epi64 (data, zcSize-cycLeftShift);
    x1 = _mm512_or_epi64 (x1, x2);
    x1 = _mm512_and_si512 (x1, bitMask);
    return x1;
}

/* This is only to cover the case when Zc>64 (and hence _mm512_srli_epi64 cannot be directly used) 
   and when Zc is not a multiple of 16 and hence method with 2 level shift using _mm512_srli_epi16 cannot be used 
   In effect only used when Zc is 88, 104, 120                                                         */
inline __m512i cycle_bit_left_shift_special(__m512i data, int16_t cycLeftShift, int16_t zcSize, int8_t zcIndex_, __m512i swapIdx0_)
{
    __m512i x1;
    int zcSizeInBytes = zcSize >> 3;
    uint8_t* pBytesData = (uint8_t *) &data;
    uint8_t* pBytesX1 = (uint8_t *) &x1;
    cycLeftShift = cycLeftShift % zcSize;
    int cycRightShift = (zcSize - cycLeftShift) % zcSize;
    int byteShift1 = cycRightShift >> 3;
    int byteShift2 = byteShift1 + 1;
    int bitShift = cycRightShift - (byteShift1 << 3);
    for (auto byteIndex = 0; byteIndex < zcSizeInBytes; byteIndex++){
        int topByte    = (byteIndex + zcSizeInBytes - byteShift1) % zcSizeInBytes;
        int botByte    = (byteIndex + zcSizeInBytes - byteShift2) % zcSizeInBytes;
        pBytesX1[byteIndex] = (pBytesData[topByte] << bitShift) | (pBytesData[botByte] >> (8 - bitShift));
    }
    for (auto byteIndex=zcSizeInBytes; byteIndex < 16;byteIndex++){
        pBytesX1[byteIndex] = 0;
    }
    return x1;
}


CYCLE_BIT_LEFT_SHIFT ldpc_select_left_shift_func(int16_t zcSize)
{
    if (zcSize >= 288)
        return cycle_bit_left_shift_from288to384;
    else if (zcSize < 64)
        return cycle_bit_left_shift_less_than_64;
    else if (zcSize == 72 || zcSize == 88 || zcSize == 104 || zcSize == 120)
        return cycle_bit_left_shift_special;
    else 
        return cycle_bit_left_shift_from144to256;
}
#endif