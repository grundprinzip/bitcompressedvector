#ifndef BCV_VERTICAL_BIT_COMPRESSION
#define BCV_VERTICAL_BIT_COMPRESSION

// SSE requirements
#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>

template <int bits>
class VerticalBitCompression
{

public:

    template<int offset>
    inline static void decompress_block(const __m128i* __restrict__ data, int* __restrict__ out, size_t* __restrict__ counter);

    template<int offset>
    inline static void cmp_eq_block(const __m128i* __restrict__ data, const int cmp, int* __restrict__ out, size_t* __restrict__ counter);

    inline static void decompress(const __m128i* __restrict__ data, int* __restrict__ out, size_t* __restrict__ counter);

    inline static void cmp_eq(const __m128i* __restrict__ data, const int cmp,  int* __restrict__ out, size_t* __restrict__ counter);
};


template<>
template<>
inline void VerticalBitCompression<1>::decompress_block<0>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x100000001, 0x100000001};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 1);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 3);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 5);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 7);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 9);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 11);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 13);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 15);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 17);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 19);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 21);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 23);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 25);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 27);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 29);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 31);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;


    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<1>::cmp_eq_block<0>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x100000001, 0x100000001};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 1);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 3);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 5);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 7);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 9);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 11);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 13);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 15);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 17);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 19);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 21);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 23);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 25);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 27);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 29);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 31);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;


    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<2>::decompress_block<0>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x300000003, 0x300000003};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;


    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<2>::cmp_eq_block<0>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x300000003, 0x300000003};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;


    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<3>::decompress_block<0>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x700000007, 0x700000007};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 3);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 9);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 15);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 21);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 27);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 2), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<3>::cmp_eq_block<0>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x700000007, 0x700000007};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 3);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 9);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 15);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 21);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 27);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 2), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<3>::decompress_block<1>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x700000007, 0x700000007};


    tmp = _mm_srli_epi32(inValue, 1);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 7);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 13);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 19);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 25);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 31);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 1), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<3>::cmp_eq_block<1>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x700000007, 0x700000007};


    tmp = _mm_srli_epi32(inValue, 1);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 7);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 13);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 19);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 25);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 31);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 1), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<3>::decompress_block<2>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x700000007, 0x700000007};


    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 5);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 11);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 17);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 23);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 29);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;


    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<3>::cmp_eq_block<2>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x700000007, 0x700000007};


    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 5);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 11);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 17);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 23);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 29);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;


    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<4>::decompress_block<0>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0xf0000000f, 0xf0000000f};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;


    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<4>::cmp_eq_block<0>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0xf0000000f, 0xf0000000f};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;


    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<5>::decompress_block<0>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1f0000001f, 0x1f0000001f};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 5);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 15);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 25);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 2), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<5>::cmp_eq_block<0>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1f0000001f, 0x1f0000001f};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 5);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 15);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 25);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 2), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<5>::decompress_block<3>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1f0000001f, 0x1f0000001f};


    tmp = _mm_srli_epi32(inValue, 3);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 13);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 23);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 4), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<5>::cmp_eq_block<3>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1f0000001f, 0x1f0000001f};


    tmp = _mm_srli_epi32(inValue, 3);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 13);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 23);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 4), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<5>::decompress_block<1>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1f0000001f, 0x1f0000001f};


    tmp = _mm_srli_epi32(inValue, 1);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 11);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 21);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 31);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 1), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<5>::cmp_eq_block<1>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1f0000001f, 0x1f0000001f};


    tmp = _mm_srli_epi32(inValue, 1);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 11);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 21);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 31);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 1), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<5>::decompress_block<4>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1f0000001f, 0x1f0000001f};


    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 9);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 19);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 29);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 3), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<5>::cmp_eq_block<4>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1f0000001f, 0x1f0000001f};


    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 9);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 19);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 29);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 3), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<5>::decompress_block<2>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1f0000001f, 0x1f0000001f};


    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 7);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 17);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 27);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;


    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<5>::cmp_eq_block<2>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1f0000001f, 0x1f0000001f};


    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 7);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 17);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 27);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;


    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<6>::decompress_block<0>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x3f0000003f, 0x3f0000003f};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 2), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<6>::cmp_eq_block<0>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x3f0000003f, 0x3f0000003f};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 2), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<6>::decompress_block<4>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x3f0000003f, 0x3f0000003f};


    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 4), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<6>::cmp_eq_block<4>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x3f0000003f, 0x3f0000003f};


    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 4), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<6>::decompress_block<2>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x3f0000003f, 0x3f0000003f};


    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;


    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<6>::cmp_eq_block<2>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x3f0000003f, 0x3f0000003f};


    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;


    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<7>::decompress_block<0>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7f0000007f, 0x7f0000007f};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 7);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 21);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 4), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<7>::cmp_eq_block<0>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7f0000007f, 0x7f0000007f};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 7);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 21);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 4), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<7>::decompress_block<3>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7f0000007f, 0x7f0000007f};


    tmp = _mm_srli_epi32(inValue, 3);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 17);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 31);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 1), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<7>::cmp_eq_block<3>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7f0000007f, 0x7f0000007f};


    tmp = _mm_srli_epi32(inValue, 3);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 17);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 31);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 1), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<7>::decompress_block<6>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7f0000007f, 0x7f0000007f};


    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 13);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 27);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 5), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<7>::cmp_eq_block<6>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7f0000007f, 0x7f0000007f};


    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 13);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 27);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 5), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<7>::decompress_block<2>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7f0000007f, 0x7f0000007f};


    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 9);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 23);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 2), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<7>::cmp_eq_block<2>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7f0000007f, 0x7f0000007f};


    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 9);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 23);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 2), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<7>::decompress_block<5>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7f0000007f, 0x7f0000007f};


    tmp = _mm_srli_epi32(inValue, 5);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 19);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 6), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<7>::cmp_eq_block<5>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7f0000007f, 0x7f0000007f};


    tmp = _mm_srli_epi32(inValue, 5);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 19);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 6), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<7>::decompress_block<1>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7f0000007f, 0x7f0000007f};


    tmp = _mm_srli_epi32(inValue, 1);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 15);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 29);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 3), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<7>::cmp_eq_block<1>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7f0000007f, 0x7f0000007f};


    tmp = _mm_srli_epi32(inValue, 1);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 15);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 29);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 3), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<7>::decompress_block<4>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7f0000007f, 0x7f0000007f};


    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 11);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 25);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;


    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<7>::cmp_eq_block<4>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7f0000007f, 0x7f0000007f};


    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 11);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 25);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;


    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<8>::decompress_block<0>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0xff000000ff, 0xff000000ff};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;


    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<8>::cmp_eq_block<0>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0xff000000ff, 0xff000000ff};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;


    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<9>::decompress_block<0>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1ff000001ff, 0x1ff000001ff};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 9);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 27);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 5), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<9>::cmp_eq_block<0>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1ff000001ff, 0x1ff000001ff};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 9);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 27);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 5), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<9>::decompress_block<4>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1ff000001ff, 0x1ff000001ff};


    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 13);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 31);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 1), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<9>::cmp_eq_block<4>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1ff000001ff, 0x1ff000001ff};


    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 13);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 31);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 1), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<9>::decompress_block<8>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1ff000001ff, 0x1ff000001ff};


    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 17);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 6), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<9>::cmp_eq_block<8>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1ff000001ff, 0x1ff000001ff};


    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 17);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 6), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<9>::decompress_block<3>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1ff000001ff, 0x1ff000001ff};


    tmp = _mm_srli_epi32(inValue, 3);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 21);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 2), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<9>::cmp_eq_block<3>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1ff000001ff, 0x1ff000001ff};


    tmp = _mm_srli_epi32(inValue, 3);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 21);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 2), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<9>::decompress_block<7>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1ff000001ff, 0x1ff000001ff};


    tmp = _mm_srli_epi32(inValue, 7);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 25);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 7), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<9>::cmp_eq_block<7>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1ff000001ff, 0x1ff000001ff};


    tmp = _mm_srli_epi32(inValue, 7);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 25);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 7), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<9>::decompress_block<2>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1ff000001ff, 0x1ff000001ff};


    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 11);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 29);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 3), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<9>::cmp_eq_block<2>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1ff000001ff, 0x1ff000001ff};


    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 11);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 29);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 3), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<9>::decompress_block<6>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1ff000001ff, 0x1ff000001ff};


    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 15);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 8), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<9>::cmp_eq_block<6>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1ff000001ff, 0x1ff000001ff};


    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 15);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 8), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<9>::decompress_block<1>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1ff000001ff, 0x1ff000001ff};


    tmp = _mm_srli_epi32(inValue, 1);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 19);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 4), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<9>::cmp_eq_block<1>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1ff000001ff, 0x1ff000001ff};


    tmp = _mm_srli_epi32(inValue, 1);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 19);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 4), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<9>::decompress_block<5>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1ff000001ff, 0x1ff000001ff};


    tmp = _mm_srli_epi32(inValue, 5);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 23);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;


    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<9>::cmp_eq_block<5>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1ff000001ff, 0x1ff000001ff};


    tmp = _mm_srli_epi32(inValue, 5);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 23);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;


    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<10>::decompress_block<0>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x3ff000003ff, 0x3ff000003ff};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 2), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<10>::cmp_eq_block<0>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x3ff000003ff, 0x3ff000003ff};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 2), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<10>::decompress_block<8>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x3ff000003ff, 0x3ff000003ff};


    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 4), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<10>::cmp_eq_block<8>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x3ff000003ff, 0x3ff000003ff};


    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 4), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<10>::decompress_block<6>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x3ff000003ff, 0x3ff000003ff};


    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 6), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<10>::cmp_eq_block<6>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x3ff000003ff, 0x3ff000003ff};


    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 6), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<10>::decompress_block<4>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x3ff000003ff, 0x3ff000003ff};


    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 8), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<10>::cmp_eq_block<4>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x3ff000003ff, 0x3ff000003ff};


    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 8), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<10>::decompress_block<2>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x3ff000003ff, 0x3ff000003ff};


    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;


    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<10>::cmp_eq_block<2>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x3ff000003ff, 0x3ff000003ff};


    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;


    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<11>::decompress_block<0>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7ff000007ff, 0x7ff000007ff};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 11);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 10), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<11>::cmp_eq_block<0>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7ff000007ff, 0x7ff000007ff};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 11);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 10), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<11>::decompress_block<1>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7ff000007ff, 0x7ff000007ff};


    tmp = _mm_srli_epi32(inValue, 1);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 23);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 9), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<11>::cmp_eq_block<1>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7ff000007ff, 0x7ff000007ff};


    tmp = _mm_srli_epi32(inValue, 1);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 23);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 9), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<11>::decompress_block<2>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7ff000007ff, 0x7ff000007ff};


    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 13);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 8), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<11>::cmp_eq_block<2>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7ff000007ff, 0x7ff000007ff};


    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 13);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 8), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<11>::decompress_block<3>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7ff000007ff, 0x7ff000007ff};


    tmp = _mm_srli_epi32(inValue, 3);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 25);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 7), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<11>::cmp_eq_block<3>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7ff000007ff, 0x7ff000007ff};


    tmp = _mm_srli_epi32(inValue, 3);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 25);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 7), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<11>::decompress_block<4>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7ff000007ff, 0x7ff000007ff};


    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 15);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 6), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<11>::cmp_eq_block<4>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7ff000007ff, 0x7ff000007ff};


    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 15);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 6), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<11>::decompress_block<5>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7ff000007ff, 0x7ff000007ff};


    tmp = _mm_srli_epi32(inValue, 5);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 27);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 5), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<11>::cmp_eq_block<5>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7ff000007ff, 0x7ff000007ff};


    tmp = _mm_srli_epi32(inValue, 5);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 27);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 5), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<11>::decompress_block<6>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7ff000007ff, 0x7ff000007ff};


    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 17);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 4), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<11>::cmp_eq_block<6>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7ff000007ff, 0x7ff000007ff};


    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 17);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 4), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<11>::decompress_block<7>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7ff000007ff, 0x7ff000007ff};


    tmp = _mm_srli_epi32(inValue, 7);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 29);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 3), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<11>::cmp_eq_block<7>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7ff000007ff, 0x7ff000007ff};


    tmp = _mm_srli_epi32(inValue, 7);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 29);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 3), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<11>::decompress_block<8>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7ff000007ff, 0x7ff000007ff};


    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 19);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 2), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<11>::cmp_eq_block<8>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7ff000007ff, 0x7ff000007ff};


    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 19);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 2), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<11>::decompress_block<9>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7ff000007ff, 0x7ff000007ff};


    tmp = _mm_srli_epi32(inValue, 9);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 31);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 1), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<11>::cmp_eq_block<9>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7ff000007ff, 0x7ff000007ff};


    tmp = _mm_srli_epi32(inValue, 9);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 31);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 1), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<11>::decompress_block<10>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7ff000007ff, 0x7ff000007ff};


    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 21);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;


    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<11>::cmp_eq_block<10>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7ff000007ff, 0x7ff000007ff};


    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 21);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;


    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<12>::decompress_block<0>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0xfff00000fff, 0xfff00000fff};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 8), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<12>::cmp_eq_block<0>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0xfff00000fff, 0xfff00000fff};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 8), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<12>::decompress_block<4>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0xfff00000fff, 0xfff00000fff};


    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 4), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<12>::cmp_eq_block<4>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0xfff00000fff, 0xfff00000fff};


    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 4), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<12>::decompress_block<8>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0xfff00000fff, 0xfff00000fff};


    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;


    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<12>::cmp_eq_block<8>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0xfff00000fff, 0xfff00000fff};


    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;


    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<13>::decompress_block<0>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 13);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 6), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<13>::cmp_eq_block<0>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 13);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 6), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<13>::decompress_block<7>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 7);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 12), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<13>::cmp_eq_block<7>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 7);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 12), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<13>::decompress_block<1>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 1);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 27);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 5), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<13>::cmp_eq_block<1>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 1);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 27);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 5), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<13>::decompress_block<8>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 21);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 11), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<13>::cmp_eq_block<8>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 21);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 11), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<13>::decompress_block<2>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 15);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 4), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<13>::cmp_eq_block<2>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 15);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 4), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<13>::decompress_block<9>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 9);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 10), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<13>::cmp_eq_block<9>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 9);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 10), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<13>::decompress_block<3>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 3);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 29);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 3), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<13>::cmp_eq_block<3>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 3);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 29);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 3), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<13>::decompress_block<10>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 23);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 9), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<13>::cmp_eq_block<10>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 23);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 9), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<13>::decompress_block<4>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 17);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 2), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<13>::cmp_eq_block<4>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 17);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 2), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<13>::decompress_block<11>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 11);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 8), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<13>::cmp_eq_block<11>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 11);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 8), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<13>::decompress_block<5>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 5);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 31);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 1), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<13>::cmp_eq_block<5>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 5);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 31);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 1), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<13>::decompress_block<12>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 25);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 7), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<13>::cmp_eq_block<12>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 25);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 7), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<13>::decompress_block<6>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 19);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;


    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<13>::cmp_eq_block<6>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x1fff00001fff, 0x1fff00001fff};


    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 19);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;


    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<14>::decompress_block<0>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x3fff00003fff, 0x3fff00003fff};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 4), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<14>::cmp_eq_block<0>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x3fff00003fff, 0x3fff00003fff};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 4), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<14>::decompress_block<10>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x3fff00003fff, 0x3fff00003fff};


    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 8), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<14>::cmp_eq_block<10>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x3fff00003fff, 0x3fff00003fff};


    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 8), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<14>::decompress_block<6>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x3fff00003fff, 0x3fff00003fff};


    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 12), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<14>::cmp_eq_block<6>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x3fff00003fff, 0x3fff00003fff};


    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 12), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<14>::decompress_block<2>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x3fff00003fff, 0x3fff00003fff};


    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 2), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<14>::cmp_eq_block<2>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x3fff00003fff, 0x3fff00003fff};


    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 2), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<14>::decompress_block<12>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x3fff00003fff, 0x3fff00003fff};


    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 6), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<14>::cmp_eq_block<12>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x3fff00003fff, 0x3fff00003fff};


    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 6), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<14>::decompress_block<8>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x3fff00003fff, 0x3fff00003fff};


    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 10), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<14>::cmp_eq_block<8>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x3fff00003fff, 0x3fff00003fff};


    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 10), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<14>::decompress_block<4>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x3fff00003fff, 0x3fff00003fff};


    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;


    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<14>::cmp_eq_block<4>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x3fff00003fff, 0x3fff00003fff};


    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;


    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<15>::decompress_block<0>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 15);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 2), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<15>::cmp_eq_block<0>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 15);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 30);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 2), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<15>::decompress_block<13>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 13);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 4), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<15>::cmp_eq_block<13>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 13);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 28);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 4), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<15>::decompress_block<11>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 11);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 6), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<15>::cmp_eq_block<11>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 11);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 26);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 6), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<15>::decompress_block<9>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 9);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 8), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<15>::cmp_eq_block<9>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 9);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 24);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 8), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<15>::decompress_block<7>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 7);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 10), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<15>::cmp_eq_block<7>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 7);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 22);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 10), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<15>::decompress_block<5>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 5);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 12), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<15>::cmp_eq_block<5>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 5);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 20);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 12), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<15>::decompress_block<3>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 3);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 14), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<15>::cmp_eq_block<3>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 3);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 18);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 14), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<15>::decompress_block<1>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 1);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 31);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 1), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<15>::cmp_eq_block<1>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 1);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 31);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 1), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<15>::decompress_block<14>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 29);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 3), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<15>::cmp_eq_block<14>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 14);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 29);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 3), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<15>::decompress_block<12>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 27);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 5), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<15>::cmp_eq_block<12>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 12);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 27);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 5), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<15>::decompress_block<10>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 25);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 7), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<15>::cmp_eq_block<10>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 10);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 25);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 7), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<15>::decompress_block<8>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 23);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 9), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<15>::cmp_eq_block<8>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 8);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 23);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 9), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<15>::decompress_block<6>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 21);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 11), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<15>::cmp_eq_block<6>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 6);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 21);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 11), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<15>::decompress_block<4>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 19);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 13), mask)));
    counter += 4;

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<15>::cmp_eq_block<4>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 4);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 19);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, 13), mask)), cmpMask));
    counter += 4;

    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<15>::decompress_block<2>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 17);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;


    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<15>::cmp_eq_block<2>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0x7fff00007fff, 0x7fff00007fff};


    tmp = _mm_srli_epi32(inValue, 2);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 17);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;


    *counter_out = counter;
}


template<>
template<>
inline void VerticalBitCompression<16>::decompress_block<0>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = {0xffff0000ffff, 0xffff0000ffff};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, outValue);
    counter += 4;


    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<16>::cmp_eq_block<0>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = {0xffff0000ffff, 0xffff0000ffff};


    outValue = _mm_and_si128(inValue, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;

    tmp = _mm_srli_epi32(inValue, 16);
    outValue = _mm_and_si128(tmp, mask);

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;


    *counter_out = counter;
}


template<>
inline void VerticalBitCompression<1>::decompress(const __m128i* __restrict__ data, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<1>::decompress_block<0>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}

template<>
inline void VerticalBitCompression<1>::cmp_eq(const __m128i* __restrict__ data, const int cmp, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<1>::cmp_eq_block<0>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}
template<>
inline void VerticalBitCompression<2>::decompress(const __m128i* __restrict__ data, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<2>::decompress_block<0>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}

template<>
inline void VerticalBitCompression<2>::cmp_eq(const __m128i* __restrict__ data, const int cmp, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<2>::cmp_eq_block<0>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}
template<>
inline void VerticalBitCompression<3>::decompress(const __m128i* __restrict__ data, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<3>::decompress_block<0>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<3>::decompress_block<1>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<3>::decompress_block<2>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}

template<>
inline void VerticalBitCompression<3>::cmp_eq(const __m128i* __restrict__ data, const int cmp, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<3>::cmp_eq_block<0>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<3>::cmp_eq_block<1>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<3>::cmp_eq_block<2>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}
template<>
inline void VerticalBitCompression<4>::decompress(const __m128i* __restrict__ data, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<4>::decompress_block<0>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}

template<>
inline void VerticalBitCompression<4>::cmp_eq(const __m128i* __restrict__ data, const int cmp, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<4>::cmp_eq_block<0>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}
template<>
inline void VerticalBitCompression<5>::decompress(const __m128i* __restrict__ data, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<5>::decompress_block<0>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<5>::decompress_block<3>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<5>::decompress_block<1>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<5>::decompress_block<4>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<5>::decompress_block<2>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}

template<>
inline void VerticalBitCompression<5>::cmp_eq(const __m128i* __restrict__ data, const int cmp, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<5>::cmp_eq_block<0>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<5>::cmp_eq_block<3>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<5>::cmp_eq_block<1>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<5>::cmp_eq_block<4>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<5>::cmp_eq_block<2>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}
template<>
inline void VerticalBitCompression<6>::decompress(const __m128i* __restrict__ data, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<6>::decompress_block<0>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<6>::decompress_block<4>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<6>::decompress_block<2>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}

template<>
inline void VerticalBitCompression<6>::cmp_eq(const __m128i* __restrict__ data, const int cmp, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<6>::cmp_eq_block<0>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<6>::cmp_eq_block<4>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<6>::cmp_eq_block<2>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}
template<>
inline void VerticalBitCompression<7>::decompress(const __m128i* __restrict__ data, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<7>::decompress_block<0>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<7>::decompress_block<3>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<7>::decompress_block<6>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<7>::decompress_block<2>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<7>::decompress_block<5>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<7>::decompress_block<1>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<7>::decompress_block<4>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}

template<>
inline void VerticalBitCompression<7>::cmp_eq(const __m128i* __restrict__ data, const int cmp, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<7>::cmp_eq_block<0>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<7>::cmp_eq_block<3>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<7>::cmp_eq_block<6>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<7>::cmp_eq_block<2>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<7>::cmp_eq_block<5>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<7>::cmp_eq_block<1>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<7>::cmp_eq_block<4>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}
template<>
inline void VerticalBitCompression<8>::decompress(const __m128i* __restrict__ data, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<8>::decompress_block<0>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}

template<>
inline void VerticalBitCompression<8>::cmp_eq(const __m128i* __restrict__ data, const int cmp, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<8>::cmp_eq_block<0>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}
template<>
inline void VerticalBitCompression<9>::decompress(const __m128i* __restrict__ data, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<9>::decompress_block<0>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<9>::decompress_block<4>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<9>::decompress_block<8>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<9>::decompress_block<3>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<9>::decompress_block<7>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<9>::decompress_block<2>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<9>::decompress_block<6>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<9>::decompress_block<1>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<9>::decompress_block<5>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}

template<>
inline void VerticalBitCompression<9>::cmp_eq(const __m128i* __restrict__ data, const int cmp, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<9>::cmp_eq_block<0>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<9>::cmp_eq_block<4>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<9>::cmp_eq_block<8>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<9>::cmp_eq_block<3>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<9>::cmp_eq_block<7>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<9>::cmp_eq_block<2>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<9>::cmp_eq_block<6>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<9>::cmp_eq_block<1>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<9>::cmp_eq_block<5>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}
template<>
inline void VerticalBitCompression<10>::decompress(const __m128i* __restrict__ data, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<10>::decompress_block<0>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<10>::decompress_block<8>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<10>::decompress_block<6>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<10>::decompress_block<4>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<10>::decompress_block<2>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}

template<>
inline void VerticalBitCompression<10>::cmp_eq(const __m128i* __restrict__ data, const int cmp, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<10>::cmp_eq_block<0>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<10>::cmp_eq_block<8>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<10>::cmp_eq_block<6>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<10>::cmp_eq_block<4>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<10>::cmp_eq_block<2>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}
template<>
inline void VerticalBitCompression<11>::decompress(const __m128i* __restrict__ data, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<11>::decompress_block<0>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<11>::decompress_block<1>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<11>::decompress_block<2>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<11>::decompress_block<3>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<11>::decompress_block<4>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<11>::decompress_block<5>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<11>::decompress_block<6>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<11>::decompress_block<7>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<11>::decompress_block<8>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<11>::decompress_block<9>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<11>::decompress_block<10>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}

template<>
inline void VerticalBitCompression<11>::cmp_eq(const __m128i* __restrict__ data, const int cmp, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<11>::cmp_eq_block<0>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<11>::cmp_eq_block<1>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<11>::cmp_eq_block<2>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<11>::cmp_eq_block<3>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<11>::cmp_eq_block<4>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<11>::cmp_eq_block<5>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<11>::cmp_eq_block<6>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<11>::cmp_eq_block<7>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<11>::cmp_eq_block<8>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<11>::cmp_eq_block<9>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<11>::cmp_eq_block<10>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}
template<>
inline void VerticalBitCompression<12>::decompress(const __m128i* __restrict__ data, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<12>::decompress_block<0>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<12>::decompress_block<4>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<12>::decompress_block<8>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}

template<>
inline void VerticalBitCompression<12>::cmp_eq(const __m128i* __restrict__ data, const int cmp, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<12>::cmp_eq_block<0>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<12>::cmp_eq_block<4>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<12>::cmp_eq_block<8>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}
template<>
inline void VerticalBitCompression<13>::decompress(const __m128i* __restrict__ data, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<13>::decompress_block<0>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::decompress_block<7>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::decompress_block<1>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::decompress_block<8>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::decompress_block<2>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::decompress_block<9>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::decompress_block<3>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::decompress_block<10>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::decompress_block<4>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::decompress_block<11>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::decompress_block<5>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::decompress_block<12>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::decompress_block<6>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}

template<>
inline void VerticalBitCompression<13>::cmp_eq(const __m128i* __restrict__ data, const int cmp, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<13>::cmp_eq_block<0>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::cmp_eq_block<7>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::cmp_eq_block<1>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::cmp_eq_block<8>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::cmp_eq_block<2>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::cmp_eq_block<9>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::cmp_eq_block<3>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::cmp_eq_block<10>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::cmp_eq_block<4>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::cmp_eq_block<11>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::cmp_eq_block<5>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::cmp_eq_block<12>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<13>::cmp_eq_block<6>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}
template<>
inline void VerticalBitCompression<14>::decompress(const __m128i* __restrict__ data, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<14>::decompress_block<0>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<14>::decompress_block<10>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<14>::decompress_block<6>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<14>::decompress_block<2>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<14>::decompress_block<12>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<14>::decompress_block<8>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<14>::decompress_block<4>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}

template<>
inline void VerticalBitCompression<14>::cmp_eq(const __m128i* __restrict__ data, const int cmp, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<14>::cmp_eq_block<0>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<14>::cmp_eq_block<10>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<14>::cmp_eq_block<6>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<14>::cmp_eq_block<2>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<14>::cmp_eq_block<12>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<14>::cmp_eq_block<8>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<14>::cmp_eq_block<4>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}
template<>
inline void VerticalBitCompression<15>::decompress(const __m128i* __restrict__ data, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<15>::decompress_block<0>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::decompress_block<13>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::decompress_block<11>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::decompress_block<9>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::decompress_block<7>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::decompress_block<5>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::decompress_block<3>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::decompress_block<1>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::decompress_block<14>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::decompress_block<12>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::decompress_block<10>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::decompress_block<8>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::decompress_block<6>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::decompress_block<4>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::decompress_block<2>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}

template<>
inline void VerticalBitCompression<15>::cmp_eq(const __m128i* __restrict__ data, const int cmp, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<15>::cmp_eq_block<0>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::cmp_eq_block<13>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::cmp_eq_block<11>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::cmp_eq_block<9>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::cmp_eq_block<7>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::cmp_eq_block<5>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::cmp_eq_block<3>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::cmp_eq_block<1>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::cmp_eq_block<14>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::cmp_eq_block<12>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::cmp_eq_block<10>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::cmp_eq_block<8>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::cmp_eq_block<6>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::cmp_eq_block<4>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    VerticalBitCompression<15>::cmp_eq_block<2>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}
template<>
inline void VerticalBitCompression<16>::decompress(const __m128i* __restrict__ data, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<16>::decompress_block<0>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}

template<>
inline void VerticalBitCompression<16>::cmp_eq(const __m128i* __restrict__ data, const int cmp, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    VerticalBitCompression<16>::cmp_eq_block<0>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    *counter = counter_out;
}

#endif

