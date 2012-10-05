{{=<% %>=}}
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

<%#bits%>

template<>
template<>
inline void VerticalBitCompression<<%bits%>>::decompress_block<<%offset%>>(const __m128i* __restrict__ data, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i mask = <%mask%>;

    <%#extracts%>

    <%#use_shift%>
    tmp = _mm_srli_epi32(inValue, <%shift%>);
    outValue = _mm_and_si128(tmp, mask);
    <%/use_shift%>
    <%#no_shift%>
    outValue = _mm_and_si128(inValue, mask);
    <%/no_shift%>

    _mm_store_si128(moving++, outValue);
    counter += 4;
    <%/extracts%>

    <%#has_overlap%>
    tmp = _mm_srli_epi32(inValue, <%shift%>);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, <%shift_left%>), mask)));
    counter += 4;
    <%/has_overlap%>

    *counter_out = counter;
}

template<>
template<>
inline void VerticalBitCompression<<%bits%>>::cmp_eq_block<<%offset%>>(const __m128i* __restrict__ data, const int cmp, int* out, size_t* __restrict__ counter_out) 
{ 
    __m128i* moving = reinterpret_cast<__m128i*>(out);
    register __m128i inValue = _mm_load_si128(data);
    register __m128i outValue;
    register __m128i tmp;
    register size_t counter = 0;

    const register __m128i cmpMask = __m128i{static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32), static_cast<int64_t>(cmp) + (static_cast<int64_t>(cmp) << 32)};
    const register __m128i mask = <%mask%>;

    <%#extracts%>

    <%#use_shift%>
    tmp = _mm_srli_epi32(inValue, <%shift%>);
    outValue = _mm_and_si128(tmp, mask);
    <%/use_shift%>
    <%#no_shift%>
    outValue = _mm_and_si128(inValue, mask);
    <%/no_shift%>

    _mm_store_si128(moving++, _mm_cmpeq_epi32(outValue, cmpMask));
    counter += 4;
    <%/extracts%>

    <%#has_overlap%>
    tmp = _mm_srli_epi32(inValue, <%shift%>);
    ++data;
    inValue = _mm_load_si128(data);
    _mm_store_si128(moving++, _mm_cmpeq_epi32(_mm_or_si128(tmp, _mm_and_si128(_mm_slli_epi32(inValue, <%shift_left%>), mask)), cmpMask));
    counter += 4;
    <%/has_overlap%>

    *counter_out = counter;
}

<%/bits%>

<%#blocks%>
template<>
inline void VerticalBitCompression<<%bits%>>::decompress(const __m128i* __restrict__ data, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    <%#offsets%>
    VerticalBitCompression<<%bits%>>::decompress_block<<%offset%>>(data++, out+counter_out, &tmp);
    counter_out += tmp;
    <%/offsets%>
    *counter = counter_out;
}

template<>
inline void VerticalBitCompression<<%bits%>>::cmp_eq(const __m128i* __restrict__ data, const int cmp, int* __restrict__ out, size_t* __restrict__ counter)
{
    size_t counter_out = 0;
    size_t tmp;
    <%#offsets%>
    VerticalBitCompression<<%bits%>>::cmp_eq_block<<%offset%>>(data++, cmp,  out+counter_out, &tmp);
    counter_out += tmp;
    <%/offsets%>
    *counter = counter_out;
}
<%/blocks%>

#endif
