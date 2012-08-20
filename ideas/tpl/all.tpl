{{=<% %>=}}

// SSE requirements
#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>

template <int bits>
class BitCompression
{

public:

    template <int offset, int block>
    inline static void decompress_block(const __m128i& data, int* output);

    template<int offset>
    inline static int remaining();

    template<int offset>
    inline static int next_offset();

    template<int offset>
    inline static const int base_shift();

    template<int offset>
    inline static int block_count();

    template<int offset>
    inline static void overlap_value(const __m128i& data, int* output);

    template<int offset, int block>
    inline static int per_block();

    template<int offset>
    inline static void decompress(const __m128i* data, int* out);
};



<%#data%>
////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: <%base_shift%>
//Offset: <%offset%>
//Bits: <%bits%>


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<<%bits%>>::remaining<<%offset%>>() { return <%elements%>; }

template<>
template<>
inline const int BitCompression<<%bits%>>::base_shift<<%offset%>>() { return <%base_shift%>; }

template<>
template<>
inline int BitCompression<<%bits%>>::next_offset<<%offset%>>() { return <%next_offset%>; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<<%bits%>>::block_count<<%offset%>>() { return <%num_extracts%>; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<<%bits%>>::overlap_value<<%offset%>>(const __m128i& data, int* output)
{
    static const __m128i shuffle_mask = {0x8080808003020100, 0x8080808080808080};
    static const __m128i and_mask = {0x1f, 0x0};
    __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    __m128i shift_right = _mm_srli_epi32(shuffeled, <%offset%>);
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shift_right, and_mask));
    //int64_t v = _mm_extract_epi32(data, 0);
    //*output = (v >> <%offset%>) & 0x1f;
}

<%#extracts%>
template<>
template<>
inline void BitCompression<<%bits%>>::decompress_block<<%offset%>, <%block%>>(const __m128i& data, int* output)
{
    static const __m128i shuffle_mask = <%shuffle%>;
    static const __m128i mull_mask = <%mullo%>;
    static const __m128i and_mask = {<%and%>, <%and%>};
    static const int shift_mask = <%shift%>;

    __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    __m128i shift_left = _mm_mullo_epi32(shuffeled, mull_mask);
    __m128i shift_right = _mm_srli_epi32(shift_left, shift_mask);

    // Always store aligned
    _mm_store_si128((__m128i*) output, _mm_and_si128(shift_right, and_mask));

    // We should never store unaligned
    //_mm_storeu_si128((__m128i*) output, _mm_and_si128(shift_right, and_mask));
}

template<>
template<>
inline int BitCompression<<%bits%>>::per_block<<%offset%>, <%block%>>()
{ 
     return <%block_elements%>;
}

<%/extracts%>

template<>
template<>
inline void BitCompression<<%bits%>>::decompress<<%offset%>>(const __m128i* block, int* out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);;
    <%#extracts%>
    BitCompression::decompress_block<<%offset%>, <%block%>>(qw_block, ctr);
    ctr += BitCompression::per_block<<%offset%>, <%block%>>();

    <%/extracts%>

    // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  <%base_shift%>);
    BitCompression::overlap_value<<%offset%>>(tmp, ctr);    
}


<%/data%>
