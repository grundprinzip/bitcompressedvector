#ifndef BCV_BITCOMPRESS_HORIZONTAL
#define BCV_BITCOMPRESS_HORIZONTAL

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
    inline static void overlap_value(const __m128i& data, int* __restrict__ output);

    template<int offset, int block>
    inline static int per_block();

    template<int offset>
    inline static void decompress(const __m128i* data, int* __restrict__ out);

    inline static void decompress_large(const __m128i* data, int* __restrict__ out, size_t* __restrict__ counter_out);
};




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 16
//Offset: 0
//Bits: 1


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<1>::remaining<0>() { return 128; }

template<>
template<>
inline const int BitCompression<1>::base_shift<0>() { return 16; }

template<>
template<>
inline int BitCompression<1>::next_offset<0>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<1>::block_count<0>() { return 32; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<1>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 0) & 0x1;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800080808000ull), static_cast<long long int>(0x8080800080808000ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800080808000ull), static_cast<long long int>(0x8080800080808000ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800180808001ull), static_cast<long long int>(0x8080800180808001ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800180808001ull), static_cast<long long int>(0x8080800180808001ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 3>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 4>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800280808002ull), static_cast<long long int>(0x8080800280808002ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 4>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 5>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800280808002ull), static_cast<long long int>(0x8080800280808002ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 5>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 6>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800380808003ull), static_cast<long long int>(0x8080800380808003ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 6>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 7>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800380808003ull), static_cast<long long int>(0x8080800380808003ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 7>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 8>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800480808004ull), static_cast<long long int>(0x8080800480808004ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 8>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 9>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800480808004ull), static_cast<long long int>(0x8080800480808004ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 9>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 10>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800580808005ull), static_cast<long long int>(0x8080800580808005ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 10>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 11>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800580808005ull), static_cast<long long int>(0x8080800580808005ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 11>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 12>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800680808006ull), static_cast<long long int>(0x8080800680808006ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 12>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 13>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800680808006ull), static_cast<long long int>(0x8080800680808006ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 13>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 14>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800780808007ull), static_cast<long long int>(0x8080800780808007ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 14>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 15>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800780808007ull), static_cast<long long int>(0x8080800780808007ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 15>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 16>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800880808008ull), static_cast<long long int>(0x8080800880808008ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 16>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 17>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800880808008ull), static_cast<long long int>(0x8080800880808008ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 17>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 18>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800980808009ull), static_cast<long long int>(0x8080800980808009ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 18>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 19>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800980808009ull), static_cast<long long int>(0x8080800980808009ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 19>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 20>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800A8080800Aull), static_cast<long long int>(0x8080800A8080800Aull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 20>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 21>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800A8080800Aull), static_cast<long long int>(0x8080800A8080800Aull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 21>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 22>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800B8080800Bull), static_cast<long long int>(0x8080800B8080800Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 22>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 23>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800B8080800Bull), static_cast<long long int>(0x8080800B8080800Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 23>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 24>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800C8080800Cull), static_cast<long long int>(0x8080800C8080800Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 24>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 25>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800C8080800Cull), static_cast<long long int>(0x8080800C8080800Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 25>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 26>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800D8080800Dull), static_cast<long long int>(0x8080800D8080800Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 26>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 27>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800D8080800Dull), static_cast<long long int>(0x8080800D8080800Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 27>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 28>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800E8080800Eull), static_cast<long long int>(0x8080800E8080800Eull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 28>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 29>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800E8080800Eull), static_cast<long long int>(0x8080800E8080800Eull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 29>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 30>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800F8080800Full), static_cast<long long int>(0x8080800F8080800Full)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 30>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<1>::decompress_block<0, 31>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800F8080800Full), static_cast<long long int>(0x8080800F8080800Full)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<1>::per_block<0, 31>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<1>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();

    BitCompression::decompress_block<0, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 2>();

    BitCompression::decompress_block<0, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 3>();

    BitCompression::decompress_block<0, 4>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 4>();

    BitCompression::decompress_block<0, 5>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 5>();

    BitCompression::decompress_block<0, 6>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 6>();

    BitCompression::decompress_block<0, 7>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 7>();

    BitCompression::decompress_block<0, 8>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 8>();

    BitCompression::decompress_block<0, 9>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 9>();

    BitCompression::decompress_block<0, 10>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 10>();

    BitCompression::decompress_block<0, 11>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 11>();

    BitCompression::decompress_block<0, 12>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 12>();

    BitCompression::decompress_block<0, 13>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 13>();

    BitCompression::decompress_block<0, 14>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 14>();

    BitCompression::decompress_block<0, 15>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 15>();

    BitCompression::decompress_block<0, 16>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 16>();

    BitCompression::decompress_block<0, 17>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 17>();

    BitCompression::decompress_block<0, 18>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 18>();

    BitCompression::decompress_block<0, 19>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 19>();

    BitCompression::decompress_block<0, 20>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 20>();

    BitCompression::decompress_block<0, 21>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 21>();

    BitCompression::decompress_block<0, 22>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 22>();

    BitCompression::decompress_block<0, 23>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 23>();

    BitCompression::decompress_block<0, 24>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 24>();

    BitCompression::decompress_block<0, 25>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 25>();

    BitCompression::decompress_block<0, 26>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 26>();

    BitCompression::decompress_block<0, 27>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 27>();

    BitCompression::decompress_block<0, 28>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 28>();

    BitCompression::decompress_block<0, 29>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 29>();

    BitCompression::decompress_block<0, 30>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 30>();

    BitCompression::decompress_block<0, 31>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 31>();


}


template<>
inline void BitCompression<1>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<1>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<1>::remaining<0>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 16
//Offset: 0
//Bits: 2


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<2>::remaining<0>() { return 64; }

template<>
template<>
inline const int BitCompression<2>::base_shift<0>() { return 16; }

template<>
template<>
inline int BitCompression<2>::next_offset<0>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<2>::block_count<0>() { return 16; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<2>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 0) & 0x3;
}

template<>
template<>
inline void BitCompression<2>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800080808000ull), static_cast<long long int>(0x8080800080808000ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x300000003ull), static_cast<long long int>(0x300000003ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<2>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<2>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800180808001ull), static_cast<long long int>(0x8080800180808001ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x300000003ull), static_cast<long long int>(0x300000003ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<2>::per_block<0, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<2>::decompress_block<0, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800280808002ull), static_cast<long long int>(0x8080800280808002ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x300000003ull), static_cast<long long int>(0x300000003ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<2>::per_block<0, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<2>::decompress_block<0, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800380808003ull), static_cast<long long int>(0x8080800380808003ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x300000003ull), static_cast<long long int>(0x300000003ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<2>::per_block<0, 3>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<2>::decompress_block<0, 4>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800480808004ull), static_cast<long long int>(0x8080800480808004ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x300000003ull), static_cast<long long int>(0x300000003ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<2>::per_block<0, 4>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<2>::decompress_block<0, 5>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800580808005ull), static_cast<long long int>(0x8080800580808005ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x300000003ull), static_cast<long long int>(0x300000003ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<2>::per_block<0, 5>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<2>::decompress_block<0, 6>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800680808006ull), static_cast<long long int>(0x8080800680808006ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x300000003ull), static_cast<long long int>(0x300000003ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<2>::per_block<0, 6>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<2>::decompress_block<0, 7>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800780808007ull), static_cast<long long int>(0x8080800780808007ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x300000003ull), static_cast<long long int>(0x300000003ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<2>::per_block<0, 7>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<2>::decompress_block<0, 8>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800880808008ull), static_cast<long long int>(0x8080800880808008ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x300000003ull), static_cast<long long int>(0x300000003ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<2>::per_block<0, 8>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<2>::decompress_block<0, 9>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800980808009ull), static_cast<long long int>(0x8080800980808009ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x300000003ull), static_cast<long long int>(0x300000003ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<2>::per_block<0, 9>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<2>::decompress_block<0, 10>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800A8080800Aull), static_cast<long long int>(0x8080800A8080800Aull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x300000003ull), static_cast<long long int>(0x300000003ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<2>::per_block<0, 10>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<2>::decompress_block<0, 11>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800B8080800Bull), static_cast<long long int>(0x8080800B8080800Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x300000003ull), static_cast<long long int>(0x300000003ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<2>::per_block<0, 11>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<2>::decompress_block<0, 12>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800C8080800Cull), static_cast<long long int>(0x8080800C8080800Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x300000003ull), static_cast<long long int>(0x300000003ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<2>::per_block<0, 12>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<2>::decompress_block<0, 13>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800D8080800Dull), static_cast<long long int>(0x8080800D8080800Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x300000003ull), static_cast<long long int>(0x300000003ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<2>::per_block<0, 13>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<2>::decompress_block<0, 14>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800E8080800Eull), static_cast<long long int>(0x8080800E8080800Eull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x300000003ull), static_cast<long long int>(0x300000003ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<2>::per_block<0, 14>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<2>::decompress_block<0, 15>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800F8080800Full), static_cast<long long int>(0x8080800F8080800Full)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x300000003ull), static_cast<long long int>(0x300000003ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<2>::per_block<0, 15>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<2>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();

    BitCompression::decompress_block<0, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 2>();

    BitCompression::decompress_block<0, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 3>();

    BitCompression::decompress_block<0, 4>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 4>();

    BitCompression::decompress_block<0, 5>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 5>();

    BitCompression::decompress_block<0, 6>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 6>();

    BitCompression::decompress_block<0, 7>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 7>();

    BitCompression::decompress_block<0, 8>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 8>();

    BitCompression::decompress_block<0, 9>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 9>();

    BitCompression::decompress_block<0, 10>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 10>();

    BitCompression::decompress_block<0, 11>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 11>();

    BitCompression::decompress_block<0, 12>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 12>();

    BitCompression::decompress_block<0, 13>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 13>();

    BitCompression::decompress_block<0, 14>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 14>();

    BitCompression::decompress_block<0, 15>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 15>();


}


template<>
inline void BitCompression<2>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<2>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<2>::remaining<0>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 0
//Bits: 3


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<3>::remaining<0>() { return 42; }

template<>
template<>
inline const int BitCompression<3>::base_shift<0>() { return 15; }

template<>
template<>
inline int BitCompression<3>::next_offset<0>() { return 1; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<3>::block_count<0>() { return 11; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<3>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 6) & 0x7;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800080808000ull), static_cast<long long int>(0x8080800180800100ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080020180808001ull), static_cast<long long int>(0x8080800280808002ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<0, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<0, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800380808003ull), static_cast<long long int>(0x8080800480800403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<0, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<0, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080050480808004ull), static_cast<long long int>(0x8080800580808005ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<0, 3>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<0, 4>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800680808006ull), static_cast<long long int>(0x8080800780800706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<0, 4>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<0, 5>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080080780808007ull), static_cast<long long int>(0x8080800880808008ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<0, 5>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<0, 6>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800980808009ull), static_cast<long long int>(0x8080800A80800A09ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<0, 6>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<0, 7>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800B0A8080800Aull), static_cast<long long int>(0x8080800B8080800Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<0, 7>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<0, 8>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800C8080800Cull), static_cast<long long int>(0x8080800D80800D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<0, 8>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<0, 9>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800E0D8080800Dull), static_cast<long long int>(0x8080800E8080800Eull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<0, 9>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<0, 10>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800F8080800Full), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<0, 10>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<3>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();

    BitCompression::decompress_block<0, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 2>();

    BitCompression::decompress_block<0, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 3>();

    BitCompression::decompress_block<0, 4>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 4>();

    BitCompression::decompress_block<0, 5>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 5>();

    BitCompression::decompress_block<0, 6>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 6>();

    BitCompression::decompress_block<0, 7>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 7>();

    BitCompression::decompress_block<0, 8>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 8>();

    BitCompression::decompress_block<0, 9>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 9>();

    BitCompression::decompress_block<0, 10>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 10>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<0>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 1
//Bits: 3


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<3>::remaining<1>() { return 42; }

template<>
template<>
inline const int BitCompression<3>::base_shift<1>() { return 15; }

template<>
template<>
inline int BitCompression<3>::next_offset<1>() { return 2; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<3>::block_count<1>() { return 11; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<3>::overlap_value<1>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 7) & 0x7;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<1, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800080808000ull), static_cast<long long int>(0x8080800180800100ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<1, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<1, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800280808001ull), static_cast<long long int>(0x8080030280808002ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<1, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<1, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800380808003ull), static_cast<long long int>(0x8080800480800403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<1, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<1, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800580808004ull), static_cast<long long int>(0x8080060580808005ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<1, 3>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<1, 4>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800680808006ull), static_cast<long long int>(0x8080800780800706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<1, 4>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<1, 5>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800880808007ull), static_cast<long long int>(0x8080090880808008ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<1, 5>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<1, 6>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800980808009ull), static_cast<long long int>(0x8080800A80800A09ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<1, 6>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<1, 7>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800B8080800Aull), static_cast<long long int>(0x80800C0B8080800Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<1, 7>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<1, 8>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800C8080800Cull), static_cast<long long int>(0x8080800D80800D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<1, 8>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<1, 9>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800E8080800Dull), static_cast<long long int>(0x80800F0E8080800Eull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<1, 9>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<1, 10>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800F8080800Full), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<1, 10>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<3>::decompress<1>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<1, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 0>();

    BitCompression::decompress_block<1, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 1>();

    BitCompression::decompress_block<1, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 2>();

    BitCompression::decompress_block<1, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 3>();

    BitCompression::decompress_block<1, 4>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 4>();

    BitCompression::decompress_block<1, 5>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 5>();

    BitCompression::decompress_block<1, 6>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 6>();

    BitCompression::decompress_block<1, 7>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 7>();

    BitCompression::decompress_block<1, 8>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 8>();

    BitCompression::decompress_block<1, 9>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 9>();

    BitCompression::decompress_block<1, 10>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 10>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<1>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 2
//Bits: 3


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<3>::remaining<2>() { return 42; }

template<>
template<>
inline const int BitCompression<3>::base_shift<2>() { return 15; }

template<>
template<>
inline int BitCompression<3>::next_offset<2>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<3>::block_count<2>() { return 11; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<3>::overlap_value<2>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0x7;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<2, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800080808000ull), static_cast<long long int>(0x8080800180808001ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<2, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<2, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800280800201ull), static_cast<long long int>(0x8080030280808002ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<2, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<2, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800380808003ull), static_cast<long long int>(0x8080800480808004ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<2, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<2, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800580800504ull), static_cast<long long int>(0x8080060580808005ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<2, 3>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<2, 4>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800680808006ull), static_cast<long long int>(0x8080800780808007ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<2, 4>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<2, 5>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800880800807ull), static_cast<long long int>(0x8080090880808008ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<2, 5>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<2, 6>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800980808009ull), static_cast<long long int>(0x8080800A8080800Aull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<2, 6>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<2, 7>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800B80800B0Aull), static_cast<long long int>(0x80800C0B8080800Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<2, 7>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<2, 8>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800C8080800Cull), static_cast<long long int>(0x8080800D8080800Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<2, 8>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<2, 9>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800E80800E0Dull), static_cast<long long int>(0x80800F0E8080800Eull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<2, 9>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<3>::decompress_block<2, 10>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800F8080800Full), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x700000007ull), static_cast<long long int>(0x700000007ull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<3>::per_block<2, 10>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<3>::decompress<2>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<2, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 0>();

    BitCompression::decompress_block<2, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 1>();

    BitCompression::decompress_block<2, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 2>();

    BitCompression::decompress_block<2, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 3>();

    BitCompression::decompress_block<2, 4>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 4>();

    BitCompression::decompress_block<2, 5>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 5>();

    BitCompression::decompress_block<2, 6>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 6>();

    BitCompression::decompress_block<2, 7>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 7>();

    BitCompression::decompress_block<2, 8>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 8>();

    BitCompression::decompress_block<2, 9>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 9>();

    BitCompression::decompress_block<2, 10>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 10>();


}


template<>
inline void BitCompression<3>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<3>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<3>::remaining<0>()  + 1 ;

    BitCompression<3>::decompress<1>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<3>::remaining<1>()  + 1 ;

    BitCompression<3>::decompress<2>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<3>::remaining<2>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 16
//Offset: 0
//Bits: 4


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<4>::remaining<0>() { return 32; }

template<>
template<>
inline const int BitCompression<4>::base_shift<0>() { return 16; }

template<>
template<>
inline int BitCompression<4>::next_offset<0>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<4>::block_count<0>() { return 8; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<4>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 0) & 0xf;
}

template<>
template<>
inline void BitCompression<4>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800080808000ull), static_cast<long long int>(0x8080800180808001ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000010ull), static_cast<long long int>(0x100000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xf0000000full), static_cast<long long int>(0xf0000000full)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<4>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<4>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800280808002ull), static_cast<long long int>(0x8080800380808003ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000010ull), static_cast<long long int>(0x100000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xf0000000full), static_cast<long long int>(0xf0000000full)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<4>::per_block<0, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<4>::decompress_block<0, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800480808004ull), static_cast<long long int>(0x8080800580808005ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000010ull), static_cast<long long int>(0x100000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xf0000000full), static_cast<long long int>(0xf0000000full)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<4>::per_block<0, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<4>::decompress_block<0, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800680808006ull), static_cast<long long int>(0x8080800780808007ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000010ull), static_cast<long long int>(0x100000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xf0000000full), static_cast<long long int>(0xf0000000full)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<4>::per_block<0, 3>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<4>::decompress_block<0, 4>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800880808008ull), static_cast<long long int>(0x8080800980808009ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000010ull), static_cast<long long int>(0x100000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xf0000000full), static_cast<long long int>(0xf0000000full)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<4>::per_block<0, 4>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<4>::decompress_block<0, 5>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800A8080800Aull), static_cast<long long int>(0x8080800B8080800Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000010ull), static_cast<long long int>(0x100000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xf0000000full), static_cast<long long int>(0xf0000000full)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<4>::per_block<0, 5>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<4>::decompress_block<0, 6>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800C8080800Cull), static_cast<long long int>(0x8080800D8080800Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000010ull), static_cast<long long int>(0x100000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xf0000000full), static_cast<long long int>(0xf0000000full)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<4>::per_block<0, 6>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<4>::decompress_block<0, 7>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800E8080800Eull), static_cast<long long int>(0x8080800F8080800Full)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000010ull), static_cast<long long int>(0x100000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xf0000000full), static_cast<long long int>(0xf0000000full)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<4>::per_block<0, 7>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<4>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();

    BitCompression::decompress_block<0, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 2>();

    BitCompression::decompress_block<0, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 3>();

    BitCompression::decompress_block<0, 4>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 4>();

    BitCompression::decompress_block<0, 5>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 5>();

    BitCompression::decompress_block<0, 6>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 6>();

    BitCompression::decompress_block<0, 7>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 7>();


}


template<>
inline void BitCompression<4>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<4>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<4>::remaining<0>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 0
//Bits: 5


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<5>::remaining<0>() { return 25; }

template<>
template<>
inline const int BitCompression<5>::base_shift<0>() { return 15; }

template<>
template<>
inline int BitCompression<5>::next_offset<0>() { return 2; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<5>::block_count<0>() { return 7; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<5>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 5) & 0x1f;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080010080808000ull), static_cast<long long int>(0x8080020180808001ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000080ull), static_cast<long long int>(0x100000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800380800302ull), static_cast<long long int>(0x8080800480800403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<0, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<0, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080060580808005ull), static_cast<long long int>(0x8080070680808006ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000080ull), static_cast<long long int>(0x100000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<0, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<0, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800880800807ull), static_cast<long long int>(0x8080800980800908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<0, 3>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<0, 4>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800B0A8080800Aull), static_cast<long long int>(0x80800C0B8080800Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000080ull), static_cast<long long int>(0x100000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<0, 4>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<0, 5>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800D80800D0Cull), static_cast<long long int>(0x8080800E80800E0Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<0, 5>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<0, 6>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080808080800Full), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000080ull), static_cast<long long int>(0x100000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<0, 6>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<5>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();

    BitCompression::decompress_block<0, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 2>();

    BitCompression::decompress_block<0, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 3>();

    BitCompression::decompress_block<0, 4>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 4>();

    BitCompression::decompress_block<0, 5>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 5>();

    BitCompression::decompress_block<0, 6>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 6>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<0>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 2
//Bits: 5


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<5>::remaining<2>() { return 25; }

template<>
template<>
inline const int BitCompression<5>::base_shift<2>() { return 15; }

template<>
template<>
inline int BitCompression<5>::next_offset<2>() { return 4; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<5>::block_count<2>() { return 7; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<5>::overlap_value<2>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 7) & 0x1f;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<2, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080010080808000ull), static_cast<long long int>(0x8080800280800201ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<2, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<2, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800380800302ull), static_cast<long long int>(0x8080050480808004ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<2, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<2, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080060580808005ull), static_cast<long long int>(0x8080800780800706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<2, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<2, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800880800807ull), static_cast<long long int>(0x80800A0980808009ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<2, 3>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<2, 4>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800B0A8080800Aull), static_cast<long long int>(0x8080800C80800C0Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<2, 4>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<2, 5>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800D80800D0Cull), static_cast<long long int>(0x80800F0E8080800Eull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<2, 5>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<2, 6>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080808080800Full), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<2, 6>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<5>::decompress<2>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<2, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 0>();

    BitCompression::decompress_block<2, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 1>();

    BitCompression::decompress_block<2, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 2>();

    BitCompression::decompress_block<2, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 3>();

    BitCompression::decompress_block<2, 4>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 4>();

    BitCompression::decompress_block<2, 5>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 5>();

    BitCompression::decompress_block<2, 6>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 6>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<2>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 4
//Bits: 5


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<5>::remaining<4>() { return 24; }

template<>
template<>
inline const int BitCompression<5>::base_shift<4>() { return 15; }

template<>
template<>
inline int BitCompression<5>::next_offset<4>() { return 1; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<5>::block_count<4>() { return 6; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<5>::overlap_value<4>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 4) & 0x1f;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<4, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800180800100ull), static_cast<long long int>(0x8080800280800201ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<4, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<4, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080040380808003ull), static_cast<long long int>(0x8080050480808004ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000080ull), static_cast<long long int>(0x100000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<4, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<4, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800680800605ull), static_cast<long long int>(0x8080800780800706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<4, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<4, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080090880808008ull), static_cast<long long int>(0x80800A0980808009ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000080ull), static_cast<long long int>(0x100000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<4, 3>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<4, 4>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800B80800B0Aull), static_cast<long long int>(0x8080800C80800C0Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<4, 4>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<4, 5>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800E0D8080800Dull), static_cast<long long int>(0x80800F0E8080800Eull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000080ull), static_cast<long long int>(0x100000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<4, 5>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<5>::decompress<4>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<4, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 0>();

    BitCompression::decompress_block<4, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 1>();

    BitCompression::decompress_block<4, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 2>();

    BitCompression::decompress_block<4, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 3>();

    BitCompression::decompress_block<4, 4>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 4>();

    BitCompression::decompress_block<4, 5>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 5>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<4>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 1
//Bits: 5


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<5>::remaining<1>() { return 25; }

template<>
template<>
inline const int BitCompression<5>::base_shift<1>() { return 15; }

template<>
template<>
inline int BitCompression<5>::next_offset<1>() { return 3; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<5>::block_count<1>() { return 7; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<5>::overlap_value<1>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 6) & 0x1f;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<1, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080010080808000ull), static_cast<long long int>(0x8080800280808001ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<1, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<1, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800380800302ull), static_cast<long long int>(0x8080050480800403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<1, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<1, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080060580808005ull), static_cast<long long int>(0x8080800780808006ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<1, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<1, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800880800807ull), static_cast<long long int>(0x80800A0980800908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<1, 3>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<1, 4>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800B0A8080800Aull), static_cast<long long int>(0x8080800C8080800Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<1, 4>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<1, 5>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800D80800D0Cull), static_cast<long long int>(0x80800F0E80800E0Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<1, 5>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<1, 6>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080808080800Full), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<1, 6>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<5>::decompress<1>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<1, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 0>();

    BitCompression::decompress_block<1, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 1>();

    BitCompression::decompress_block<1, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 2>();

    BitCompression::decompress_block<1, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 3>();

    BitCompression::decompress_block<1, 4>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 4>();

    BitCompression::decompress_block<1, 5>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 5>();

    BitCompression::decompress_block<1, 6>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 6>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<1>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 3
//Bits: 5


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<5>::remaining<3>() { return 25; }

template<>
template<>
inline const int BitCompression<5>::base_shift<3>() { return 15; }

template<>
template<>
inline int BitCompression<5>::next_offset<3>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<5>::block_count<3>() { return 7; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<5>::overlap_value<3>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0x1f;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<3, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800180808000ull), static_cast<long long int>(0x8080800280800201ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<3, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<3, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080040380800302ull), static_cast<long long int>(0x8080050480808004ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<3, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<3, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800680808005ull), static_cast<long long int>(0x8080800780800706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<3, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<3, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080090880800807ull), static_cast<long long int>(0x80800A0980808009ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<3, 3>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<3, 4>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800B8080800Aull), static_cast<long long int>(0x8080800C80800C0Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<3, 4>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<3, 5>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800E0D80800D0Cull), static_cast<long long int>(0x80800F0E8080800Eull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<3, 5>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<5>::decompress_block<3, 6>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080808080800Full), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1f0000001full), static_cast<long long int>(0x1f0000001full)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<5>::per_block<3, 6>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<5>::decompress<3>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<3, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 0>();

    BitCompression::decompress_block<3, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 1>();

    BitCompression::decompress_block<3, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 2>();

    BitCompression::decompress_block<3, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 3>();

    BitCompression::decompress_block<3, 4>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 4>();

    BitCompression::decompress_block<3, 5>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 5>();

    BitCompression::decompress_block<3, 6>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 6>();


}


template<>
inline void BitCompression<5>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<5>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<5>::remaining<0>()  + 1 ;

    BitCompression<5>::decompress<2>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<5>::remaining<2>()  + 1 ;

    BitCompression<5>::decompress<4>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<5>::remaining<4>()  + 1 ;

    BitCompression<5>::decompress<1>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<5>::remaining<1>()  + 1 ;

    BitCompression<5>::decompress<3>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<5>::remaining<3>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 0
//Bits: 6


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<6>::remaining<0>() { return 21; }

template<>
template<>
inline const int BitCompression<6>::base_shift<0>() { return 15; }

template<>
template<>
inline int BitCompression<6>::next_offset<0>() { return 4; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<6>::block_count<0>() { return 6; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<6>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 6) & 0x3f;
}

template<>
template<>
inline void BitCompression<6>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080010080808000ull), static_cast<long long int>(0x8080800280800201ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000040ull), static_cast<long long int>(0x1000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3f0000003full), static_cast<long long int>(0x3f0000003full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<6>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<6>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080040380808003ull), static_cast<long long int>(0x8080800580800504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000040ull), static_cast<long long int>(0x1000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3f0000003full), static_cast<long long int>(0x3f0000003full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<6>::per_block<0, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<6>::decompress_block<0, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080070680808006ull), static_cast<long long int>(0x8080800880800807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000040ull), static_cast<long long int>(0x1000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3f0000003full), static_cast<long long int>(0x3f0000003full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<6>::per_block<0, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<6>::decompress_block<0, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800A0980808009ull), static_cast<long long int>(0x8080800B80800B0Aull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000040ull), static_cast<long long int>(0x1000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3f0000003full), static_cast<long long int>(0x3f0000003full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<6>::per_block<0, 3>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<6>::decompress_block<0, 4>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800D0C8080800Cull), static_cast<long long int>(0x8080800E80800E0Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000040ull), static_cast<long long int>(0x1000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3f0000003full), static_cast<long long int>(0x3f0000003full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<6>::per_block<0, 4>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<6>::decompress_block<0, 5>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080808080800Full), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000040ull), static_cast<long long int>(0x1000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3f0000003full), static_cast<long long int>(0x3f0000003full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<6>::per_block<0, 5>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<6>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();

    BitCompression::decompress_block<0, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 2>();

    BitCompression::decompress_block<0, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 3>();

    BitCompression::decompress_block<0, 4>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 4>();

    BitCompression::decompress_block<0, 5>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 5>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<0>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 4
//Bits: 6


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<6>::remaining<4>() { return 20; }

template<>
template<>
inline const int BitCompression<6>::base_shift<4>() { return 15; }

template<>
template<>
inline int BitCompression<6>::next_offset<4>() { return 2; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<6>::block_count<4>() { return 5; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<6>::overlap_value<4>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 4) & 0x3f;
}

template<>
template<>
inline void BitCompression<6>::decompress_block<4, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800180800100ull), static_cast<long long int>(0x8080030280808002ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000004ull), static_cast<long long int>(0x100000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3f0000003full), static_cast<long long int>(0x3f0000003full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<6>::per_block<4, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<6>::decompress_block<4, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800480800403ull), static_cast<long long int>(0x8080060580808005ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000004ull), static_cast<long long int>(0x100000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3f0000003full), static_cast<long long int>(0x3f0000003full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<6>::per_block<4, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<6>::decompress_block<4, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800780800706ull), static_cast<long long int>(0x8080090880808008ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000004ull), static_cast<long long int>(0x100000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3f0000003full), static_cast<long long int>(0x3f0000003full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<6>::per_block<4, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<6>::decompress_block<4, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800A80800A09ull), static_cast<long long int>(0x80800C0B8080800Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000004ull), static_cast<long long int>(0x100000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3f0000003full), static_cast<long long int>(0x3f0000003full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<6>::per_block<4, 3>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<6>::decompress_block<4, 4>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800D80800D0Cull), static_cast<long long int>(0x80800F0E8080800Eull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000004ull), static_cast<long long int>(0x100000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3f0000003full), static_cast<long long int>(0x3f0000003full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<6>::per_block<4, 4>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<6>::decompress<4>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<4, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 0>();

    BitCompression::decompress_block<4, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 1>();

    BitCompression::decompress_block<4, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 2>();

    BitCompression::decompress_block<4, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 3>();

    BitCompression::decompress_block<4, 4>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 4>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<4>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 2
//Bits: 6


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<6>::remaining<2>() { return 21; }

template<>
template<>
inline const int BitCompression<6>::base_shift<2>() { return 15; }

template<>
template<>
inline int BitCompression<6>::next_offset<2>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<6>::block_count<2>() { return 6; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<6>::overlap_value<2>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0x3f;
}

template<>
template<>
inline void BitCompression<6>::decompress_block<2, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800180808000ull), static_cast<long long int>(0x8080030280800201ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000010ull), static_cast<long long int>(0x400000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3f0000003full), static_cast<long long int>(0x3f0000003full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<6>::per_block<2, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<6>::decompress_block<2, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800480808003ull), static_cast<long long int>(0x8080060580800504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000010ull), static_cast<long long int>(0x400000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3f0000003full), static_cast<long long int>(0x3f0000003full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<6>::per_block<2, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<6>::decompress_block<2, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800780808006ull), static_cast<long long int>(0x8080090880800807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000010ull), static_cast<long long int>(0x400000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3f0000003full), static_cast<long long int>(0x3f0000003full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<6>::per_block<2, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<6>::decompress_block<2, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800A80808009ull), static_cast<long long int>(0x80800C0B80800B0Aull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000010ull), static_cast<long long int>(0x400000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3f0000003full), static_cast<long long int>(0x3f0000003full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<6>::per_block<2, 3>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<6>::decompress_block<2, 4>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800D8080800Cull), static_cast<long long int>(0x80800F0E80800E0Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000010ull), static_cast<long long int>(0x400000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3f0000003full), static_cast<long long int>(0x3f0000003full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<6>::per_block<2, 4>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<6>::decompress_block<2, 5>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080808080800Full), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000010ull), static_cast<long long int>(0x400000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3f0000003full), static_cast<long long int>(0x3f0000003full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<6>::per_block<2, 5>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<6>::decompress<2>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<2, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 0>();

    BitCompression::decompress_block<2, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 1>();

    BitCompression::decompress_block<2, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 2>();

    BitCompression::decompress_block<2, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 3>();

    BitCompression::decompress_block<2, 4>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 4>();

    BitCompression::decompress_block<2, 5>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 5>();


}


template<>
inline void BitCompression<6>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<6>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<6>::remaining<0>()  + 1 ;

    BitCompression<6>::decompress<4>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<6>::remaining<4>()  + 1 ;

    BitCompression<6>::decompress<2>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<6>::remaining<2>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 0
//Bits: 7


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<7>::remaining<0>() { return 18; }

template<>
template<>
inline const int BitCompression<7>::base_shift<0>() { return 15; }

template<>
template<>
inline int BitCompression<7>::next_offset<0>() { return 5; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<7>::block_count<0>() { return 5; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<7>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 6) & 0x7f;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080010080808000ull), static_cast<long long int>(0x8080030280800201ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000080ull), static_cast<long long int>(0x400000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080050480800403ull), static_cast<long long int>(0x8080800680800605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<0, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<0, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080080780808007ull), static_cast<long long int>(0x80800A0980800908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000080ull), static_cast<long long int>(0x400000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<0, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<0, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800C0B80800B0Aull), static_cast<long long int>(0x8080800D80800D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<0, 3>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<0, 4>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800F0E8080800Eull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000080ull), static_cast<long long int>(0x400000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<0, 4>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<7>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();

    BitCompression::decompress_block<0, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 2>();

    BitCompression::decompress_block<0, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 3>();

    BitCompression::decompress_block<0, 4>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 4>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<0>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 5
//Bits: 7


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<7>::remaining<5>() { return 17; }

template<>
template<>
inline const int BitCompression<7>::base_shift<5>() { return 14; }

template<>
template<>
inline int BitCompression<7>::next_offset<5>() { return 3; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<7>::block_count<5>() { return 5; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<7>::overlap_value<5>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 12) & 0x7f;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<5, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080020180800100ull), static_cast<long long int>(0x8080040380800302ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<5, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<5, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800580808004ull), static_cast<long long int>(0x8080070680800605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000040ull), static_cast<long long int>(0x200000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<5, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<5, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080090880800807ull), static_cast<long long int>(0x80800B0A80800A09ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<5, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<5, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800C8080800Bull), static_cast<long long int>(0x80800E0D80800D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000040ull), static_cast<long long int>(0x200000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<5, 3>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<5, 4>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080808080800F0Eull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<5, 4>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<7>::decompress<5>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<5, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 0>();

    BitCompression::decompress_block<5, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 1>();

    BitCompression::decompress_block<5, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 2>();

    BitCompression::decompress_block<5, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 3>();

    BitCompression::decompress_block<5, 4>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 4>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<5>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 3
//Bits: 7


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<7>::remaining<3>() { return 17; }

template<>
template<>
inline const int BitCompression<7>::base_shift<3>() { return 14; }

template<>
template<>
inline int BitCompression<7>::next_offset<3>() { return 1; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<7>::block_count<3>() { return 5; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<7>::overlap_value<3>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 10) & 0x7f;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<3, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080020180800100ull), static_cast<long long int>(0x8080800380808002ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<3, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<3, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080050480800403ull), static_cast<long long int>(0x8080070680800605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<3, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<3, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080090880800807ull), static_cast<long long int>(0x8080800A80808009ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<3, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<3, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800C0B80800B0Aull), static_cast<long long int>(0x80800E0D80800D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<3, 3>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<3, 4>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080808080800F0Eull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<3, 4>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<7>::decompress<3>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<3, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 0>();

    BitCompression::decompress_block<3, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 1>();

    BitCompression::decompress_block<3, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 2>();

    BitCompression::decompress_block<3, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 3>();

    BitCompression::decompress_block<3, 4>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 4>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<3>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 1
//Bits: 7


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<7>::remaining<1>() { return 18; }

template<>
template<>
inline const int BitCompression<7>::base_shift<1>() { return 15; }

template<>
template<>
inline int BitCompression<7>::next_offset<1>() { return 6; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<7>::block_count<1>() { return 5; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<7>::overlap_value<1>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 7) & 0x7f;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<1, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800180808000ull), static_cast<long long int>(0x8080030280800201ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000040ull), static_cast<long long int>(0x200000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<1, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<1, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080050480800403ull), static_cast<long long int>(0x8080070680800605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<1, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<1, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800880808007ull), static_cast<long long int>(0x80800A0980800908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000040ull), static_cast<long long int>(0x200000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<1, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<1, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800C0B80800B0Aull), static_cast<long long int>(0x80800E0D80800D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<1, 3>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<1, 4>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800F8080800Eull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000040ull), static_cast<long long int>(0x200000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<1, 4>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<7>::decompress<1>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<1, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 0>();

    BitCompression::decompress_block<1, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 1>();

    BitCompression::decompress_block<1, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 2>();

    BitCompression::decompress_block<1, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 3>();

    BitCompression::decompress_block<1, 4>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 4>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<1>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 6
//Bits: 7


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<7>::remaining<6>() { return 17; }

template<>
template<>
inline const int BitCompression<7>::base_shift<6>() { return 14; }

template<>
template<>
inline int BitCompression<7>::next_offset<6>() { return 4; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<7>::block_count<6>() { return 5; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<7>::overlap_value<6>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 13) & 0x7f;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<6, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080020180800100ull), static_cast<long long int>(0x8080040380800302ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<6, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<6, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800580800504ull), static_cast<long long int>(0x8080070680808006ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000020ull), static_cast<long long int>(0x100000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<6, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<6, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080090880800807ull), static_cast<long long int>(0x80800B0A80800A09ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<6, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<6, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800C80800C0Bull), static_cast<long long int>(0x80800E0D8080800Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000020ull), static_cast<long long int>(0x100000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<6, 3>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<6, 4>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080808080800F0Eull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<6, 4>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<7>::decompress<6>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<6, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 0>();

    BitCompression::decompress_block<6, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 1>();

    BitCompression::decompress_block<6, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 2>();

    BitCompression::decompress_block<6, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 3>();

    BitCompression::decompress_block<6, 4>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 4>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<6>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 4
//Bits: 7


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<7>::remaining<4>() { return 17; }

template<>
template<>
inline const int BitCompression<7>::base_shift<4>() { return 14; }

template<>
template<>
inline int BitCompression<7>::next_offset<4>() { return 2; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<7>::block_count<4>() { return 5; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<7>::overlap_value<4>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 11) & 0x7f;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<4, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080020180800100ull), static_cast<long long int>(0x8080800380800302ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<4, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<4, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080050480808004ull), static_cast<long long int>(0x8080070680800605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000080ull), static_cast<long long int>(0x400000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<4, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<4, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080090880800807ull), static_cast<long long int>(0x8080800A80800A09ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<4, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<4, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800C0B8080800Bull), static_cast<long long int>(0x80800E0D80800D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000080ull), static_cast<long long int>(0x400000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<4, 3>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<4, 4>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080808080800F0Eull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<4, 4>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<7>::decompress<4>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<4, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 0>();

    BitCompression::decompress_block<4, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 1>();

    BitCompression::decompress_block<4, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 2>();

    BitCompression::decompress_block<4, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 3>();

    BitCompression::decompress_block<4, 4>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 4>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<4>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 2
//Bits: 7


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<7>::remaining<2>() { return 18; }

template<>
template<>
inline const int BitCompression<7>::base_shift<2>() { return 15; }

template<>
template<>
inline int BitCompression<7>::next_offset<2>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<7>::block_count<2>() { return 5; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<7>::overlap_value<2>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0x7f;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<2, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800180800100ull), static_cast<long long int>(0x8080030280808002ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000020ull), static_cast<long long int>(0x100000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<2, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<2, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080050480800403ull), static_cast<long long int>(0x8080070680800605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<2, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<2, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800880800807ull), static_cast<long long int>(0x80800A0980808009ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000020ull), static_cast<long long int>(0x100000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<2, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<2, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800C0B80800B0Aull), static_cast<long long int>(0x80800E0D80800D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<2, 3>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<7>::decompress_block<2, 4>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800F80800F0Eull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000020ull), static_cast<long long int>(0x100000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7f0000007full), static_cast<long long int>(0x7f0000007full)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<7>::per_block<2, 4>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<7>::decompress<2>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<2, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 0>();

    BitCompression::decompress_block<2, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 1>();

    BitCompression::decompress_block<2, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 2>();

    BitCompression::decompress_block<2, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 3>();

    BitCompression::decompress_block<2, 4>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 4>();


}


template<>
inline void BitCompression<7>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<7>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<7>::remaining<0>()  + 1 ;

    BitCompression<7>::decompress<5>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<7>::remaining<5>()  + 1 ;

    BitCompression<7>::decompress<3>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<7>::remaining<3>()  + 1 ;

    BitCompression<7>::decompress<1>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<7>::remaining<1>()  + 1 ;

    BitCompression<7>::decompress<6>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<7>::remaining<6>()  + 1 ;

    BitCompression<7>::decompress<4>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<7>::remaining<4>()  + 1 ;

    BitCompression<7>::decompress<2>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<7>::remaining<2>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 16
//Offset: 0
//Bits: 8


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<8>::remaining<0>() { return 16; }

template<>
template<>
inline const int BitCompression<8>::base_shift<0>() { return 16; }

template<>
template<>
inline int BitCompression<8>::next_offset<0>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<8>::block_count<0>() { return 4; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<8>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 0) & 0xff;
}

template<>
template<>
inline void BitCompression<8>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800180808000ull), static_cast<long long int>(0x8080800380808002ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xff000000ffull), static_cast<long long int>(0xff000000ffull)};
    static const int shift_mask = 0;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<8>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<8>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800580808004ull), static_cast<long long int>(0x8080800780808006ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xff000000ffull), static_cast<long long int>(0xff000000ffull)};
    static const int shift_mask = 0;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<8>::per_block<0, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<8>::decompress_block<0, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800980808008ull), static_cast<long long int>(0x8080800B8080800Aull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xff000000ffull), static_cast<long long int>(0xff000000ffull)};
    static const int shift_mask = 0;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<8>::per_block<0, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<8>::decompress_block<0, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080800D8080800Cull), static_cast<long long int>(0x8080800F8080800Eull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xff000000ffull), static_cast<long long int>(0xff000000ffull)};
    static const int shift_mask = 0;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<8>::per_block<0, 3>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<8>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();

    BitCompression::decompress_block<0, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 2>();

    BitCompression::decompress_block<0, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 3>();


}


template<>
inline void BitCompression<8>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<8>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<8>::remaining<0>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 0
//Bits: 9


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<9>::remaining<0>() { return 14; }

template<>
template<>
inline const int BitCompression<9>::base_shift<0>() { return 15; }

template<>
template<>
inline int BitCompression<9>::next_offset<0>() { return 7; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<9>::block_count<0>() { return 4; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<9>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 6) & 0x1ff;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080020180800100ull), static_cast<long long int>(0x8080040380800302ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080060580800504ull), static_cast<long long int>(0x8080080780800706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<0, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<0, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800B0A80800A09ull), static_cast<long long int>(0x80800D0C80800C0Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<0, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<0, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800F0E80800E0Dull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<0, 3>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<9>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();

    BitCompression::decompress_block<0, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 2>();

    BitCompression::decompress_block<0, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 3>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<0>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 7
//Bits: 9


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<9>::remaining<7>() { return 13; }

template<>
template<>
inline const int BitCompression<9>::base_shift<7>() { return 14; }

template<>
template<>
inline int BitCompression<9>::next_offset<7>() { return 5; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<9>::block_count<7>() { return 4; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<9>::overlap_value<7>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 12) & 0x1ff;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<7, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080030280800100ull), static_cast<long long int>(0x8080050480800403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000001ull), static_cast<long long int>(0x2000000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<7, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<7, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080070680800605ull), static_cast<long long int>(0x8080090880800807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<7, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<7, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800C0B80800A09ull), static_cast<long long int>(0x80800E0D80800D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000001ull), static_cast<long long int>(0x2000000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<7, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<7, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080808080800F0Eull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<7, 3>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<9>::decompress<7>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<7, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<7, 0>();

    BitCompression::decompress_block<7, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<7, 1>();

    BitCompression::decompress_block<7, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<7, 2>();

    BitCompression::decompress_block<7, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<7, 3>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<7>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 5
//Bits: 9


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<9>::remaining<5>() { return 13; }

template<>
template<>
inline const int BitCompression<9>::base_shift<5>() { return 14; }

template<>
template<>
inline int BitCompression<9>::next_offset<5>() { return 3; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<9>::block_count<5>() { return 4; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<9>::overlap_value<5>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 10) & 0x1ff;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<5, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080020180800100ull), static_cast<long long int>(0x8080050480800302ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000004ull), static_cast<long long int>(0x8000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<5, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<5, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080070680800605ull), static_cast<long long int>(0x8080090880800807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<5, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<5, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800B0A80800A09ull), static_cast<long long int>(0x80800E0D80800C0Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000004ull), static_cast<long long int>(0x8000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<5, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<5, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080808080800F0Eull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<5, 3>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<9>::decompress<5>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<5, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 0>();

    BitCompression::decompress_block<5, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 1>();

    BitCompression::decompress_block<5, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 2>();

    BitCompression::decompress_block<5, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 3>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<5>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 3
//Bits: 9


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<9>::remaining<3>() { return 13; }

template<>
template<>
inline const int BitCompression<9>::base_shift<3>() { return 14; }

template<>
template<>
inline int BitCompression<9>::next_offset<3>() { return 1; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<9>::block_count<3>() { return 4; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<9>::overlap_value<3>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0x1ff;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<3, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080020180800100ull), static_cast<long long int>(0x8080040380800302ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<3, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<3, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080070680800504ull), static_cast<long long int>(0x8080090880800807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000001ull), static_cast<long long int>(0x2000000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<3, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<3, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800B0A80800A09ull), static_cast<long long int>(0x80800D0C80800C0Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<3, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<3, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080808080800E0Dull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000001ull), static_cast<long long int>(0x2000000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<3, 3>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<9>::decompress<3>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<3, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 0>();

    BitCompression::decompress_block<3, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 1>();

    BitCompression::decompress_block<3, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 2>();

    BitCompression::decompress_block<3, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 3>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<3>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 1
//Bits: 9


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<9>::remaining<1>() { return 14; }

template<>
template<>
inline const int BitCompression<9>::base_shift<1>() { return 15; }

template<>
template<>
inline int BitCompression<9>::next_offset<1>() { return 8; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<9>::block_count<1>() { return 4; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<9>::overlap_value<1>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 7) & 0x1ff;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<1, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080020180800100ull), static_cast<long long int>(0x8080040380800302ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<1, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<1, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080060580800504ull), static_cast<long long int>(0x8080090880800706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000004ull), static_cast<long long int>(0x8000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<1, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<1, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800B0A80800A09ull), static_cast<long long int>(0x80800D0C80800C0Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<1, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<1, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800F0E80800E0Dull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000004ull), static_cast<long long int>(0x8000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<1, 3>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<9>::decompress<1>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<1, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 0>();

    BitCompression::decompress_block<1, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 1>();

    BitCompression::decompress_block<1, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 2>();

    BitCompression::decompress_block<1, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 3>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<1>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 8
//Bits: 9


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<9>::remaining<8>() { return 13; }

template<>
template<>
inline const int BitCompression<9>::base_shift<8>() { return 14; }

template<>
template<>
inline int BitCompression<9>::next_offset<8>() { return 6; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<9>::block_count<8>() { return 4; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<9>::overlap_value<8>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 13) & 0x1ff;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<8, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080030280800201ull), static_cast<long long int>(0x8080050480800403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<8, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<8, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080070680800605ull), static_cast<long long int>(0x8080090880800807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<8, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<8, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800C0B80800B0Aull), static_cast<long long int>(0x80800E0D80800D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<8, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<8, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080808080800F0Eull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<8, 3>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<9>::decompress<8>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<8, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 0>();

    BitCompression::decompress_block<8, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 1>();

    BitCompression::decompress_block<8, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 2>();

    BitCompression::decompress_block<8, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 3>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<8>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 6
//Bits: 9


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<9>::remaining<6>() { return 13; }

template<>
template<>
inline const int BitCompression<9>::base_shift<6>() { return 14; }

template<>
template<>
inline int BitCompression<9>::next_offset<6>() { return 4; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<9>::block_count<6>() { return 4; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<9>::overlap_value<6>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 11) & 0x1ff;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<6, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080020180800100ull), static_cast<long long int>(0x8080050480800403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000002ull), static_cast<long long int>(0x4000000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<6, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<6, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080070680800605ull), static_cast<long long int>(0x8080090880800807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<6, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<6, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800B0A80800A09ull), static_cast<long long int>(0x80800E0D80800D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000002ull), static_cast<long long int>(0x4000000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<6, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<6, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080808080800F0Eull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<6, 3>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<9>::decompress<6>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<6, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 0>();

    BitCompression::decompress_block<6, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 1>();

    BitCompression::decompress_block<6, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 2>();

    BitCompression::decompress_block<6, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 3>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<6>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 4
//Bits: 9


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<9>::remaining<4>() { return 13; }

template<>
template<>
inline const int BitCompression<9>::base_shift<4>() { return 14; }

template<>
template<>
inline int BitCompression<9>::next_offset<4>() { return 2; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<9>::block_count<4>() { return 4; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<9>::overlap_value<4>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 9) & 0x1ff;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<4, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080020180800100ull), static_cast<long long int>(0x8080040380800302ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<4, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<4, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080070680800605ull), static_cast<long long int>(0x8080090880800807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<4, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<4, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800B0A80800A09ull), static_cast<long long int>(0x80800D0C80800C0Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<4, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<4, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080808080800F0Eull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<4, 3>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<9>::decompress<4>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<4, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 0>();

    BitCompression::decompress_block<4, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 1>();

    BitCompression::decompress_block<4, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 2>();

    BitCompression::decompress_block<4, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 3>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<4>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 2
//Bits: 9


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<9>::remaining<2>() { return 14; }

template<>
template<>
inline const int BitCompression<9>::base_shift<2>() { return 15; }

template<>
template<>
inline int BitCompression<9>::next_offset<2>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<9>::block_count<2>() { return 4; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<9>::overlap_value<2>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0x1ff;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<2, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080020180800100ull), static_cast<long long int>(0x8080040380800302ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<2, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<2, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080060580800504ull), static_cast<long long int>(0x8080090880800807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000002ull), static_cast<long long int>(0x4000000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<2, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<2, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800B0A80800A09ull), static_cast<long long int>(0x80800D0C80800C0Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<2, 2>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<9>::decompress_block<2, 3>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800F0E80800E0Dull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000002ull), static_cast<long long int>(0x4000000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ff000001ffull), static_cast<long long int>(0x1ff000001ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<9>::per_block<2, 3>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<9>::decompress<2>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<2, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 0>();

    BitCompression::decompress_block<2, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 1>();

    BitCompression::decompress_block<2, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 2>();

    BitCompression::decompress_block<2, 3>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 3>();


}


template<>
inline void BitCompression<9>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<9>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<9>::remaining<0>()  + 1 ;

    BitCompression<9>::decompress<7>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<9>::remaining<7>()  + 1 ;

    BitCompression<9>::decompress<5>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<9>::remaining<5>()  + 1 ;

    BitCompression<9>::decompress<3>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<9>::remaining<3>()  + 1 ;

    BitCompression<9>::decompress<1>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<9>::remaining<1>()  + 1 ;

    BitCompression<9>::decompress<8>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<9>::remaining<8>()  + 1 ;

    BitCompression<9>::decompress<6>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<9>::remaining<6>()  + 1 ;

    BitCompression<9>::decompress<4>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<9>::remaining<4>()  + 1 ;

    BitCompression<9>::decompress<2>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<9>::remaining<2>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 0
//Bits: 10


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<10>::remaining<0>() { return 12; }

template<>
template<>
inline const int BitCompression<10>::base_shift<0>() { return 15; }

template<>
template<>
inline int BitCompression<10>::next_offset<0>() { return 2; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<10>::block_count<0>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<10>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 0) & 0x3ff;
}

template<>
template<>
inline void BitCompression<10>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080020180800100ull), static_cast<long long int>(0x8080040380800302ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ff000003ffull), static_cast<long long int>(0x3ff000003ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<10>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<10>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080070680800605ull), static_cast<long long int>(0x8080090880800807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ff000003ffull), static_cast<long long int>(0x3ff000003ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<10>::per_block<0, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<10>::decompress_block<0, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800C0B80800B0Aull), static_cast<long long int>(0x80800E0D80800D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ff000003ffull), static_cast<long long int>(0x3ff000003ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<10>::per_block<0, 2>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<10>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();

    BitCompression::decompress_block<0, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<0>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 2
//Bits: 10


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<10>::remaining<2>() { return 12; }

template<>
template<>
inline const int BitCompression<10>::base_shift<2>() { return 15; }

template<>
template<>
inline int BitCompression<10>::next_offset<2>() { return 4; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<10>::block_count<2>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<10>::overlap_value<2>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 2) & 0x3ff;
}

template<>
template<>
inline void BitCompression<10>::decompress_block<2, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080020180800100ull), static_cast<long long int>(0x8080050480800302ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000010ull), static_cast<long long int>(0x4000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ff000003ffull), static_cast<long long int>(0x3ff000003ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<10>::per_block<2, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<10>::decompress_block<2, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080070680800605ull), static_cast<long long int>(0x80800A0980800807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000010ull), static_cast<long long int>(0x4000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ff000003ffull), static_cast<long long int>(0x3ff000003ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<10>::per_block<2, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<10>::decompress_block<2, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800C0B80800B0Aull), static_cast<long long int>(0x80800F0E80800D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000010ull), static_cast<long long int>(0x4000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ff000003ffull), static_cast<long long int>(0x3ff000003ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<10>::per_block<2, 2>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<10>::decompress<2>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<2, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 0>();

    BitCompression::decompress_block<2, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 1>();

    BitCompression::decompress_block<2, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<2>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 4
//Bits: 10


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<10>::remaining<4>() { return 12; }

template<>
template<>
inline const int BitCompression<10>::base_shift<4>() { return 15; }

template<>
template<>
inline int BitCompression<10>::next_offset<4>() { return 6; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<10>::block_count<4>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<10>::overlap_value<4>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 4) & 0x3ff;
}

template<>
template<>
inline void BitCompression<10>::decompress_block<4, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080020180800100ull), static_cast<long long int>(0x8080050480800403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000004ull), static_cast<long long int>(0x1000000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ff000003ffull), static_cast<long long int>(0x3ff000003ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<10>::per_block<4, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<10>::decompress_block<4, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080070680800605ull), static_cast<long long int>(0x80800A0980800908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000004ull), static_cast<long long int>(0x1000000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ff000003ffull), static_cast<long long int>(0x3ff000003ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<10>::per_block<4, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<10>::decompress_block<4, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800C0B80800B0Aull), static_cast<long long int>(0x80800F0E80800E0Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000004ull), static_cast<long long int>(0x1000000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ff000003ffull), static_cast<long long int>(0x3ff000003ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<10>::per_block<4, 2>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<10>::decompress<4>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<4, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 0>();

    BitCompression::decompress_block<4, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 1>();

    BitCompression::decompress_block<4, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<4>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 6
//Bits: 10


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<10>::remaining<6>() { return 12; }

template<>
template<>
inline const int BitCompression<10>::base_shift<6>() { return 15; }

template<>
template<>
inline int BitCompression<10>::next_offset<6>() { return 8; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<10>::block_count<6>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<10>::overlap_value<6>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 6) & 0x3ff;
}

template<>
template<>
inline void BitCompression<10>::decompress_block<6, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080030280800100ull), static_cast<long long int>(0x8080050480800403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000001ull), static_cast<long long int>(0x400000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ff000003ffull), static_cast<long long int>(0x3ff000003ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<10>::per_block<6, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<10>::decompress_block<6, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080080780800605ull), static_cast<long long int>(0x80800A0980800908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000001ull), static_cast<long long int>(0x400000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ff000003ffull), static_cast<long long int>(0x3ff000003ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<10>::per_block<6, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<10>::decompress_block<6, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800D0C80800B0Aull), static_cast<long long int>(0x80800F0E80800E0Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000001ull), static_cast<long long int>(0x400000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ff000003ffull), static_cast<long long int>(0x3ff000003ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<10>::per_block<6, 2>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<10>::decompress<6>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<6, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 0>();

    BitCompression::decompress_block<6, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 1>();

    BitCompression::decompress_block<6, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<6>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 8
//Bits: 10


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<10>::remaining<8>() { return 12; }

template<>
template<>
inline const int BitCompression<10>::base_shift<8>() { return 15; }

template<>
template<>
inline int BitCompression<10>::next_offset<8>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<10>::block_count<8>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<10>::overlap_value<8>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0x3ff;
}

template<>
template<>
inline void BitCompression<10>::decompress_block<8, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080030280800201ull), static_cast<long long int>(0x8080050480800403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ff000003ffull), static_cast<long long int>(0x3ff000003ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<10>::per_block<8, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<10>::decompress_block<8, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080080780800706ull), static_cast<long long int>(0x80800A0980800908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ff000003ffull), static_cast<long long int>(0x3ff000003ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<10>::per_block<8, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<10>::decompress_block<8, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800D0C80800C0Bull), static_cast<long long int>(0x80800F0E80800E0Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ff000003ffull), static_cast<long long int>(0x3ff000003ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<10>::per_block<8, 2>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<10>::decompress<8>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<8, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 0>();

    BitCompression::decompress_block<8, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 1>();

    BitCompression::decompress_block<8, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 2>();


}


template<>
inline void BitCompression<10>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<10>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<10>::remaining<0>()  + 1 ;

    BitCompression<10>::decompress<2>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<10>::remaining<2>()  + 1 ;

    BitCompression<10>::decompress<4>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<10>::remaining<4>()  + 1 ;

    BitCompression<10>::decompress<6>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<10>::remaining<6>()  + 1 ;

    BitCompression<10>::decompress<8>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<10>::remaining<8>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 0
//Bits: 11


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<11>::remaining<0>() { return 11; }

template<>
template<>
inline const int BitCompression<11>::base_shift<0>() { return 15; }

template<>
template<>
inline int BitCompression<11>::next_offset<0>() { return 4; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<11>::block_count<0>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<11>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 1) & 0x7ff;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080020180800100ull), static_cast<long long int>(0x8080050480040302ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8008070680800605ull), static_cast<long long int>(0x80800A0980800908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<0, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<0, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800D0C80800C0Bull), static_cast<long long int>(0x80808080800F0E0Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<0, 2>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<11>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();

    BitCompression::decompress_block<0, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<0>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 4
//Bits: 11


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<11>::remaining<4>() { return 11; }

template<>
template<>
inline const int BitCompression<11>::base_shift<4>() { return 15; }

template<>
template<>
inline int BitCompression<11>::next_offset<4>() { return 8; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<11>::block_count<4>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<11>::overlap_value<4>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 5) & 0x7ff;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<4, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8003020180800100ull), static_cast<long long int>(0x8080050480800403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<4, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<4, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080080780800706ull), static_cast<long long int>(0x80800B0A800A0908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<4, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<4, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800E0D0C80800C0Bull), static_cast<long long int>(0x8080808080800F0Eull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<4, 2>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<11>::decompress<4>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<4, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 0>();

    BitCompression::decompress_block<4, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 1>();

    BitCompression::decompress_block<4, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<4>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 8
//Bits: 11


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<11>::remaining<8>() { return 10; }

template<>
template<>
inline const int BitCompression<11>::base_shift<8>() { return 13; }

template<>
template<>
inline int BitCompression<11>::next_offset<8>() { return 1; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<11>::block_count<8>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<11>::overlap_value<8>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 14) & 0x7ff;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<8, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080030280800201ull), static_cast<long long int>(0x8080060580050403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<8, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<8, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8009080780800706ull), static_cast<long long int>(0x80800B0A80800A09ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<8, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<8, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800E0D80800D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<8, 2>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<11>::decompress<8>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<8, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 0>();

    BitCompression::decompress_block<8, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 1>();

    BitCompression::decompress_block<8, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<8>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 1
//Bits: 11


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<11>::remaining<1>() { return 11; }

template<>
template<>
inline const int BitCompression<11>::base_shift<1>() { return 15; }

template<>
template<>
inline int BitCompression<11>::next_offset<1>() { return 5; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<11>::block_count<1>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<11>::overlap_value<1>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 2) & 0x7ff;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<1, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080020180800100ull), static_cast<long long int>(0x8080050480040302ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<1, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<1, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080080780800605ull), static_cast<long long int>(0x800B0A0980800908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<1, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<1, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800D0C80800C0Bull), static_cast<long long int>(0x80808080800F0E0Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<1, 2>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<11>::decompress<1>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<1, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 0>();

    BitCompression::decompress_block<1, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 1>();

    BitCompression::decompress_block<1, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<1>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 5
//Bits: 11


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<11>::remaining<5>() { return 11; }

template<>
template<>
inline const int BitCompression<11>::base_shift<5>() { return 15; }

template<>
template<>
inline int BitCompression<11>::next_offset<5>() { return 9; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<11>::block_count<5>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<11>::overlap_value<5>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 6) & 0x7ff;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<5, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080030280800100ull), static_cast<long long int>(0x8006050480800403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<5, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<5, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080080780800706ull), static_cast<long long int>(0x80800B0A800A0908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<5, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<5, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800E0D80800C0Bull), static_cast<long long int>(0x8080808080800F0Eull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<5, 2>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<11>::decompress<5>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<5, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 0>();

    BitCompression::decompress_block<5, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 1>();

    BitCompression::decompress_block<5, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<5>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 9
//Bits: 11


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<11>::remaining<9>() { return 10; }

template<>
template<>
inline const int BitCompression<11>::base_shift<9>() { return 13; }

template<>
template<>
inline int BitCompression<11>::next_offset<9>() { return 2; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<11>::block_count<9>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<11>::overlap_value<9>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 15) & 0x7ff;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<9, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080030280800201ull), static_cast<long long int>(0x8080060580050403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<9, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<9, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080090880800706ull), static_cast<long long int>(0x800C0B0A80800A09ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<9, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<9, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800E0D80800D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<9, 2>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<11>::decompress<9>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<9, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<9, 0>();

    BitCompression::decompress_block<9, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<9, 1>();

    BitCompression::decompress_block<9, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<9, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<9>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 2
//Bits: 11


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<11>::remaining<2>() { return 11; }

template<>
template<>
inline const int BitCompression<11>::base_shift<2>() { return 15; }

template<>
template<>
inline int BitCompression<11>::next_offset<2>() { return 6; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<11>::block_count<2>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<11>::overlap_value<2>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 3) & 0x7ff;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<2, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080020180800100ull), static_cast<long long int>(0x8080050480800403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<2, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<2, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080080780070605ull), static_cast<long long int>(0x800B0A0980800908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<2, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<2, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800D0C80800C0Bull), static_cast<long long int>(0x8080808080800F0Eull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<2, 2>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<11>::decompress<2>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<2, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 0>();

    BitCompression::decompress_block<2, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 1>();

    BitCompression::decompress_block<2, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<2>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 6
//Bits: 11


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<11>::remaining<6>() { return 11; }

template<>
template<>
inline const int BitCompression<11>::base_shift<6>() { return 15; }

template<>
template<>
inline int BitCompression<11>::next_offset<6>() { return 10; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<11>::block_count<6>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<11>::overlap_value<6>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 7) & 0x7ff;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<6, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080030280020100ull), static_cast<long long int>(0x8006050480800403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<6, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<6, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080080780800706ull), static_cast<long long int>(0x80800B0A80800A09ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<6, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<6, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800E0D800D0C0Bull), static_cast<long long int>(0x8080808080800F0Eull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<6, 2>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<11>::decompress<6>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<6, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 0>();

    BitCompression::decompress_block<6, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 1>();

    BitCompression::decompress_block<6, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<6>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 10
//Bits: 11


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<11>::remaining<10>() { return 10; }

template<>
template<>
inline const int BitCompression<11>::base_shift<10>() { return 13; }

template<>
template<>
inline int BitCompression<11>::next_offset<10>() { return 3; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<11>::block_count<10>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<11>::overlap_value<10>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 16) & 0x7ff;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<10, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080030280800201ull), static_cast<long long int>(0x8080060580800504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<10, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<10, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080090880080706ull), static_cast<long long int>(0x800C0B0A80800A09ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<10, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<10, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800E0D80800D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<10, 2>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<11>::decompress<10>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<10, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 0>();

    BitCompression::decompress_block<10, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 1>();

    BitCompression::decompress_block<10, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<10>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 3
//Bits: 11


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<11>::remaining<3>() { return 11; }

template<>
template<>
inline const int BitCompression<11>::base_shift<3>() { return 15; }

template<>
template<>
inline int BitCompression<11>::next_offset<3>() { return 7; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<11>::block_count<3>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<11>::overlap_value<3>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 4) & 0x7ff;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<3, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8003020180800100ull), static_cast<long long int>(0x8080050480800403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<3, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<3, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080080780070605ull), static_cast<long long int>(0x80800B0A80800908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000001ull), static_cast<long long int>(0x8000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<3, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<3, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800E0D0C80800C0Bull), static_cast<long long int>(0x8080808080800F0Eull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<3, 2>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<11>::decompress<3>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<3, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 0>();

    BitCompression::decompress_block<3, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 1>();

    BitCompression::decompress_block<3, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<3>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 7
//Bits: 11


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<11>::remaining<7>() { return 11; }

template<>
template<>
inline const int BitCompression<11>::base_shift<7>() { return 15; }

template<>
template<>
inline int BitCompression<11>::next_offset<7>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<11>::block_count<7>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<11>::overlap_value<7>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0x7ff;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<7, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080030280020100ull), static_cast<long long int>(0x8080060580800403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000001ull), static_cast<long long int>(0x8000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<7, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<7, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8009080780800706ull), static_cast<long long int>(0x80800B0A80800A09ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<7, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<11>::decompress_block<7, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800E0D800D0C0Bull), static_cast<long long int>(0x8080808080800F0Eull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000001ull), static_cast<long long int>(0x8000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ff000007ffull), static_cast<long long int>(0x7ff000007ffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<11>::per_block<7, 2>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<11>::decompress<7>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<7, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<7, 0>();

    BitCompression::decompress_block<7, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<7, 1>();

    BitCompression::decompress_block<7, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<7, 2>();


}


template<>
inline void BitCompression<11>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<11>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<11>::remaining<0>()  + 1 ;

    BitCompression<11>::decompress<4>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<11>::remaining<4>()  + 1 ;

    BitCompression<11>::decompress<8>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<11>::remaining<8>()  + 1 ;

    BitCompression<11>::decompress<1>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<11>::remaining<1>()  + 1 ;

    BitCompression<11>::decompress<5>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<11>::remaining<5>()  + 1 ;

    BitCompression<11>::decompress<9>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<11>::remaining<9>()  + 1 ;

    BitCompression<11>::decompress<2>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<11>::remaining<2>()  + 1 ;

    BitCompression<11>::decompress<6>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<11>::remaining<6>()  + 1 ;

    BitCompression<11>::decompress<10>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<11>::remaining<10>()  + 1 ;

    BitCompression<11>::decompress<3>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<11>::remaining<3>()  + 1 ;

    BitCompression<11>::decompress<7>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<11>::remaining<7>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 0
//Bits: 12


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<12>::remaining<0>() { return 10; }

template<>
template<>
inline const int BitCompression<12>::base_shift<0>() { return 15; }

template<>
template<>
inline int BitCompression<12>::next_offset<0>() { return 4; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<12>::block_count<0>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<12>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 0) & 0xfff;
}

template<>
template<>
inline void BitCompression<12>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080020180800100ull), static_cast<long long int>(0x8080050480800403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000010ull), static_cast<long long int>(0x100000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xfff00000fffull), static_cast<long long int>(0xfff00000fffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<12>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<12>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080080780800706ull), static_cast<long long int>(0x80800B0A80800A09ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000010ull), static_cast<long long int>(0x100000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xfff00000fffull), static_cast<long long int>(0xfff00000fffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<12>::per_block<0, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<12>::decompress_block<0, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800E0D80800D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000010ull), static_cast<long long int>(0x100000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xfff00000fffull), static_cast<long long int>(0xfff00000fffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<12>::per_block<0, 2>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<12>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();

    BitCompression::decompress_block<0, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<0>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 4
//Bits: 12


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<12>::remaining<4>() { return 10; }

template<>
template<>
inline const int BitCompression<12>::base_shift<4>() { return 15; }

template<>
template<>
inline int BitCompression<12>::next_offset<4>() { return 8; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<12>::block_count<4>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<12>::overlap_value<4>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 4) & 0xfff;
}

template<>
template<>
inline void BitCompression<12>::decompress_block<4, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080030280800100ull), static_cast<long long int>(0x8080060580800403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000001ull), static_cast<long long int>(0x1000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xfff00000fffull), static_cast<long long int>(0xfff00000fffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<12>::per_block<4, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<12>::decompress_block<4, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080090880800706ull), static_cast<long long int>(0x80800C0B80800A09ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000001ull), static_cast<long long int>(0x1000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xfff00000fffull), static_cast<long long int>(0xfff00000fffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<12>::per_block<4, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<12>::decompress_block<4, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800F0E80800D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000001ull), static_cast<long long int>(0x1000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xfff00000fffull), static_cast<long long int>(0xfff00000fffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<12>::per_block<4, 2>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<12>::decompress<4>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<4, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 0>();

    BitCompression::decompress_block<4, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 1>();

    BitCompression::decompress_block<4, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<4>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 8
//Bits: 12


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<12>::remaining<8>() { return 10; }

template<>
template<>
inline const int BitCompression<12>::base_shift<8>() { return 15; }

template<>
template<>
inline int BitCompression<12>::next_offset<8>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<12>::block_count<8>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<12>::overlap_value<8>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0xfff;
}

template<>
template<>
inline void BitCompression<12>::decompress_block<8, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080030280800201ull), static_cast<long long int>(0x8080060580800504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000010ull), static_cast<long long int>(0x100000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xfff00000fffull), static_cast<long long int>(0xfff00000fffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<12>::per_block<8, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<12>::decompress_block<8, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080090880800807ull), static_cast<long long int>(0x80800C0B80800B0Aull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000010ull), static_cast<long long int>(0x100000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xfff00000fffull), static_cast<long long int>(0xfff00000fffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<12>::per_block<8, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<12>::decompress_block<8, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800F0E80800E0Dull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000010ull), static_cast<long long int>(0x100000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xfff00000fffull), static_cast<long long int>(0xfff00000fffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<12>::per_block<8, 2>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<12>::decompress<8>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<8, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 0>();

    BitCompression::decompress_block<8, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 1>();

    BitCompression::decompress_block<8, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 2>();


}


template<>
inline void BitCompression<12>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<12>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<12>::remaining<0>()  + 1 ;

    BitCompression<12>::decompress<4>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<12>::remaining<4>()  + 1 ;

    BitCompression<12>::decompress<8>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<12>::remaining<8>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 0
//Bits: 13


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<13>::remaining<0>() { return 9; }

template<>
template<>
inline const int BitCompression<13>::base_shift<0>() { return 14; }

template<>
template<>
inline int BitCompression<13>::next_offset<0>() { return 2; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<13>::block_count<0>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<13>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 5) & 0x1fff;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8003020180800100ull), static_cast<long long int>(0x8006050480800403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000080ull), static_cast<long long int>(0x100000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080090880080706ull), static_cast<long long int>(0x80800C0B800B0A09ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<0, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<0, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080808080800E0Dull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000080ull), static_cast<long long int>(0x100000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<0, 2>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<13>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();

    BitCompression::decompress_block<0, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<0>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 2
//Bits: 13


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<13>::remaining<2>() { return 9; }

template<>
template<>
inline const int BitCompression<13>::base_shift<2>() { return 14; }

template<>
template<>
inline int BitCompression<13>::next_offset<2>() { return 4; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<13>::block_count<2>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<13>::overlap_value<2>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 7) & 0x1fff;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<2, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8003020180800100ull), static_cast<long long int>(0x8080060580050403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<2, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<2, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080090880080706ull), static_cast<long long int>(0x800D0C0B80800B0Aull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<2, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<2, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080808080800E0Dull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<2, 2>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<13>::decompress<2>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<2, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 0>();

    BitCompression::decompress_block<2, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 1>();

    BitCompression::decompress_block<2, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<2>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 4
//Bits: 13


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<13>::remaining<4>() { return 9; }

template<>
template<>
inline const int BitCompression<13>::base_shift<4>() { return 14; }

template<>
template<>
inline int BitCompression<13>::next_offset<4>() { return 6; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<13>::block_count<4>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<13>::overlap_value<4>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 9) & 0x1fff;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<4, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080030280020100ull), static_cast<long long int>(0x8080060580050403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<4, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<4, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800A090880800807ull), static_cast<long long int>(0x800D0C0B80800B0Aull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000080ull), static_cast<long long int>(0x100000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<4, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<4, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800F0E0Dull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<4, 2>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<13>::decompress<4>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<4, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 0>();

    BitCompression::decompress_block<4, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 1>();

    BitCompression::decompress_block<4, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<4>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 6
//Bits: 13


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<13>::remaining<6>() { return 9; }

template<>
template<>
inline const int BitCompression<13>::base_shift<6>() { return 14; }

template<>
template<>
inline int BitCompression<13>::next_offset<6>() { return 8; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<13>::block_count<6>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<13>::overlap_value<6>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 11) & 0x1fff;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<6, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080030280020100ull), static_cast<long long int>(0x8007060580800504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<6, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<6, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800A090880800807ull), static_cast<long long int>(0x80800D0C800C0B0Aull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<6, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<6, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800F0E0Dull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<6, 2>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<13>::decompress<6>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<6, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 0>();

    BitCompression::decompress_block<6, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 1>();

    BitCompression::decompress_block<6, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<6>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 8
//Bits: 13


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<13>::remaining<8>() { return 9; }

template<>
template<>
inline const int BitCompression<13>::base_shift<8>() { return 14; }

template<>
template<>
inline int BitCompression<13>::next_offset<8>() { return 10; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<13>::block_count<8>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<13>::overlap_value<8>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 13) & 0x1fff;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<8, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280800201ull), static_cast<long long int>(0x8007060580800504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000080ull), static_cast<long long int>(0x100000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<8, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<8, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800A0980090807ull), static_cast<long long int>(0x80800D0C800C0B0Aull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<8, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<8, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080808080800F0Eull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000080ull), static_cast<long long int>(0x100000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<8, 2>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<13>::decompress<8>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<8, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 0>();

    BitCompression::decompress_block<8, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 1>();

    BitCompression::decompress_block<8, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<8>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 10
//Bits: 13


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<13>::remaining<10>() { return 9; }

template<>
template<>
inline const int BitCompression<13>::base_shift<10>() { return 14; }

template<>
template<>
inline int BitCompression<13>::next_offset<10>() { return 12; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<13>::block_count<10>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<13>::overlap_value<10>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 15) & 0x1fff;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<10, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280800201ull), static_cast<long long int>(0x8080070680060504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<10, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<10, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800A0980090807ull), static_cast<long long int>(0x800E0D0C80800C0Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<10, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<10, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080808080800F0Eull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<10, 2>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<13>::decompress<10>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<10, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 0>();

    BitCompression::decompress_block<10, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 1>();

    BitCompression::decompress_block<10, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<10>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 12
//Bits: 13


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<13>::remaining<12>() { return 8; }

template<>
template<>
inline const int BitCompression<13>::base_shift<12>() { return 13; }

template<>
template<>
inline int BitCompression<13>::next_offset<12>() { return 1; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<13>::block_count<12>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<13>::overlap_value<12>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 12) & 0x1fff;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<12, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080040380030201ull), static_cast<long long int>(0x8080070680060504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<12, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<12, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800B0A0980800908ull), static_cast<long long int>(0x800E0D0C80800C0Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000080ull), static_cast<long long int>(0x100000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<12, 1>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<13>::decompress<12>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<12, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<12, 0>();

    BitCompression::decompress_block<12, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<12, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<12>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 1
//Bits: 13


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<13>::remaining<1>() { return 9; }

template<>
template<>
inline const int BitCompression<13>::base_shift<1>() { return 14; }

template<>
template<>
inline int BitCompression<13>::next_offset<1>() { return 3; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<13>::block_count<1>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<13>::overlap_value<1>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 6) & 0x1fff;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<1, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8003020180800100ull), static_cast<long long int>(0x8080060580800403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<1, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<1, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080090880080706ull), static_cast<long long int>(0x800D0C0B800B0A09ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<1, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<1, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080808080800E0Dull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<1, 2>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<13>::decompress<1>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<1, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 0>();

    BitCompression::decompress_block<1, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 1>();

    BitCompression::decompress_block<1, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<1>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 3
//Bits: 13


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<13>::remaining<3>() { return 9; }

template<>
template<>
inline const int BitCompression<13>::base_shift<3>() { return 14; }

template<>
template<>
inline int BitCompression<13>::next_offset<3>() { return 5; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<13>::block_count<3>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<13>::overlap_value<3>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0x1fff;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<3, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080030280800100ull), static_cast<long long int>(0x8080060580050403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<3, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<3, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800A090880080706ull), static_cast<long long int>(0x800D0C0B80800B0Aull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<3, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<3, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080808080800E0Dull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<3, 2>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<13>::decompress<3>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<3, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 0>();

    BitCompression::decompress_block<3, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 1>();

    BitCompression::decompress_block<3, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<3>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 5
//Bits: 13


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<13>::remaining<5>() { return 9; }

template<>
template<>
inline const int BitCompression<13>::base_shift<5>() { return 14; }

template<>
template<>
inline int BitCompression<13>::next_offset<5>() { return 7; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<13>::block_count<5>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<13>::overlap_value<5>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 10) & 0x1fff;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<5, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080030280020100ull), static_cast<long long int>(0x8007060580050403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<5, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<5, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800A090880800807ull), static_cast<long long int>(0x80800D0C80800B0Aull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<5, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<5, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800F0E0Dull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<5, 2>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<13>::decompress<5>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<5, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 0>();

    BitCompression::decompress_block<5, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 1>();

    BitCompression::decompress_block<5, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<5>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 7
//Bits: 13


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<13>::remaining<7>() { return 9; }

template<>
template<>
inline const int BitCompression<13>::base_shift<7>() { return 14; }

template<>
template<>
inline int BitCompression<13>::next_offset<7>() { return 9; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<13>::block_count<7>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<13>::overlap_value<7>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 12) & 0x1fff;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<7, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280020100ull), static_cast<long long int>(0x8007060580800504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<7, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<7, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800A0980800807ull), static_cast<long long int>(0x80800D0C800C0B0Aull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<7, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<7, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800F0E0Dull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<7, 2>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<13>::decompress<7>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<7, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<7, 0>();

    BitCompression::decompress_block<7, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<7, 1>();

    BitCompression::decompress_block<7, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<7, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<7>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 9
//Bits: 13


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<13>::remaining<9>() { return 9; }

template<>
template<>
inline const int BitCompression<13>::base_shift<9>() { return 14; }

template<>
template<>
inline int BitCompression<13>::next_offset<9>() { return 11; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<13>::block_count<9>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<13>::overlap_value<9>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 14) & 0x1fff;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<9, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280800201ull), static_cast<long long int>(0x8080070680800504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<9, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<9, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800A0980090807ull), static_cast<long long int>(0x800E0D0C800C0B0Aull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<9, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<9, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080808080800F0Eull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<9, 2>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<13>::decompress<9>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<9, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<9, 0>();

    BitCompression::decompress_block<9, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<9, 1>();

    BitCompression::decompress_block<9, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<9, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<9>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 11
//Bits: 13


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<13>::remaining<11>() { return 9; }

template<>
template<>
inline const int BitCompression<13>::base_shift<11>() { return 14; }

template<>
template<>
inline int BitCompression<13>::next_offset<11>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<13>::block_count<11>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<13>::overlap_value<11>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 16) & 0x1fff;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<11, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080040380800201ull), static_cast<long long int>(0x8080070680060504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<11, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<11, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800B0A0980090807ull), static_cast<long long int>(0x800E0D0C80800C0Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<11, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<13>::decompress_block<11, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080808080800F0Eull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fff00001fffull), static_cast<long long int>(0x1fff00001fffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<13>::per_block<11, 2>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<13>::decompress<11>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<11, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<11, 0>();

    BitCompression::decompress_block<11, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<11, 1>();

    BitCompression::decompress_block<11, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<11, 2>();


}


template<>
inline void BitCompression<13>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<13>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<13>::remaining<0>()  + 1 ;

    BitCompression<13>::decompress<2>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<13>::remaining<2>()  + 1 ;

    BitCompression<13>::decompress<4>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<13>::remaining<4>()  + 1 ;

    BitCompression<13>::decompress<6>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<13>::remaining<6>()  + 1 ;

    BitCompression<13>::decompress<8>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<13>::remaining<8>()  + 1 ;

    BitCompression<13>::decompress<10>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<13>::remaining<10>()  + 1 ;

    BitCompression<13>::decompress<12>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<13>::remaining<12>()  + 1 ;

    BitCompression<13>::decompress<1>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<13>::remaining<1>()  + 1 ;

    BitCompression<13>::decompress<3>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<13>::remaining<3>()  + 1 ;

    BitCompression<13>::decompress<5>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<13>::remaining<5>()  + 1 ;

    BitCompression<13>::decompress<7>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<13>::remaining<7>()  + 1 ;

    BitCompression<13>::decompress<9>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<13>::remaining<9>()  + 1 ;

    BitCompression<13>::decompress<11>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<13>::remaining<11>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 0
//Bits: 14


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<14>::remaining<0>() { return 9; }

template<>
template<>
inline const int BitCompression<14>::base_shift<0>() { return 15; }

template<>
template<>
inline int BitCompression<14>::next_offset<0>() { return 12; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<14>::block_count<0>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<14>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 6) & 0x3fff;
}

template<>
template<>
inline void BitCompression<14>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8003020180800100ull), static_cast<long long int>(0x8080060580050403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000040ull), static_cast<long long int>(0x1000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fff00003fffull), static_cast<long long int>(0x3fff00003fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<14>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<14>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800A090880800807ull), static_cast<long long int>(0x80800D0C800C0B0Aull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000040ull), static_cast<long long int>(0x1000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fff00003fffull), static_cast<long long int>(0x3fff00003fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<14>::per_block<0, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<14>::decompress_block<0, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080808080800F0Eull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000040ull), static_cast<long long int>(0x1000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fff00003fffull), static_cast<long long int>(0x3fff00003fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<14>::per_block<0, 2>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<14>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();

    BitCompression::decompress_block<0, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 2>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<0>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 12
//Bits: 14


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<14>::remaining<12>() { return 8; }

template<>
template<>
inline const int BitCompression<14>::base_shift<12>() { return 14; }

template<>
template<>
inline int BitCompression<14>::next_offset<12>() { return 10; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<14>::block_count<12>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<14>::overlap_value<12>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 12) & 0x3fff;
}

template<>
template<>
inline void BitCompression<14>::decompress_block<12, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080040380030201ull), static_cast<long long int>(0x8008070680800605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000004ull), static_cast<long long int>(0x100000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fff00003fffull), static_cast<long long int>(0x3fff00003fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<14>::per_block<12, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<14>::decompress_block<12, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800B0A800A0908ull), static_cast<long long int>(0x800F0E0D80800D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000004ull), static_cast<long long int>(0x100000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fff00003fffull), static_cast<long long int>(0x3fff00003fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<14>::per_block<12, 1>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<14>::decompress<12>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<12, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<12, 0>();

    BitCompression::decompress_block<12, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<12, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<12>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 10
//Bits: 14


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<14>::remaining<10>() { return 8; }

template<>
template<>
inline const int BitCompression<14>::base_shift<10>() { return 14; }

template<>
template<>
inline int BitCompression<14>::next_offset<10>() { return 8; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<14>::block_count<10>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<14>::overlap_value<10>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 10) & 0x3fff;
}

template<>
template<>
inline void BitCompression<14>::decompress_block<10, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080040380800201ull), static_cast<long long int>(0x8008070680060504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000010ull), static_cast<long long int>(0x400000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fff00003fffull), static_cast<long long int>(0x3fff00003fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<14>::per_block<10, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<14>::decompress_block<10, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800B0A80800908ull), static_cast<long long int>(0x800F0E0D800D0C0Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000010ull), static_cast<long long int>(0x400000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fff00003fffull), static_cast<long long int>(0x3fff00003fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<14>::per_block<10, 1>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<14>::decompress<10>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<10, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 0>();

    BitCompression::decompress_block<10, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<10>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 8
//Bits: 14


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<14>::remaining<8>() { return 8; }

template<>
template<>
inline const int BitCompression<14>::base_shift<8>() { return 14; }

template<>
template<>
inline int BitCompression<14>::next_offset<8>() { return 6; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<14>::block_count<8>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<14>::overlap_value<8>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0x3fff;
}

template<>
template<>
inline void BitCompression<14>::decompress_block<8, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280800201ull), static_cast<long long int>(0x8080070680060504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000040ull), static_cast<long long int>(0x1000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fff00003fffull), static_cast<long long int>(0x3fff00003fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<14>::per_block<8, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<14>::decompress_block<8, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800B0A0980800908ull), static_cast<long long int>(0x80800E0D800D0C0Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000040ull), static_cast<long long int>(0x1000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fff00003fffull), static_cast<long long int>(0x3fff00003fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<14>::per_block<8, 1>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<14>::decompress<8>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<8, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 0>();

    BitCompression::decompress_block<8, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<8>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 6
//Bits: 14


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<14>::remaining<6>() { return 8; }

template<>
template<>
inline const int BitCompression<14>::base_shift<6>() { return 14; }

template<>
template<>
inline int BitCompression<14>::next_offset<6>() { return 4; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<14>::block_count<6>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<14>::overlap_value<6>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 6) & 0x3fff;
}

template<>
template<>
inline void BitCompression<14>::decompress_block<6, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280020100ull), static_cast<long long int>(0x8080070680800504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000001ull), static_cast<long long int>(0x4000000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fff00003fffull), static_cast<long long int>(0x3fff00003fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<14>::per_block<6, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<14>::decompress_block<6, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800B0A0980090807ull), static_cast<long long int>(0x80800E0D80800C0Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000001ull), static_cast<long long int>(0x4000000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fff00003fffull), static_cast<long long int>(0x3fff00003fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<14>::per_block<6, 1>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<14>::decompress<6>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<6, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 0>();

    BitCompression::decompress_block<6, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<6>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 4
//Bits: 14


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<14>::remaining<4>() { return 8; }

template<>
template<>
inline const int BitCompression<14>::base_shift<4>() { return 14; }

template<>
template<>
inline int BitCompression<14>::next_offset<4>() { return 2; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<14>::block_count<4>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<14>::overlap_value<4>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 4) & 0x3fff;
}

template<>
template<>
inline void BitCompression<14>::decompress_block<4, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080030280020100ull), static_cast<long long int>(0x8007060580800504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000004ull), static_cast<long long int>(0x100000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fff00003fffull), static_cast<long long int>(0x3fff00003fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<14>::per_block<4, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<14>::decompress_block<4, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800A0980090807ull), static_cast<long long int>(0x800E0D0C80800C0Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000004ull), static_cast<long long int>(0x100000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fff00003fffull), static_cast<long long int>(0x3fff00003fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<14>::per_block<4, 1>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<14>::decompress<4>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<4, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 0>();

    BitCompression::decompress_block<4, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<4>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 2
//Bits: 14


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<14>::remaining<2>() { return 9; }

template<>
template<>
inline const int BitCompression<14>::base_shift<2>() { return 15; }

template<>
template<>
inline int BitCompression<14>::next_offset<2>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<14>::block_count<2>() { return 3; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<14>::overlap_value<2>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0x3fff;
}

template<>
template<>
inline void BitCompression<14>::decompress_block<2, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080030280800100ull), static_cast<long long int>(0x8007060580050403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000010ull), static_cast<long long int>(0x400000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fff00003fffull), static_cast<long long int>(0x3fff00003fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<14>::per_block<2, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<14>::decompress_block<2, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800A0980800807ull), static_cast<long long int>(0x800E0D0C800C0B0Aull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000010ull), static_cast<long long int>(0x400000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fff00003fffull), static_cast<long long int>(0x3fff00003fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<14>::per_block<2, 1>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<14>::decompress_block<2, 2>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080808080800F0Eull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000010ull), static_cast<long long int>(0x400000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fff00003fffull), static_cast<long long int>(0x3fff00003fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<14>::per_block<2, 2>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<14>::decompress<2>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<2, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 0>();

    BitCompression::decompress_block<2, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 1>();

    BitCompression::decompress_block<2, 2>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 2>();


}


template<>
inline void BitCompression<14>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<14>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<14>::remaining<0>()  + 1 ;

    BitCompression<14>::decompress<12>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<14>::remaining<12>()  + 1 ;

    BitCompression<14>::decompress<10>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<14>::remaining<10>()  + 1 ;

    BitCompression<14>::decompress<8>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<14>::remaining<8>()  + 1 ;

    BitCompression<14>::decompress<6>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<14>::remaining<6>()  + 1 ;

    BitCompression<14>::decompress<4>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<14>::remaining<4>()  + 1 ;

    BitCompression<14>::decompress<2>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<14>::remaining<2>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 0
//Bits: 15


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<15>::remaining<0>() { return 8; }

template<>
template<>
inline const int BitCompression<15>::base_shift<0>() { return 15; }

template<>
template<>
inline int BitCompression<15>::next_offset<0>() { return 7; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<15>::block_count<0>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<15>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 0) & 0x7fff;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8003020180800100ull), static_cast<long long int>(0x8007060580050403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000080ull), static_cast<long long int>(0x400000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800B0A0980090807ull), static_cast<long long int>(0x80800E0D800D0C0Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<0, 1>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<15>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<0>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 7
//Bits: 15


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<15>::remaining<7>() { return 8; }

template<>
template<>
inline const int BitCompression<15>::base_shift<7>() { return 15; }

template<>
template<>
inline int BitCompression<15>::next_offset<7>() { return 14; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<15>::block_count<7>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<15>::overlap_value<7>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 7) & 0x7fff;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<7, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280020100ull), static_cast<long long int>(0x8008070680060504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<7, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<7, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800C0B0A800A0908ull), static_cast<long long int>(0x80800F0E80800D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<7, 1>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<15>::decompress<7>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<7, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<7, 0>();

    BitCompression::decompress_block<7, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<7, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<7>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 14
//Bits: 15


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<15>::remaining<14>() { return 7; }

template<>
template<>
inline const int BitCompression<15>::base_shift<14>() { return 13; }

template<>
template<>
inline int BitCompression<15>::next_offset<14>() { return 6; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<15>::block_count<14>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<15>::overlap_value<14>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 15) & 0x7fff;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<14, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380030201ull), static_cast<long long int>(0x8009080780070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<14, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<14, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800C0B800B0A09ull), static_cast<long long int>(0x8080808080800E0Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000020ull), static_cast<long long int>(0x100000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<14, 1>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<15>::decompress<14>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<14, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<14, 0>();

    BitCompression::decompress_block<14, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<14, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<14>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 6
//Bits: 15


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<15>::remaining<6>() { return 8; }

template<>
template<>
inline const int BitCompression<15>::base_shift<6>() { return 15; }

template<>
template<>
inline int BitCompression<15>::next_offset<6>() { return 13; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<15>::block_count<6>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<15>::overlap_value<6>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 6) & 0x7fff;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<6, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280020100ull), static_cast<long long int>(0x8008070680060504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<6, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<6, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800B0A800A0908ull), static_cast<long long int>(0x800F0E0D80800D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000020ull), static_cast<long long int>(0x100000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<6, 1>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<15>::decompress<6>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<6, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 0>();

    BitCompression::decompress_block<6, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<6>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 13
//Bits: 15


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<15>::remaining<13>() { return 7; }

template<>
template<>
inline const int BitCompression<15>::base_shift<13>() { return 13; }

template<>
template<>
inline int BitCompression<15>::next_offset<13>() { return 5; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<15>::block_count<13>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<15>::overlap_value<13>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 14) & 0x7fff;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<13, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380030201ull), static_cast<long long int>(0x8009080780070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<13, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<13, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800C0B80800A09ull), static_cast<long long int>(0x80808080800E0D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000040ull), static_cast<long long int>(0x200000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<13, 1>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<15>::decompress<13>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<13, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<13, 0>();

    BitCompression::decompress_block<13, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<13, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<13>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 5
//Bits: 15


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<15>::remaining<5>() { return 8; }

template<>
template<>
inline const int BitCompression<15>::base_shift<5>() { return 15; }

template<>
template<>
inline int BitCompression<15>::next_offset<5>() { return 12; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<15>::block_count<5>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<15>::overlap_value<5>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 5) & 0x7fff;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<5, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280020100ull), static_cast<long long int>(0x8008070680060504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<5, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<5, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800B0A80800908ull), static_cast<long long int>(0x800F0E0D800D0C0Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000040ull), static_cast<long long int>(0x200000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<5, 1>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<15>::decompress<5>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<5, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 0>();

    BitCompression::decompress_block<5, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<5>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 12
//Bits: 15


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<15>::remaining<12>() { return 7; }

template<>
template<>
inline const int BitCompression<15>::base_shift<12>() { return 13; }

template<>
template<>
inline int BitCompression<15>::next_offset<12>() { return 4; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<15>::block_count<12>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<15>::overlap_value<12>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 13) & 0x7fff;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<12, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380030201ull), static_cast<long long int>(0x8080080780070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<12, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<12, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800C0B0A80800A09ull), static_cast<long long int>(0x80808080800E0D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000080ull), static_cast<long long int>(0x400000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<12, 1>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<15>::decompress<12>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<12, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<12, 0>();

    BitCompression::decompress_block<12, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<12, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<12>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 4
//Bits: 15


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<15>::remaining<4>() { return 8; }

template<>
template<>
inline const int BitCompression<15>::base_shift<4>() { return 15; }

template<>
template<>
inline int BitCompression<15>::next_offset<4>() { return 11; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<15>::block_count<4>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<15>::overlap_value<4>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 4) & 0x7fff;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<4, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280020100ull), static_cast<long long int>(0x8080070680060504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<4, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<4, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800B0A0980800908ull), static_cast<long long int>(0x800F0E0D800D0C0Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000080ull), static_cast<long long int>(0x400000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<4, 1>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<15>::decompress<4>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<4, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 0>();

    BitCompression::decompress_block<4, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<4>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 11
//Bits: 15


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<15>::remaining<11>() { return 7; }

template<>
template<>
inline const int BitCompression<15>::base_shift<11>() { return 13; }

template<>
template<>
inline int BitCompression<15>::next_offset<11>() { return 3; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<15>::block_count<11>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<15>::overlap_value<11>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 12) & 0x7fff;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<11, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380030201ull), static_cast<long long int>(0x8080080780800605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<11, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<11, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800C0B0A800A0908ull), static_cast<long long int>(0x80808080800E0D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<11, 1>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<15>::decompress<11>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<11, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<11, 0>();

    BitCompression::decompress_block<11, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<11, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<11>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 3
//Bits: 15


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<15>::remaining<3>() { return 8; }

template<>
template<>
inline const int BitCompression<15>::base_shift<3>() { return 15; }

template<>
template<>
inline int BitCompression<15>::next_offset<3>() { return 10; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<15>::block_count<3>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<15>::overlap_value<3>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 3) & 0x7fff;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<3, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280020100ull), static_cast<long long int>(0x8080070680800504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<3, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<3, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800B0A0980090807ull), static_cast<long long int>(0x800F0E0D800D0C0Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<3, 1>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<15>::decompress<3>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<3, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 0>();

    BitCompression::decompress_block<3, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<3>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 10
//Bits: 15


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<15>::remaining<10>() { return 7; }

template<>
template<>
inline const int BitCompression<15>::base_shift<10>() { return 13; }

template<>
template<>
inline int BitCompression<15>::next_offset<10>() { return 2; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<15>::block_count<10>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<15>::overlap_value<10>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 11) & 0x7fff;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<10, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080040380030201ull), static_cast<long long int>(0x8008070680800605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000020ull), static_cast<long long int>(0x100000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<10, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<10, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800C0B0A800A0908ull), static_cast<long long int>(0x80808080800E0D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<10, 1>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<15>::decompress<10>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<10, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 0>();

    BitCompression::decompress_block<10, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<10>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 2
//Bits: 15


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<15>::remaining<2>() { return 8; }

template<>
template<>
inline const int BitCompression<15>::base_shift<2>() { return 15; }

template<>
template<>
inline int BitCompression<15>::next_offset<2>() { return 9; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<15>::block_count<2>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<15>::overlap_value<2>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 2) & 0x7fff;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<2, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080030280020100ull), static_cast<long long int>(0x8007060580800504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000020ull), static_cast<long long int>(0x100000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<2, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<2, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800B0A0980090807ull), static_cast<long long int>(0x800F0E0D800D0C0Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<2, 1>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<15>::decompress<2>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<2, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 0>();

    BitCompression::decompress_block<2, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<2>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 9
//Bits: 15


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<15>::remaining<9>() { return 7; }

template<>
template<>
inline const int BitCompression<15>::base_shift<9>() { return 13; }

template<>
template<>
inline int BitCompression<15>::next_offset<9>() { return 1; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<15>::block_count<9>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<15>::overlap_value<9>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 10) & 0x7fff;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<9, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080040380800201ull), static_cast<long long int>(0x8008070680060504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000040ull), static_cast<long long int>(0x200000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<9, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<9, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800C0B0A800A0908ull), static_cast<long long int>(0x80808080800E0D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<9, 1>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<15>::decompress<9>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<9, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<9, 0>();

    BitCompression::decompress_block<9, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<9, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<9>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 1
//Bits: 15


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<15>::remaining<1>() { return 8; }

template<>
template<>
inline const int BitCompression<15>::base_shift<1>() { return 15; }

template<>
template<>
inline int BitCompression<15>::next_offset<1>() { return 8; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<15>::block_count<1>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<15>::overlap_value<1>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 1) & 0x7fff;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<1, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080030280800100ull), static_cast<long long int>(0x8007060580050403ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000040ull), static_cast<long long int>(0x200000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<1, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<1, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800B0A0980090807ull), static_cast<long long int>(0x800F0E0D800D0C0Bull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<1, 1>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<15>::decompress<1>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<1, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 0>();

    BitCompression::decompress_block<1, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<1>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 8
//Bits: 15


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<15>::remaining<8>() { return 8; }

template<>
template<>
inline const int BitCompression<15>::base_shift<8>() { return 15; }

template<>
template<>
inline int BitCompression<15>::next_offset<8>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<15>::block_count<8>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<15>::overlap_value<8>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0x7fff;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<8, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280800201ull), static_cast<long long int>(0x8008070680060504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000080ull), static_cast<long long int>(0x400000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<8, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<15>::decompress_block<8, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800C0B0A800A0908ull), static_cast<long long int>(0x80800F0E800E0D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fff00007fffull), static_cast<long long int>(0x7fff00007fffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<15>::per_block<8, 1>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<15>::decompress<8>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<8, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 0>();

    BitCompression::decompress_block<8, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 1>();


}


template<>
inline void BitCompression<15>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<15>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<15>::remaining<0>()  + 1 ;

    BitCompression<15>::decompress<7>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<15>::remaining<7>()  + 1 ;

    BitCompression<15>::decompress<14>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<15>::remaining<14>()  + 1 ;

    BitCompression<15>::decompress<6>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<15>::remaining<6>()  + 1 ;

    BitCompression<15>::decompress<13>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<15>::remaining<13>()  + 1 ;

    BitCompression<15>::decompress<5>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<15>::remaining<5>()  + 1 ;

    BitCompression<15>::decompress<12>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<15>::remaining<12>()  + 1 ;

    BitCompression<15>::decompress<4>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<15>::remaining<4>()  + 1 ;

    BitCompression<15>::decompress<11>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<15>::remaining<11>()  + 1 ;

    BitCompression<15>::decompress<3>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<15>::remaining<3>()  + 1 ;

    BitCompression<15>::decompress<10>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<15>::remaining<10>()  + 1 ;

    BitCompression<15>::decompress<2>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<15>::remaining<2>()  + 1 ;

    BitCompression<15>::decompress<9>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<15>::remaining<9>()  + 1 ;

    BitCompression<15>::decompress<1>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<15>::remaining<1>()  + 1 ;

    BitCompression<15>::decompress<8>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<15>::remaining<8>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 16
//Offset: 0
//Bits: 16


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<16>::remaining<0>() { return 8; }

template<>
template<>
inline const int BitCompression<16>::base_shift<0>() { return 16; }

template<>
template<>
inline int BitCompression<16>::next_offset<0>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<16>::block_count<0>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<16>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 0) & 0xffff;
}

template<>
template<>
inline void BitCompression<16>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8080030280800100ull), static_cast<long long int>(0x8080070680800504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xffff0000ffffull), static_cast<long long int>(0xffff0000ffffull)};
    static const int shift_mask = 0;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<16>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<16>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80800B0A80800908ull), static_cast<long long int>(0x80800F0E80800D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xffff0000ffffull), static_cast<long long int>(0xffff0000ffffull)};
    static const int shift_mask = 0;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<16>::per_block<0, 1>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<16>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();


}


template<>
inline void BitCompression<16>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<16>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<16>::remaining<0>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 0
//Bits: 17


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<17>::remaining<0>() { return 7; }

template<>
template<>
inline const int BitCompression<17>::base_shift<0>() { return 14; }

template<>
template<>
inline int BitCompression<17>::next_offset<0>() { return 8; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<17>::block_count<0>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<17>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 7) & 0x1ffff;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280020100ull), static_cast<long long int>(0x8008070680060504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800C0B0A800A0908ull), static_cast<long long int>(0x80808080800E0D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<0, 1>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<17>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<0>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 8
//Bits: 17


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<17>::remaining<8>() { return 7; }

template<>
template<>
inline const int BitCompression<17>::base_shift<8>() { return 14; }

template<>
template<>
inline int BitCompression<17>::next_offset<8>() { return 16; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<17>::block_count<8>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<17>::overlap_value<8>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 15) & 0x1ffff;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<8, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380030201ull), static_cast<long long int>(0x8009080780070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<8, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<8, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800D0C0B800B0A09ull), static_cast<long long int>(0x80808080800F0E0Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<8, 1>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<17>::decompress<8>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<8, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 0>();

    BitCompression::decompress_block<8, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<8>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 16
//Bits: 17


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<17>::remaining<16>() { return 6; }

template<>
template<>
inline const int BitCompression<17>::base_shift<16>() { return 12; }

template<>
template<>
inline int BitCompression<17>::next_offset<16>() { return 7; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<17>::block_count<16>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<17>::overlap_value<16>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 22) & 0x1ffff;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<16, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8006050480040302ull), static_cast<long long int>(0x800A090880080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<16, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<16, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800E0D0C800C0B0Aull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<16, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<17>::decompress<16>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<16, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<16, 0>();

    BitCompression::decompress_block<16, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<16, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<16>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 7
//Bits: 17


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<17>::remaining<7>() { return 7; }

template<>
template<>
inline const int BitCompression<17>::base_shift<7>() { return 14; }

template<>
template<>
inline int BitCompression<17>::next_offset<7>() { return 15; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<17>::block_count<7>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<17>::overlap_value<7>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 14) & 0x1ffff;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<7, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380020100ull), static_cast<long long int>(0x8009080780070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000001ull), static_cast<long long int>(0x2000000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<7, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<7, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800D0C0B800B0A09ull), static_cast<long long int>(0x80808080800F0E0Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<7, 1>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<17>::decompress<7>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<7, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<7, 0>();

    BitCompression::decompress_block<7, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<7, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<7>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 15
//Bits: 17


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<17>::remaining<15>() { return 6; }

template<>
template<>
inline const int BitCompression<17>::base_shift<15>() { return 12; }

template<>
template<>
inline int BitCompression<17>::next_offset<15>() { return 6; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<17>::block_count<15>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<17>::overlap_value<15>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 21) & 0x1ffff;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<15, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8006050480030201ull), static_cast<long long int>(0x800A090880080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000001ull), static_cast<long long int>(0x2000000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<15, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<15, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800E0D0C800C0B0Aull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<15, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<17>::decompress<15>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<15, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<15, 0>();

    BitCompression::decompress_block<15, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<15, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<15>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 6
//Bits: 17


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<17>::remaining<6>() { return 7; }

template<>
template<>
inline const int BitCompression<17>::base_shift<6>() { return 14; }

template<>
template<>
inline int BitCompression<17>::next_offset<6>() { return 14; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<17>::block_count<6>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<17>::overlap_value<6>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 13) & 0x1ffff;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<6, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280020100ull), static_cast<long long int>(0x8009080780070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000002ull), static_cast<long long int>(0x4000000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<6, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<6, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800D0C0B800B0A09ull), static_cast<long long int>(0x80808080800F0E0Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<6, 1>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<17>::decompress<6>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<6, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 0>();

    BitCompression::decompress_block<6, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<6>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 14
//Bits: 17


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<17>::remaining<14>() { return 6; }

template<>
template<>
inline const int BitCompression<17>::base_shift<14>() { return 12; }

template<>
template<>
inline int BitCompression<17>::next_offset<14>() { return 5; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<17>::block_count<14>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<17>::overlap_value<14>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 20) & 0x1ffff;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<14, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380030201ull), static_cast<long long int>(0x800A090880080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000002ull), static_cast<long long int>(0x4000000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<14, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<14, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800E0D0C800C0B0Aull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<14, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<17>::decompress<14>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<14, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<14, 0>();

    BitCompression::decompress_block<14, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<14, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<14>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 5
//Bits: 17


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<17>::remaining<5>() { return 7; }

template<>
template<>
inline const int BitCompression<17>::base_shift<5>() { return 14; }

template<>
template<>
inline int BitCompression<17>::next_offset<5>() { return 13; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<17>::block_count<5>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<17>::overlap_value<5>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 12) & 0x1ffff;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<5, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280020100ull), static_cast<long long int>(0x8009080780060504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000004ull), static_cast<long long int>(0x8000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<5, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<5, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800D0C0B800B0A09ull), static_cast<long long int>(0x80808080800F0E0Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<5, 1>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<17>::decompress<5>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<5, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 0>();

    BitCompression::decompress_block<5, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<5>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 13
//Bits: 17


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<17>::remaining<13>() { return 6; }

template<>
template<>
inline const int BitCompression<17>::base_shift<13>() { return 12; }

template<>
template<>
inline int BitCompression<17>::next_offset<13>() { return 4; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<17>::block_count<13>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<17>::overlap_value<13>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 19) & 0x1ffff;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<13, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380030201ull), static_cast<long long int>(0x800A090880070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000004ull), static_cast<long long int>(0x8000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<13, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<13, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800E0D0C800C0B0Aull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<13, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<17>::decompress<13>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<13, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<13, 0>();

    BitCompression::decompress_block<13, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<13, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<13>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 4
//Bits: 17


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<17>::remaining<4>() { return 7; }

template<>
template<>
inline const int BitCompression<17>::base_shift<4>() { return 14; }

template<>
template<>
inline int BitCompression<17>::next_offset<4>() { return 12; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<17>::block_count<4>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<17>::overlap_value<4>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 11) & 0x1ffff;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<4, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280020100ull), static_cast<long long int>(0x8008070680060504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<4, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<4, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800D0C0B800B0A09ull), static_cast<long long int>(0x80808080800F0E0Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<4, 1>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<17>::decompress<4>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<4, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 0>();

    BitCompression::decompress_block<4, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<4>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 12
//Bits: 17


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<17>::remaining<12>() { return 6; }

template<>
template<>
inline const int BitCompression<17>::base_shift<12>() { return 12; }

template<>
template<>
inline int BitCompression<17>::next_offset<12>() { return 3; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<17>::block_count<12>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<17>::overlap_value<12>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 18) & 0x1ffff;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<12, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380030201ull), static_cast<long long int>(0x8009080780070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<12, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<12, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800E0D0C800C0B0Aull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<12, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<17>::decompress<12>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<12, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<12, 0>();

    BitCompression::decompress_block<12, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<12, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<12>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 3
//Bits: 17


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<17>::remaining<3>() { return 7; }

template<>
template<>
inline const int BitCompression<17>::base_shift<3>() { return 14; }

template<>
template<>
inline int BitCompression<17>::next_offset<3>() { return 11; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<17>::block_count<3>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<17>::overlap_value<3>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 10) & 0x1ffff;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<3, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280020100ull), static_cast<long long int>(0x8008070680060504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<3, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<3, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800D0C0B800A0908ull), static_cast<long long int>(0x80808080800F0E0Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000001ull), static_cast<long long int>(0x2000000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<3, 1>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<17>::decompress<3>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<3, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 0>();

    BitCompression::decompress_block<3, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<3>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 11
//Bits: 17


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<17>::remaining<11>() { return 6; }

template<>
template<>
inline const int BitCompression<17>::base_shift<11>() { return 12; }

template<>
template<>
inline int BitCompression<17>::next_offset<11>() { return 2; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<17>::block_count<11>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<17>::overlap_value<11>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 17) & 0x1ffff;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<11, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380030201ull), static_cast<long long int>(0x8009080780070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<11, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<11, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800E0D0C800B0A09ull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000001ull), static_cast<long long int>(0x2000000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<11, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<17>::decompress<11>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<11, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<11, 0>();

    BitCompression::decompress_block<11, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<11, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<11>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 2
//Bits: 17


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<17>::remaining<2>() { return 7; }

template<>
template<>
inline const int BitCompression<17>::base_shift<2>() { return 14; }

template<>
template<>
inline int BitCompression<17>::next_offset<2>() { return 10; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<17>::block_count<2>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<17>::overlap_value<2>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 9) & 0x1ffff;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<2, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280020100ull), static_cast<long long int>(0x8008070680060504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<2, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<2, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800C0B0A800A0908ull), static_cast<long long int>(0x80808080800F0E0Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000002ull), static_cast<long long int>(0x4000000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<2, 1>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<17>::decompress<2>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<2, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 0>();

    BitCompression::decompress_block<2, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<2>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 10
//Bits: 17


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<17>::remaining<10>() { return 6; }

template<>
template<>
inline const int BitCompression<17>::base_shift<10>() { return 12; }

template<>
template<>
inline int BitCompression<17>::next_offset<10>() { return 1; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<17>::block_count<10>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<17>::overlap_value<10>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 16) & 0x1ffff;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<10, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380030201ull), static_cast<long long int>(0x8009080780070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<10, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<10, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800D0C0B800B0A09ull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000002ull), static_cast<long long int>(0x4000000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<10, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<17>::decompress<10>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<10, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 0>();

    BitCompression::decompress_block<10, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<10>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 1
//Bits: 17


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<17>::remaining<1>() { return 7; }

template<>
template<>
inline const int BitCompression<17>::base_shift<1>() { return 14; }

template<>
template<>
inline int BitCompression<17>::next_offset<1>() { return 9; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<17>::block_count<1>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<17>::overlap_value<1>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0x1ffff;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<1, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280020100ull), static_cast<long long int>(0x8008070680060504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<1, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<1, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800C0B0A800A0908ull), static_cast<long long int>(0x80808080800E0D0Cull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000004ull), static_cast<long long int>(0x8000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<1, 1>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<17>::decompress<1>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<1, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 0>();

    BitCompression::decompress_block<1, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<1>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 9
//Bits: 17


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<17>::remaining<9>() { return 7; }

template<>
template<>
inline const int BitCompression<17>::base_shift<9>() { return 14; }

template<>
template<>
inline int BitCompression<17>::next_offset<9>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<17>::block_count<9>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<17>::overlap_value<9>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 16) & 0x1ffff;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<9, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380030201ull), static_cast<long long int>(0x8009080780070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<9, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<17>::decompress_block<9, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800D0C0B800B0A09ull), static_cast<long long int>(0x80808080800F0E0Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000004ull), static_cast<long long int>(0x8000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffff0001ffffull), static_cast<long long int>(0x1ffff0001ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<17>::per_block<9, 1>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<17>::decompress<9>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<9, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<9, 0>();

    BitCompression::decompress_block<9, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<9, 1>();


}


template<>
inline void BitCompression<17>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<17>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<17>::remaining<0>()  + 1 ;

    BitCompression<17>::decompress<8>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<17>::remaining<8>()  + 1 ;

    BitCompression<17>::decompress<16>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<17>::remaining<16>()  + 1 ;

    BitCompression<17>::decompress<7>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<17>::remaining<7>()  + 1 ;

    BitCompression<17>::decompress<15>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<17>::remaining<15>()  + 1 ;

    BitCompression<17>::decompress<6>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<17>::remaining<6>()  + 1 ;

    BitCompression<17>::decompress<14>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<17>::remaining<14>()  + 1 ;

    BitCompression<17>::decompress<5>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<17>::remaining<5>()  + 1 ;

    BitCompression<17>::decompress<13>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<17>::remaining<13>()  + 1 ;

    BitCompression<17>::decompress<4>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<17>::remaining<4>()  + 1 ;

    BitCompression<17>::decompress<12>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<17>::remaining<12>()  + 1 ;

    BitCompression<17>::decompress<3>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<17>::remaining<3>()  + 1 ;

    BitCompression<17>::decompress<11>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<17>::remaining<11>()  + 1 ;

    BitCompression<17>::decompress<2>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<17>::remaining<2>()  + 1 ;

    BitCompression<17>::decompress<10>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<17>::remaining<10>()  + 1 ;

    BitCompression<17>::decompress<1>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<17>::remaining<1>()  + 1 ;

    BitCompression<17>::decompress<9>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<17>::remaining<9>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 0
//Bits: 18


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<18>::remaining<0>() { return 7; }

template<>
template<>
inline const int BitCompression<18>::base_shift<0>() { return 15; }

template<>
template<>
inline int BitCompression<18>::next_offset<0>() { return 16; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<18>::block_count<0>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<18>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 6) & 0x3ffff;
}

template<>
template<>
inline void BitCompression<18>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280020100ull), static_cast<long long int>(0x8008070680060504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffff0003ffffull), static_cast<long long int>(0x3ffff0003ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<18>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<18>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800D0C0B800B0A09ull), static_cast<long long int>(0x80808080800F0E0Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffff0003ffffull), static_cast<long long int>(0x3ffff0003ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<18>::per_block<0, 1>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<18>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<0>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 16
//Bits: 18


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<18>::remaining<16>() { return 6; }

template<>
template<>
inline const int BitCompression<18>::base_shift<16>() { return 13; }

template<>
template<>
inline int BitCompression<18>::next_offset<16>() { return 14; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<18>::block_count<16>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<18>::overlap_value<16>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 20) & 0x3ffff;
}

template<>
template<>
inline void BitCompression<18>::decompress_block<16, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8006050480040302ull), static_cast<long long int>(0x800A090880080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffff0003ffffull), static_cast<long long int>(0x3ffff0003ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<18>::per_block<16, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<18>::decompress_block<16, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800F0E0D800D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffff0003ffffull), static_cast<long long int>(0x3ffff0003ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<18>::per_block<16, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<18>::decompress<16>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<16, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<16, 0>();

    BitCompression::decompress_block<16, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<16, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<16>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 14
//Bits: 18


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<18>::remaining<14>() { return 6; }

template<>
template<>
inline const int BitCompression<18>::base_shift<14>() { return 13; }

template<>
template<>
inline int BitCompression<18>::next_offset<14>() { return 12; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<18>::block_count<14>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<18>::overlap_value<14>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 18) & 0x3ffff;
}

template<>
template<>
inline void BitCompression<18>::decompress_block<14, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8006050480030201ull), static_cast<long long int>(0x800A090880080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000001ull), static_cast<long long int>(0x400000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffff0003ffffull), static_cast<long long int>(0x3ffff0003ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<18>::per_block<14, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<18>::decompress_block<14, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800F0E0D800C0B0Aull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000001ull), static_cast<long long int>(0x400000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffff0003ffffull), static_cast<long long int>(0x3ffff0003ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<18>::per_block<14, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<18>::decompress<14>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<14, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<14, 0>();

    BitCompression::decompress_block<14, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<14, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<14>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 12
//Bits: 18


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<18>::remaining<12>() { return 6; }

template<>
template<>
inline const int BitCompression<18>::base_shift<12>() { return 13; }

template<>
template<>
inline int BitCompression<18>::next_offset<12>() { return 10; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<18>::block_count<12>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<18>::overlap_value<12>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 16) & 0x3ffff;
}

template<>
template<>
inline void BitCompression<18>::decompress_block<12, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380030201ull), static_cast<long long int>(0x800A090880080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000004ull), static_cast<long long int>(0x1000000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffff0003ffffull), static_cast<long long int>(0x3ffff0003ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<18>::per_block<12, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<18>::decompress_block<12, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800E0D0C800C0B0Aull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000004ull), static_cast<long long int>(0x1000000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffff0003ffffull), static_cast<long long int>(0x3ffff0003ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<18>::per_block<12, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<18>::decompress<12>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<12, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<12, 0>();

    BitCompression::decompress_block<12, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<12, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<12>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 10
//Bits: 18


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<18>::remaining<10>() { return 6; }

template<>
template<>
inline const int BitCompression<18>::base_shift<10>() { return 13; }

template<>
template<>
inline int BitCompression<18>::next_offset<10>() { return 8; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<18>::block_count<10>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<18>::overlap_value<10>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 14) & 0x3ffff;
}

template<>
template<>
inline void BitCompression<18>::decompress_block<10, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380030201ull), static_cast<long long int>(0x800A090880070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000010ull), static_cast<long long int>(0x4000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffff0003ffffull), static_cast<long long int>(0x3ffff0003ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<18>::per_block<10, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<18>::decompress_block<10, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800E0D0C800C0B0Aull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000010ull), static_cast<long long int>(0x4000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffff0003ffffull), static_cast<long long int>(0x3ffff0003ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<18>::per_block<10, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<18>::decompress<10>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<10, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 0>();

    BitCompression::decompress_block<10, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<10>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 8
//Bits: 18


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<18>::remaining<8>() { return 6; }

template<>
template<>
inline const int BitCompression<18>::base_shift<8>() { return 13; }

template<>
template<>
inline int BitCompression<18>::next_offset<8>() { return 6; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<18>::block_count<8>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<18>::overlap_value<8>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 12) & 0x3ffff;
}

template<>
template<>
inline void BitCompression<18>::decompress_block<8, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380030201ull), static_cast<long long int>(0x8009080780070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffff0003ffffull), static_cast<long long int>(0x3ffff0003ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<18>::per_block<8, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<18>::decompress_block<8, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800E0D0C800C0B0Aull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffff0003ffffull), static_cast<long long int>(0x3ffff0003ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<18>::per_block<8, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<18>::decompress<8>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<8, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 0>();

    BitCompression::decompress_block<8, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<8>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 6
//Bits: 18


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<18>::remaining<6>() { return 6; }

template<>
template<>
inline const int BitCompression<18>::base_shift<6>() { return 13; }

template<>
template<>
inline int BitCompression<18>::next_offset<6>() { return 4; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<18>::block_count<6>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<18>::overlap_value<6>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 10) & 0x3ffff;
}

template<>
template<>
inline void BitCompression<18>::decompress_block<6, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380020100ull), static_cast<long long int>(0x8009080780070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000001ull), static_cast<long long int>(0x400000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffff0003ffffull), static_cast<long long int>(0x3ffff0003ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<18>::per_block<6, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<18>::decompress_block<6, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800E0D0C800B0A09ull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000001ull), static_cast<long long int>(0x400000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffff0003ffffull), static_cast<long long int>(0x3ffff0003ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<18>::per_block<6, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<18>::decompress<6>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<6, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 0>();

    BitCompression::decompress_block<6, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<6>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 4
//Bits: 18


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<18>::remaining<4>() { return 6; }

template<>
template<>
inline const int BitCompression<18>::base_shift<4>() { return 13; }

template<>
template<>
inline int BitCompression<18>::next_offset<4>() { return 2; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<18>::block_count<4>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<18>::overlap_value<4>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0x3ffff;
}

template<>
template<>
inline void BitCompression<18>::decompress_block<4, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280020100ull), static_cast<long long int>(0x8009080780070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000004ull), static_cast<long long int>(0x1000000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffff0003ffffull), static_cast<long long int>(0x3ffff0003ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<18>::per_block<4, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<18>::decompress_block<4, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800D0C0B800B0A09ull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000004ull), static_cast<long long int>(0x1000000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffff0003ffffull), static_cast<long long int>(0x3ffff0003ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<18>::per_block<4, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<18>::decompress<4>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<4, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 0>();

    BitCompression::decompress_block<4, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<4>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 2
//Bits: 18


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<18>::remaining<2>() { return 7; }

template<>
template<>
inline const int BitCompression<18>::base_shift<2>() { return 15; }

template<>
template<>
inline int BitCompression<18>::next_offset<2>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<18>::block_count<2>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<18>::overlap_value<2>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0x3ffff;
}

template<>
template<>
inline void BitCompression<18>::decompress_block<2, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280020100ull), static_cast<long long int>(0x8009080780060504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000010ull), static_cast<long long int>(0x4000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffff0003ffffull), static_cast<long long int>(0x3ffff0003ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<18>::per_block<2, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<18>::decompress_block<2, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800D0C0B800B0A09ull), static_cast<long long int>(0x80808080800F0E0Dull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000010ull), static_cast<long long int>(0x4000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffff0003ffffull), static_cast<long long int>(0x3ffff0003ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<18>::per_block<2, 1>()
{ 
     return 3;
}


template<>
template<>
inline void BitCompression<18>::decompress<2>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<2, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 0>();

    BitCompression::decompress_block<2, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 1>();


}


template<>
inline void BitCompression<18>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<18>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<18>::remaining<0>()  + 1 ;

    BitCompression<18>::decompress<16>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<18>::remaining<16>()  + 1 ;

    BitCompression<18>::decompress<14>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<18>::remaining<14>()  + 1 ;

    BitCompression<18>::decompress<12>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<18>::remaining<12>()  + 1 ;

    BitCompression<18>::decompress<10>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<18>::remaining<10>()  + 1 ;

    BitCompression<18>::decompress<8>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<18>::remaining<8>()  + 1 ;

    BitCompression<18>::decompress<6>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<18>::remaining<6>()  + 1 ;

    BitCompression<18>::decompress<4>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<18>::remaining<4>()  + 1 ;

    BitCompression<18>::decompress<2>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<18>::remaining<2>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 0
//Bits: 19


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<19>::remaining<0>() { return 6; }

template<>
template<>
inline const int BitCompression<19>::base_shift<0>() { return 14; }

template<>
template<>
inline int BitCompression<19>::next_offset<0>() { return 5; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<19>::block_count<0>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<19>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 2) & 0x7ffff;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280020100ull), static_cast<long long int>(0x8009080707060504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0E0D0C0B800B0A09ull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<0, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<19>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<0>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 5
//Bits: 19


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<19>::remaining<5>() { return 6; }

template<>
template<>
inline const int BitCompression<19>::base_shift<5>() { return 14; }

template<>
template<>
inline int BitCompression<19>::next_offset<5>() { return 10; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<19>::block_count<5>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<19>::overlap_value<5>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 7) & 0x7ffff;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<5, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380020100ull), static_cast<long long int>(0x0A09080780070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<5, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<5, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800E0D0C800C0B0Aull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<5, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<19>::decompress<5>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<5, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 0>();

    BitCompression::decompress_block<5, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<5>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 10
//Bits: 19


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<19>::remaining<10>() { return 6; }

template<>
template<>
inline const int BitCompression<19>::base_shift<10>() { return 14; }

template<>
template<>
inline int BitCompression<19>::next_offset<10>() { return 15; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<19>::block_count<10>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<19>::overlap_value<10>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 12) & 0x7ffff;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<10, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380030201ull), static_cast<long long int>(0x800A090880080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<10, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<10, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800F0E0D0D0C0B0Aull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<10, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<19>::decompress<10>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<10, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 0>();

    BitCompression::decompress_block<10, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<10>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 11
//Offset: 15
//Bits: 19


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<19>::remaining<15>() { return 5; }

template<>
template<>
inline const int BitCompression<19>::base_shift<15>() { return 11; }

template<>
template<>
inline int BitCompression<19>::next_offset<15>() { return 1; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<19>::block_count<15>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<19>::overlap_value<15>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 22) & 0x7ffff;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<15, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8006050404030201ull), static_cast<long long int>(0x800B0A0980080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000001ull), static_cast<long long int>(0x8000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<15, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<15, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<15, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<19>::decompress<15>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<15, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<15, 0>();

    BitCompression::decompress_block<15, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<15, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  11);
    BitCompression::overlap_value<15>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 1
//Bits: 19


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<19>::remaining<1>() { return 6; }

template<>
template<>
inline const int BitCompression<19>::base_shift<1>() { return 14; }

template<>
template<>
inline int BitCompression<19>::next_offset<1>() { return 6; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<19>::block_count<1>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<19>::overlap_value<1>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 3) & 0x7ffff;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<1, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280020100ull), static_cast<long long int>(0x8009080707060504ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<1, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<1, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800E0D0C800B0A09ull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<1, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<19>::decompress<1>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<1, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 0>();

    BitCompression::decompress_block<1, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<1>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 6
//Bits: 19


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<19>::remaining<6>() { return 6; }

template<>
template<>
inline const int BitCompression<19>::base_shift<6>() { return 14; }

template<>
template<>
inline int BitCompression<19>::next_offset<6>() { return 11; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<19>::block_count<6>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<19>::overlap_value<6>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0x7ffff;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<6, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040303020100ull), static_cast<long long int>(0x0A09080780070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<6, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<6, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800E0D0C800C0B0Aull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<6, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<19>::decompress<6>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<6, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 0>();

    BitCompression::decompress_block<6, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<6>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 11
//Bits: 19


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<19>::remaining<11>() { return 6; }

template<>
template<>
inline const int BitCompression<19>::base_shift<11>() { return 14; }

template<>
template<>
inline int BitCompression<19>::next_offset<11>() { return 16; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<19>::block_count<11>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<19>::overlap_value<11>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 13) & 0x7ffff;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<11, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040380030201ull), static_cast<long long int>(0x800A090880080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<11, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<11, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800F0E0D0D0C0B0Aull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000001ull), static_cast<long long int>(0x8000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<11, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<19>::decompress<11>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<11, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<11, 0>();

    BitCompression::decompress_block<11, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<11, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<11>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 11
//Offset: 16
//Bits: 19


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<19>::remaining<16>() { return 5; }

template<>
template<>
inline const int BitCompression<19>::base_shift<16>() { return 11; }

template<>
template<>
inline int BitCompression<19>::next_offset<16>() { return 2; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<19>::block_count<16>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<19>::overlap_value<16>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 23) & 0x7ffff;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<16, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8006050480040302ull), static_cast<long long int>(0x800B0A0909080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<16, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<16, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<16, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<19>::decompress<16>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<16, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<16, 0>();

    BitCompression::decompress_block<16, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<16, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  11);
    BitCompression::overlap_value<16>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 2
//Bits: 19


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<19>::remaining<2>() { return 6; }

template<>
template<>
inline const int BitCompression<19>::base_shift<2>() { return 14; }

template<>
template<>
inline int BitCompression<19>::next_offset<2>() { return 7; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<19>::block_count<2>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<19>::overlap_value<2>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 4) & 0x7ffff;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<2, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280020100ull), static_cast<long long int>(0x8009080780070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<2, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<2, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800E0D0C0C0B0A09ull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<2, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<19>::decompress<2>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<2, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 0>();

    BitCompression::decompress_block<2, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<2>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 7
//Bits: 19


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<19>::remaining<7>() { return 6; }

template<>
template<>
inline const int BitCompression<19>::base_shift<7>() { return 14; }

template<>
template<>
inline int BitCompression<19>::next_offset<7>() { return 12; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<19>::block_count<7>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<19>::overlap_value<7>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 9) & 0x7ffff;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<7, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040303020100ull), static_cast<long long int>(0x800A090880070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000001ull), static_cast<long long int>(0x8000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<7, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<7, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0F0E0D0C800C0B0Aull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<7, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<19>::decompress<7>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<7, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<7, 0>();

    BitCompression::decompress_block<7, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<7, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<7>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 12
//Bits: 19


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<19>::remaining<12>() { return 6; }

template<>
template<>
inline const int BitCompression<19>::base_shift<12>() { return 14; }

template<>
template<>
inline int BitCompression<19>::next_offset<12>() { return 17; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<19>::block_count<12>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<19>::overlap_value<12>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 14) & 0x7ffff;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<12, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040380030201ull), static_cast<long long int>(0x800A090880080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<12, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<12, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800F0E0D800D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<12, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<19>::decompress<12>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<12, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<12, 0>();

    BitCompression::decompress_block<12, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<12, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<12>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 11
//Offset: 17
//Bits: 19


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<19>::remaining<17>() { return 5; }

template<>
template<>
inline const int BitCompression<19>::base_shift<17>() { return 11; }

template<>
template<>
inline int BitCompression<19>::next_offset<17>() { return 3; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<19>::block_count<17>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<19>::overlap_value<17>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 24) & 0x7ffff;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<17, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8006050480040302ull), static_cast<long long int>(0x800B0A0909080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<17, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<17, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<17, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<19>::decompress<17>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<17, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<17, 0>();

    BitCompression::decompress_block<17, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<17, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  11);
    BitCompression::overlap_value<17>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 3
//Bits: 19


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<19>::remaining<3>() { return 6; }

template<>
template<>
inline const int BitCompression<19>::base_shift<3>() { return 14; }

template<>
template<>
inline int BitCompression<19>::next_offset<3>() { return 8; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<19>::block_count<3>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<19>::overlap_value<3>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 5) & 0x7ffff;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<3, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0504030280020100ull), static_cast<long long int>(0x8009080780070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<3, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<3, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800E0D0C0C0B0A09ull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000001ull), static_cast<long long int>(0x8000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<3, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<19>::decompress<3>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<3, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 0>();

    BitCompression::decompress_block<3, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<3>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 8
//Bits: 19


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<19>::remaining<8>() { return 6; }

template<>
template<>
inline const int BitCompression<19>::base_shift<8>() { return 14; }

template<>
template<>
inline int BitCompression<19>::next_offset<8>() { return 13; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<19>::block_count<8>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<19>::overlap_value<8>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 10) & 0x7ffff;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<8, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380030201ull), static_cast<long long int>(0x800A090808070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<8, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<8, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0F0E0D0C800C0B0Aull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<8, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<19>::decompress<8>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<8, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 0>();

    BitCompression::decompress_block<8, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<8>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 13
//Bits: 19


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<19>::remaining<13>() { return 6; }

template<>
template<>
inline const int BitCompression<19>::base_shift<13>() { return 14; }

template<>
template<>
inline int BitCompression<19>::next_offset<13>() { return 18; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<19>::block_count<13>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<19>::overlap_value<13>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 15) & 0x7ffff;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<13, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8006050480030201ull), static_cast<long long int>(0x0B0A090880080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<13, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<13, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800F0E0D800D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<13, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<19>::decompress<13>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<13, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<13, 0>();

    BitCompression::decompress_block<13, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<13, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<13>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 11
//Offset: 18
//Bits: 19


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<19>::remaining<18>() { return 5; }

template<>
template<>
inline const int BitCompression<19>::base_shift<18>() { return 11; }

template<>
template<>
inline int BitCompression<19>::next_offset<18>() { return 4; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<19>::block_count<18>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<19>::overlap_value<18>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 25) & 0x7ffff;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<18, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8006050480040302ull), static_cast<long long int>(0x800B0A0980090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<18, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<18, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800E0D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<18, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<19>::decompress<18>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<18, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<18, 0>();

    BitCompression::decompress_block<18, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<18, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  11);
    BitCompression::overlap_value<18>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 4
//Bits: 19


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<19>::remaining<4>() { return 6; }

template<>
template<>
inline const int BitCompression<19>::base_shift<4>() { return 14; }

template<>
template<>
inline int BitCompression<19>::next_offset<4>() { return 9; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<19>::block_count<4>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<19>::overlap_value<4>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 6) & 0x7ffff;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<4, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0504030280020100ull), static_cast<long long int>(0x8009080780070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<4, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<4, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800E0D0C800C0B0Aull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<4, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<19>::decompress<4>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<4, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 0>();

    BitCompression::decompress_block<4, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<4>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 9
//Bits: 19


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<19>::remaining<9>() { return 6; }

template<>
template<>
inline const int BitCompression<19>::base_shift<9>() { return 14; }

template<>
template<>
inline int BitCompression<19>::next_offset<9>() { return 14; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<19>::block_count<9>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<19>::overlap_value<9>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 11) & 0x7ffff;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<9, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380030201ull), static_cast<long long int>(0x800A090808070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000040ull), static_cast<long long int>(0x2000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<9, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<9, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800F0E0D800C0B0Aull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<9, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<19>::decompress<9>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<9, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<9, 0>();

    BitCompression::decompress_block<9, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<9, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<9>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 14
//Bits: 19


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<19>::remaining<14>() { return 6; }

template<>
template<>
inline const int BitCompression<19>::base_shift<14>() { return 14; }

template<>
template<>
inline int BitCompression<19>::next_offset<14>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<19>::block_count<14>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<19>::overlap_value<14>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 16) & 0x7ffff;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<14, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8006050404030201ull), static_cast<long long int>(0x0B0A090880080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000002ull), static_cast<long long int>(0x100000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<14, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<19>::decompress_block<14, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800F0E0D800D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000008ull), static_cast<long long int>(0x400000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7ffff0007ffffull), static_cast<long long int>(0x7ffff0007ffffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<19>::per_block<14, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<19>::decompress<14>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<14, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<14, 0>();

    BitCompression::decompress_block<14, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<14, 1>();


}


template<>
inline void BitCompression<19>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<19>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<19>::remaining<0>()  + 1 ;

    BitCompression<19>::decompress<5>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<19>::remaining<5>()  + 1 ;

    BitCompression<19>::decompress<10>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<19>::remaining<10>()  + 1 ;

    BitCompression<19>::decompress<15>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<19>::remaining<15>()  + 1 ;

    BitCompression<19>::decompress<1>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<19>::remaining<1>()  + 1 ;

    BitCompression<19>::decompress<6>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<19>::remaining<6>()  + 1 ;

    BitCompression<19>::decompress<11>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<19>::remaining<11>()  + 1 ;

    BitCompression<19>::decompress<16>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<19>::remaining<16>()  + 1 ;

    BitCompression<19>::decompress<2>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<19>::remaining<2>()  + 1 ;

    BitCompression<19>::decompress<7>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<19>::remaining<7>()  + 1 ;

    BitCompression<19>::decompress<12>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<19>::remaining<12>()  + 1 ;

    BitCompression<19>::decompress<17>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<19>::remaining<17>()  + 1 ;

    BitCompression<19>::decompress<3>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<19>::remaining<3>()  + 1 ;

    BitCompression<19>::decompress<8>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<19>::remaining<8>()  + 1 ;

    BitCompression<19>::decompress<13>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<19>::remaining<13>()  + 1 ;

    BitCompression<19>::decompress<18>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<19>::remaining<18>()  + 1 ;

    BitCompression<19>::decompress<4>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<19>::remaining<4>()  + 1 ;

    BitCompression<19>::decompress<9>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<19>::remaining<9>()  + 1 ;

    BitCompression<19>::decompress<14>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<19>::remaining<14>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 0
//Bits: 20


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<20>::remaining<0>() { return 6; }

template<>
template<>
inline const int BitCompression<20>::base_shift<0>() { return 15; }

template<>
template<>
inline int BitCompression<20>::next_offset<0>() { return 12; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<20>::block_count<0>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<20>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 0) & 0xfffff;
}

template<>
template<>
inline void BitCompression<20>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8004030280020100ull), static_cast<long long int>(0x8009080780070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000010ull), static_cast<long long int>(0x100000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xfffff000fffffull), static_cast<long long int>(0xfffff000fffffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<20>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<20>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800E0D0C800C0B0Aull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000010ull), static_cast<long long int>(0x100000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xfffff000fffffull), static_cast<long long int>(0xfffff000fffffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<20>::per_block<0, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<20>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<0>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 12
//Bits: 20


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<20>::remaining<12>() { return 5; }

template<>
template<>
inline const int BitCompression<20>::base_shift<12>() { return 12; }

template<>
template<>
inline int BitCompression<20>::next_offset<12>() { return 4; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<20>::block_count<12>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<20>::overlap_value<12>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 16) & 0xfffff;
}

template<>
template<>
inline void BitCompression<20>::decompress_block<12, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8006050480030201ull), static_cast<long long int>(0x800B0A0980080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000001ull), static_cast<long long int>(0x1000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xfffff000fffffull), static_cast<long long int>(0xfffff000fffffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<20>::per_block<12, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<20>::decompress_block<12, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000001ull), static_cast<long long int>(0x1000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xfffff000fffffull), static_cast<long long int>(0xfffff000fffffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<20>::per_block<12, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<20>::decompress<12>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<12, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<12, 0>();

    BitCompression::decompress_block<12, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<12, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<12>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 4
//Bits: 20


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<20>::remaining<4>() { return 6; }

template<>
template<>
inline const int BitCompression<20>::base_shift<4>() { return 15; }

template<>
template<>
inline int BitCompression<20>::next_offset<4>() { return 16; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<20>::block_count<4>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<20>::overlap_value<4>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 4) & 0xfffff;
}

template<>
template<>
inline void BitCompression<20>::decompress_block<4, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380020100ull), static_cast<long long int>(0x800A090880070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000001ull), static_cast<long long int>(0x1000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xfffff000fffffull), static_cast<long long int>(0xfffff000fffffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<20>::per_block<4, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<20>::decompress_block<4, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800F0E0D800C0B0Aull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000001ull), static_cast<long long int>(0x1000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xfffff000fffffull), static_cast<long long int>(0xfffff000fffffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<20>::per_block<4, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<20>::decompress<4>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<4, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 0>();

    BitCompression::decompress_block<4, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<4>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 16
//Bits: 20


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<20>::remaining<16>() { return 5; }

template<>
template<>
inline const int BitCompression<20>::base_shift<16>() { return 12; }

template<>
template<>
inline int BitCompression<20>::next_offset<16>() { return 8; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<20>::block_count<16>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<20>::overlap_value<16>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 20) & 0xfffff;
}

template<>
template<>
inline void BitCompression<20>::decompress_block<16, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8006050480040302ull), static_cast<long long int>(0x800B0A0980090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000010ull), static_cast<long long int>(0x100000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xfffff000fffffull), static_cast<long long int>(0xfffff000fffffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<20>::per_block<16, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<20>::decompress_block<16, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000010ull), static_cast<long long int>(0x100000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xfffff000fffffull), static_cast<long long int>(0xfffff000fffffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<20>::per_block<16, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<20>::decompress<16>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<16, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<16, 0>();

    BitCompression::decompress_block<16, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<16, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<16>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 8
//Bits: 20


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<20>::remaining<8>() { return 6; }

template<>
template<>
inline const int BitCompression<20>::base_shift<8>() { return 15; }

template<>
template<>
inline int BitCompression<20>::next_offset<8>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<20>::block_count<8>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<20>::overlap_value<8>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0xfffff;
}

template<>
template<>
inline void BitCompression<20>::decompress_block<8, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380030201ull), static_cast<long long int>(0x800A090880080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000010ull), static_cast<long long int>(0x100000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xfffff000fffffull), static_cast<long long int>(0xfffff000fffffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<20>::per_block<8, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<20>::decompress_block<8, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800F0E0D800D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000010ull), static_cast<long long int>(0x100000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xfffff000fffffull), static_cast<long long int>(0xfffff000fffffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<20>::per_block<8, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<20>::decompress<8>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<8, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 0>();

    BitCompression::decompress_block<8, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 1>();


}


template<>
inline void BitCompression<20>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<20>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<20>::remaining<0>()  + 1 ;

    BitCompression<20>::decompress<12>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<20>::remaining<12>()  + 1 ;

    BitCompression<20>::decompress<4>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<20>::remaining<4>()  + 1 ;

    BitCompression<20>::decompress<16>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<20>::remaining<16>()  + 1 ;

    BitCompression<20>::decompress<8>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<20>::remaining<8>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 0
//Bits: 21


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<21>::remaining<0>() { return 6; }

template<>
template<>
inline const int BitCompression<21>::base_shift<0>() { return 15; }

template<>
template<>
inline int BitCompression<21>::next_offset<0>() { return 19; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<21>::block_count<0>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<21>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 6) & 0x1fffff;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0504030280020100ull), static_cast<long long int>(0x0A09080780070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000080ull), static_cast<long long int>(0x100000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800F0E0D0D0C0B0Aull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<0, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<21>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<0>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 19
//Bits: 21


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<21>::remaining<19>() { return 5; }

template<>
template<>
inline const int BitCompression<21>::base_shift<19>() { return 13; }

template<>
template<>
inline int BitCompression<21>::next_offset<19>() { return 17; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<21>::block_count<19>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<21>::overlap_value<19>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 20) & 0x1fffff;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<19, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8007060580040302ull), static_cast<long long int>(0x800C0B0A0A090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<19, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<19, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800F0E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<19, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<21>::decompress<19>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<19, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<19, 0>();

    BitCompression::decompress_block<19, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<19, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<19>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 17
//Bits: 21


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<21>::remaining<17>() { return 5; }

template<>
template<>
inline const int BitCompression<21>::base_shift<17>() { return 13; }

template<>
template<>
inline int BitCompression<21>::next_offset<17>() { return 15; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<21>::block_count<17>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<21>::overlap_value<17>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 18) & 0x1fffff;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<17, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050480040302ull), static_cast<long long int>(0x800C0B0A80090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<17, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<17, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800F0E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<17, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<21>::decompress<17>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<17, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<17, 0>();

    BitCompression::decompress_block<17, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<17, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<17>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 15
//Bits: 21


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<21>::remaining<15>() { return 5; }

template<>
template<>
inline const int BitCompression<21>::base_shift<15>() { return 13; }

template<>
template<>
inline int BitCompression<21>::next_offset<15>() { return 13; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<21>::block_count<15>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<21>::overlap_value<15>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 16) & 0x1fffff;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<15, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050404030201ull), static_cast<long long int>(0x0C0B0A0980090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<15, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<15, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<15, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<21>::decompress<15>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<15, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<15, 0>();

    BitCompression::decompress_block<15, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<15, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<15>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 13
//Bits: 21


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<21>::remaining<13>() { return 5; }

template<>
template<>
inline const int BitCompression<21>::base_shift<13>() { return 13; }

template<>
template<>
inline int BitCompression<21>::next_offset<13>() { return 11; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<21>::block_count<13>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<21>::overlap_value<13>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 14) & 0x1fffff;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<13, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8006050404030201ull), static_cast<long long int>(0x0C0B0A0909080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<13, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<13, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<13, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<21>::decompress<13>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<13, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<13, 0>();

    BitCompression::decompress_block<13, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<13, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<13>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 11
//Bits: 21


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<21>::remaining<11>() { return 5; }

template<>
template<>
inline const int BitCompression<21>::base_shift<11>() { return 13; }

template<>
template<>
inline int BitCompression<21>::next_offset<11>() { return 9; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<21>::block_count<11>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<21>::overlap_value<11>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 12) & 0x1fffff;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<11, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8006050480030201ull), static_cast<long long int>(0x800B0A0909080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<11, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<11, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800E0D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<11, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<21>::decompress<11>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<11, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<11, 0>();

    BitCompression::decompress_block<11, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<11, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<11>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 9
//Bits: 21


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<21>::remaining<9>() { return 5; }

template<>
template<>
inline const int BitCompression<21>::base_shift<9>() { return 13; }

template<>
template<>
inline int BitCompression<21>::next_offset<9>() { return 7; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<21>::block_count<9>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<21>::overlap_value<9>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 10) & 0x1fffff;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<9, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040380030201ull), static_cast<long long int>(0x800B0A0980080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<9, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<9, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800E0D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<9, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<21>::decompress<9>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<9, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<9, 0>();

    BitCompression::decompress_block<9, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<9, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<9>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 7
//Bits: 21


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<21>::remaining<7>() { return 5; }

template<>
template<>
inline const int BitCompression<21>::base_shift<7>() { return 13; }

template<>
template<>
inline int BitCompression<21>::next_offset<7>() { return 5; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<21>::block_count<7>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<21>::overlap_value<7>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0x1fffff;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<7, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040303020100ull), static_cast<long long int>(0x0B0A090880080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<7, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<7, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<7, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<21>::decompress<7>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<7, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<7, 0>();

    BitCompression::decompress_block<7, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<7, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<7>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 5
//Bits: 21


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<21>::remaining<5>() { return 5; }

template<>
template<>
inline const int BitCompression<21>::base_shift<5>() { return 13; }

template<>
template<>
inline int BitCompression<21>::next_offset<5>() { return 3; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<21>::block_count<5>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<21>::overlap_value<5>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 6) & 0x1fffff;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<5, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040303020100ull), static_cast<long long int>(0x0B0A090808070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<5, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<5, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<5, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<21>::decompress<5>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<5, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 0>();

    BitCompression::decompress_block<5, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<5>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 3
//Bits: 21


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<21>::remaining<3>() { return 5; }

template<>
template<>
inline const int BitCompression<21>::base_shift<3>() { return 13; }

template<>
template<>
inline int BitCompression<21>::next_offset<3>() { return 1; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<21>::block_count<3>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<21>::overlap_value<3>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 4) & 0x1fffff;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<3, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380020100ull), static_cast<long long int>(0x800A090808070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<3, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<3, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800D0C0B0Aull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<3, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<21>::decompress<3>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<3, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 0>();

    BitCompression::decompress_block<3, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<3>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 1
//Bits: 21


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<21>::remaining<1>() { return 6; }

template<>
template<>
inline const int BitCompression<21>::base_shift<1>() { return 15; }

template<>
template<>
inline int BitCompression<21>::next_offset<1>() { return 20; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<21>::block_count<1>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<21>::overlap_value<1>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 7) & 0x1fffff;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<1, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0504030280020100ull), static_cast<long long int>(0x800A090880070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<1, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<1, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800F0E0D0D0C0B0Aull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<1, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<21>::decompress<1>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<1, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 0>();

    BitCompression::decompress_block<1, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<1>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 20
//Bits: 21


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<21>::remaining<20>() { return 5; }

template<>
template<>
inline const int BitCompression<21>::base_shift<20>() { return 13; }

template<>
template<>
inline int BitCompression<21>::next_offset<20>() { return 18; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<21>::block_count<20>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<21>::overlap_value<20>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 21) & 0x1fffff;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<20, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8007060505040302ull), static_cast<long long int>(0x800C0B0A0A090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<20, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<20, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800F0E0Dull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000080ull), static_cast<long long int>(0x100000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<20, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<21>::decompress<20>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<20, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<20, 0>();

    BitCompression::decompress_block<20, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<20, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<20>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 18
//Bits: 21


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<21>::remaining<18>() { return 5; }

template<>
template<>
inline const int BitCompression<21>::base_shift<18>() { return 13; }

template<>
template<>
inline int BitCompression<21>::next_offset<18>() { return 16; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<21>::block_count<18>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<21>::overlap_value<18>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 19) & 0x1fffff;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<18, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050480040302ull), static_cast<long long int>(0x800C0B0A0A090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<18, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<18, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800F0E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<18, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<21>::decompress<18>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<18, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<18, 0>();

    BitCompression::decompress_block<18, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<18, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<18>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 16
//Bits: 21


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<21>::remaining<16>() { return 5; }

template<>
template<>
inline const int BitCompression<21>::base_shift<16>() { return 13; }

template<>
template<>
inline int BitCompression<21>::next_offset<16>() { return 14; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<21>::block_count<16>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<21>::overlap_value<16>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 17) & 0x1fffff;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<16, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050480040302ull), static_cast<long long int>(0x0C0B0A0980090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000080ull), static_cast<long long int>(0x100000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<16, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<16, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800F0E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<16, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<21>::decompress<16>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<16, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<16, 0>();

    BitCompression::decompress_block<16, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<16, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<16>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 14
//Bits: 21


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<21>::remaining<14>() { return 5; }

template<>
template<>
inline const int BitCompression<21>::base_shift<14>() { return 13; }

template<>
template<>
inline int BitCompression<21>::next_offset<14>() { return 12; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<21>::block_count<14>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<21>::overlap_value<14>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 15) & 0x1fffff;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<14, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8006050404030201ull), static_cast<long long int>(0x0C0B0A0980090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<14, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<14, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<14, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<21>::decompress<14>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<14, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<14, 0>();

    BitCompression::decompress_block<14, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<14, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<14>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 12
//Bits: 21


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<21>::remaining<12>() { return 5; }

template<>
template<>
inline const int BitCompression<21>::base_shift<12>() { return 13; }

template<>
template<>
inline int BitCompression<21>::next_offset<12>() { return 10; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<21>::block_count<12>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<21>::overlap_value<12>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 13) & 0x1fffff;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<12, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8006050404030201ull), static_cast<long long int>(0x800B0A0909080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<12, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<12, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000080ull), static_cast<long long int>(0x100000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<12, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<21>::decompress<12>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<12, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<12, 0>();

    BitCompression::decompress_block<12, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<12, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<12>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 10
//Bits: 21


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<21>::remaining<10>() { return 5; }

template<>
template<>
inline const int BitCompression<21>::base_shift<10>() { return 13; }

template<>
template<>
inline int BitCompression<21>::next_offset<10>() { return 8; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<21>::block_count<10>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<21>::overlap_value<10>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 11) & 0x1fffff;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<10, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040380030201ull), static_cast<long long int>(0x800B0A0909080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<10, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<10, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800E0D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<10, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<21>::decompress<10>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<10, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 0>();

    BitCompression::decompress_block<10, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<10>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 8
//Bits: 21


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<21>::remaining<8>() { return 5; }

template<>
template<>
inline const int BitCompression<21>::base_shift<8>() { return 13; }

template<>
template<>
inline int BitCompression<21>::next_offset<8>() { return 6; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<21>::block_count<8>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<21>::overlap_value<8>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 9) & 0x1fffff;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<8, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040380030201ull), static_cast<long long int>(0x0B0A090880080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000080ull), static_cast<long long int>(0x100000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<8, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<8, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800E0D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<8, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<21>::decompress<8>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<8, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 0>();

    BitCompression::decompress_block<8, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<8>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 6
//Bits: 21


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<21>::remaining<6>() { return 5; }

template<>
template<>
inline const int BitCompression<21>::base_shift<6>() { return 13; }

template<>
template<>
inline int BitCompression<21>::next_offset<6>() { return 4; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<21>::block_count<6>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<21>::overlap_value<6>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 7) & 0x1fffff;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<6, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040303020100ull), static_cast<long long int>(0x0B0A090880080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<6, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<6, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<6, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<21>::decompress<6>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<6, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 0>();

    BitCompression::decompress_block<6, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<6>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 4
//Bits: 21


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<21>::remaining<4>() { return 5; }

template<>
template<>
inline const int BitCompression<21>::base_shift<4>() { return 13; }

template<>
template<>
inline int BitCompression<21>::next_offset<4>() { return 2; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<21>::block_count<4>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<21>::overlap_value<4>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 5) & 0x1fffff;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<4, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040303020100ull), static_cast<long long int>(0x800A090808070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x2000000004ull), static_cast<long long int>(0x800000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<4, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<4, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000080ull), static_cast<long long int>(0x100000020ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<4, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<21>::decompress<4>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<4, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 0>();

    BitCompression::decompress_block<4, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<4>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 2
//Bits: 21


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<21>::remaining<2>() { return 6; }

template<>
template<>
inline const int BitCompression<21>::base_shift<2>() { return 15; }

template<>
template<>
inline int BitCompression<21>::next_offset<2>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<21>::block_count<2>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<21>::overlap_value<2>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0x1fffff;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<2, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0504030280020100ull), static_cast<long long int>(0x800A090808070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000020ull), static_cast<long long int>(0x4000000008ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<2, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<21>::decompress_block<2, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x800F0E0D0D0C0B0Aull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x800000001ull), static_cast<long long int>(0x200000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1fffff001fffffull), static_cast<long long int>(0x1fffff001fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<21>::per_block<2, 1>()
{ 
     return 2;
}


template<>
template<>
inline void BitCompression<21>::decompress<2>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<2, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 0>();

    BitCompression::decompress_block<2, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 1>();


}


template<>
inline void BitCompression<21>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<21>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<21>::remaining<0>()  + 1 ;

    BitCompression<21>::decompress<19>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<21>::remaining<19>()  + 1 ;

    BitCompression<21>::decompress<17>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<21>::remaining<17>()  + 1 ;

    BitCompression<21>::decompress<15>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<21>::remaining<15>()  + 1 ;

    BitCompression<21>::decompress<13>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<21>::remaining<13>()  + 1 ;

    BitCompression<21>::decompress<11>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<21>::remaining<11>()  + 1 ;

    BitCompression<21>::decompress<9>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<21>::remaining<9>()  + 1 ;

    BitCompression<21>::decompress<7>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<21>::remaining<7>()  + 1 ;

    BitCompression<21>::decompress<5>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<21>::remaining<5>()  + 1 ;

    BitCompression<21>::decompress<3>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<21>::remaining<3>()  + 1 ;

    BitCompression<21>::decompress<1>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<21>::remaining<1>()  + 1 ;

    BitCompression<21>::decompress<20>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<21>::remaining<20>()  + 1 ;

    BitCompression<21>::decompress<18>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<21>::remaining<18>()  + 1 ;

    BitCompression<21>::decompress<16>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<21>::remaining<16>()  + 1 ;

    BitCompression<21>::decompress<14>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<21>::remaining<14>()  + 1 ;

    BitCompression<21>::decompress<12>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<21>::remaining<12>()  + 1 ;

    BitCompression<21>::decompress<10>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<21>::remaining<10>()  + 1 ;

    BitCompression<21>::decompress<8>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<21>::remaining<8>()  + 1 ;

    BitCompression<21>::decompress<6>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<21>::remaining<6>()  + 1 ;

    BitCompression<21>::decompress<4>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<21>::remaining<4>()  + 1 ;

    BitCompression<21>::decompress<2>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<21>::remaining<2>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 0
//Bits: 22


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<22>::remaining<0>() { return 5; }

template<>
template<>
inline const int BitCompression<22>::base_shift<0>() { return 13; }

template<>
template<>
inline int BitCompression<22>::next_offset<0>() { return 4; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<22>::block_count<0>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<22>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 6) & 0x3fffff;
}

template<>
template<>
inline void BitCompression<22>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0504030280020100ull), static_cast<long long int>(0x800A090808070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000040ull), static_cast<long long int>(0x1000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fffff003fffffull), static_cast<long long int>(0x3fffff003fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<22>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<22>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000040ull), static_cast<long long int>(0x1000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fffff003fffffull), static_cast<long long int>(0x3fffff003fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<22>::per_block<0, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<22>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<0>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 4
//Bits: 22


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<22>::remaining<4>() { return 5; }

template<>
template<>
inline const int BitCompression<22>::base_shift<4>() { return 13; }

template<>
template<>
inline int BitCompression<22>::next_offset<4>() { return 8; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<22>::block_count<4>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<22>::overlap_value<4>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 10) & 0x3fffff;
}

template<>
template<>
inline void BitCompression<22>::decompress_block<4, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040303020100ull), static_cast<long long int>(0x0B0A090880080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000004ull), static_cast<long long int>(0x100000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fffff003fffffull), static_cast<long long int>(0x3fffff003fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<22>::per_block<4, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<22>::decompress_block<4, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800E0D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000004ull), static_cast<long long int>(0x100000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fffff003fffffull), static_cast<long long int>(0x3fffff003fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<22>::per_block<4, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<22>::decompress<4>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<4, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 0>();

    BitCompression::decompress_block<4, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<4>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 8
//Bits: 22


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<22>::remaining<8>() { return 5; }

template<>
template<>
inline const int BitCompression<22>::base_shift<8>() { return 13; }

template<>
template<>
inline int BitCompression<22>::next_offset<8>() { return 12; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<22>::block_count<8>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<22>::overlap_value<8>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 14) & 0x3fffff;
}

template<>
template<>
inline void BitCompression<22>::decompress_block<8, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040380030201ull), static_cast<long long int>(0x800B0A0909080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000040ull), static_cast<long long int>(0x1000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fffff003fffffull), static_cast<long long int>(0x3fffff003fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<22>::per_block<8, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<22>::decompress_block<8, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000040ull), static_cast<long long int>(0x1000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fffff003fffffull), static_cast<long long int>(0x3fffff003fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<22>::per_block<8, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<22>::decompress<8>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<8, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 0>();

    BitCompression::decompress_block<8, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<8>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 12
//Bits: 22


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<22>::remaining<12>() { return 5; }

template<>
template<>
inline const int BitCompression<22>::base_shift<12>() { return 13; }

template<>
template<>
inline int BitCompression<22>::next_offset<12>() { return 16; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<22>::block_count<12>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<22>::overlap_value<12>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 18) & 0x3fffff;
}

template<>
template<>
inline void BitCompression<22>::decompress_block<12, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8006050404030201ull), static_cast<long long int>(0x0C0B0A0980090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000004ull), static_cast<long long int>(0x100000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fffff003fffffull), static_cast<long long int>(0x3fffff003fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<22>::per_block<12, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<22>::decompress_block<12, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800F0E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000004ull), static_cast<long long int>(0x100000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fffff003fffffull), static_cast<long long int>(0x3fffff003fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<22>::per_block<12, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<22>::decompress<12>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<12, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<12, 0>();

    BitCompression::decompress_block<12, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<12, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<12>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 16
//Bits: 22


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<22>::remaining<16>() { return 5; }

template<>
template<>
inline const int BitCompression<22>::base_shift<16>() { return 13; }

template<>
template<>
inline int BitCompression<22>::next_offset<16>() { return 20; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<22>::block_count<16>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<22>::overlap_value<16>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 22) & 0x3fffff;
}

template<>
template<>
inline void BitCompression<22>::decompress_block<16, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050480040302ull), static_cast<long long int>(0x800C0B0A0A090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000040ull), static_cast<long long int>(0x1000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fffff003fffffull), static_cast<long long int>(0x3fffff003fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<22>::per_block<16, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<22>::decompress_block<16, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800F0E0Dull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000040ull), static_cast<long long int>(0x1000000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fffff003fffffull), static_cast<long long int>(0x3fffff003fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<22>::per_block<16, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<22>::decompress<16>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<16, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<16, 0>();

    BitCompression::decompress_block<16, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<16, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<16>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 11
//Offset: 20
//Bits: 22


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<22>::remaining<20>() { return 4; }

template<>
template<>
inline const int BitCompression<22>::base_shift<20>() { return 11; }

template<>
template<>
inline int BitCompression<22>::next_offset<20>() { return 2; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<22>::block_count<20>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<22>::overlap_value<20>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 20) & 0x3fffff;
}

template<>
template<>
inline void BitCompression<22>::decompress_block<20, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8007060505040302ull), static_cast<long long int>(0x0D0C0B0A800A0908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000004ull), static_cast<long long int>(0x100000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fffff003fffffull), static_cast<long long int>(0x3fffff003fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<22>::per_block<20, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<22>::decompress<20>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<20, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<20, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  11);
    BitCompression::overlap_value<20>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 2
//Bits: 22


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<22>::remaining<2>() { return 5; }

template<>
template<>
inline const int BitCompression<22>::base_shift<2>() { return 13; }

template<>
template<>
inline int BitCompression<22>::next_offset<2>() { return 6; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<22>::block_count<2>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<22>::overlap_value<2>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0x3fffff;
}

template<>
template<>
inline void BitCompression<22>::decompress_block<2, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380020100ull), static_cast<long long int>(0x0B0A090808070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000010ull), static_cast<long long int>(0x400000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fffff003fffffull), static_cast<long long int>(0x3fffff003fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<22>::per_block<2, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<22>::decompress_block<2, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000010ull), static_cast<long long int>(0x400000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fffff003fffffull), static_cast<long long int>(0x3fffff003fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<22>::per_block<2, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<22>::decompress<2>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<2, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 0>();

    BitCompression::decompress_block<2, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<2>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 6
//Bits: 22


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<22>::remaining<6>() { return 5; }

template<>
template<>
inline const int BitCompression<22>::base_shift<6>() { return 13; }

template<>
template<>
inline int BitCompression<22>::next_offset<6>() { return 10; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<22>::block_count<6>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<22>::overlap_value<6>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 12) & 0x3fffff;
}

template<>
template<>
inline void BitCompression<22>::decompress_block<6, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040303020100ull), static_cast<long long int>(0x800B0A0980080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000001ull), static_cast<long long int>(0x4000000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fffff003fffffull), static_cast<long long int>(0x3fffff003fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<22>::per_block<6, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<22>::decompress_block<6, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800E0D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000001ull), static_cast<long long int>(0x4000000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fffff003fffffull), static_cast<long long int>(0x3fffff003fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<22>::per_block<6, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<22>::decompress<6>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<6, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 0>();

    BitCompression::decompress_block<6, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<6>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 10
//Bits: 22


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<22>::remaining<10>() { return 5; }

template<>
template<>
inline const int BitCompression<22>::base_shift<10>() { return 13; }

template<>
template<>
inline int BitCompression<22>::next_offset<10>() { return 14; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<22>::block_count<10>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<22>::overlap_value<10>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 16) & 0x3fffff;
}

template<>
template<>
inline void BitCompression<22>::decompress_block<10, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8006050480030201ull), static_cast<long long int>(0x0C0B0A0909080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000010ull), static_cast<long long int>(0x400000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fffff003fffffull), static_cast<long long int>(0x3fffff003fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<22>::per_block<10, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<22>::decompress_block<10, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000010ull), static_cast<long long int>(0x400000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fffff003fffffull), static_cast<long long int>(0x3fffff003fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<22>::per_block<10, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<22>::decompress<10>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<10, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 0>();

    BitCompression::decompress_block<10, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<10>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 14
//Bits: 22


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<22>::remaining<14>() { return 5; }

template<>
template<>
inline const int BitCompression<22>::base_shift<14>() { return 13; }

template<>
template<>
inline int BitCompression<22>::next_offset<14>() { return 18; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<22>::block_count<14>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<22>::overlap_value<14>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 20) & 0x3fffff;
}

template<>
template<>
inline void BitCompression<22>::decompress_block<14, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050404030201ull), static_cast<long long int>(0x800C0B0A80090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000001ull), static_cast<long long int>(0x4000000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fffff003fffffull), static_cast<long long int>(0x3fffff003fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<22>::per_block<14, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<22>::decompress_block<14, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800F0E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000001ull), static_cast<long long int>(0x4000000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fffff003fffffull), static_cast<long long int>(0x3fffff003fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<22>::per_block<14, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<22>::decompress<14>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<14, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<14, 0>();

    BitCompression::decompress_block<14, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<14, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<14>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 18
//Bits: 22


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<22>::remaining<18>() { return 5; }

template<>
template<>
inline const int BitCompression<22>::base_shift<18>() { return 13; }

template<>
template<>
inline int BitCompression<22>::next_offset<18>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<22>::block_count<18>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<22>::overlap_value<18>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 24) & 0x3fffff;
}

template<>
template<>
inline void BitCompression<22>::decompress_block<18, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8007060580040302ull), static_cast<long long int>(0x0D0C0B0A0A090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000010ull), static_cast<long long int>(0x400000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fffff003fffffull), static_cast<long long int>(0x3fffff003fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<22>::per_block<18, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<22>::decompress_block<18, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800F0E0Dull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000010ull), static_cast<long long int>(0x400000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3fffff003fffffull), static_cast<long long int>(0x3fffff003fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<22>::per_block<18, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<22>::decompress<18>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<18, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<18, 0>();

    BitCompression::decompress_block<18, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<18, 1>();


}


template<>
inline void BitCompression<22>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<22>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<22>::remaining<0>()  + 1 ;

    BitCompression<22>::decompress<4>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<22>::remaining<4>()  + 1 ;

    BitCompression<22>::decompress<8>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<22>::remaining<8>()  + 1 ;

    BitCompression<22>::decompress<12>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<22>::remaining<12>()  + 1 ;

    BitCompression<22>::decompress<16>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<22>::remaining<16>()  + 1 ;

    BitCompression<22>::decompress<20>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<22>::remaining<20>()  + 1 ;

    BitCompression<22>::decompress<2>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<22>::remaining<2>()  + 1 ;

    BitCompression<22>::decompress<6>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<22>::remaining<6>()  + 1 ;

    BitCompression<22>::decompress<10>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<22>::remaining<10>()  + 1 ;

    BitCompression<22>::decompress<14>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<22>::remaining<14>()  + 1 ;

    BitCompression<22>::decompress<18>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<22>::remaining<18>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 0
//Bits: 23


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<23>::remaining<0>() { return 5; }

template<>
template<>
inline const int BitCompression<23>::base_shift<0>() { return 14; }

template<>
template<>
inline int BitCompression<23>::next_offset<0>() { return 10; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<23>::block_count<0>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<23>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 3) & 0x7fffff;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0504030280020100ull), static_cast<long long int>(0x0B0A090808070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000080ull), static_cast<long long int>(0x400000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800E0D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<0, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<23>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<0>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 10
//Bits: 23


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<23>::remaining<10>() { return 5; }

template<>
template<>
inline const int BitCompression<23>::base_shift<10>() { return 14; }

template<>
template<>
inline int BitCompression<23>::next_offset<10>() { return 20; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<23>::block_count<10>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<23>::overlap_value<10>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 13) & 0x7fffff;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<10, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8006050404030201ull), static_cast<long long int>(0x0C0B0A0980090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000020ull), static_cast<long long int>(0x100000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<10, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<10, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800F0E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<10, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<23>::decompress<10>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<10, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 0>();

    BitCompression::decompress_block<10, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<10>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 11
//Offset: 20
//Bits: 23


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<23>::remaining<20>() { return 4; }

template<>
template<>
inline const int BitCompression<23>::base_shift<20>() { return 11; }

template<>
template<>
inline int BitCompression<23>::next_offset<20>() { return 7; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<23>::block_count<20>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<23>::overlap_value<20>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 24) & 0x7fffff;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<20, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0807060505040302ull), static_cast<long long int>(0x800D0C0B0B0A0908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<20, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<23>::decompress<20>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<20, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<20, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  11);
    BitCompression::overlap_value<20>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 7
//Bits: 23


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<23>::remaining<7>() { return 5; }

template<>
template<>
inline const int BitCompression<23>::base_shift<7>() { return 14; }

template<>
template<>
inline int BitCompression<23>::next_offset<7>() { return 17; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<23>::block_count<7>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<23>::overlap_value<7>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 10) & 0x7fffff;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<7, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040303020100ull), static_cast<long long int>(0x0C0B0A0909080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<7, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<7, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800F0E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<7, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<23>::decompress<7>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<7, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<7, 0>();

    BitCompression::decompress_block<7, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<7, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<7>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 11
//Offset: 17
//Bits: 23


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<23>::remaining<17>() { return 4; }

template<>
template<>
inline const int BitCompression<23>::base_shift<17>() { return 11; }

template<>
template<>
inline int BitCompression<23>::next_offset<17>() { return 4; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<23>::block_count<17>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<23>::overlap_value<17>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 21) & 0x7fffff;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<17, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8007060580040302ull), static_cast<long long int>(0x0D0C0B0A0A090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000040ull), static_cast<long long int>(0x200000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<17, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<23>::decompress<17>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<17, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<17, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  11);
    BitCompression::overlap_value<17>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 4
//Bits: 23


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<23>::remaining<4>() { return 5; }

template<>
template<>
inline const int BitCompression<23>::base_shift<4>() { return 14; }

template<>
template<>
inline int BitCompression<23>::next_offset<4>() { return 14; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<23>::block_count<4>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<23>::overlap_value<4>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 7) & 0x7fffff;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<4, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040303020100ull), static_cast<long long int>(0x800B0A0909080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<4, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<4, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000080ull), static_cast<long long int>(0x400000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<4, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<23>::decompress<4>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<4, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 0>();

    BitCompression::decompress_block<4, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<4>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 11
//Offset: 14
//Bits: 23


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<23>::remaining<14>() { return 4; }

template<>
template<>
inline const int BitCompression<23>::base_shift<14>() { return 11; }

template<>
template<>
inline int BitCompression<23>::next_offset<14>() { return 1; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<23>::block_count<14>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<23>::overlap_value<14>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 18) & 0x7fffff;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<14, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050404030201ull), static_cast<long long int>(0x0D0C0B0A0A090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<14, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<23>::decompress<14>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<14, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<14, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  11);
    BitCompression::overlap_value<14>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 1
//Bits: 23


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<23>::remaining<1>() { return 5; }

template<>
template<>
inline const int BitCompression<23>::base_shift<1>() { return 14; }

template<>
template<>
inline int BitCompression<23>::next_offset<1>() { return 11; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<23>::block_count<1>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<23>::overlap_value<1>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 4) & 0x7fffff;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<1, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380020100ull), static_cast<long long int>(0x0B0A090808070605ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000040ull), static_cast<long long int>(0x200000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<1, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<1, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800E0D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<1, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<23>::decompress<1>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<1, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 0>();

    BitCompression::decompress_block<1, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<1>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 11
//Bits: 23


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<23>::remaining<11>() { return 5; }

template<>
template<>
inline const int BitCompression<23>::base_shift<11>() { return 14; }

template<>
template<>
inline int BitCompression<23>::next_offset<11>() { return 21; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<23>::block_count<11>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<23>::overlap_value<11>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 14) & 0x7fffff;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<11, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050404030201ull), static_cast<long long int>(0x800C0B0A80090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<11, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<11, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800F0E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<11, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<23>::decompress<11>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<11, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<11, 0>();

    BitCompression::decompress_block<11, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<11, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<11>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 11
//Offset: 21
//Bits: 23


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<23>::remaining<21>() { return 4; }

template<>
template<>
inline const int BitCompression<23>::base_shift<21>() { return 11; }

template<>
template<>
inline int BitCompression<23>::next_offset<21>() { return 8; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<23>::block_count<21>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<23>::overlap_value<21>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 25) & 0x7fffff;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<21, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0807060505040302ull), static_cast<long long int>(0x0E0D0C0B0B0A0908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<21, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<23>::decompress<21>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<21, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<21, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  11);
    BitCompression::overlap_value<21>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 8
//Bits: 23


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<23>::remaining<8>() { return 5; }

template<>
template<>
inline const int BitCompression<23>::base_shift<8>() { return 14; }

template<>
template<>
inline int BitCompression<23>::next_offset<8>() { return 18; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<23>::block_count<8>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<23>::overlap_value<8>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 11) & 0x7fffff;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<8, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040380030201ull), static_cast<long long int>(0x0C0B0A0909080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000080ull), static_cast<long long int>(0x400000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<8, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<8, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800F0E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<8, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<23>::decompress<8>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<8, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 0>();

    BitCompression::decompress_block<8, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<8>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 11
//Offset: 18
//Bits: 23


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<23>::remaining<18>() { return 4; }

template<>
template<>
inline const int BitCompression<23>::base_shift<18>() { return 11; }

template<>
template<>
inline int BitCompression<23>::next_offset<18>() { return 5; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<23>::block_count<18>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<23>::overlap_value<18>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 22) & 0x7fffff;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<18, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8007060505040302ull), static_cast<long long int>(0x0D0C0B0A800A0908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000020ull), static_cast<long long int>(0x100000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<18, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<23>::decompress<18>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<18, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<18, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  11);
    BitCompression::overlap_value<18>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 5
//Bits: 23


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<23>::remaining<5>() { return 5; }

template<>
template<>
inline const int BitCompression<23>::base_shift<5>() { return 14; }

template<>
template<>
inline int BitCompression<23>::next_offset<5>() { return 15; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<23>::block_count<5>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<23>::overlap_value<5>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0x7fffff;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<5, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040303020100ull), static_cast<long long int>(0x0C0B0A0909080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<5, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<5, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000040ull), static_cast<long long int>(0x200000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<5, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<23>::decompress<5>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<5, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 0>();

    BitCompression::decompress_block<5, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<5>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 11
//Offset: 15
//Bits: 23


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<23>::remaining<15>() { return 4; }

template<>
template<>
inline const int BitCompression<23>::base_shift<15>() { return 11; }

template<>
template<>
inline int BitCompression<23>::next_offset<15>() { return 2; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<23>::block_count<15>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<23>::overlap_value<15>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 19) & 0x7fffff;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<15, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050404030201ull), static_cast<long long int>(0x0D0C0B0A0A090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<15, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<23>::decompress<15>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<15, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<15, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  11);
    BitCompression::overlap_value<15>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 2
//Bits: 23


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<23>::remaining<2>() { return 5; }

template<>
template<>
inline const int BitCompression<23>::base_shift<2>() { return 14; }

template<>
template<>
inline int BitCompression<23>::next_offset<2>() { return 12; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<23>::block_count<2>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<23>::overlap_value<2>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 5) & 0x7fffff;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<2, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040303020100ull), static_cast<long long int>(0x0B0A090880080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000020ull), static_cast<long long int>(0x100000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<2, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<2, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800E0D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<2, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<23>::decompress<2>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<2, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 0>();

    BitCompression::decompress_block<2, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<2>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 12
//Bits: 23


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<23>::remaining<12>() { return 5; }

template<>
template<>
inline const int BitCompression<23>::base_shift<12>() { return 14; }

template<>
template<>
inline int BitCompression<23>::next_offset<12>() { return 22; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<23>::block_count<12>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<23>::overlap_value<12>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 15) & 0x7fffff;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<12, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050404030201ull), static_cast<long long int>(0x800C0B0A0A090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<12, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<12, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800F0E0Dull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000080ull), static_cast<long long int>(0x400000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<12, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<23>::decompress<12>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<12, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<12, 0>();

    BitCompression::decompress_block<12, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<12, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<12>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 11
//Offset: 22
//Bits: 23


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<23>::remaining<22>() { return 4; }

template<>
template<>
inline const int BitCompression<23>::base_shift<22>() { return 11; }

template<>
template<>
inline int BitCompression<23>::next_offset<22>() { return 9; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<23>::block_count<22>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<23>::overlap_value<22>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 26) & 0x7fffff;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<22, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0807060505040302ull), static_cast<long long int>(0x0E0D0C0B0B0A0908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<22, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<23>::decompress<22>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<22, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<22, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  11);
    BitCompression::overlap_value<22>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 9
//Bits: 23


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<23>::remaining<9>() { return 5; }

template<>
template<>
inline const int BitCompression<23>::base_shift<9>() { return 14; }

template<>
template<>
inline int BitCompression<23>::next_offset<9>() { return 19; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<23>::block_count<9>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<23>::overlap_value<9>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 12) & 0x7fffff;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<9, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8006050480030201ull), static_cast<long long int>(0x0C0B0A0909080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000040ull), static_cast<long long int>(0x200000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<9, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<9, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800F0E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<9, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<23>::decompress<9>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<9, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<9, 0>();

    BitCompression::decompress_block<9, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<9, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<9>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 11
//Offset: 19
//Bits: 23


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<23>::remaining<19>() { return 4; }

template<>
template<>
inline const int BitCompression<23>::base_shift<19>() { return 11; }

template<>
template<>
inline int BitCompression<23>::next_offset<19>() { return 6; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<23>::block_count<19>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<23>::overlap_value<19>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 23) & 0x7fffff;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<19, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0807060505040302ull), static_cast<long long int>(0x800D0C0B800A0908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<19, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<23>::decompress<19>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<19, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<19, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  11);
    BitCompression::overlap_value<19>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 6
//Bits: 23


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<23>::remaining<6>() { return 5; }

template<>
template<>
inline const int BitCompression<23>::base_shift<6>() { return 14; }

template<>
template<>
inline int BitCompression<23>::next_offset<6>() { return 16; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<23>::block_count<6>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<23>::overlap_value<6>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 9) & 0x7fffff;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<6, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040303020100ull), static_cast<long long int>(0x0C0B0A0909080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<6, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<6, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800F0E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000020ull), static_cast<long long int>(0x100000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<6, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<23>::decompress<6>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<6, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 0>();

    BitCompression::decompress_block<6, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<6>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 11
//Offset: 16
//Bits: 23


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<23>::remaining<16>() { return 4; }

template<>
template<>
inline const int BitCompression<23>::base_shift<16>() { return 11; }

template<>
template<>
inline int BitCompression<23>::next_offset<16>() { return 3; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<23>::block_count<16>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<23>::overlap_value<16>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 20) & 0x7fffff;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<16, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050480040302ull), static_cast<long long int>(0x0D0C0B0A0A090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000080ull), static_cast<long long int>(0x400000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<16, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<23>::decompress<16>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<16, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<16, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  11);
    BitCompression::overlap_value<16>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 3
//Bits: 23


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<23>::remaining<3>() { return 5; }

template<>
template<>
inline const int BitCompression<23>::base_shift<3>() { return 14; }

template<>
template<>
inline int BitCompression<23>::next_offset<3>() { return 13; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<23>::block_count<3>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<23>::overlap_value<3>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 6) & 0x7fffff;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<3, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040303020100ull), static_cast<long long int>(0x800B0A0980080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<3, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<3, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800E0D0C0Bull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<3, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<23>::decompress<3>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<3, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 0>();

    BitCompression::decompress_block<3, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  14);
    BitCompression::overlap_value<3>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 14
//Offset: 13
//Bits: 23


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<23>::remaining<13>() { return 5; }

template<>
template<>
inline const int BitCompression<23>::base_shift<13>() { return 14; }

template<>
template<>
inline int BitCompression<23>::next_offset<13>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<23>::block_count<13>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<23>::overlap_value<13>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 16) & 0x7fffff;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<13, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050404030201ull), static_cast<long long int>(0x0D0C0B0A0A090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000001ull), static_cast<long long int>(0x800000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<13, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<23>::decompress_block<13, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800F0E0Dull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000040ull), static_cast<long long int>(0x200000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x7fffff007fffffull), static_cast<long long int>(0x7fffff007fffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<23>::per_block<13, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<23>::decompress<13>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<13, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<13, 0>();

    BitCompression::decompress_block<13, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<13, 1>();


}


template<>
inline void BitCompression<23>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<23>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<23>::remaining<0>()  + 1 ;

    BitCompression<23>::decompress<10>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<23>::remaining<10>()  + 1 ;

    BitCompression<23>::decompress<20>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<23>::remaining<20>()  + 1 ;

    BitCompression<23>::decompress<7>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<23>::remaining<7>()  + 1 ;

    BitCompression<23>::decompress<17>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<23>::remaining<17>()  + 1 ;

    BitCompression<23>::decompress<4>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<23>::remaining<4>()  + 1 ;

    BitCompression<23>::decompress<14>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<23>::remaining<14>()  + 1 ;

    BitCompression<23>::decompress<1>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<23>::remaining<1>()  + 1 ;

    BitCompression<23>::decompress<11>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<23>::remaining<11>()  + 1 ;

    BitCompression<23>::decompress<21>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<23>::remaining<21>()  + 1 ;

    BitCompression<23>::decompress<8>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<23>::remaining<8>()  + 1 ;

    BitCompression<23>::decompress<18>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<23>::remaining<18>()  + 1 ;

    BitCompression<23>::decompress<5>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<23>::remaining<5>()  + 1 ;

    BitCompression<23>::decompress<15>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<23>::remaining<15>()  + 1 ;

    BitCompression<23>::decompress<2>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<23>::remaining<2>()  + 1 ;

    BitCompression<23>::decompress<12>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<23>::remaining<12>()  + 1 ;

    BitCompression<23>::decompress<22>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<23>::remaining<22>()  + 1 ;

    BitCompression<23>::decompress<9>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<23>::remaining<9>()  + 1 ;

    BitCompression<23>::decompress<19>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<23>::remaining<19>()  + 1 ;

    BitCompression<23>::decompress<6>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<23>::remaining<6>()  + 1 ;

    BitCompression<23>::decompress<16>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<23>::remaining<16>()  + 1 ;

    BitCompression<23>::decompress<3>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<23>::remaining<3>()  + 1 ;

    BitCompression<23>::decompress<13>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<23>::remaining<13>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 0
//Bits: 24


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<24>::remaining<0>() { return 5; }

template<>
template<>
inline const int BitCompression<24>::base_shift<0>() { return 15; }

template<>
template<>
inline int BitCompression<24>::next_offset<0>() { return 16; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<24>::block_count<0>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<24>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 0) & 0xffffff;
}

template<>
template<>
inline void BitCompression<24>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8005040380020100ull), static_cast<long long int>(0x800B0A0980080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xffffff00ffffffull), static_cast<long long int>(0xffffff00ffffffull)};
    static const int shift_mask = 0;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<24>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<24>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xffffff00ffffffull), static_cast<long long int>(0xffffff00ffffffull)};
    static const int shift_mask = 0;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<24>::per_block<0, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<24>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<0>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 16
//Bits: 24


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<24>::remaining<16>() { return 4; }

template<>
template<>
inline const int BitCompression<24>::base_shift<16>() { return 12; }

template<>
template<>
inline int BitCompression<24>::next_offset<16>() { return 8; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<24>::block_count<16>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<24>::overlap_value<16>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 16) & 0xffffff;
}

template<>
template<>
inline void BitCompression<24>::decompress_block<16, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8007060580040302ull), static_cast<long long int>(0x800D0C0B800A0908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xffffff00ffffffull), static_cast<long long int>(0xffffff00ffffffull)};
    static const int shift_mask = 0;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<24>::per_block<16, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<24>::decompress<16>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<16, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<16, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<16>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 8
//Bits: 24


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<24>::remaining<8>() { return 5; }

template<>
template<>
inline const int BitCompression<24>::base_shift<8>() { return 15; }

template<>
template<>
inline int BitCompression<24>::next_offset<8>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<24>::block_count<8>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<24>::overlap_value<8>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0xffffff;
}

template<>
template<>
inline void BitCompression<24>::decompress_block<8, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x8006050480030201ull), static_cast<long long int>(0x800C0B0A80090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xffffff00ffffffull), static_cast<long long int>(0xffffff00ffffffull)};
    static const int shift_mask = 0;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<24>::per_block<8, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<24>::decompress_block<8, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x80808080800F0E0Dull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000001ull), static_cast<long long int>(0x100000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0xffffff00ffffffull), static_cast<long long int>(0xffffff00ffffffull)};
    static const int shift_mask = 0;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<24>::per_block<8, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<24>::decompress<8>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<8, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 0>();

    BitCompression::decompress_block<8, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 1>();


}


template<>
inline void BitCompression<24>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<24>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<24>::remaining<0>()  + 1 ;

    BitCompression<24>::decompress<16>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<24>::remaining<16>()  + 1 ;

    BitCompression<24>::decompress<8>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<24>::remaining<8>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 0
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<0>() { return 5; }

template<>
template<>
inline const int BitCompression<25>::base_shift<0>() { return 15; }

template<>
template<>
inline int BitCompression<25>::next_offset<0>() { return 22; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<0>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 5) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040303020100ull), static_cast<long long int>(0x0C0B0A0909080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<0, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<0, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800F0E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<0, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<25>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();

    BitCompression::decompress_block<0, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<0>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 22
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<22>() { return 4; }

template<>
template<>
inline const int BitCompression<25>::base_shift<22>() { return 12; }

template<>
template<>
inline int BitCompression<25>::next_offset<22>() { return 19; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<22>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<22>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 26) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<22, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0807060505040302ull), static_cast<long long int>(0x0F0E0D0C0C0B0A09ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000002ull), static_cast<long long int>(0x4000000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<22, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<25>::decompress<22>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<22, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<22, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<22>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 19
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<19>() { return 4; }

template<>
template<>
inline const int BitCompression<25>::base_shift<19>() { return 12; }

template<>
template<>
inline int BitCompression<25>::next_offset<19>() { return 16; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<19>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<19>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 23) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<19, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0807060505040302ull), static_cast<long long int>(0x0E0D0C0B0B0A0908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<19, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<25>::decompress<19>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<19, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<19, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<19>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 16
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<16>() { return 4; }

template<>
template<>
inline const int BitCompression<25>::base_shift<16>() { return 12; }

template<>
template<>
inline int BitCompression<25>::next_offset<16>() { return 13; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<16>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<16>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 20) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<16, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0807060505040302ull), static_cast<long long int>(0x0E0D0C0B0B0A0908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<16, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<25>::decompress<16>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<16, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<16, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<16>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 13
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<13>() { return 4; }

template<>
template<>
inline const int BitCompression<25>::base_shift<13>() { return 12; }

template<>
template<>
inline int BitCompression<25>::next_offset<13>() { return 10; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<13>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<13>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 17) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<13, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050404030201ull), static_cast<long long int>(0x0E0D0C0B0A090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000004ull), static_cast<long long int>(0x8000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<13, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<25>::decompress<13>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<13, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<13, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<13>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 10
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<10>() { return 4; }

template<>
template<>
inline const int BitCompression<25>::base_shift<10>() { return 12; }

template<>
template<>
inline int BitCompression<25>::next_offset<10>() { return 7; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<10>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<10>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 14) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<10, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050404030201ull), static_cast<long long int>(0x0D0C0B0A0A090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<10, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<25>::decompress<10>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<10, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<10>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 7
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<7>() { return 4; }

template<>
template<>
inline const int BitCompression<25>::base_shift<7>() { return 12; }

template<>
template<>
inline int BitCompression<25>::next_offset<7>() { return 4; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<7>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<7>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 11) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<7, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050403020100ull), static_cast<long long int>(0x0D0C0B0A0A090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000001ull), static_cast<long long int>(0x2000000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<7, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<25>::decompress<7>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<7, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<7, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<7>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 4
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<4>() { return 4; }

template<>
template<>
inline const int BitCompression<25>::base_shift<4>() { return 12; }

template<>
template<>
inline int BitCompression<25>::next_offset<4>() { return 1; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<4>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<4>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<4, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040303020100ull), static_cast<long long int>(0x0C0B0A0909080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<4, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<25>::decompress<4>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<4, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<4>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 1
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<1>() { return 5; }

template<>
template<>
inline const int BitCompression<25>::base_shift<1>() { return 15; }

template<>
template<>
inline int BitCompression<25>::next_offset<1>() { return 23; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<1>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<1>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 6) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<1, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040303020100ull), static_cast<long long int>(0x0C0B0A0909080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<1, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<1, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800F0E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000004ull), static_cast<long long int>(0x8000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<1, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<25>::decompress<1>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<1, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 0>();

    BitCompression::decompress_block<1, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<1, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<1>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 23
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<23>() { return 4; }

template<>
template<>
inline const int BitCompression<25>::base_shift<23>() { return 12; }

template<>
template<>
inline int BitCompression<25>::next_offset<23>() { return 20; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<23>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<23>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 27) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<23, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0908070605040302ull), static_cast<long long int>(0x0F0E0D0C0C0B0A09ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000001ull), static_cast<long long int>(0x2000000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<23, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<25>::decompress<23>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<23, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<23, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<23>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 20
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<20>() { return 4; }

template<>
template<>
inline const int BitCompression<25>::base_shift<20>() { return 12; }

template<>
template<>
inline int BitCompression<25>::next_offset<20>() { return 17; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<20>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<20>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 24) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<20, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0807060505040302ull), static_cast<long long int>(0x0E0D0C0B0B0A0908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<20, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<25>::decompress<20>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<20, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<20, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<20>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 17
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<17>() { return 4; }

template<>
template<>
inline const int BitCompression<25>::base_shift<17>() { return 12; }

template<>
template<>
inline int BitCompression<25>::next_offset<17>() { return 14; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<17>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<17>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 21) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<17, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0807060505040302ull), static_cast<long long int>(0x0E0D0C0B0B0A0908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<17, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<25>::decompress<17>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<17, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<17, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<17>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 14
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<14>() { return 4; }

template<>
template<>
inline const int BitCompression<25>::base_shift<14>() { return 12; }

template<>
template<>
inline int BitCompression<25>::next_offset<14>() { return 11; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<14>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<14>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 18) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<14, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050404030201ull), static_cast<long long int>(0x0E0D0C0B0B0A0908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000002ull), static_cast<long long int>(0x4000000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<14, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<25>::decompress<14>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<14, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<14, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<14>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 11
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<11>() { return 4; }

template<>
template<>
inline const int BitCompression<25>::base_shift<11>() { return 12; }

template<>
template<>
inline int BitCompression<25>::next_offset<11>() { return 8; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<11>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<11>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 15) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<11, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050404030201ull), static_cast<long long int>(0x0D0C0B0A0A090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<11, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<25>::decompress<11>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<11, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<11, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<11>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 8
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<8>() { return 4; }

template<>
template<>
inline const int BitCompression<25>::base_shift<8>() { return 12; }

template<>
template<>
inline int BitCompression<25>::next_offset<8>() { return 5; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<8>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<8>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 12) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<8, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050404030201ull), static_cast<long long int>(0x0D0C0B0A0A090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<8, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<25>::decompress<8>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<8, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<8>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 5
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<5>() { return 4; }

template<>
template<>
inline const int BitCompression<25>::base_shift<5>() { return 12; }

template<>
template<>
inline int BitCompression<25>::next_offset<5>() { return 2; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<5>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<5>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 9) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<5, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040303020100ull), static_cast<long long int>(0x0D0C0B0A09080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000004ull), static_cast<long long int>(0x8000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<5, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<25>::decompress<5>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<5, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<5, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<5>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 2
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<2>() { return 5; }

template<>
template<>
inline const int BitCompression<25>::base_shift<2>() { return 15; }

template<>
template<>
inline int BitCompression<25>::next_offset<2>() { return 24; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<2>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<2>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 7) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<2, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040303020100ull), static_cast<long long int>(0x0C0B0A0909080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<2, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<2, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800F0E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000002ull), static_cast<long long int>(0x4000000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<2, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<25>::decompress<2>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<2, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 0>();

    BitCompression::decompress_block<2, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 1>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  15);
    BitCompression::overlap_value<2>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 24
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<24>() { return 4; }

template<>
template<>
inline const int BitCompression<25>::base_shift<24>() { return 12; }

template<>
template<>
inline int BitCompression<25>::next_offset<24>() { return 21; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<24>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<24>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 28) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<24, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0908070606050403ull), static_cast<long long int>(0x0F0E0D0C0C0B0A09ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 3;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<24, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<25>::decompress<24>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<24, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<24, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<24>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 21
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<21>() { return 4; }

template<>
template<>
inline const int BitCompression<25>::base_shift<21>() { return 12; }

template<>
template<>
inline int BitCompression<25>::next_offset<21>() { return 18; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<21>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<21>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 25) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<21, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0807060505040302ull), static_cast<long long int>(0x0F0E0D0C0B0A0908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x200000004ull), static_cast<long long int>(0x8000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<21, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<25>::decompress<21>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<21, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<21, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<21>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 18
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<18>() { return 4; }

template<>
template<>
inline const int BitCompression<25>::base_shift<18>() { return 12; }

template<>
template<>
inline int BitCompression<25>::next_offset<18>() { return 15; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<18>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<18>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 22) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<18, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0807060505040302ull), static_cast<long long int>(0x0E0D0C0B0B0A0908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 5;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<18, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<25>::decompress<18>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<18, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<18, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<18>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 15
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<15>() { return 4; }

template<>
template<>
inline const int BitCompression<25>::base_shift<15>() { return 12; }

template<>
template<>
inline int BitCompression<25>::next_offset<15>() { return 12; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<15>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<15>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 19) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<15, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0807060504030201ull), static_cast<long long int>(0x0E0D0C0B0B0A0908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000001ull), static_cast<long long int>(0x2000000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<15, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<25>::decompress<15>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<15, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<15, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<15>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 12
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<12>() { return 4; }

template<>
template<>
inline const int BitCompression<25>::base_shift<12>() { return 12; }

template<>
template<>
inline int BitCompression<25>::next_offset<12>() { return 9; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<12>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<12>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 16) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<12, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050404030201ull), static_cast<long long int>(0x0D0C0B0A0A090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<12, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<25>::decompress<12>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<12, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<12, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<12>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 9
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<9>() { return 4; }

template<>
template<>
inline const int BitCompression<25>::base_shift<9>() { return 12; }

template<>
template<>
inline int BitCompression<25>::next_offset<9>() { return 6; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<9>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<9>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 13) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<9, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050404030201ull), static_cast<long long int>(0x0D0C0B0A0A090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 4;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<9, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<25>::decompress<9>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<9, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<9, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<9>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 12
//Offset: 6
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<6>() { return 4; }

template<>
template<>
inline const int BitCompression<25>::base_shift<6>() { return 12; }

template<>
template<>
inline int BitCompression<25>::next_offset<6>() { return 3; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<6>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<6>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 10) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<6, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040303020100ull), static_cast<long long int>(0x0D0C0B0A0A090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000002ull), static_cast<long long int>(0x4000000080ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<6, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<25>::decompress<6>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<6, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  12);
    BitCompression::overlap_value<6>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 15
//Offset: 3
//Bits: 25


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<25>::remaining<3>() { return 5; }

template<>
template<>
inline const int BitCompression<25>::base_shift<3>() { return 15; }

template<>
template<>
inline int BitCompression<25>::next_offset<3>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<25>::block_count<3>() { return 2; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<25>::overlap_value<3>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0x1ffffff;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<3, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040303020100ull), static_cast<long long int>(0x0C0B0A0909080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000008ull), static_cast<long long int>(0x100000002ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<3, 0>()
{ 
     return 4;
}

template<>
template<>
inline void BitCompression<25>::decompress_block<3, 1>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x808080800F0E0D0Cull), static_cast<long long int>(0x8080808080808080ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x8000000001ull), static_cast<long long int>(0x2000000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x1ffffff01ffffffull), static_cast<long long int>(0x1ffffff01ffffffull)};
    static const int shift_mask = 7;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<25>::per_block<3, 1>()
{ 
     return 1;
}


template<>
template<>
inline void BitCompression<25>::decompress<3>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<3, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 0>();

    BitCompression::decompress_block<3, 1>(qw_block, ctr);
    ctr += BitCompression::per_block<3, 1>();


}


template<>
inline void BitCompression<25>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<25>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<0>()  + 1 ;

    BitCompression<25>::decompress<22>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<22>()  + 1 ;

    BitCompression<25>::decompress<19>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<19>()  + 1 ;

    BitCompression<25>::decompress<16>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<16>()  + 1 ;

    BitCompression<25>::decompress<13>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<13>()  + 1 ;

    BitCompression<25>::decompress<10>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<10>()  + 1 ;

    BitCompression<25>::decompress<7>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<7>()  + 1 ;

    BitCompression<25>::decompress<4>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<4>()  + 1 ;

    BitCompression<25>::decompress<1>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<1>()  + 1 ;

    BitCompression<25>::decompress<23>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<23>()  + 1 ;

    BitCompression<25>::decompress<20>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<20>()  + 1 ;

    BitCompression<25>::decompress<17>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<17>()  + 1 ;

    BitCompression<25>::decompress<14>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<14>()  + 1 ;

    BitCompression<25>::decompress<11>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<11>()  + 1 ;

    BitCompression<25>::decompress<8>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<8>()  + 1 ;

    BitCompression<25>::decompress<5>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<5>()  + 1 ;

    BitCompression<25>::decompress<2>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<2>()  + 1 ;

    BitCompression<25>::decompress<24>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<24>()  + 1 ;

    BitCompression<25>::decompress<21>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<21>()  + 1 ;

    BitCompression<25>::decompress<18>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<18>()  + 1 ;

    BitCompression<25>::decompress<15>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<15>()  + 1 ;

    BitCompression<25>::decompress<12>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<12>()  + 1 ;

    BitCompression<25>::decompress<9>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<9>()  + 1 ;

    BitCompression<25>::decompress<6>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<6>()  + 1 ;

    BitCompression<25>::decompress<3>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<25>::remaining<3>() ;

    *counter_out = tmp_counter;
}




////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 0
//Bits: 26


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<26>::remaining<0>() { return 4; }

template<>
template<>
inline const int BitCompression<26>::base_shift<0>() { return 13; }

template<>
template<>
inline int BitCompression<26>::next_offset<0>() { return 2; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<26>::block_count<0>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<26>::overlap_value<0>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 0) & 0x3ffffff;
}

template<>
template<>
inline void BitCompression<26>::decompress_block<0, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040303020100ull), static_cast<long long int>(0x0C0B0A0909080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffffff03ffffffull), static_cast<long long int>(0x3ffffff03ffffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<26>::per_block<0, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<26>::decompress<0>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<0, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<0, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<0>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 2
//Bits: 26


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<26>::remaining<2>() { return 4; }

template<>
template<>
inline const int BitCompression<26>::base_shift<2>() { return 13; }

template<>
template<>
inline int BitCompression<26>::next_offset<2>() { return 4; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<26>::block_count<2>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<26>::overlap_value<2>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 2) & 0x3ffffff;
}

template<>
template<>
inline void BitCompression<26>::decompress_block<2, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040303020100ull), static_cast<long long int>(0x0D0C0B0A09080706ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000010ull), static_cast<long long int>(0x4000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffffff03ffffffull), static_cast<long long int>(0x3ffffff03ffffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<26>::per_block<2, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<26>::decompress<2>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<2, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<2, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<2>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 4
//Bits: 26


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<26>::remaining<4>() { return 4; }

template<>
template<>
inline const int BitCompression<26>::base_shift<4>() { return 13; }

template<>
template<>
inline int BitCompression<26>::next_offset<4>() { return 6; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<26>::block_count<4>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<26>::overlap_value<4>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 4) & 0x3ffffff;
}

template<>
template<>
inline void BitCompression<26>::decompress_block<4, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0605040303020100ull), static_cast<long long int>(0x0D0C0B0A0A090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000004ull), static_cast<long long int>(0x1000000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffffff03ffffffull), static_cast<long long int>(0x3ffffff03ffffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<26>::per_block<4, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<26>::decompress<4>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<4, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<4, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<4>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 6
//Bits: 26


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<26>::remaining<6>() { return 4; }

template<>
template<>
inline const int BitCompression<26>::base_shift<6>() { return 13; }

template<>
template<>
inline int BitCompression<26>::next_offset<6>() { return 8; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<26>::block_count<6>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<26>::overlap_value<6>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 6) & 0x3ffffff;
}

template<>
template<>
inline void BitCompression<26>::decompress_block<6, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050403020100ull), static_cast<long long int>(0x0D0C0B0A0A090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000001ull), static_cast<long long int>(0x400000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffffff03ffffffull), static_cast<long long int>(0x3ffffff03ffffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<26>::per_block<6, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<26>::decompress<6>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<6, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<6, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<6>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 8
//Bits: 26


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<26>::remaining<8>() { return 4; }

template<>
template<>
inline const int BitCompression<26>::base_shift<8>() { return 13; }

template<>
template<>
inline int BitCompression<26>::next_offset<8>() { return 10; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<26>::block_count<8>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<26>::overlap_value<8>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 8) & 0x3ffffff;
}

template<>
template<>
inline void BitCompression<26>::decompress_block<8, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050404030201ull), static_cast<long long int>(0x0D0C0B0A0A090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffffff03ffffffull), static_cast<long long int>(0x3ffffff03ffffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<26>::per_block<8, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<26>::decompress<8>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<8, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<8, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<8>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 10
//Bits: 26


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<26>::remaining<10>() { return 4; }

template<>
template<>
inline const int BitCompression<26>::base_shift<10>() { return 13; }

template<>
template<>
inline int BitCompression<26>::next_offset<10>() { return 12; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<26>::block_count<10>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<26>::overlap_value<10>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 10) & 0x3ffffff;
}

template<>
template<>
inline void BitCompression<26>::decompress_block<10, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050404030201ull), static_cast<long long int>(0x0E0D0C0B0A090807ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000010ull), static_cast<long long int>(0x4000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffffff03ffffffull), static_cast<long long int>(0x3ffffff03ffffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<26>::per_block<10, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<26>::decompress<10>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<10, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<10, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<10>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 12
//Bits: 26


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<26>::remaining<12>() { return 4; }

template<>
template<>
inline const int BitCompression<26>::base_shift<12>() { return 13; }

template<>
template<>
inline int BitCompression<26>::next_offset<12>() { return 14; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<26>::block_count<12>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<26>::overlap_value<12>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 12) & 0x3ffffff;
}

template<>
template<>
inline void BitCompression<26>::decompress_block<12, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0706050404030201ull), static_cast<long long int>(0x0E0D0C0B0B0A0908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000004ull), static_cast<long long int>(0x1000000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffffff03ffffffull), static_cast<long long int>(0x3ffffff03ffffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<26>::per_block<12, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<26>::decompress<12>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<12, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<12, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<12>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 14
//Bits: 26


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<26>::remaining<14>() { return 4; }

template<>
template<>
inline const int BitCompression<26>::base_shift<14>() { return 13; }

template<>
template<>
inline int BitCompression<26>::next_offset<14>() { return 16; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<26>::block_count<14>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<26>::overlap_value<14>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 14) & 0x3ffffff;
}

template<>
template<>
inline void BitCompression<26>::decompress_block<14, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0807060504030201ull), static_cast<long long int>(0x0E0D0C0B0B0A0908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000001ull), static_cast<long long int>(0x400000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffffff03ffffffull), static_cast<long long int>(0x3ffffff03ffffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<26>::per_block<14, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<26>::decompress<14>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<14, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<14, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<14>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 16
//Bits: 26


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<26>::remaining<16>() { return 4; }

template<>
template<>
inline const int BitCompression<26>::base_shift<16>() { return 13; }

template<>
template<>
inline int BitCompression<26>::next_offset<16>() { return 18; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<26>::block_count<16>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<26>::overlap_value<16>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 16) & 0x3ffffff;
}

template<>
template<>
inline void BitCompression<26>::decompress_block<16, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0807060505040302ull), static_cast<long long int>(0x0E0D0C0B0B0A0908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffffff03ffffffull), static_cast<long long int>(0x3ffffff03ffffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<26>::per_block<16, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<26>::decompress<16>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<16, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<16, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<16>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 18
//Bits: 26


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<26>::remaining<18>() { return 4; }

template<>
template<>
inline const int BitCompression<26>::base_shift<18>() { return 13; }

template<>
template<>
inline int BitCompression<26>::next_offset<18>() { return 20; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<26>::block_count<18>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<26>::overlap_value<18>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 18) & 0x3ffffff;
}

template<>
template<>
inline void BitCompression<26>::decompress_block<18, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0807060505040302ull), static_cast<long long int>(0x0F0E0D0C0B0A0908ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x400000010ull), static_cast<long long int>(0x4000000001ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffffff03ffffffull), static_cast<long long int>(0x3ffffff03ffffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<26>::per_block<18, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<26>::decompress<18>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<18, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<18, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<18>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 20
//Bits: 26


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<26>::remaining<20>() { return 4; }

template<>
template<>
inline const int BitCompression<26>::base_shift<20>() { return 13; }

template<>
template<>
inline int BitCompression<26>::next_offset<20>() { return 22; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<26>::block_count<20>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<26>::overlap_value<20>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 20) & 0x3ffffff;
}

template<>
template<>
inline void BitCompression<26>::decompress_block<20, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0807060505040302ull), static_cast<long long int>(0x0F0E0D0C0C0B0A09ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x100000004ull), static_cast<long long int>(0x1000000040ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffffff03ffffffull), static_cast<long long int>(0x3ffffff03ffffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<26>::per_block<20, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<26>::decompress<20>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<20, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<20, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<20>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 22
//Bits: 26


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<26>::remaining<22>() { return 4; }

template<>
template<>
inline const int BitCompression<26>::base_shift<22>() { return 13; }

template<>
template<>
inline int BitCompression<26>::next_offset<22>() { return 24; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<26>::block_count<22>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<26>::overlap_value<22>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 22) & 0x3ffffff;
}

template<>
template<>
inline void BitCompression<26>::decompress_block<22, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0908070605040302ull), static_cast<long long int>(0x0F0E0D0C0C0B0A09ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x4000000001ull), static_cast<long long int>(0x400000010ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffffff03ffffffull), static_cast<long long int>(0x3ffffff03ffffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<26>::per_block<22, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<26>::decompress<22>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<22, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<22, 0>();


    // // extract last element
    __m128i tmp = _mm_alignr_epi8(_mm_load_si128(block + 1), qw_block,  13);
    BitCompression::overlap_value<22>(tmp, ctr); 
}


////////////////////////////////////////////////////////////////////////////////
//Generate shuffle masks for"
//Base Shift: 13
//Offset: 24
//Bits: 26


// Number of elements per m128 block
template<>
template<>
inline int BitCompression<26>::remaining<24>() { return 4; }

template<>
template<>
inline const int BitCompression<26>::base_shift<24>() { return 13; }

template<>
template<>
inline int BitCompression<26>::next_offset<24>() { return 0; }

// Next Offset based on the current offset
template<>
template<>
inline int BitCompression<26>::block_count<24>() { return 1; }

// Extract the last value which overlaps from the m128 register
template<>
template<>
inline void BitCompression<26>::overlap_value<24>(const __m128i& data, int* __restrict__ output)
{
    int64_t v = _mm_extract_epi32(data, 0);
    *output = (v >> 24) & 0x3ffffff;
}

template<>
template<>
inline void BitCompression<26>::decompress_block<24, 0>(const __m128i& data, int* __restrict__ output)
{
    static const __m128i shuffle_mask = {static_cast<long long int>(0x0908070606050403ull), static_cast<long long int>(0x0F0E0D0C0C0B0A09ull)};
    static const __m128i mull_mask = {static_cast<long long int>(0x1000000040ull), static_cast<long long int>(0x100000004ull)};
    static const __m128i and_mask = {static_cast<long long int>(0x3ffffff03ffffffull), static_cast<long long int>(0x3ffffff03ffffffull)};
    static const int shift_mask = 6;

    register __m128i shuffeled = _mm_shuffle_epi8(data, shuffle_mask);
    shuffeled = _mm_mullo_epi32(shuffeled, mull_mask);
    shuffeled = _mm_srli_epi32(shuffeled, shift_mask);    
    
    //_mm_store_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
    _mm_storeu_si128((__m128i*) output, _mm_and_si128(shuffeled, and_mask));
}

template<>
template<>
inline int BitCompression<26>::per_block<24, 0>()
{ 
     return 4;
}


template<>
template<>
inline void BitCompression<26>::decompress<24>(const __m128i* block, int* __restrict__ out)
{
    int *ctr = out;
    const register __m128i qw_block = _mm_load_si128(block);

    BitCompression::decompress_block<24, 0>(qw_block, ctr);
    ctr += BitCompression::per_block<24, 0>();


}


template<>
inline void BitCompression<26>::decompress_large(const __m128i* data, int* out, size_t* __restrict__ counter_out) 
{ 
    size_t tmp_counter = 0;

    const __m128i *data_moving = data;
    BitCompression<26>::decompress<0>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<26>::remaining<0>()  + 1 ;

    BitCompression<26>::decompress<2>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<26>::remaining<2>()  + 1 ;

    BitCompression<26>::decompress<4>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<26>::remaining<4>()  + 1 ;

    BitCompression<26>::decompress<6>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<26>::remaining<6>()  + 1 ;

    BitCompression<26>::decompress<8>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<26>::remaining<8>()  + 1 ;

    BitCompression<26>::decompress<10>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<26>::remaining<10>()  + 1 ;

    BitCompression<26>::decompress<12>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<26>::remaining<12>()  + 1 ;

    BitCompression<26>::decompress<14>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<26>::remaining<14>()  + 1 ;

    BitCompression<26>::decompress<16>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<26>::remaining<16>()  + 1 ;

    BitCompression<26>::decompress<18>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<26>::remaining<18>()  + 1 ;

    BitCompression<26>::decompress<20>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<26>::remaining<20>()  + 1 ;

    BitCompression<26>::decompress<22>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<26>::remaining<22>()  + 1 ;

    BitCompression<26>::decompress<24>(data_moving++, out + tmp_counter);
    tmp_counter += BitCompression<26>::remaining<24>() ;

    *counter_out = tmp_counter;
}


#endif

