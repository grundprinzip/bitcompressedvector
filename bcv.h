#ifndef BCV_BCV_H
#define BCV_BCV_H

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// SSE requirements
#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>

#define CACHE_LINE_SIZE 64

#ifndef NDEBUG

#define DEBUG(msg) std::cout << msg << std::endl;
#define DEBUG_M128(m) std::cout << (uint64_t) _mm_extract_epi64(m, 0) << "  " << (uint64_t) _mm_extract_epi64(m, 1) << std::endl;

#else

#define DEBUG(msg)

#endif

#include <assert.h>
#include "decompress.h"
#include "decompress2.h"

/*

 This class provides a numeric bit compressed vector. 
 Basically it provides a drop-in replacement for the standard std::vector. However,
 the number of bits allocated per value cannot be changed afterwards.


*/
template<typename T, uint64_t B>
class BitCompressedVector
{
    
public:

    typedef T   value_type;
    typedef T&  value_type_ref;
    typedef T*  value_type_ptr;

    typedef BitCompressedVector<T,B> vector_type;


    /*
    * Constructor
    */
    explicit BitCompressedVector(size_t size=0): _reserved(0), _size(0), _data(0), _allocated_blocks(0)
    {       
        reserve(size);

        // If the size is set to a value large 0 we assume
        // a repeated initialization
        if (size > 0)
            _size = size;
    }

    

    /* copy constructor */
    BitCompressedVector(const BitCompressedVector& other): _reserved(other._reserved),
    _size(other._size), _allocated_blocks(other._allocated_blocks), _data(nullptr)
    {
        _data = _allocate(_allocated_blocks);
        std::memcpy(_data, other._data, sizeof(data_t) * _allocated_blocks);
    }

    /* assignemnt operator */
    vector_type& operator=(BitCompressedVector other)
    {
        std::swap(_reserved, other._reserved);
        std::swap(_size, other._size);
        std::swap(_allocated_blocks, other._allocated_blocks);
        std::swap(_data, other._data);

        return *this;
    }
    
    /* Destructor */
    ~BitCompressedVector()
    {
        free(_data);
    }



    /*
    * Returns the used size of the vector
    */
    inline uint64_t size() const
    {
        return _size;
    }

    /*
    * Returns the reserved size of the vector
    */
    inline uint64_t capacity() const
    {
        return _reserved;
    }


    /*
     *  Original get method based on the index
     */
    inline value_type get(const size_t index) const;


    /* 

    This method returns a list of extracted values from the vector.The number
    of elements is variadic end depends on the number of elements inside a
    single block.

    Typicallay we try to extract at least a single cache line

    This operation is unchecked and will not throw an out of range exception
    in case the access is illegal

     */
    inline void mget(const size_t index, value_type_ptr data, size_t *actual) const;

    /*
     *  Set method to set a value
     *
     * This method will throw an out of range exception in case the access is illegal
     */
    inline void set(const size_t index, const value_type v);

    /*
    * Append a new value to the vector
    */
    inline void push_back(const value_type v)
    {
        if (_size + 1 > _reserved)
            reserve((_size + 1) * 2);

        // Increase the size
        set(_size++, v);
    }

    /*
    * Perform a resize of the vector.
    *
    * Currently, the size can only be increased not decreased.
    */
    inline void reserve(size_t size) 
    {
        if (_blocks(size) > _allocated_blocks)
        {
            data_t *newMemory = _allocate(_blocks(size));
            // Copy the data from the old data partition to the new partition
            std::memcpy(newMemory, _data, sizeof(data_t) * _allocated_blocks);

            // Allocate memory
            _allocated_blocks = _blocks(size);
            _reserved = _allocated_blocks * _width / B;

            // Swap pointers
            _data = newMemory;

        }
    }

    /*
        This small class is a simple proxy class that let's us handle reference 
        values to indizes in the bitvector without actually having a direct reference
    */
    struct BitVectorProxy
    {
        size_t _index;
        BitCompressedVector<T, B> *_vector;

        BitVectorProxy(size_t idx, BitCompressedVector<T, B> *v): _index(idx), _vector(v)
        {}

        // Implicit conversion operator used for rvalues of T
        inline operator const T () const 
        {
            return _vector->get(_index);
        }

        // Usins the Proxy to set the value using the subscript as an lvalue
        inline BitVectorProxy& operator= (const T& rvalue)
        {
            _vector->set(_index, rvalue);
            return *this;
        }

    };

    /*
     * Shortcut method for get(size_t index)
     */
    inline const BitVectorProxy operator[] (const size_t index) const
    {
        return BitVectorProxy(index, this);
    }

    inline BitVectorProxy operator[] (const size_t index)
    {
        return BitVectorProxy(index, this);
    }


private:

    // data type
    typedef uint64_t data_t;
    
    // Width determines the number of bits used to encode the block data type
    static const uint8_t _width = sizeof(data_t) * 8;

    // Pointer to the data, aligned
    data_t *_data __attribute__((aligned(16))) ;

    size_t _reserved;

    size_t _size;

    size_t _allocated_blocks;

    // get the position of an index inside the list of data values
    inline size_t _getPos(size_t index, size_t width=_width) const
    {
        return (index * B) / width;
    }

    // get the offset of an index inside a block
    inline size_t _getOffset(size_t index, size_t base) const
    {
        return (index * B) - base;
    }

    // returns the offset mask for any given index
    inline data_t buildMask(size_t index) const
    {
        return (index * B) % _width;
    }

    // returns the number of blocks required for the number of
    // elements to store
    inline size_t _blocks(size_t elements) const
    {
        // Simple ceil implementation
        return ((elements * B) + _width - 1 )/ _width;
    }

    /*
    * Allocate a number of blocks and zero out the memory
    */
    inline data_t* _allocate(size_t blocks) const 
    {
        data_t *newMemory = nullptr;
        posix_memalign((void**) &newMemory, 16, blocks * sizeof(data_t));
        memset(newMemory, 0, blocks * sizeof(data_t));
        return newMemory;
    }

public:

    data_t* getData(){ return _data; }

};


/**
*/
template<typename T, uint64_t B>
void BitCompressedVector<T, B>::set(const size_t index, const value_type v)
{
    if (index >= _size)
        throw std::out_of_range("Access not permitted");


    // Allocate new memory if required
    uint64_t pos = _getPos(index);
    uint64_t offset = _getOffset(index, pos * _width);
    uint64_t bounds = _width - offset;

    uint64_t mask, baseMask;
    baseMask = (1ull << B) - 1ull;
    mask = ~(baseMask << offset);


    _data[pos] &= mask;
    _data[pos] = _data[pos] | ((uint64_t) v << offset);

    if (bounds < B)
    {
        mask = ~(baseMask << offset); // we have a an overflow here thatswhy we do not need to care about the original stuff

       _data[pos + 1] &= mask; // clear bits
       _data[pos + 1] |= v >> bounds; // set bits and shift by the number of bits we already inserted
    }
}

/**
*/
template<typename T, uint64_t B>
typename BitCompressedVector<T, B>::value_type BitCompressedVector<T, B>::get(const size_t index) const
{
    value_type result;
    register uint64_t mask;

    register uint64_t pos = _getPos(index);
    register uint64_t offset = _getOffset(index, pos * _width);
    register uint64_t bounds = _width - offset; // This is almost static expression, that could be handled with a switch case

    mask = (1ull << B) - 1;
    register data_t block = _data[pos];
    block >>= offset;

    result = (mask & block);

    if (bounds < B)
    {
        offset = B - bounds;
        mask = (1ull << offset) - 1;

        result |= (mask & _data[pos + 1]) << bounds;
    }
    return result;
}

/**
*/
template<typename T, uint64_t B>
void BitCompressedVector<T, B>::mget(const size_t index, value_type_ptr data, size_t *actual) const
{
    //assert(128 % index == 0);
    BitCompression<B>::decompress_large(((const __m128i*) _data) + _getPos(index, 128), data, actual);
}


/**************************************************************************************************************/

template<typename T, uint64_t B>
class BitCompressedVectorVertical
{
    
public:

    typedef T   value_type;
    typedef T&  value_type_ref;
    typedef T*  value_type_ptr;

    typedef BitCompressedVectorVertical<T,B> vector_type;

    /*
    * Constructor
    */
    BitCompressedVectorVertical(size_t size=0): _reserved(size), _size(0), _allocated_blocks(0), _data(nullptr)
    {   
        reserve(size);

        // If the size is set to a value large 0 we assume
        // a repeated initialization
        if (size > 0)
            _size = size;        
    }

    /* copy constructor */
    BitCompressedVectorVertical(const BitCompressedVectorVertical& other):
        _reserved(other._reserved), _size(other._size), _allocated_blocks(other._allocated_blocks),
        _data(nullptr)
    {
        _data = _allocate(_allocated_blocks);
        std::memcpy(_data, other._data, sizeof(data_t) * _allocated_blocks);
    }

    /* assignemnt operator */
    vector_type& operator=(BitCompressedVectorVertical other)
    {
        std::swap(_reserved, other._reserved);
        std::swap(_size, other._size);
        std::swap(_allocated_blocks, other._allocated_blocks);
        std::swap(_data, other._data);

        return *this;
    }

    ~BitCompressedVectorVertical()
    {
        free(_data);
    }

    /*
     *  Original get method based on the index
     */
    inline value_type get(const size_t index) const;

    /*
    * Returns the used size of the vector
    */
    inline uint64_t size() const
    {
        return _size;
    }

    /*
    * Returns the reserved size of the vector
    */
    inline uint64_t capacity() const
    {
        return _reserved;
    }

    /*
    * Append a new value to the vector
    */
    inline void push_back(const value_type v)
    {
        if (_size + 1 > _reserved)
            reserve((_size + 1) * 2);

        // Increase the size
        set(_size++, v);
    }

    /*
    * Perform a resize of the vector.
    *
    * Currently, the size can only be increased not decreased.
    */
    inline void reserve(size_t size) 
    {
        if (_blocks(size) > _allocated_blocks)
        {
            data_t *newMemory = _allocate(_blocks(size));
            // Copy the data from the old data partition to the new partition
            std::memcpy(newMemory, _data, sizeof(data_t) * _allocated_blocks);

            // Allocate memory
            _allocated_blocks = _blocks(size);
            _reserved = _allocated_blocks * _width / B;

            // Swap pointers
            _data = newMemory;

        }
    }

    /* 

    This method returns a list of extracted values from the vector.The number
    of elements is variadic end depends on the number of elements inside a
    single block.

    Typicallay we try to extract at least a single cache line

     */
    inline void mget(const size_t index, value_type_ptr __restrict__ data, size_t* __restrict__ actual) const;

    /*
     *  Set method to set a value
     */
    inline void set(const size_t index, const value_type v);


    inline void cmp_eq_bv(const size_t index, const value_type v, value_type_ptr __restrict__ data, size_t* __restrict__ actual) const;

    /*
        This small class is a simple proxy class that let's us handle reference 
        values to indizes in the bitvector without actually having a direct reference
    */
    struct BitVectorProxy
    {
        size_t _index;
        BitCompressedVectorVertical<T, B> *_vector;

        BitVectorProxy(size_t idx, BitCompressedVectorVertical<T, B> *v): _index(idx), _vector(v)
        {}

        // Implicit conversion operator used for rvalues of T
        inline operator const T () const 
        {
            return _vector->get(_index);
        }

        // Usins the Proxy to set the value using the subscript as an lvalue
        inline BitVectorProxy& operator= (const T& rvalue)
        {
            _vector->set(_index, rvalue);
            return *this;
        }

    };

    /*
     * Shortcut method for get(size_t index)
     */
    inline const BitVectorProxy operator[] (const size_t index) const
    {
        return BitVectorProxy(index, this);
    }

    inline BitVectorProxy operator[] (const size_t index)
    {
        return BitVectorProxy(index, this);
    }


private:


    // data type
    typedef __m128i data_t;
    
    // Width determines the number of bits used to encode the block data type
    static const uint8_t _width = sizeof(data_t) * 8;

    // How many elements are we storing in a 32bit small block
    static const uint8_t _extracts = 4;
    static const uint8_t _extract_bits = 32;
    static const uint8_t _elements_per_small_block = (sizeof(uint32_t)*8) / B;
    static const uint8_t _elements_per_large_block = _extracts * _elements_per_small_block;

    // Pointer to the data, aligned
    data_t *_data __attribute__((aligned(16))) ;

    size_t _reserved;
    size_t _size;

    size_t _allocated_blocks;

    // returns the number of blocks required for the number of
    // elements to store
    inline size_t _blocks(size_t elements) const
    {
        // Simple ceil implementation
        return ((elements * B) + _width - 1 )/ _width;
    }

    /*
    * Allocate a number of blocks and zero out the memory
    */
    inline data_t* _allocate(size_t blocks) const 
    {
        data_t *newMemory = nullptr;
        posix_memalign((void**) &newMemory, 16, blocks * sizeof(data_t));
        memset(newMemory, 0, blocks * sizeof(data_t));
        return newMemory;
    }
    
public:

    data_t* getData(){ return _data; }

};

/**
*/
template<typename T, uint64_t B>
void BitCompressedVectorVertical<T, B>::set(const size_t index, const value_type v)
{
    if (index >= _size)
        throw std::out_of_range("Index out of range");

    register size_t block = (index / _extracts) * B / _extract_bits;
    register size_t in_block = index % _extracts;
    register size_t offset_in_block = ((index / _extracts) * B) % _extract_bits;

    register __m128i in = _data[block];

    // Shuffle the mask
    register __m128i mask = {static_cast<uint32_t>((static_cast<int32_t>(1) << B) - 1) << (offset_in_block), 0};
    register __m128i shuffled_mask;

    switch(in_block)
    {
        case 0: shuffled_mask = mask; break;
        case 1: shuffled_mask = _mm_shuffle_epi32(mask, 243); break;
        case 2: shuffled_mask = _mm_shuffle_epi32(mask, 975); break;
        case 3: shuffled_mask = _mm_shuffle_epi32(mask, 3903); break;
    }

    // Shuffle the value
    in = _mm_and_si128(in, ~shuffled_mask);

    // Shuffle the value
    __m128i out = {static_cast<uint32_t>(v) << (offset_in_block), 0};
    switch(in_block)
    {
        case 0: break;
        case 1: out = _mm_shuffle_epi32(out, 243); break;
        case 2: out = _mm_shuffle_epi32(out, 975); break;
        case 3: out = _mm_shuffle_epi32(out, 3903); break;
    }

    in = _mm_or_si128(in, out);
    _mm_store_si128(_data + block, in);

    if (_extract_bits - offset_in_block < B)
    {
        size_t diff = (_extract_bits - offset_in_block);
        mask = __m128i{((1ll << B) - 1) >> diff, 0};
        switch(in_block)
        {
            case 0: mask = mask; break;
            case 1: mask = _mm_shuffle_epi32(mask, 243); break;
            case 2: mask = _mm_shuffle_epi32(mask, 975); break;
            case 3: mask = _mm_shuffle_epi32(mask, 3903); break;
        }
        in = _data[block + 1];
        in = _mm_and_si128(in, ~mask);

        out = __m128i{v >> diff, 0};
        switch(in_block)
        {
            case 0: break;
            case 1: out = _mm_shuffle_epi32(out, 243); break;
            case 2: out = _mm_shuffle_epi32(out, 975); break;
            case 3: out = _mm_shuffle_epi32(out, 3903); break;
        }

        _mm_store_si128(_data + block + 1, _mm_or_si128(in, out));
    }
}

/**
*/
template<typename T, uint64_t B>
typename BitCompressedVectorVertical<T, B>::value_type BitCompressedVectorVertical<T, B>::get(const size_t index) const
{  
    // Shuffle masks
    static const int _shuffle2 = 1;
    static const int _shuffle3 = 2;
    static const int _shuffle4 = 3;

    register size_t block = (index / _extracts) * B / _extract_bits;
    register size_t in_block = index % _extracts;
    register size_t offset_in_block = ((index / _extracts) * B) % _extract_bits;

    static int32_t mask = (1 << B) - 1;


    uint32_t part;

    switch(in_block)
    {
        case 0: part = _mm_cvtsi128_si32(_data[block]); break;
        case 1: part = _mm_cvtsi128_si32(_mm_shuffle_epi32(_data[block], _shuffle2)); break;
        case 2: part = _mm_cvtsi128_si32(_mm_shuffle_epi32(_data[block], _shuffle3)); break;
        case 3: part = _mm_cvtsi128_si32(_mm_shuffle_epi32(_data[block], _shuffle4)); break;
    }
    
    part >>= offset_in_block;
    part &= mask;

    if (_extract_bits - offset_in_block < B)
    {
        // Get the block + 1
        int32_t upper;
        switch(in_block)
        {
            case 0: upper = _mm_cvtsi128_si32(_data[block+1]); break;
            case 1: upper = _mm_cvtsi128_si32(_mm_shuffle_epi32(_data[block+1], _shuffle2)); break;
            case 2: upper = _mm_cvtsi128_si32(_mm_shuffle_epi32(_data[block+1], _shuffle3)); break;
            case 3: upper = _mm_cvtsi128_si32(_mm_shuffle_epi32(_data[block+1], _shuffle4)); break;
        }
        size_t diff = _extract_bits - offset_in_block;
        upper = (upper << diff) & mask;
        part |= upper;
    }

    return part;
}

/**
*/
template<typename T, uint64_t B>
void BitCompressedVectorVertical<T, B>::mget(const size_t index, value_type_ptr __restrict__ data, size_t* __restrict__ actual) const
{
    register size_t block = (index / _extracts) * B / _extract_bits;
    VerticalBitCompression<B>::decompress(_data + block, data, actual);    
}

template<typename T, uint64_t B>
inline void BitCompressedVectorVertical<T,B>::cmp_eq_bv(const size_t index, const value_type cmp, value_type_ptr __restrict__ data, size_t* __restrict__ actual) const
{
    register size_t block = (index / _extracts) * B / _extract_bits;
    VerticalBitCompression<B>::cmp_eq(_data + block, cmp, data, actual);    
}

#endif // BCV_BCV_H
