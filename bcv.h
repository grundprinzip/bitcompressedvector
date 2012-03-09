#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bcv_defines.h"
#include "mask.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>


#ifndef NDEBUG
#define DEBUG(msg) std::cout << msg << std::endl;
#else
#define DEBUG(msg)
#endif

/*

 This class provides a numeric bit compressed vector. 
 Basically it provides a drop-in replacement for the standard std::vector. However,
 the number of bits allocated per value cannot be changed afterwards.


*/
template<typename T>
class BitCompressedVector
{
	
public:

	typedef T   value_type;
	typedef T&  value_type_ref;
	typedef T*  value_type_ptr;


	/*
	* Constructor
	*/
	BitCompressedVector(size_t size, unsigned char bits): _bits(bits), _reserved(size)
	{		
		_allocated_blocks = (size * bits) / (sizeof(data_t) * 8) + 1;
		posix_memalign((void**) &_data, 64, _allocated_blocks * sizeof(data_t));
        memset(_data, 0, _allocated_blocks * sizeof(data_t));
	}

	~BitCompressedVector()
	{
		free(_data);
	}

	/*
	 *	Original get method based on the index
	 */
	inline value_type get(const size_t index) const;


    /* 

    This method returns a list of extracted values from the vector.The number
    of elements is variadic end depends on the number of elements inside a
    single block.

    Typicallay we try to extract at least a single cache line

     */
    inline void mget(const size_t index, value_type_ptr data, size_t *actual) const;


	/*
	 *	Set method to set a value
	 */
	inline void set(const size_t index, const value_type v);

    /*
     * Shortcut method for get(size_t index)
     */
    inline value_type operator[] (const size_t index) const
    {
        return get(index);
    }

private:


	typedef uint8_t byte;
	typedef uint32_t data_t;

    static const uint8_t _width = sizeof(data_t) * 8;

	// Number of bits to use
	byte _bits;

	// Pointer to the data
	data_t *_data;

	size_t _reserved;

    size_t _allocated_blocks;

	// get the position of an index inside the list of data values
	inline size_t _getPos(size_t index) const
	{
		return (index * _bits) / _width;
	}

    // get the offset of an index inside a block
	inline size_t _getOffset(size_t index, size_t base) const
	{
		return (index * _bits) - base;
	}

};


template<typename T>
void BitCompressedVector<T>::mget(const size_t index, value_type_ptr data, size_t *actual) const
{

    // Number of blocks to extract
    data_t num_blocks = CACHE_LINE_SIZE / sizeof(data_t);

    // First get the initial values
    data_t pos = _getPos(index);
    data_t mask = 0;

    // Running values for the loop
    data_t currentValue;
    size_t currentIndex = index;

    data_t offset = _getOffset(currentIndex, pos * _width);
    data_t bounds = _width - offset;
    CREATE_MASK(_bits, mask, offset);

    *actual = 0;

    size_t upper = _allocated_blocks < pos + num_blocks ? _allocated_blocks : pos + num_blocks;
    while(pos < upper)
    {
        currentValue = (mask & _data[pos]) >> offset;

        if (__builtin_expect(bounds > _bits, 1))
        {
            bounds -= _bits;
            offset += _bits;
            mask <<= _bits;

        } else {

            data_t b = _bits - bounds;
            CREATE_MASK(b, mask, 0);
            currentValue |= (mask & _data[pos + 1]) << bounds;

            // Increment pos
            ++pos;
            offset = b;
            bounds = _width - offset;
            CREATE_MASK(_bits, mask, offset);
        } 
        
        // Append current value
        data[*actual] = currentValue;
        *actual += 1;
    }
}


template<typename T>
void BitCompressedVector<T>::set(const size_t index, const value_type v)
{
	data_t pos = _getPos(index);
	data_t offset = _getOffset(index, pos * _width);
	data_t bounds = _width - offset;
	
    data_t mask;
    CREATE_MASK(_bits, mask, offset);
    mask = ~mask;

	_data[pos] &= mask; 
	_data[pos] = _data[pos] | ((data_t) v << offset);

	if (bounds < _bits)
	{
        CREATE_MASK(_bits, mask, offset);
        mask = ~mask;

	   _data[pos + 1] &= mask; // clear bits
       _data[pos + 1] |= v >> bounds; // set bits and shift by the number of bits we already inserted
	}

}

template<typename T>
typename BitCompressedVector<T>::value_type BitCompressedVector<T>::get(const size_t index) const
{
	value_type result;
    data_t mask;

	data_t pos = _getPos(index);
	data_t offset = _getOffset(index, pos * _width);
	data_t bounds = _width - offset; // This is almost static expression, that could be handled with a switch case
	
    CREATE_MASK(_bits, mask, offset);

	result = (mask & _data[pos]) >> offset;

	if (bounds < _bits)
	{
        data_t b = _bits - bounds;
        CREATE_MASK(b, mask, 0);
		result |= (mask & _data[pos + 1]) << bounds;
	} 
	return result;
}
