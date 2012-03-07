#include <memory>
#include <stdexcept>
#include <numeric>

#ifndef NDEBUG
#define DEBUG(msg) std::cout << msg << std::endl;
#else
#define DEBUG(msg)
#endif

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
		size_t allocated_size = (size * bits) / (sizeof(data_t) * 8) + 1;
		posix_memalign((void**) &_data, 64, allocated_size * sizeof(data_t));
        memset(_data, 0, allocated_size);
	}

	~BitCompressedVector()
	{
		free(_data);
	}

	/*
		Original get method based on the index
	*/
	inline value_type get(size_t index) const;

	/*
		Set method to set a value
	*/
	inline void set(size_t index, value_type v);

private:


	typedef uint8_t byte;
	typedef uint64_t data_t;

	// Number of bits to use
	byte _bits;

	// Pointer to the data
	data_t *_data;

	size_t _reserved;

	data_t createMask(size_t offset, byte bits) const;

        // get the position of an index inside the list of data values
	inline size_t _getPos(size_t index) const
	{
		return (index * _bits) / (sizeof(data_t) * 8);
	}

        // get the offset of an index inside a block
	inline size_t _getOffset(size_t index) const
	{
		return (index * _bits) % (sizeof(data_t) * 8);
	}

};

template<typename T>
void BitCompressedVector<T>::set(size_t index, value_type v)
{
	data_t pos = _getPos(index);
	data_t offset = _getOffset(index);
	data_t bounds = (sizeof(data_t) * 8) - offset;

	data_t mask = ~createMask(offset, _bits);
    DEBUG("pos " << pos << " offset " << offset << " bounds " << bounds << " mask " << mask);
	_data[pos] &= mask; 
	_data[pos] = _data[pos] | (v << offset);

	if (bounds < _bits)
	{
	   mask = ~createMask(0, _bits - bounds); // create inverted mask
	   _data[pos + 1] &= mask; // clear bits
       _data[pos + 1] |= v >> bounds; // set bits and shift by the number of bits we already inserted
	}

}

template<typename T>
typename BitCompressedVector<T>::value_type BitCompressedVector<T>::get(size_t index) const
{
	value_type result;
	data_t pos = _getPos(index);
	data_t offset = _getOffset(index);
	data_t bounds = (sizeof(data_t) * 8) - offset; // This is almost static expression, that could be handled with a switch case

	data_t mask = createMask(offset, _bits);
    DEBUG("pos " << pos << " offset " << offset << " bounds " << bounds << " mask " << mask);

	result = (mask & _data[pos]) >> offset;

	if (bounds < _bits)
	{
		mask = createMask(0, _bits - bounds);
		result |= (mask & _data[pos + 1]) << bounds;
	} 
	return result;
}
