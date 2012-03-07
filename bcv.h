#include <iostream>
#include <memory>
#include <stdexcept>

#include "mask.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>


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

    static const uint8_t _width = sizeof(data_t) * 8;

	// Number of bits to use
	byte _bits;

	// Pointer to the data
	data_t *_data;

	size_t _reserved;

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
typename BitCompressedVector<T>::value_type BitCompressedVector<T>::get(size_t index) const
{
    value_type result;

    data_t pos = (index * _bits) / _width;
    data_t offset = (index * _bits) - (pos * _width);
    data_t bounds = _width - offset; 
   
    data_t mask;
    switch(_bits)
    {
        case 1: mask  = CreateMask<1>::mask(offset); break;
        case 2: mask  = CreateMask<2>::mask(offset); break;
        case 3: mask  = CreateMask<3>::mask(offset); break;
        case 4: mask  = CreateMask<4>::mask(offset); break;
        case 5: mask  = CreateMask<5>::mask(offset); break;
        case 6: mask  = CreateMask<6>::mask(offset); break;
        case 7: mask  = CreateMask<7>::mask(offset); break;
        case 8: mask  = CreateMask<8>::mask(offset); break;
        case 9: mask  = CreateMask<9>::mask(offset); break;
        case 10: mask = CreateMask<10>::mask(offset); break;
        case 11: mask = CreateMask<11>::mask(offset); break;
        case 12: mask = CreateMask<12>::mask(offset); break;
        case 13: mask = CreateMask<13>::mask(offset); break;
        case 14: mask = CreateMask<14>::mask(offset); break;
        case 15: mask = CreateMask<15>::mask(offset); break;
        case 16: mask = CreateMask<16>::mask(offset); break;
        case 17: mask = CreateMask<17>::mask(offset); break;
        case 18: mask = CreateMask<18>::mask(offset); break;
        case 19: mask = CreateMask<19>::mask(offset); break;
        case 20: mask = CreateMask<20>::mask(offset); break;
        case 21: mask = CreateMask<21>::mask(offset); break;
        case 22: mask = CreateMask<22>::mask(offset); break;
        case 23: mask = CreateMask<23>::mask(offset); break;
        case 24: mask = CreateMask<24>::mask(offset); break;
        case 25: mask = CreateMask<25>::mask(offset); break;
        case 26: mask = CreateMask<26>::mask(offset); break;
        case 27: mask = CreateMask<27>::mask(offset); break;
        case 28: mask = CreateMask<28>::mask(offset); break;
        case 29: mask = CreateMask<29>::mask(offset); break;
        case 30: mask = CreateMask<30>::mask(offset); break;
        case 31: mask = CreateMask<31>::mask(offset); break;
        case 32: mask = CreateMask<32>::mask(offset); break;
        case 33: mask = CreateMask<33>::mask(offset); break;
        case 34: mask = CreateMask<34>::mask(offset); break;
        case 35: mask = CreateMask<35>::mask(offset); break;
        case 36: mask = CreateMask<36>::mask(offset); break;
        case 37: mask = CreateMask<37>::mask(offset); break;
        case 38: mask = CreateMask<38>::mask(offset); break;
        case 39: mask = CreateMask<39>::mask(offset); break;
        case 40: mask = CreateMask<40>::mask(offset); break;
        case 41: mask = CreateMask<41>::mask(offset); break;
        case 42: mask = CreateMask<42>::mask(offset); break;
        case 43: mask = CreateMask<43>::mask(offset); break;
        case 44: mask = CreateMask<44>::mask(offset); break;
        case 45: mask = CreateMask<45>::mask(offset); break;
        case 46: mask = CreateMask<46>::mask(offset); break;
        case 47: mask = CreateMask<47>::mask(offset); break;
        case 48: mask = CreateMask<48>::mask(offset); break;
        case 49: mask = CreateMask<49>::mask(offset); break;
        case 50: mask = CreateMask<50>::mask(offset); break;
        case 51: mask = CreateMask<51>::mask(offset); break;
        case 52: mask = CreateMask<52>::mask(offset); break;
        case 53: mask = CreateMask<53>::mask(offset); break;
        case 54: mask = CreateMask<54>::mask(offset); break;
        case 55: mask = CreateMask<55>::mask(offset); break;
        case 56: mask = CreateMask<56>::mask(offset); break;
        case 57: mask = CreateMask<57>::mask(offset); break;
        case 58: mask = CreateMask<58>::mask(offset); break;
        case 59: mask = CreateMask<59>::mask(offset); break;
        case 60: mask = CreateMask<60>::mask(offset); break;
        case 61: mask = CreateMask<61>::mask(offset); break;
        case 62: mask = CreateMask<62>::mask(offset); break;
        case 63: mask = CreateMask<63>::mask(offset); break;
        case 64: mask = CreateMask<64>::mask(offset); break;        
    }

    result = (mask & _data[pos]) >> offset;

    if (bounds < _bits)
    {
        switch((_bits - bounds))
        {
            case 1: mask  = ~CreateMask<1>::mask(0); break;
            case 2: mask  = ~CreateMask<2>::mask(0); break;
            case 3: mask  = ~CreateMask<3>::mask(0); break;
            case 4: mask  = ~CreateMask<4>::mask(0); break;
            case 5: mask  = ~CreateMask<5>::mask(0); break;
            case 6: mask  = ~CreateMask<6>::mask(0); break;
            case 7: mask  = ~CreateMask<7>::mask(0); break;
            case 8: mask  = ~CreateMask<8>::mask(0); break;
            case 9: mask  = ~CreateMask<9>::mask(0); break;
            case 10: mask = ~CreateMask<10>::mask(0); break;
            case 11: mask = ~CreateMask<11>::mask(0); break;
            case 12: mask = ~CreateMask<12>::mask(0); break;
            case 13: mask = ~CreateMask<13>::mask(0); break;
            case 14: mask = ~CreateMask<14>::mask(0); break;
            case 15: mask = ~CreateMask<15>::mask(0); break;
            case 16: mask = ~CreateMask<16>::mask(0); break;
            case 17: mask = ~CreateMask<17>::mask(0); break;
            case 18: mask = ~CreateMask<18>::mask(0); break;
            case 19: mask = ~CreateMask<19>::mask(0); break;
            case 20: mask = ~CreateMask<20>::mask(0); break;
            case 21: mask = ~CreateMask<21>::mask(0); break;
            case 22: mask = ~CreateMask<22>::mask(0); break;
            case 23: mask = ~CreateMask<23>::mask(0); break;
            case 24: mask = ~CreateMask<24>::mask(0); break;
            case 25: mask = ~CreateMask<25>::mask(0); break;
            case 26: mask = ~CreateMask<26>::mask(0); break;
            case 27: mask = ~CreateMask<27>::mask(0); break;
            case 28: mask = ~CreateMask<28>::mask(0); break;
            case 29: mask = ~CreateMask<29>::mask(0); break;
            case 30: mask = ~CreateMask<30>::mask(0); break;
            case 31: mask = ~CreateMask<31>::mask(0); break;
            case 32: mask = ~CreateMask<32>::mask(0); break;
            case 33: mask = ~CreateMask<33>::mask(0); break;
            case 34: mask = ~CreateMask<34>::mask(0); break;
            case 35: mask = ~CreateMask<35>::mask(0); break;
            case 36: mask = ~CreateMask<36>::mask(0); break;
            case 37: mask = ~CreateMask<37>::mask(0); break;
            case 38: mask = ~CreateMask<38>::mask(0); break;
            case 39: mask = ~CreateMask<39>::mask(0); break;
            case 40: mask = ~CreateMask<40>::mask(0); break;
            case 41: mask = ~CreateMask<41>::mask(0); break;
            case 42: mask = ~CreateMask<42>::mask(0); break;
            case 43: mask = ~CreateMask<43>::mask(0); break;
            case 44: mask = ~CreateMask<44>::mask(0); break;
            case 45: mask = ~CreateMask<45>::mask(0); break;
            case 46: mask = ~CreateMask<46>::mask(0); break;
            case 47: mask = ~CreateMask<47>::mask(0); break;
            case 48: mask = ~CreateMask<48>::mask(0); break;
            case 49: mask = ~CreateMask<49>::mask(0); break;
            case 50: mask = ~CreateMask<50>::mask(0); break;
            case 51: mask = ~CreateMask<51>::mask(0); break;
            case 52: mask = ~CreateMask<52>::mask(0); break;
            case 53: mask = ~CreateMask<53>::mask(0); break;
            case 54: mask = ~CreateMask<54>::mask(0); break;
            case 55: mask = ~CreateMask<55>::mask(0); break;
            case 56: mask = ~CreateMask<56>::mask(0); break;
            case 57: mask = ~CreateMask<57>::mask(0); break;
            case 58: mask = ~CreateMask<58>::mask(0); break;
            case 59: mask = ~CreateMask<59>::mask(0); break;
            case 60: mask = ~CreateMask<60>::mask(0); break;
            case 61: mask = ~CreateMask<61>::mask(0); break;
            case 62: mask = ~CreateMask<62>::mask(0); break;
            case 63: mask = ~CreateMask<63>::mask(0); break;
            case 64: mask = ~CreateMask<64>::mask(0); break;        
        }
        result |= (mask & _data[pos + 1]) << bounds;
    } 
    return result;
}


template<typename T>
void BitCompressedVector<T>::set(size_t index, value_type v)
{
	data_t pos = _getPos(index);
	data_t offset = _getOffset(index, pos * _width);
	data_t bounds = _width - offset;
	
    data_t mask;
    switch(_bits)
    {
        case 1: mask  = ~CreateMask<1>::mask(offset); break;
        case 2: mask  = ~CreateMask<2>::mask(offset); break;
        case 3: mask  = ~CreateMask<3>::mask(offset); break;
        case 4: mask  = ~CreateMask<4>::mask(offset); break;
        case 5: mask  = ~CreateMask<5>::mask(offset); break;
        case 6: mask  = ~CreateMask<6>::mask(offset); break;
        case 7: mask  = ~CreateMask<7>::mask(offset); break;
        case 8: mask  = ~CreateMask<8>::mask(offset); break;
        case 9: mask  = ~CreateMask<9>::mask(offset); break;
        case 10: mask = ~CreateMask<10>::mask(offset); break;
        case 11: mask = ~CreateMask<11>::mask(offset); break;
        case 12: mask = ~CreateMask<12>::mask(offset); break;
        case 13: mask = ~CreateMask<13>::mask(offset); break;
        case 14: mask = ~CreateMask<14>::mask(offset); break;
        case 15: mask = ~CreateMask<15>::mask(offset); break;
        case 16: mask = ~CreateMask<16>::mask(offset); break;
        case 17: mask = ~CreateMask<17>::mask(offset); break;
        case 18: mask = ~CreateMask<18>::mask(offset); break;
        case 19: mask = ~CreateMask<19>::mask(offset); break;
        case 20: mask = ~CreateMask<20>::mask(offset); break;
        case 21: mask = ~CreateMask<21>::mask(offset); break;
        case 22: mask = ~CreateMask<22>::mask(offset); break;
        case 23: mask = ~CreateMask<23>::mask(offset); break;
        case 24: mask = ~CreateMask<24>::mask(offset); break;
        case 25: mask = ~CreateMask<25>::mask(offset); break;
        case 26: mask = ~CreateMask<26>::mask(offset); break;
        case 27: mask = ~CreateMask<27>::mask(offset); break;
        case 28: mask = ~CreateMask<28>::mask(offset); break;
        case 29: mask = ~CreateMask<29>::mask(offset); break;
        case 30: mask = ~CreateMask<30>::mask(offset); break;
        case 31: mask = ~CreateMask<31>::mask(offset); break;
        case 32: mask = ~CreateMask<32>::mask(offset); break;
        case 33: mask = ~CreateMask<33>::mask(offset); break;
        case 34: mask = ~CreateMask<34>::mask(offset); break;
        case 35: mask = ~CreateMask<35>::mask(offset); break;
        case 36: mask = ~CreateMask<36>::mask(offset); break;
        case 37: mask = ~CreateMask<37>::mask(offset); break;
        case 38: mask = ~CreateMask<38>::mask(offset); break;
        case 39: mask = ~CreateMask<39>::mask(offset); break;
        case 40: mask = ~CreateMask<40>::mask(offset); break;
        case 41: mask = ~CreateMask<41>::mask(offset); break;
        case 42: mask = ~CreateMask<42>::mask(offset); break;
        case 43: mask = ~CreateMask<43>::mask(offset); break;
        case 44: mask = ~CreateMask<44>::mask(offset); break;
        case 45: mask = ~CreateMask<45>::mask(offset); break;
        case 46: mask = ~CreateMask<46>::mask(offset); break;
        case 47: mask = ~CreateMask<47>::mask(offset); break;
        case 48: mask = ~CreateMask<48>::mask(offset); break;
        case 49: mask = ~CreateMask<49>::mask(offset); break;
        case 50: mask = ~CreateMask<50>::mask(offset); break;
        case 51: mask = ~CreateMask<51>::mask(offset); break;
        case 52: mask = ~CreateMask<52>::mask(offset); break;
        case 53: mask = ~CreateMask<53>::mask(offset); break;
        case 54: mask = ~CreateMask<54>::mask(offset); break;
        case 55: mask = ~CreateMask<55>::mask(offset); break;
        case 56: mask = ~CreateMask<56>::mask(offset); break;
        case 57: mask = ~CreateMask<57>::mask(offset); break;
        case 58: mask = ~CreateMask<58>::mask(offset); break;
        case 59: mask = ~CreateMask<59>::mask(offset); break;
        case 60: mask = ~CreateMask<60>::mask(offset); break;
        case 61: mask = ~CreateMask<61>::mask(offset); break;
        case 62: mask = ~CreateMask<62>::mask(offset); break;
        case 63: mask = ~CreateMask<63>::mask(offset); break;
        case 64: mask = ~CreateMask<64>::mask(offset); break;        
    }

	_data[pos] &= mask; 
	_data[pos] = _data[pos] | ((data_t) v << offset);

	if (bounds < _bits)
	{
        switch((_bits - bounds))
        {
            case 1: mask  = ~CreateMask<1>::mask(0); break;
            case 2: mask  = ~CreateMask<2>::mask(0); break;
            case 3: mask  = ~CreateMask<3>::mask(0); break;
            case 4: mask  = ~CreateMask<4>::mask(0); break;
            case 5: mask  = ~CreateMask<5>::mask(0); break;
            case 6: mask  = ~CreateMask<6>::mask(0); break;
            case 7: mask  = ~CreateMask<7>::mask(0); break;
            case 8: mask  = ~CreateMask<8>::mask(0); break;
            case 9: mask  = ~CreateMask<9>::mask(0); break;
            case 10: mask = ~CreateMask<10>::mask(0); break;
            case 11: mask = ~CreateMask<11>::mask(0); break;
            case 12: mask = ~CreateMask<12>::mask(0); break;
            case 13: mask = ~CreateMask<13>::mask(0); break;
            case 14: mask = ~CreateMask<14>::mask(0); break;
            case 15: mask = ~CreateMask<15>::mask(0); break;
            case 16: mask = ~CreateMask<16>::mask(0); break;
            case 17: mask = ~CreateMask<17>::mask(0); break;
            case 18: mask = ~CreateMask<18>::mask(0); break;
            case 19: mask = ~CreateMask<19>::mask(0); break;
            case 20: mask = ~CreateMask<20>::mask(0); break;
            case 21: mask = ~CreateMask<21>::mask(0); break;
            case 22: mask = ~CreateMask<22>::mask(0); break;
            case 23: mask = ~CreateMask<23>::mask(0); break;
            case 24: mask = ~CreateMask<24>::mask(0); break;
            case 25: mask = ~CreateMask<25>::mask(0); break;
            case 26: mask = ~CreateMask<26>::mask(0); break;
            case 27: mask = ~CreateMask<27>::mask(0); break;
            case 28: mask = ~CreateMask<28>::mask(0); break;
            case 29: mask = ~CreateMask<29>::mask(0); break;
            case 30: mask = ~CreateMask<30>::mask(0); break;
            case 31: mask = ~CreateMask<31>::mask(0); break;
            case 32: mask = ~CreateMask<32>::mask(0); break;
            case 33: mask = ~CreateMask<33>::mask(0); break;
            case 34: mask = ~CreateMask<34>::mask(0); break;
            case 35: mask = ~CreateMask<35>::mask(0); break;
            case 36: mask = ~CreateMask<36>::mask(0); break;
            case 37: mask = ~CreateMask<37>::mask(0); break;
            case 38: mask = ~CreateMask<38>::mask(0); break;
            case 39: mask = ~CreateMask<39>::mask(0); break;
            case 40: mask = ~CreateMask<40>::mask(0); break;
            case 41: mask = ~CreateMask<41>::mask(0); break;
            case 42: mask = ~CreateMask<42>::mask(0); break;
            case 43: mask = ~CreateMask<43>::mask(0); break;
            case 44: mask = ~CreateMask<44>::mask(0); break;
            case 45: mask = ~CreateMask<45>::mask(0); break;
            case 46: mask = ~CreateMask<46>::mask(0); break;
            case 47: mask = ~CreateMask<47>::mask(0); break;
            case 48: mask = ~CreateMask<48>::mask(0); break;
            case 49: mask = ~CreateMask<49>::mask(0); break;
            case 50: mask = ~CreateMask<50>::mask(0); break;
            case 51: mask = ~CreateMask<51>::mask(0); break;
            case 52: mask = ~CreateMask<52>::mask(0); break;
            case 53: mask = ~CreateMask<53>::mask(0); break;
            case 54: mask = ~CreateMask<54>::mask(0); break;
            case 55: mask = ~CreateMask<55>::mask(0); break;
            case 56: mask = ~CreateMask<56>::mask(0); break;
            case 57: mask = ~CreateMask<57>::mask(0); break;
            case 58: mask = ~CreateMask<58>::mask(0); break;
            case 59: mask = ~CreateMask<59>::mask(0); break;
            case 60: mask = ~CreateMask<60>::mask(0); break;
            case 61: mask = ~CreateMask<61>::mask(0); break;
            case 62: mask = ~CreateMask<62>::mask(0); break;
            case 63: mask = ~CreateMask<63>::mask(0); break;
            case 64: mask = ~CreateMask<64>::mask(0); break;        
        }
	   _data[pos + 1] &= mask; // clear bits
       _data[pos + 1] |= v >> bounds; // set bits and shift by the number of bits we already inserted
	}

}

