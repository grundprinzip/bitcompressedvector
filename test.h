#ifndef BCV_TEST_H

#include "bcv.h"

void runTests();


template<class C, int BITS>
void fill(C& v, size_t size)
{
    for(size_t i=0; i < size; ++i)
        v[i] = i % (1UL << BITS);
}

#endif // BCV_TEST_H