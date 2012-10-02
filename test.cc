#include "test.h"

#include <assert.h>

#define TEST_BITS 5ull

void test_set(long SIZE)
{
    std::cout << "[TEST ] set/get interleaved ..." << std::flush;
    BitCompressedVector<int, TEST_BITS> v(SIZE);
    for(size_t i=0; i < SIZE; ++i)
    {
        int a = i % (1UL << TEST_BITS);
        v.set(i, a);
        assert(a == v.get(i));
    }
    std::cout << " OK" << std::endl;
}

void test_get(long SIZE)
{
    std::cout << "[TEST ] set/get separated ..." << std::flush;
    BitCompressedVector<int, TEST_BITS> v(SIZE);
    for(size_t i=0; i < SIZE; ++i)
    {
        int a = i % (1UL << TEST_BITS);
        v.set(i, a);        
    }

    for(size_t i=0; i < SIZE; ++i)
    {
        int a = i % (1UL << TEST_BITS);
        assert(a == v.get(i));
    }
    std::cout << " OK" << std::endl;
}

void test_mget(long SIZE)
{
    std::cout << "[TEST ] set/mget separated ..." << std::flush;
    long sum = 0, sum2 = 0;
    BitCompressedVector<int, TEST_BITS> v(SIZE);
    for(size_t i=0; i < SIZE; ++i)
    {
        int a = i % (1UL << TEST_BITS);
        v.set(i, a);        
        sum += a;
    }

    size_t alloca = TEST_BITS * 1000 / 8;
    int *tmp = (int*) malloc(sizeof(int) * alloca);

    for(size_t i=0; i < SIZE; )       
    {
        size_t actual = 0;
        v.mget(i, (int*) tmp, &actual);

        for(size_t j=0; j < actual && i < SIZE; ++j, ++i)
        {
            int a = i % (1UL << TEST_BITS);
            sum2 += tmp[j];

            assert(a == tmp[j]);
        }
    }
    free(tmp);
    assert(sum == sum2);
    std::cout << " OK" << std::endl;
}

#include "decompress2.h"

void test_vertical(long SIZE)
{
    std::cout << "[TEST ] vertical ..." << std::flush;
    long sum = 0, sum2 = 0;
    const size_t mybits = 4;
    BitCompressedVector<int, mybits> v(SIZE);
    for(size_t i=0; i < SIZE; ++i)
    {
        int a = i % (1UL << mybits);
        v.set(i, a);        
        sum += a;
    }

    size_t alloca = mybits * 1000 / 8;
    int *tmp = (int*) malloc(sizeof(int) * alloca);
    size_t actual = 0;

    VerticalBitCompression<4>::decompress((__m128i*) v.getData(), tmp, &actual);

    size_t distance = 32 / 4;

    for(size_t i=0; i<actual; ++i)
    {
        //std::cout << i << " " << tmp[i] << " " << (i/4 + (i%4) * distance) % (1UL << mybits) << " " << v[(i/4 + (i%4) * distance)] << std::endl;
        assert( tmp[i] == ((i/4 + (i%4) * distance) % (1UL << mybits)) );
        assert( ((i/4 + (i%4) * distance) % (1UL << mybits)) == v[(i/4 + (i%4) * distance)]);
    }
}




void runTests()
{
	int64_t size = 10000;
	test_get(size);
	test_set(size);
	test_mget(size);
    test_vertical(size);

}