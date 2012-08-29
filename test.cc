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




void runTests()
{
	int64_t size = 10000;
	test_get(size);
	test_set(size);
	test_mget(size);

}