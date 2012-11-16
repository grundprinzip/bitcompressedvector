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

void test_set_vertical(long SIZE)
{
    std::cout << "[TEST ] set/get interleaved ..." << std::flush;
    BitCompressedVectorVertical<int, TEST_BITS> v(SIZE);
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

void test_get_vertical(long SIZE)
{
    std::cout << "[TEST ] set/get separated ..." << std::flush;
    BitCompressedVectorVertical<int, TEST_BITS> v(SIZE);
    for(size_t i=0; i < SIZE; ++i)
    {
        int a = i % (1UL << TEST_BITS);
        v.set(i, a);
        if (i >= 77)
            assert(13 == v.get(77));
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
    std::cout << " OK" << std::endl;
}

void test_vertical_overlap(long SIZE)
{
    std::cout << "[TEST ] vertical overlap ..." << std::flush;
    long sum = 0, sum2 = 0;
    const size_t mybits = 5;
    BitCompressedVectorVertical<int, mybits> v(SIZE);
    for(size_t i=0; i < SIZE; ++i)
    {
        int a = i % (1UL << mybits);
        v.set(i, a);        
        sum += a;
    }

    size_t alloca = mybits * 1000 / 8;
    int *tmp = (int*) malloc(sizeof(int) * alloca);
    size_t actual = 0;

    for(size_t i=0; i<SIZE;)
    {
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

void test_vertical_cmp(long SIZE)
{
    std::cout << "[TEST ] vertical cmp_eq_bv ..." << std::flush;
    long sum = 0, sum2 = 0;
    const size_t mybits = 5;
    BitCompressedVectorVertical<int, mybits> v(SIZE);
    for(size_t i=0; i < SIZE; ++i)
    {
        int a = i % (1UL << mybits);
        v.set(i, a);        
        sum += a;
    }

    size_t alloca = mybits * 1000 / 8;
    int *tmp = (int*) malloc(sizeof(int) * alloca);
    size_t actual = 0;

    for(size_t i=0; i<SIZE;)
    {
        v.cmp_eq_bv(i, 1, (int*) tmp, &actual);

        for(size_t j=0; j < actual && i < SIZE; ++j, ++i)
        {
            int a = i % (1UL << TEST_BITS);
            if (a == 1)
            {
                assert(-1 == tmp[j]);
            } else {
                assert(0 == tmp[j]);
            }
        }
    }
    free(tmp);
    std::cout << " OK" << std::endl;
}

void test_allocation()
{
    BitCompressedVector<int, TEST_BITS> v;
    std::cout << "[TEST ] allocation ..." << std::flush;
    
    uint64_t reserved = 64/TEST_BITS;
    assert( 0ul == v.size());
    assert(0 == v.capacity());

    for(size_t i=0; i < reserved+1; ++i)
        v.push_back(i%TEST_BITS);

    assert((reserved+1) == v.size());

    uint64_t alloca = (3*64)/ TEST_BITS;
    assert(alloca == v.capacity());

    std::cout << " OK" << std::endl;
}

void test_large_allocation()
{
    BitCompressedVector<int, TEST_BITS> v;
    std::cout << "[TEST ] allocation large..." << std::flush;
    
    assert( 0ul == v.size());
    assert(0 == v.capacity());

    for(size_t i=0; i < 10000; ++i)
        v.push_back(i%TEST_BITS);

    assert(10000 == v.size());
    std::cout << " OK" << std::endl;
}

void test_allocation_vertical()
{
    BitCompressedVectorVertical<int, TEST_BITS> v;
    std::cout << "[TEST ] allocation vertical..." << std::flush;
    
    uint64_t reserved = 128/TEST_BITS;
    assert( 0ul == v.size());
    assert(0 == v.capacity());

    for(size_t i=0; i < reserved+1; ++i)
        v.push_back(i%TEST_BITS);

    assert((reserved+1) == v.size());

    uint64_t alloca = (3*128)/ TEST_BITS;
    assert(alloca == v.capacity());

    std::cout << " OK" << std::endl;
}

void test_large_allocation_vert()
{
    BitCompressedVectorVertical<int, TEST_BITS> v;
    std::cout << "[TEST ] allocation vertical large ..." << std::flush;
    
    assert( 0ul == v.size());
    assert(0 == v.capacity());

    for(size_t i=0; i < 10000; ++i)
        v.push_back(i%TEST_BITS);

    assert(10000 == v.size());
    std::cout << " OK" << std::endl;
}

void test_assignment_constructors()
{
    std::cout << "[TEST ] assignment..." << std::flush;
    BitCompressedVector<uint32_t, 5> a(100), b;
    b = a;

    assert(a.size() == b.size());
    assert(a.getData() != b.getData());
    std::cout << " OK" << std::endl;
}

void test_copy_constructors()
{
    std::cout << "[TEST ] copy constructors..." << std::flush;
    BitCompressedVector<uint32_t, 5> a(100), b;
    b = BitCompressedVector<uint32_t, 5>(a);

    assert(a.size() == b.size());
    assert(a.getData() != b.getData());
    std::cout << " OK" << std::endl;
}

void test_assignment_constructors_vert()
{
    std::cout << "[TEST ] assignment vertical..." << std::flush;
    BitCompressedVectorVertical<uint32_t, 5> a(100), b;
    b = a;

    assert(a.size() == b.size());
    assert(a.getData() != b.getData());
    std::cout << " OK" << std::endl;
}

void test_copy_constructors_vert()
{
    std::cout << "[TEST ] copy constructors vertical..." << std::flush;
    BitCompressedVectorVertical<uint32_t, 5> a(100), b;
    b = BitCompressedVectorVertical<uint32_t, 5>(a);

    assert(a.size() == b.size());
    assert(a.getData() != b.getData());
    std::cout << " OK" << std::endl;
}


void runTests()
{
	int64_t size = 10000;
	test_get(size);
	test_set(size);
    test_set_vertical(size);
    test_get_vertical(size);
	test_mget(size);
    test_vertical(size);
    test_vertical_overlap(size);
    test_vertical_cmp(size);
    test_allocation();
    test_large_allocation();
    test_allocation_vertical();
    test_large_allocation_vert();
    test_assignment_constructors();
    test_copy_constructors();
    test_assignment_constructors_vert();
    test_copy_constructors_vert();

}