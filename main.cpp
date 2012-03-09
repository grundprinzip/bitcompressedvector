#include "bcv.h"
#include "Timer.h"
#include "PapiTracer.h"

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <vector>

#define BITS 3

void test_set(long SIZE)
{
    std::cout << "[TEST ] set/get interleaved ..." << std::flush;
    BitCompressedVector<int> v(SIZE, BITS);
    for(size_t i=0; i < SIZE; ++i)
    {
        int a = i % (1 << BITS);
        v.set(i, a);
        assert(a == v.get(i));
    }
    std::cout << " OK" << std::endl;
}

void test_get(long SIZE)
{
    std::cout << "[TEST ] set/get separated ..." << std::flush;
    BitCompressedVector<int> v(SIZE, BITS);
    for(size_t i=0; i < SIZE; ++i)
    {
        int a = i % (1 << BITS);
        v.set(i, a);        
    }

    for(size_t i=0; i < SIZE; ++i)
    {
        int a = i % (1 << BITS);
        assert(a == v.get(i));
    }
    std::cout << " OK" << std::endl;
}

void test_mget(long SIZE)
{
    std::cout << "[TEST ] set/mget separated ..." << std::flush;
    long sum = 0, sum2 = 0;
    BitCompressedVector<int> v(SIZE, BITS);
    for(size_t i=0; i < SIZE; ++i)
    {
        int a = i % (1 << BITS);
        v.set(i, a);        
        sum += a;
    }

    size_t alloca = ((64 / BITS)+1) * 8;
    int *tmp = (int*) malloc(sizeof(int) * alloca);

    for(size_t i=0; i < SIZE; )       
    {
        size_t actual = 0;
        v.mget(i, (int*) tmp, &actual);
        for(size_t j=0; j < actual; ++j)
            sum2 += tmp[j];
        
        i += actual;

    }
    free(tmp);

    assert(sum == sum2);
    std::cout << " OK" << std::endl;
}


template<class C>
void fill(C& v, size_t size)
{
    for(size_t i=0; i < size; ++i)
        v[i] = i % (1 << BITS);
}

void performance(size_t size)
{
    BitCompressedVector<int> v(size, BITS);
    std::vector<int> v2(size);

    fill(v, size);
    fill(v2, size);

    double a,b,c,d;

    Timer t;
    long long res = 0;

    ///////////////////////////////////////////////////////////////////////////
    t.start();
    for(size_t i=0; i < size; i+=1)  
    {
        res += v.get(i);                
    }
    t.stop();
    std::cout << res << " get time " << (a = t.elapsed_time()) << std::endl;

    ///////////////////////////////////////////////////////////////////////////
    res = 0;
    t.start();
    for(size_t i=0; i < size; i+=1)  
    {
        res += v[i];                
    }
    t.stop();
    std::cout << res << " get[] time " << (b = t.elapsed_time()) << std::endl;


    ///////////////////////////////////////////////////////////////////////////
    res = 0;
    size_t alloca = ((64 / BITS)+1) * 8;
    int *tmp = (int*) malloc(sizeof(int) * alloca);

    size_t actual;
    t.start();
    for(size_t i=0; i < size; )       
    {
        actual = 0;
        v.mget(i, (int*) tmp, &actual);
        for(size_t j=0; j < actual; ++j)
            res += tmp[j];
        
        i += actual;

    }
    t.stop();
    std::cout << res << " mget time " << (b = t.elapsed_time()) << std::endl;
    free(tmp);

    ///////////////////////////////////////////////////////////////////////////
    res = 0;
    t.start();
    for(size_t i=0; i < size; i+=1)  
    {
        res += v2[i];                
    }
    t.stop();
    std::cout << res << " vector time " << (d = t.elapsed_time()) << std::endl;
}


int main(int argc, char* argv[])
{
    // Setting size
    long SIZE = atol(argv[1]);

    test_set(SIZE);
    test_get(SIZE);
    test_mget(SIZE);


    performance(SIZE);


     //    t.start();
     //    int flags = PapiTracer::start();
     //    for(size_t i=0; i < SIZE; )       
     //    {
     //        size_t actual = 0;
     //        v.mget(i, (int*) &tmp, &actual);
     //        for(size_t j=0; j < actual; ++j)
     //            res += tmp[j];
            
     //        i += actual;

     //    }
     //    PapiTracer::result_t r = PapiTracer::stop(flags);
     //    t.stop();
     //    free(tmp);
     //    std::cout << res << " mget time " << (b = t.elapsed_time()) << std::endl;
     //    std::cout << r.first << " CYC " << r.second << std::endl;

   
	return 0;
}