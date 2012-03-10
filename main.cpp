#include "bcv.h"
#include "Timer.h"
#include "PapiTracer.h"

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <vector>

#define BITS 5

void test_set(long SIZE)
{
    std::cout << "[TEST ] set/get interleaved ..." << std::flush;
    BitCompressedVector<int> v(SIZE, BITS);
    for(size_t i=0; i < SIZE; ++i)
    {
        int a = i % (1UL << BITS);
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
        int a = i % (1UL << BITS);
        v.set(i, a);        
    }

    for(size_t i=0; i < SIZE; ++i)
    {
        int a = i % (1UL << BITS);
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
        int a = i % (1UL << BITS);
        v.set(i, a);        
        sum += a;
    }

    size_t alloca = ((64 / BITS)+1) * 8;
    int *tmp = (int*) malloc(sizeof(int) * alloca);

    for(size_t i=0; i < SIZE; )       
    {
        size_t actual = 0;
        v.mget(i, (int*) tmp, &actual);
        for(size_t j=0; j < actual; ++j, ++i)
        {
            int a = i % (1UL << BITS);
            sum2 += tmp[j];

            assert(a == tmp[j]);
        }
        
        

    }
    free(tmp);
    assert(sum == sum2);
    std::cout << " OK" << std::endl;
}

void test_mget_fixed(long SIZE)
{
    std::cout << "[TEST ] set/mget_fixed separated ..." << std::flush;
    long sum = 0, sum2 = 0;
    BitCompressedVector<int> v(SIZE, BITS);
    for(size_t i=0; i < SIZE; ++i)
    {
        int a = i % (1UL << BITS);
        v.set(i, a);        
        sum += a;
    }

    int *tmp = (int*) malloc(sizeof(int) * 20);

    for(size_t i=0; i < SIZE; )       
    {
        size_t actual = 16;
        v.mget_fixed(i, tmp, &actual);
        
        for(size_t j=0; j < actual; ++j)
            sum2 += tmp[j];

        // sum2 += tmp[0];
        // sum2 += tmp[1];
        // sum2 += tmp[2];
        // sum2 += tmp[3];
        // sum2 += tmp[4];
        // sum2 += tmp[5];
        // sum2 += tmp[6];
        // sum2 += tmp[7];
        // sum2 += tmp[8];
        // sum2 += tmp[9];
        // sum2 += tmp[10];
        // sum2 += tmp[11];
        // sum2 += tmp[12];
        // sum2 += tmp[13];
        // sum2 += tmp[14];
        // sum2 += tmp[15];
        // sum2 += tmp[16];
        // sum2 += tmp[17];
        // sum2 += tmp[18];
        // sum2 += tmp[19];
        
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
        v[i] = i % (1UL << BITS);
}

void performance(size_t size)
{
    BitCompressedVector<int> v(size, BITS);
    std::vector<int> v2(size);

    fill(v, size);
    fill(v2, size);

    double a,b,c,d,e;

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
    //int flags = PapiTracer::start();
    for(size_t i=0; i < size; )       
    {
        actual = 0;
        v.mget(i, tmp, &actual);
        for(size_t j=0; j < actual; ++j)
            res += tmp[j];
        
        i += actual;

    }
    //PapiTracer::result_t papi = PapiTracer::stop(flags);
    t.stop();
    std::cout << res << " mget time " << (c = t.elapsed_time()) << std::endl;
    //std::cout << papi.first << " " << papi.second << std::endl;
    free(tmp);

    ///////////////////////////////////////////////////////////////////////////
    tmp = (int*) malloc(sizeof(int) * 16);
    res = 0;
    t.start();
    actual = 0;
    for(size_t i=0; i < size; )       
    {        
        actual = 16;
        v.mget_fixed(i, tmp, &actual);
        
        // for(size_t j=0; j < actual; ++j)
        //     res += tmp[j];
        
        res += tmp[0];
        res += tmp[1];
        res += tmp[2];
        res += tmp[3];
        res += tmp[4];
        res += tmp[5];
        res += tmp[6];
        res += tmp[7];
        res += tmp[8];
        res += tmp[9];
        res += tmp[10];
        res += tmp[11];
        res += tmp[12];
        res += tmp[13];
        res += tmp[14];
        res += tmp[15];

        i += actual;

    }
    t.stop();
    std::cout << res << " mget fixed time " << (d = t.elapsed_time()) << std::endl;
    free(tmp);

    ///////////////////////////////////////////////////////////////////////////
    res = 0;
    t.start();
    for(size_t i=0; i < size; i+=1)  
    {
        res += v2[i];                
    }
    t.stop();
    std::cout << res << " vector time " << (e = t.elapsed_time()) << std::endl;
}


int main(int argc, char* argv[])
{
    // Setting size
    long SIZE = atol(argv[1]);

    test_set(SIZE);
    test_get(SIZE);
    test_mget(SIZE);
    test_mget_fixed(SIZE);

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