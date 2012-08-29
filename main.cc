#include "bcv.h"
#include "test.h"
#include "Timer.h"
#include "PapiTracer.h"

#include <algorithm>
#include <assert.h>
#include <iostream>
#include <vector>

#define BITS 5ull

#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>

void performance(size_t size)
{
    BitCompressedVector<int, BITS> v(size);
    std::vector<int> v2(size);

    fill<BitCompressedVector<int, BITS>, BITS>(v, size);
    fill<std::vector<int>, BITS>(v2, size);

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
    size_t alloca = 1000;
    int *tmp = (int*) malloc(sizeof(int) * alloca);

    size_t actual;
    t.start();
    for(size_t i=0; i < size; )       
    {
        actual = 0;
        v.mget(i, tmp, &actual);
        //for(size_t j=0; j < actual & i < size; ++j)
        res += tmp[0];
        
        i += actual;

    }
    t.stop();
    std::cout << res << " mget time " << (c = t.elapsed_time()) << std::endl;
    //std::cout << papi.first << " " << papi.second << std::endl;
    free(tmp);

    
    ///////////////////////////////////////////////////////////////////////////
    res = 0;
    t.start();
    for(size_t i=0; i < size; i+=16)  
    {
        res += v2[i];
        res += v2[i+1];
        res += v2[i+2];
        res += v2[i+3];
        res += v2[i+4];
        res += v2[i+5];
        res += v2[i+6];
        res += v2[i+7];
        res += v2[i+8];
        res += v2[i+9];
        res += v2[i+10];
        res += v2[i+11];
        res += v2[i+12];
        res += v2[i+13];
        res += v2[i+14];
        res += v2[i+15];
    }
    t.stop();
    std::cout << res << " vector time " << (e = t.elapsed_time()) << std::endl;
}


int main(int argc, char* argv[])
{
    // Setting size
    long SIZE = atol(argv[1]);

    //pshufb_test(SIZE);
    #ifndef NDEBUG
    runTests();
    #endif

    performance(SIZE);

	return 0;
}