#include "bcv.h"
#include "Timer.h"

#include <algorithm>
#include <iostream>
#include <vector>

#define BITS 5

int main(int argc, char* argv[])
{
        // Setting size
        long SIZE = atol(argv[1]);

        Timer t;
	   BitCompressedVector<int> v(SIZE, BITS);
        vector<int> v2;

        double a,b,c;

        long long res = 0;

    	for(size_t i=0; i < SIZE; ++i)
    	{
    		v.set(i, i % (1 << BITS));
                    v2.push_back(i % (1 << BITS));
    	}

        t.start();
    	for(size_t i=0; i < SIZE; i+=1)	
    	{
    		res += v.get(i);                
    	}
        t.stop();

        std::cout << res << " get time " << (a = t.elapsed_time()) << std::endl;
        res = 0;

        int data[110];

        t.start();
        for(size_t i=0; i < SIZE; )       
        {
            size_t actual = 0;
            v.mget(i, (int*) &data, &actual);
            for(size_t j=0; j < actual; ++j)
                res += data[j];
            
            i += actual;

        }
        t.stop();
        std::cout << res << " mget time " << (b = t.elapsed_time()) << std::endl;

        res = 0;
        t.start();
        for(size_t i=0; i < SIZE; ++i)       
        {
                res += v2[i];
        }
        t.stop();
        std::cout << res << " vector time " << (c = t.elapsed_time()) << std::endl;

        std::cout << a / c << std::endl;
        std::cout << b / c << std::endl;
	
        if (argc > 2)
        {
                std::cout << "random access" << std::endl;
                std::vector<size_t> vPosList;
                for(size_t i=0; i < SIZE; ++i)
                        vPosList.push_back(i);

                std::random_shuffle(vPosList.begin(), vPosList.end());

                res = 0;
                t.start();
                for(size_t i=0; i < SIZE; ++i)       
                {
                        res += v.get(vPosList[i]);
                }
                t.stop();
                std::cout << res << " time " << (a = t.elapsed_time()) << std::endl;

                res = 0;
                t.start();
                for(size_t i=0; i < SIZE; ++i)       
                {
                        res += v2[vPosList[i]];
                }
                t.stop();
                std::cout << res << " time " << (b = t.elapsed_time()) << std::endl;
                std::cout << a / b << std::endl;
        }

	return 0;
}