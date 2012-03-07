#include "bcv.h"
#include "Timer.h"

#include <algorithm>
#include <iostream>
#include <vector>

#define SIZE 100000000

int main()
{
        Timer t;
	BitCompressedVector<int> v(SIZE, 3);
        vector<int> v2;

        long long res = 0;

	for(size_t i=0; i < SIZE; ++i)
	{
		v.set(i, i % 8);
                v2.push_back(i % 8);
	}

        t.start();
	for(size_t i=0; i < SIZE; ++i)	
	{
		res += v.get(i);
	}
        t.stop();
        std::cout << res << " time " << t.elapsed_time() << std::endl;

        res = 0;
        t.start();
        for(size_t i=0; i < SIZE; ++i)       
        {
                res += v2[i];
        }
        t.stop();
        std::cout << res << " time " << t.elapsed_time() << std::endl;
	

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
        std::cout << res << " time " << t.elapsed_time() << std::endl;

        res = 0;
        t.start();
        for(size_t i=0; i < SIZE; ++i)       
        {
                res += v2[vPosList[i]];
        }
        t.stop();
        std::cout << res << " time " << t.elapsed_time() << std::endl;


	return 0;
}