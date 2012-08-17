#include <stdlib.h>
#include <stdio.h>
#include <iostream>

// SSE requirements
#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>


#include <stdexcept>

#include "../Timer.h"
#include "../test.h"
#include "../bcv.h"

#define BITS 5
#define TUPLES 1000000000

//#define DEBUG_M128(m) std::cout << (uint64_t) _mm_extract_epi64(m, 0) << "  " << (uint64_t) _mm_extract_epi64(m, 1) << std::endl;

#define DEBUG_OUT(m) std::cout << m[0] << " " << m[1] << " " << m[2] << " "  << m[3] << " " << m[4] << " " << m[5] << " " << m[6] << " " << m[7] << " " << m[8] << " " << m[9] << " " << m[10] << " " << m[11] << " " << m[12] << " " << m[13] << " " << m[14] << " " << m[15] << std::endl;



#include "special.h"

int main(int argc, char* argv[])
{

  uint64_t some = 0;

    BitCompressedVector<int, BITS> v(TUPLES);
    fill<BitCompressedVector<int, BITS>, BITS>(v, TUPLES);

    __m128i *data = (__m128i*) v.getData();

    // The algorithm should always extract at max 128 values at a time
    const int out_size = TUPLES;

    // allocate and align memory
    int *out = NULL;
    posix_memalign((void**) &out, 64, out_size * sizeof(int));
    memset(out, 0, out_size * sizeof(int));
    
    // Main loop follows here
    uint64_t counter = 0;
    uint64_t block = 0;

    __m128i tmp = data[block];
    
    // Offset used to where we are at
    int offset = 0;

    const __m128i *data_moving = data;

    Timer t;
    t.start();

    while (counter < TUPLES)
    {
      switch (offset)
      {
        case 0:
          BitCompression<BITS>::decompress<0>(data_moving, out);
          offset = BitCompression<BITS>::next_offset<0>();
          counter += BitCompression<BITS>::remaining<0>() + 1;
          break;
        case 1:
          BitCompression<BITS>::decompress<1>(data_moving, out);
          offset = BitCompression<BITS>::next_offset<1>();
          counter += BitCompression<BITS>::remaining<1>() + 1;
          break;
        case 2:
          BitCompression<BITS>::decompress<2>(data_moving, out);
          offset = BitCompression<BITS>::next_offset<2>();
          counter += BitCompression<BITS>::remaining<2>() + 1;
          break;
        case 3:
          BitCompression<BITS>::decompress<3>(data_moving, out);
          offset = BitCompression<BITS>::next_offset<3>();
          counter += BitCompression<BITS>::remaining<3>() + 1;
          break;
        case 4:
          BitCompression<BITS>::decompress<4>(data_moving, out);
          offset = BitCompression<BITS>::next_offset<4>();
          counter += BitCompression<BITS>::remaining<4>() + 1;
          break;
      }

        //some += *out;
        ++data_moving;
    }

    t.stop();
    std::cout << " get time " << ( t.elapsed_time()) << std::endl;
    std::cout << some << std::endl;

    // for(size_t i=0; i < TUPLES; ++i)
    //   std::cout << out[i] << " ";

    // std::cout << std::endl;

    
    // __m128i tmp;

    // Timer t;
    // t.start();
    // // Get the first value
    // tmp = data[0];

    // // Start loop
    // decompress<5>::run<0,0>(tmp, out);
    // decompress<5>::run<0,1>(tmp, out + 4);
    // decompress<5>::run<0,2>(tmp, out + 8);
    // decompress<5>::run<0,3>(tmp, out + 12); 

    // DEBUG_OUT(out);

    // // Add 16
    // memset(out, 0, 16 * sizeof(int));
    // decompress<5>::run<0,4>(tmp, out);
    // decompress<5>::run<0,5>(tmp, out + 4); 
    // decompress<5>::run<0,6>(tmp, out + 8); 

    // DEBUG_OUT(out);

    // // Align and shift
    // DEBUG_M128(tmp);
    // DEBUG_M128(data[1]);

    // tmp = _mm_alignr_epi8(data[1], tmp, 15);

    // DEBUG_M128(tmp);
    // std::cout << v[25] << std::endl;


    // int upper = num_blocks - 1;
    // int i = 1;
    // do
    // {
    //     std::cout << "--" << std::endl;
    //     DEBUG_M128(tmp);
    //     DEBUG_M128(data[i+1]);

    //     memset(out, 0, 16 * sizeof(int));
    //     // Next remaining is 24 based on the previous offset
    //     decompress<5>::run<5,0>(tmp, out);
    //     decompress<5>::run<5,1>(tmp, out + 4);
    //     decompress<5>::run<5,2>(tmp, out + 8);
    //     decompress<5>::run<5,3>(tmp, out + 12);

    //     DEBUG_OUT(out);
        
    //     memset(out, 0, 16 * sizeof(int));
    //     decompress<5>::run<5,4>(tmp, out);
    //     decompress<5>::run<5,5>(tmp, out + 4); 
        
    //     DEBUG_OUT(out);

    //     // Align and shift
    //     tmp = _mm_alignr_epi8(data[++i], tmp, 15);
        
    // } while (i < upper);
    // t.stop();
    // std::cout << " get time " << ( t.elapsed_time()) << std::endl;

    // std::cout << out[0] << " " 
    // << out[1] << " "
    // << out[2] << " "
    // << out[3] << std::endl;

	return 0;
}
