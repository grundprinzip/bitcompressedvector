#include <stdint.h>

template<int M>
struct CreateMask
{

	static inline uint64_t mask()
	{
		uint64_t result = 0;
		for(size_t i=0; i < M; ++i)
		{
			if (i > 0)
				result << 1;

			result += 1;
		}

		return result;
	}
	
};
