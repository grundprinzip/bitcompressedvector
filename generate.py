
def generate_create_mask(bits):

	buf = ""
	# For all bits
	for b in range(1, bits + 1):
		buf += """

	
template<>
struct CreateMask<%d>
{

    static inline uint64_t mask()
	{
""" % b
			
		result = 0
		for i in range(b):
			if i > 0:
				result <<= 1
			result += 1
		buf += "        return " + str(result) + "ULL;"
		buf += """    
    }
};

"""
	return buf


print generate_create_mask(64)
print "#endif //BCV_MASK_H"