
def generate_create_mask(bits):

	buf = "\n\n";

	buf += """
template<typename T>
typename BitCompressedVector<T>::data_t BitCompressedVector<T>::createMask(size_t offset, byte bits) const
{
    switch(bits)
    {
"""

	# For all bits
	for b in range(1, bits + 1):
			
		buf += "        case " + str(b) + ":\n"
		result = 0
		for i in range(b):
			if i > 0:
				result <<= 1
			result += 1
		buf += "        return " + str(result) + "ULL << offset;\n"
		buf += "        break;\n"

	buf += """    }
}"""
	return buf


print generate_create_mask(64)