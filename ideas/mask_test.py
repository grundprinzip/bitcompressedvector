import fractions
import math
import pystache

renderer = pystache.Renderer(search_dirs="tpl")

def format(val):
    if val > 15:
        raise Exception("Error with val", val)
    if val < 10:
        return "0"+str(val)
    if val == 10:
        return "0A"
    if val == 11:
        return "0B"
    if val == 12:
        return "0C"
    if val == 13:
        return "0D"
    if val == 14:
        return "0E"
    if val == 15:
        return "0F"


def build_mask(index_in_vector, bits, elements, offset):
    """
    This method builds the shuffle masks that we 
    use to extract the data words from the 128 bit element
    """

    if index_in_vector < elements:
        val = ""
        # Check how many bytes we need to touch, by checking the first bit
        start_byte = (index_in_vector * bits + offset) / 8
        # We check the position of the last bit of the word
        end_byte = (index_in_vector * bits + bits - 1 + offset) / 8

        num_bytes = end_byte - start_byte + 1


        # If we have more than 4 bytes we need additional handling. In any case we
        # can shift the upper half of the qw to the right to align it to byte
        # boundaries. Then we extract the lower half, shift it and add it with 
        # the masked higher part
        if num_bytes > 4:
            raise Exception("bytes", num_bytes)

        # Format the upper part with zeros
        for x in range(4 - num_bytes):
            val += "80"

        # the lower part contains the shuffle mask
        for x in reversed(range(num_bytes)):
            val += format((index_in_vector * bits + offset) / 8 + x)

        return val
    else:
        return "80808080"



################################################################################
# Special cases:
# The most complicated case is the 5 byte spanning value case which occurs for 
# the following bit cases :
#      - 27,29,30,31
# Question is if we just ignore these cases and use 32bit compression then
all_data = {}
all_data["bits"] = []

for bits in range(1, 27):

    bit_elements = {}
    bit_elements["data"] = []
    bit_elements["bits"] = str(bits)
    size = 128

    for o in range(bits/fractions.gcd(bits, size)):
    
        # Based on the number of iterations we can calculate which one we need to look at now
        rest = o*size - (o * size / bits) * bits
    
        # offset is the offset from the new value
        offset = rest#size - rest - base_shift * 8
    
        # we want to generate all possible shuffle instructions
        # based on the usable bits
        elements = (size-offset) / bits 
        extracts = int(math.ceil(elements / 4.0))
        num_bytes = int(math.ceil(bits / 8.0))
    
        # base shift is used for concatenating the two m128i values 
        base_shift = (elements * bits) / 8
    
        # What is the next offset
        next_offset = (bits - (size - (offset + elements * bits))) % bits
        
        element_data = {}
        
        # print "//","-" * 80
        # print "//Rest", rest
        # print "//Generate shuffle masks for"
        # print "//Base Shift:", base_shift
        # print "//Offset:", offset
        # print "//Bits:", bits
    
        element_data["elements"] = elements
        element_data["offset"] = offset
        element_data["bits"] = bits
        element_data["base_shift"] = base_shift
        element_data["next_offset"] = next_offset
        element_data["num_extracts"] = extracts
        element_data["overlap_shift"] = (elements * bits) % 8 + offset
        element_data["overlap_mask"] = hex(2**bits -1)
        element_data["extracts"] = []
    
        # Generate the shuffle masks for extracting from one single 16 byte register 
        # without offset
    
        for i in range(extracts):
    
            extracts_data = {}
            extracts_data["bits"] = bits
            extracts_data["block"] = i
            extracts_data["offset"] = offset
            extracts_data["elements"] = elements
            extracts_data["block_elements"] = 0
    
            val = "{0x"
    
            for cnt in [i*4+1, i*4]:
                val += build_mask(cnt, bits, elements, offset)
                extracts_data["block_elements"] += 1 if cnt < elements else 0
    
    
            val += ", 0x"
            for cnt in [i*4+3, i*4 + 2]:
                val += build_mask(cnt, bits, elements, offset)
                extracts_data["block_elements"] += 1 if cnt < elements else 0
    
            val += "}"
        
            #print val
            extracts_data["shuffle"] = val
    
    
            # The mul masks are a baisc independent shift left, if we multiply by
            # power of twos
            #print "shift mull mask"
    
            # Generate the bit offsets to the preceeding bytes
            all_offsets = [((i * 4 + k) * bits + offset) % 8 for k in range(4)]
    
            #print "// All offsets", all_offsets
            # This is the shift target
            max_invalid_bits = max(all_offsets)
            
            # For each element derive the independent shift level, by 
            # substracting the current offset from the maximum offset
            mullo = [2**(max_invalid_bits - k) for k in all_offsets]
            
            # Generate buffers of m128i
            buf = "{"
            val = (mullo[1] << 32) + mullo[0]
            buf += str(hex(val)) + ", "
    
            val = (mullo[3] << 32) + mullo[2]
            buf += str(hex(val)) + "}"
    
            #print buf
            extracts_data["mullo"] = buf
    
            #print "shift mask"
            #print max_invalid_bits
            extracts_data["shift"] = max_invalid_bits
    
            #print "and mask"
    
            anded = 2**bits - 1
    
            #print hex((anded << 32) + (2**bits -1))
            extracts_data["and"] = str(hex((anded << 32) + (2**bits -1)))
    
            element_data["extracts"].append(extracts_data)
            ########################################################################
    
        
        bit_elements["data"].append(element_data)
    
    # We have to reorder data , so that the offsets are in the correct order
    new_order = []
    old_order = bit_elements["data"]
    
    new_order.append(old_order[0])
    next = old_order[0]["next_offset"]
    for x in range(len(old_order)-1):
        t = [a for a in old_order if a["offset"] == next]
        next = t[0]["next_offset"]
        new_order.append(t[0])

    
           
    bit_elements["data"] = new_order
    # Append to the bit list
    all_data["bits"].append(bit_elements)


print renderer.render_path("tpl/all.tpl", all_data)






