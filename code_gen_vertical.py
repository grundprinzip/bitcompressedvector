import math
import pystache
import fractions

renderer = pystache.Renderer(search_dirs="tpl")

REGISTER_WIDTH = 128
TYPE_WIDTH = 32

def mask_bits(val):
	return (1 << val) - 1

def build_128bit_val(val1, val2, val3, val4):
	return "{%s, %s}" %( hex(mask_bits(val3) + (mask_bits(val4) << 32))  ,hex(mask_bits(val1) + (mask_bits(val2) << 32)))
	


def generate_vertical(offset, bits, type_width, size):
	"""
	In contrast to the other bit packing approach in this version we use 
	vertical bit packing this means that we assume that our data layout is 
	as follows for a 128 bit register for 4 bit integers

	1,5,9,...,2,6,10,....

	As a result we expect to extract always four integers with one load
	"""
	data = {}
	data["offset"] = offset
	data["bits"] = bits
	data["mask"] = build_128bit_val(bits, bits, bits, bits)

	# Number of elements per extract operation
	elements_per_cycle = 4

	# Number of extractions
	extractions = (type_width-offset) / bits

	# Mask for extraction
	mask = (1 << bits) - 1

	# Start the outer loop
	data["extracts"] = []
	for i in range(extractions):

		extract = {}
		# This is the outer loop per block of four extractions
		extract["shift"] = i * bits + offset
		extract["use_shift"] = extract["shift"] > 0
		extract["no_shift"] = extract["shift"] == 0
				
		data["extracts"].append(extract)

	#print "---"

	#print len(data["extracts"])* bits + offset
	#print type_width

	# In the end we need to check if there is an overlap
	if len(data["extracts"]) * bits + offset < type_width:
		data["has_overlap"] = {}

		#old_part = type_width - (extractions * bits)
		#new_part = bits - old_part
		
		data["has_overlap"]["and_mask"] = build_128bit_val(bits, bits, bits, bits)
		data["has_overlap"]["shift"] = (extractions * bits + offset)
		data["has_overlap"]["shift_left"] = (type_width - (extractions * bits + offset))
	else:
		pass


	return data


all_data = {}
all_data["bits"] = []
all_data["blocks"] = []

for bits in range(1,16+1):

	single = {}
	single["bits"] = bits
	single["offsets"] = []
	for x in range(bits / fractions.gcd(bits, TYPE_WIDTH)):
		# Please somebody should simplify this...
		offset = (bits - ((32 - ((((x * TYPE_WIDTH) / bits) * bits) % 32)) % 32)) % bits
		single["offsets"].append({"offset":offset})

		all_data["bits"].append(generate_vertical(offset, bits, TYPE_WIDTH, REGISTER_WIDTH))

	all_data["blocks"].append(single)

print renderer.render_path("tpl/vertical.tpl", all_data)

	
