import math
import pystache

renderer = pystache.Renderer(search_dirs="tpl")

REGISTER_WIDTH = 128
TYPE_WIDTH = 32

def mask_bits(val):
	return (1 << val) - 1

def build_128bit_val(val1, val2, val3, val4):
	return "{%s, %s}" %( hex(mask_bits(val3) + (mask_bits(val4) << 32))  ,hex(mask_bits(val1) + (mask_bits(val2) << 32)))
	


def generate_vertical(bits, type_width, size):
	"""
	In contrast to the other bit packing approach in this version we use 
	vertical bit packing this means that we assume that our data layout is 
	as follows for a 128 bit register for 4 bit integers

	1,5,9,...,2,6,10,....

	As a result we expect to extract always four integers with one load
	"""
	data = {}
	data["bits"] = bits
	data["mask"] = build_128bit_val(bits, bits, bits, bits)

	# Number of elements per extract operation
	elements_per_cycle = size / type_width


	# Number of extractions
	extractions = size / bits

	# Mask for extraction
	mask = (1 << bits) - 1

	# Start the outer loop
	data["extracts"] = []
	for i in range(extractions/elements_per_cycle):

		extract = {}
		# This is the outer loop per block of four extractions
		extract["shift"] = i * bits
		extract["use_shift"] = extract["shift"] > 0
		extract["no_shift"] = extract["shift"] == 0
				
		data["extracts"].append(extract)

	# In the end we need to check if there is an overlap
	if len(data["extracts"]) * elements_per_cycle * bits < size:
		data["has_overlap"] = {}

		base_value = extractions/elements_per_cycle * bits
		mask = ((1 << (bits - (type_width - base_value))) - 1) << type_width - base_value

		data["has_overlap"]["and_mask"] = build_128bit_val(mask, mask, mask, mask)
		data["has_overlap"]["shift"] = base_value
		data["has_overlap"]["shift_left"] = bits - (type_width - base_value)


	return data


all_data = {}
all_data["bits"] = []


#for bits in range(1,TYPE_WIDTH + 1):
#for bits in range(5,6):
for bits in [2,4,8,16]:
	all_data["bits"].append(generate_vertical(bits, TYPE_WIDTH, REGISTER_WIDTH))

print renderer.render_path("tpl/vertical.tpl", all_data)

	
