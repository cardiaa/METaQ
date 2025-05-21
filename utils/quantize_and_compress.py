import torch
import math
import gzip
import zstandard as zstd
import struct
from heapq import heappush, heappop, heapify
from decimal import Decimal
from collections import Counter

def compute_entropy(string):
    """
    Function to compute the entropy of a given string.
    """
    # Count the frequency of each character in the string
    frequencies = Counter(string)
    # Calculate the total length of the string
    total_length = len(string)
    
    # Compute entropy
    entropy = 0
    for freq in frequencies.values():
        probability = freq / total_length  # Compute probability of each character
        entropy -= probability * math.log2(probability)  # Apply entropy formula
    
    return entropy * total_length  # Return the entropy weighted by string length

def compute_entropy_new(string, pruning_threshold):
    """
    Function to compute a modified entropy measure for a list of numeric values,
    based on pruning threshold and entropy of significant values.
    """
    # Create binary map based on pruning threshold
    binary_map = [1 if abs(val) >= pruning_threshold else 0 for val in string]
    n = len(binary_map)
    m = sum(binary_map)

    if m == 0:
        entropy_new_formula = 0
    else:
        # Extract values above pruning threshold
        non_zero_weights = [val for val in string if abs(val) >= pruning_threshold]
        frequencies = Counter(non_zero_weights)
        total = len(non_zero_weights)

        # Entropy of non-zero elements
        entropy_non_zeros = 0
        for freq in frequencies.values():
            probability = freq / total
            entropy_non_zeros -= probability * math.log2(probability)

        # Final entropy formula combining two terms
        entropy_new_formula = m * (2 + math.ceil(math.log2(n / m))) + entropy_non_zeros

    return entropy_new_formula

def quantize_weights_center(weights, v, v_centers):
    """
    Function for weight quantization using central value.
    Quantizes weights based on the central value of the buckets in vector v.
    """
    indices = torch.bucketize(weights, v, right=False) - 1
    indices = torch.clamp(indices, min=0, max=len(v_centers) - 1)  # Ensure indices are valid
    return v_centers[indices]

def encode(symb2freq):
    """Huffman encode the given dict mapping symbols to weights"""
    heap = [[wt, [sym, ""]] for sym, wt in symb2freq.items()]
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def decode(encoded_text, huffman_code):
    """
    Decodes the given encoded text using the provided Huffman code.
    """
    code_to_symbol = {code: symbol for symbol, code in huffman_code}
    decoded_text = []
    temp_code = ""
    for bit in encoded_text:
        temp_code += bit
        if temp_code in code_to_symbol:
            decoded_text.append(code_to_symbol[temp_code])
            temp_code = ""
    return decoded_text

def encode_arithmetic(symb2freq):
    """
    Performs arithmetic encoding for a dictionary of symbols with their respective frequencies.
    """
    total_freq = sum(symb2freq.values())  # Compute total frequency
    prob_intervals = {}  # Dictionary to store probability intervals
    low = Decimal(0.0)
    
    for sym, freq in sorted(symb2freq.items()):  # Iterate through symbols in sorted order
        prob_intervals[sym] = (low, low + Decimal(freq) / Decimal(total_freq))  # Assign probability range
        low = prob_intervals[sym][1]  # Update lower bound for next symbol
    
    return prob_intervals  # Return dictionary of intervals

def encode_arithmetic_text(symbol_list, prob_intervals):
    """
    Encodes a list of symbols using arithmetic encoding.
    """
    low = Decimal(0.0)
    high = Decimal(1.0)
    
    for sym in symbol_list:  # Process each symbol in the list
        range_ = high - low  # Compute current range
        high = low + range_ * prob_intervals[sym][1]  # Update upper bound
        low = low + range_ * prob_intervals[sym][0]  # Update lower bound
    
    R = high - low  # Final range after encoding
    dimension = - R.ln() / Decimal(2).ln()  # Compute encoding length in bits
    return (low + high) / 2, dimension  # Return the midpoint representation and bit size

def decode_arithmetic(encoded_value, symb2freq, length_of_symbols):
    """
    Decodes an arithmetic-encoded value given the symbol frequency dictionary.
    """
    prob_intervals = encode_arithmetic(symb2freq)  # Recompute probability intervals
    decoded_symbols = []
    
    for _ in range(length_of_symbols):  # Decode for the original length of symbols
        for sym, (low, high) in prob_intervals.items():  # Iterate through interval mappings
            if low <= encoded_value < high:  # Check if value falls in symbol range
                decoded_symbols.append(sym)  # Append decoded symbol
                range_ = high - low  # Compute new range
                encoded_value = (encoded_value - low) / range_  # Normalize value
                break
        else:
            raise ValueError(f"Unable to decode symbol. encoded_value: {encoded_value}")
    
    return decoded_symbols  # Return decoded symbol list

# Compresses with gzip-9
def compress_gzip(data):
    return gzip.compress(data, compresslevel=9)

# Compresses with zstd-22
def compress_zstd(data):
    cctx = zstd.ZstdCompressor(level=22)  # Creates a compressor object with level 22
    return cctx.compress(data)

# Decompresses with gzip-9
def decompress_gzip(data):
    decompressed = gzip.decompress(data)
    return list(struct.unpack(f'{len(decompressed) // 4}f', decompressed))  # Casts to float

# Decompresses with zstd-22
def decompress_zstd(data):
    dctx = zstd.ZstdDecompressor()  # Creates a compressor object
    decompressed = dctx.decompress(data)
    return list(struct.unpack(f'{len(decompressed) // 4}f', decompressed))  # Casts to float

def compare_lists(list1, list2, tollerance=1e-4):
    if len(list1) != len(list2):
        return False
    return all(abs(a - b) <= tollerance for a, b in zip(list1, list2))