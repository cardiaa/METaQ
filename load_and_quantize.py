import torch  
import os  
import numpy as np
import time
import struct
import torch.nn as nn 
from torchvision import datasets, transforms  
from torch.utils.data import DataLoader  
from collections import defaultdict
from decimal import getcontext
from utils.networks import LeNet5  
from utils.trainer import test_accuracy  
from utils.quantize_and_compress import compute_entropy, quantize_weights_center, encode, decode
from utils.quantize_and_compress import encode_arithmetic, encode_arithmetic_text, decode_arithmetic
from utils.quantize_and_compress import compare_lists, compress_gzip, compress_zstd, decompress_gzip, decompress_zstd
from utils.networks import json_to_model  

# Main entry point of the script
if __name__ == "__main__":
    # Initialize the computing device: use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the MNIST test dataset with transformation (convert images to tensors)
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Create a DataLoader for the test set with batch size 1000
    testloader = DataLoader(testset, batch_size=1000, shuffle=False)

    # Load the network architecture from a JSON file and move it to the selected device
    json_file = "utils/lenet5.json"
    model = json_to_model(json_file).to(device)

    # Compute the file size of the architecture JSON in bits
    architecture_size_bytes = os.path.getsize(json_file)
    architecture_size_bits = architecture_size_bytes * 8

    # Output JSON file information
    print(f"\nNetwork architecture saved in: {json_file}")
    print(f"Network architecture size: {architecture_size_bits} bits\n")

    # Define model weight file and path
    model_name = input("SELECT THE NAME OF A MODEL IN THE FOLDER "
                       "BestModelBeforeQuantization TO QUANTIZE AND COMPRESS.\n"
                       "model_name: ")
    model_path = f"BestModelsBeforeQuantization/{model_name}.pth"

    # Define weight quantization parameters
    w0 = -0.11  # Base weight value
    r = 1.114  # Range parameter
    min_w = w0 - r  
    max_w = w0 + r  

    # Load the pre-trained model weights
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)  # Load weights into the model

    # Flatten and concatenate all model parameters into a single tensor
    w_saved = torch.cat([param.data.view(-1) for param in model.parameters()])

    # Set model to evaluation mode (disable dropout, batch norm updates, etc.)
    model.eval()

    # Count the number of unique weight values in the model
    num_unique_weights_saved = torch.unique(w_saved).numel()
    
    # Compute and display the original model accuracy on the test set
    original_accuracy = test_accuracy(model, testloader, device)
    print(f"Number of unique weights: {num_unique_weights_saved}")
    print(f"Model accuracy: {original_accuracy:.2f}%")

    # Encode weight values into a list
    encoded_list = []
    for elem in w_saved:
        loaded_element = float(elem)
        if loaded_element == -0.0:
            loaded_element = 0.0  # Normalize -0.0 to 0.0 to consider them as a unique element
        encoded_list.append(loaded_element)

    # Compute the entropy of the encoded weight values
    entropy = round(compute_entropy(encoded_list)) + 1
    print(f"Model entropy: {entropy} bits\n")

    QuantAcc = []
    QuantEntr = []

    print("Select a range [c1, c2] for C (pay attention: the range must contain at least 10 values for C).")
    c1 = int(input("c1: "))
    c2 = int(input("c2: "))
    
    for C in range(c1, c2 + 1):
        # Construct vector v
        v = torch.linspace(min_w, max_w - (max_w - min_w)/C, steps=C)

        # Compute central values of the buckets
        v_centers = (v[:-1] + v[1:]) / 2
        v_centers = torch.cat([v_centers, v[-1:]])  # Add final value to handle the last bucket

        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)  # Adjust device accordingly
        model.load_state_dict(state_dict)

        # Extract model weights
        w_saved = torch.cat([param.data.view(-1) for param in model.parameters()])

        # Quantize weights using central values
        w_quantized = quantize_weights_center(w_saved, v, v_centers)

        # Replace quantized weights in the model
        start_idx = 0
        for param in model.parameters():
            numel = param.data.numel()
            param.data = w_quantized[start_idx:start_idx + numel].view(param.data.size())
            start_idx += numel

        # Evaluate quantized model
        model.eval()
        num_unique_weights_quantized = torch.unique(w_quantized).numel()
        quantized_accuracy = test_accuracy(model, testloader, device)

        # Compute entropy of the quantized string
        encoded_list = [float(elem) if float(elem) != -0.0 else 0.0 for elem in w_quantized]
        entropy = round(compute_entropy(encoded_list)) + 1
        
        QuantAcc.append(quantized_accuracy)
        QuantEntr.append(entropy)
        
        print(f"C = {C} analyzed")
        
    # Print results for the best 10 models
    sorted_indices = np.argsort(QuantAcc)
    for i in range(1, 10):
        print(f"Quantization at C={sorted_indices[-i] + c1}, Accuracy:{QuantAcc[sorted_indices[-i]]}, Entropy:{QuantEntr[sorted_indices[-i]]}")

    C = int(input("\nSelect the size of quantization.\nC: "))

    # Construct vector v
    v = torch.linspace(min_w, max_w - (max_w - min_w)/C, steps=C)

    # Compute central values of the buckets
    v_centers = (v[:-1] + v[1:]) / 2
    v_centers = torch.cat([v_centers, v[-1:]])  # Add final value to handle the last bucket

    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)  # Adjust device accordingly
    model.load_state_dict(state_dict)

    # Extract model weights
    w_saved = torch.cat([param.data.view(-1) for param in model.parameters()])

    # Quantize weights using central values
    w_quantized = quantize_weights_center(w_saved, v, v_centers)

    # Replace quantized weights in the model
    start_idx = 0
    for param in model.parameters():
        numel = param.data.numel()
        param.data = w_quantized[start_idx:start_idx + numel].view(param.data.size())
        start_idx += numel

    # Evaluate quantized model
    model.eval()
    num_unique_weights_quantized = torch.unique(w_quantized).numel()
    quantized_accuracy = test_accuracy(model, testloader, device)

    print(f"\nNumber of unique weights in the quantized model: {num_unique_weights_quantized}")
    print(f"Quantized model accuracy: {quantized_accuracy:.2f}%")

    # Compute entropy of the quantized string
    encoded_list = [float(elem) if float(elem) != -0.0 else 0.0 for elem in w_quantized]
    entropy = round(compute_entropy(encoded_list)) + 1
    print(f"Entropy of the string multiplied by its length: {entropy}")

    model_path = "BestModelsAfterQuantization/" + model_name + "_quantized.pth"
    torch.save(model.state_dict(), model_path)
    print("\nModel quantized saved!")
    
    compression_algo = input("Select one of the following compression algorithm (it is recommended to use Z):\n\n"
                             "H: for Huffman coding\n"
                             "A: for Arithmetic coding\n"
                             "G: for gzip-9\n"
                             "Z: for zstd-22\n")
    
    # Apply Huffman encoding
    if(compression_algo.upper() == 'H'):
        print("... Start Huffman Coding ...")
        symb2freq = defaultdict(int)
        for sym in encoded_list:
            symb2freq[sym] += 1
        huff = encode(symb2freq)
        encoded_huffman = ''.join([dict(huff)[sym] for sym in encoded_list])
        decoded_list = decode(encoded_huffman, huff)

        if encoded_list == decoded_list:
            print("\nEncoding successful!")
        else:
            print("Error during encoding!")

        print("Compressed size in bits:", len(encoded_huffman))
        print("Uncompressed size in bits:", len(encoded_list) * 32)
        print("\nCompression ratio:", 100 * round(len(encoded_huffman) / (len(encoded_list) * 32), 4), "%")
    elif(compression_algo.upper() == 'A'):
        # Set higher precision for decimal operations
        getcontext().prec = round(len(encoded_list) * 1.4)

        # Create the frequency dictionary
        symb2freq = defaultdict(int)
        for sym in encoded_list:
            symb2freq[sym] += 1  # Count symbol occurrences

        # Perform arithmetic encoding
        start_time = time.time()
        print("...STARTING ARITHMETIC CODING...")
        prob_intervals = encode_arithmetic(symb2freq)  # Compute probability intervals
        encoded_value, dimension = encode_arithmetic_text(encoded_list, prob_intervals)  # Encode data
        final_time = time.time() - start_time
        print(f'Encoding time: {final_time:.2f} seconds')

        # Decode the encoded text
        print("...STARTING DECODING...")
        start_time = time.time()
        decoded_symbols = decode_arithmetic(encoded_value, symb2freq, len(encoded_list))  # Decode data
        final_time = time.time() - start_time
        print(f'Decoding time: {final_time:.2f} seconds')

        # Verify that encoding and decoding are correct
        if encoded_list == decoded_symbols:
            print("\nENCODING SUCCESSFUL!")
        else:
            print(f"Encoding error! Decoded: {decoded_symbols}")

        print("\nArithmetic Coding achieves", round(dimension) + 1, "bits.")

        # Compute original file size
        bits_per_float = 32  # Assuming 32-bit float representation
        dimensione_originale = len(encoded_list) * bits_per_float  # Total bit size

        # Compute compression ratio
        compression_ratio = (round(dimension) + 1) / dimensione_originale

        # Output compression ratio
        print(f"Compression ratio: {compression_ratio:.4%}")

        print("ARITHMETIC CODING REACHES n*H?", entropy == round(dimension) + 1)

    elif(compression_algo.upper() == 'G'):
        
        # Converts float list in byte
        input_bytes = b''.join(struct.pack('f', num) for num in encoded_list)
        
        # Compression
        gzip_compressed = compress_gzip(input_bytes)
        
        # Decompression
        gzip_decompressed = decompress_gzip(gzip_compressed)

        # Verifies
        if compare_lists(encoded_list, gzip_decompressed):
            print("\nENCODING SUCCESSFUL!")
        else:
            print(f"Encoding error! Decoded")

        # Calculates dimensions
        original_size_bits = len(input_bytes) * 8
        gzip_size = len(gzip_compressed) * 8

        # Compression ratio
        gzip_ratio = gzip_size / original_size_bits

        # Output delle dimensioni e del rapporto di compressione
        print(f"\n\nOriginal dimension: {original_size_bits} bits")
        print(f"Gzip-9 compressed dimension: {gzip_size} bits (Compression Ratio: {gzip_ratio:.2%})")

    elif(compression_algo.upper() == 'Z'):
        
        # Converts float list in byte
        input_bytes = b''.join(struct.pack('f', num) for num in encoded_list)
        
        # Compression
        zstd_compressed = compress_zstd(input_bytes)
        
        # Decompression
        zstd_decompressed = decompress_zstd(zstd_compressed)

        # Verifies
        if compare_lists(encoded_list, zstd_decompressed):
            print("\nENCODING SUCCESSFUL!")
        else:
            print(f"Encoding error! Decoded")

        # Calculates dimensions
        original_size_bits = len(input_bytes) * 8
        zstd_size = len(zstd_compressed) * 8

        # Compression ratio
        zstd_ratio = zstd_size / original_size_bits

        # Output delle dimensioni e del rapporto di compressione
        print(f"\n\nOriginal dimension: {original_size_bits} bits")
        print(f"Zstd-22 compressed dimension: {zstd_size} bits (Compression Ratio: {zstd_ratio:.2%})")