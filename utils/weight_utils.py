import torch  

def initialize_weights(model, min_w, max_w):
    """
    Initializes the weights of a given model using a uniform distribution.

    Args:
        model (torch.nn.Module): The neural network model whose weights need initialization.
        min_w (float): Minimum value for weight initialization.
        max_w (float): Maximum value for weight initialization.

    Returns:
        None
    """
    for param in model.parameters():
        torch.nn.init.uniform_(param, a=min_w, b=max_w)

def quantize_weights_center(weights, v, v_centers):
    """
    Quantizes the weights based on the central values of the quantization vector buckets.

    Args:
        weights (torch.Tensor): Tensor containing the model's weights.
        v (torch.Tensor): Quantization vector defining bucket boundaries.
        v_centers (torch.Tensor): Central values of the buckets in the quantization vector.

    Returns:
        torch.Tensor: Tensor of quantized weights.
    """
    indices = torch.bucketize(weights, v, right=False) - 1
    indices = torch.clamp(indices, min=0, max=len(v_centers) - 1)  # Ensure indices are within valid range
    return v_centers[indices]
