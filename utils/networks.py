import torch.nn as nn
import torch
import json

# Define the LeNet-5 architecture
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        # First convolutional layer:
        # Input: 1 channel (e.g., grayscale image)
        # Output: 32 feature maps
        # Kernel size: 5x5, padding of 2 to maintain spatial dimensions
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        
        # First pooling layer:
        # Average Pooling with a 2x2 window, reduces spatial dimensions by half
        self.pool1 = nn.AvgPool2d(2)
        
        # Second convolutional layer:
        # Input: 32 feature maps
        # Output: 64 feature maps
        # Kernel size: 5x5, padding of 2 to maintain spatial dimensions
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        
        # Second pooling layer:
        # Average Pooling with a 2x2 window, reduces spatial dimensions by half
        self.pool2 = nn.AvgPool2d(2)
        
        # Fully connected layer:
        # Input: 64 feature maps of size 7x7 flattened to a vector
        # Output: 512 neurons
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        
        # Dropout layer for regularization with a probability of 0.5
        self.dropout = nn.Dropout(0.5)
        
        # Final fully connected layer:
        # Output: 10 classes (e.g., for digit classification)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Apply first convolution + ReLU activation
        x = torch.relu(self.conv1(x))
        
        # Apply first pooling
        x = self.pool1(x)
        
        # Apply second convolution + ReLU activation
        x = torch.relu(self.conv2(x))
        
        # Apply second pooling
        x = self.pool2(x)
        
        # Flatten the tensor for fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        
        # Apply first fully connected layer + ReLU activation
        x = torch.relu(self.fc1(x))
        
        # Apply dropout for regularization
        x = self.dropout(x)
        
        # Apply final fully connected layer (logits for 10 classes)
        x = self.fc2(x)
        
        return x
    
def model_to_json(model):
    """
    Function to convert a PyTorch model to a JSON representation
    """
    layers = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Consider only terminal modules (not containers)
            layer = {"name": name, "type": module.__class__.__name__}
            
            # Add specific parameters based on the layer type
            if isinstance(module, nn.Conv2d):
                layer.update({
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_size": module.kernel_size,
                    "stride": module.stride,
                    "padding": module.padding
                })
            elif isinstance(module, nn.AvgPool2d):
                layer.update({
                    "kernel_size": module.kernel_size,
                    "stride": module.stride
                })
            elif isinstance(module, nn.Linear):
                layer.update({
                    "in_features": module.in_features,
                    "out_features": module.out_features
                })
            layers.append(layer)
    return {"layers": layers}

def json_to_model(json_file):
    """
    Function to load a model from a JSON file
    """
    with open(json_file, "r") as f:
        model_data = json.load(f)
    
    # Manually create a LeNet-5 model instance
    model = LeNet5()  # Directly instantiate the LeNet-5 class
    layers = model.children()  # Retrieve model layers
    
    for layer_data, layer in zip(model_data["layers"], layers):
        layer_type = layer_data["type"]
        
        if isinstance(layer, nn.Conv2d):
            # Randomly initialize weights and biases for Conv2d layers
            layer.weight.data = torch.randn_like(layer.weight.data)
            layer.bias.data = torch.randn_like(layer.bias.data)
        
        elif isinstance(layer, nn.Linear):
            # Randomly initialize weights and biases for Linear layers
            layer.weight.data = torch.randn_like(layer.weight.data)
            layer.bias.data = torch.randn_like(layer.bias.data)
        
        # Additional logic can be added to update other layer types if necessary
        
    return model

