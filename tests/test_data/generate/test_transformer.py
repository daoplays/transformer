import torch
import torch.nn as nn
import numpy as np

def save_tensor(name, tensor):
    np.savetxt(f'{name}.txt', tensor.detach().numpy(), fmt='%.17f')

def print_tensor(name, tensor):
    print(f"{name}:\n{tensor.detach().numpy()}\n")

# Set random seed for reproducibility
torch.manual_seed(42)
torch.set_default_dtype(torch.float64)

# Hyperparameters
d_model = 512
nhead = 8
dim_feedforward = 2048
num_layers = 3 
seq_length = 10
batch_size = 1
dropout = 0.0  # Set dropout to 0

# Create TransformerEncoder with 3 layers
encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

# Generate random input
X = torch.randn(batch_size, seq_length, d_model)

# Save input
save_tensor("transformer_input", X.squeeze(0))
print_tensor("Input", X)

# Forward pass
with torch.no_grad():
    output = transformer_encoder(X)

# Save output
save_tensor("transformer_output", output.squeeze(0))
print_tensor("Output", output)

# Save weights and biases for each layer
for layer_idx, layer in enumerate(transformer_encoder.layers):
    # Self-attention weights and biases
    save_tensor(f"layer_{layer_idx}_self_attn_in_proj_weight", layer.self_attn.in_proj_weight)
    save_tensor(f"layer_{layer_idx}_self_attn_in_proj_bias", layer.self_attn.in_proj_bias)
    save_tensor(f"layer_{layer_idx}_self_attn_out_proj_weight", layer.self_attn.out_proj.weight)
    save_tensor(f"layer_{layer_idx}_self_attn_out_proj_bias", layer.self_attn.out_proj.bias)

    # Layer norm weights and biases
    save_tensor(f"layer_{layer_idx}_norm1_weight", layer.norm1.weight)  # This is gamma
    save_tensor(f"layer_{layer_idx}_norm1_bias", layer.norm1.bias)      # This is beta
    save_tensor(f"layer_{layer_idx}_norm2_weight", layer.norm2.weight)  # This is gamma
    save_tensor(f"layer_{layer_idx}_norm2_bias", layer.norm2.bias)      # This is beta

    # Feedforward weights and biases
    save_tensor(f"layer_{layer_idx}_ff_linear1_weight", layer.linear1.weight)
    save_tensor(f"layer_{layer_idx}_ff_linear1_bias", layer.linear1.bias)
    save_tensor(f"layer_{layer_idx}_ff_linear2_weight", layer.linear2.weight)
    save_tensor(f"layer_{layer_idx}_ff_linear2_bias", layer.linear2.bias)

print("Test data generated and saved.")

# Print shapes for verification
print(f"Input shape: {X.shape}")
print(f"Output shape: {output.shape}")
print(f"self_attn_in_proj_weight shape: {layer.self_attn.in_proj_weight.shape}")
print(f"linear1_weight shape: {layer.linear1.weight.shape}")
print(f"linear2_weight shape: {layer.linear2.weight.shape}")

# Additional information
print(f"\nActivation function: {layer.activation.__class__.__name__}")
print(f"Layer normalization epsilon: {layer.norm1.eps}")