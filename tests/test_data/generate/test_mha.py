import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def print_tensor(name, tensor):
    print(f"{name}:\n{tensor.detach().numpy()}\n")

def save_tensor(name, tensor):
    np.savetxt(f'{name}.txt', tensor.detach().numpy(), fmt='%.9f')  # Changed to space-separated

# Set parameters
d_model = 8
num_heads = 2
seq_length = 10

# Create PyTorch MultiheadAttention
mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

# Generate random input
input = torch.randn(1, seq_length, d_model)  # Add batch dimension
print_tensor("Input", input.squeeze(0))  # Remove batch dimension for printing
save_tensor("mha_input", input.squeeze(0))

# Get the weights and biases
in_proj_weight = mha.in_proj_weight
in_proj_bias = mha.in_proj_bias
out_proj_weight = mha.out_proj.weight
out_proj_bias = mha.out_proj.bias

print_tensor("in_proj_weight", in_proj_weight)
print_tensor("in_proj_bias", in_proj_bias)
print_tensor("out_proj_weight", out_proj_weight)
print_tensor("out_proj_bias", out_proj_bias)

# Save weights and biases
save_tensor("mha_in_proj_weight", in_proj_weight)
save_tensor("mha_in_proj_bias", in_proj_bias)
save_tensor("mha_out_proj_weight", out_proj_weight)
save_tensor("mha_out_proj_bias", out_proj_bias)

# Compute Q, K, V
def compute_qkv(x, w, b):
    return F.linear(x, w, b)

qkv = compute_qkv(input, in_proj_weight, in_proj_bias)
q, k, v = qkv.chunk(3, dim=-1)

# Remove batch dimension for consistency with C++ implementation
q = q.squeeze(0)
k = k.squeeze(0)
v = v.squeeze(0)

print_tensor("Q", q.transpose(0, 1))
print_tensor("K", k.transpose(0, 1))
print_tensor("V", v.transpose(0, 1))

# Save Q, K, V
save_tensor("mha_q", q)
save_tensor("mha_k", k)
save_tensor("mha_v", v)

# Forward pass using PyTorch's MultiheadAttention
output_pytorch, _ = mha(input, input, input)
output_pytorch = output_pytorch.squeeze(0)  # Remove batch dimension

print_tensor("PyTorch Output", output_pytorch)
save_tensor("mha_output", output_pytorch)

# Manual computation using Q, K, V
def manual_attention(q, k, v, num_heads, out_proj_weight, out_proj_bias):
    d_k = q.size(-1) // num_heads
    q = q.view(seq_length, num_heads, d_k)
    k = k.view(seq_length, num_heads, d_k)
    v = v.view(seq_length, num_heads, d_k)

    # Transpose for attention dot product: (seq_length, num_heads, d_k) -> (num_heads, seq_length, d_k)
    q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)

    # Compute attention
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)

    attn_weights = F.softmax(attn_weights, dim=-1)

    attn_output = torch.matmul(attn_weights, v)

    # Transpose and reshape: (num_heads, seq_length, d_k) -> (seq_length, d_model)
    attn_output = attn_output.transpose(0, 1).contiguous().view(seq_length, d_model)
    print(out_proj_weight)

    # Apply output projection
    output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    return output

output_manual = manual_attention(q, k, v, num_heads, out_proj_weight, out_proj_bias)
print_tensor("Manual Output", output_manual)

# Compare PyTorch and manual outputs
diff = torch.abs(output_pytorch - output_manual)
print_tensor("Difference", diff)

if torch.allclose(output_pytorch, output_manual, atol=1e-5):
    print("Manual computation matches PyTorch output!")
else:
    print("Manual computation does not match PyTorch output.")
    print(f"Max difference: {diff.max().item()}")
    print(f"Mean difference: {diff.mean().item()}")

print("Test data generated and saved.")