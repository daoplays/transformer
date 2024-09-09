import torch
import torch.nn as nn
import numpy as np

def save_tensor(name, tensor):
    np.savetxt(f'{name}.txt', tensor.detach().numpy(), fmt='%.17f') 


# Set random seed for reproducibility
torch.manual_seed(42)
torch.set_default_dtype(torch.float64)

# Hyperparameters
d_model = 512
nhead = 8
dim_feedforward = 2048
seq_length = 10

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        ff_output = self.ff(x)
        x = x + ff_output
        x = self.norm2(x)
        return x

# Create encoder layer
encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward)

# Generate random input
X = torch.randn(1, seq_length, d_model)

# Forward pass
with torch.no_grad():
    attn_output, _ = encoder_layer.self_attn(X, X, X)
    norm1_output = encoder_layer.norm1(X + attn_output)
    ff_output = encoder_layer.ff(norm1_output)
    final_output = encoder_layer.norm2(norm1_output + ff_output)

# Save input
save_tensor("encoder_input", X.squeeze(0))

# Save self-attention weights and biases
save_tensor("self_attn_in_proj_weight", encoder_layer.self_attn.in_proj_weight)
save_tensor("self_attn_in_proj_bias", encoder_layer.self_attn.in_proj_bias)
save_tensor("self_attn_out_proj_weight", encoder_layer.self_attn.out_proj.weight)
save_tensor("self_attn_out_proj_bias", encoder_layer.self_attn.out_proj.bias)

# Save layer norm weights and biases
save_tensor("norm1_weight", encoder_layer.norm1.weight)
save_tensor("norm1_bias", encoder_layer.norm1.bias)
save_tensor("norm2_weight", encoder_layer.norm2.weight)
save_tensor("norm2_bias", encoder_layer.norm2.bias)

# Save feed-forward weights and biases
save_tensor("ff_linear1_weight", encoder_layer.ff[0].weight)
save_tensor("ff_linear1_bias", encoder_layer.ff[0].bias)
save_tensor("ff_linear2_weight", encoder_layer.ff[2].weight)
save_tensor("ff_linear2_bias", encoder_layer.ff[2].bias)

# Save intermediate outputs
save_tensor("attn_output", attn_output.squeeze(0))
save_tensor("norm1_output", norm1_output.squeeze(0))
save_tensor("ff_output", ff_output.squeeze(0))
save_tensor("final_output", final_output.squeeze(0))

print("Encoder layer data generated and saved.")