import torch
import torch.nn as nn
import numpy as np

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        print(attn_output)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

# Set up parameters
d_model = 512
nhead = 8
dim_feedforward = 2048
seq_length = 10
batch_size = 2

# Create the encoder layer
encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward)

# Set the random seed for reproducibility
torch.manual_seed(0)

# Generate a random input tensor
input_tensor = torch.randn(seq_length, batch_size, d_model)

# Set the model to evaluation mode (no dropout)
encoder_layer.eval()

# Ensure double precision
encoder_layer.double()
input_tensor = input_tensor.double()

# Forward pass
with torch.no_grad():
    output = encoder_layer(input_tensor)

# Save input tensor to file
np.savetxt('encoder_input.txt', input_tensor.numpy().reshape(-1, d_model), fmt='%.10f')

# Save output tensor to file
np.savetxt('encoder_output.txt', output.numpy().reshape(-1, d_model), fmt='%.10f')

print("Input shape:", input_tensor.shape)
print("Output shape:", output.shape)
print("Data saved to encoder_input.txt and encoder_output.txt")