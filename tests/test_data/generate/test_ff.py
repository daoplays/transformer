import torch
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define dimensions
d_model = 512
d_ff = 2048
batch_size = 32
seq_length = 10

# Create a feed-forward layer
class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.linear2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))

# Instantiate the feed-forward layer
ff = FeedForward(d_model, d_ff)

# Generate random input
input_data = torch.randn(batch_size, seq_length, d_model)

# Get output
with torch.no_grad():
    output = ff(input_data)

# Also save as numpy arrays for easier C++ reading
np.savetxt('w1.txt', ff.linear1.weight.detach().numpy())
np.savetxt('b1.txt', ff.linear1.bias.detach().numpy())
np.savetxt('w2.txt', ff.linear2.weight.detach().numpy())
np.savetxt('b2.txt', ff.linear2.bias.detach().numpy())
np.savetxt('input.txt', input_data.detach().numpy().reshape(-1, d_model))
np.savetxt('output.txt', output.detach().numpy().reshape(-1, d_model))

print("Test data generated and saved.")