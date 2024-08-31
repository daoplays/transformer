import torch
import torch.nn as nn
import numpy as np

class AttentionHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model, d_model, bias=True)
        self.key = nn.Linear(d_model, d_model, bias=True)
        self.value = nn.Linear(d_model, d_model, bias=True)
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        print(Q)
        print(K)
        print(V)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_model)

        print(scores)
        attention_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, V)

# Set parameters
d_model = 512
seq_length = 10
batch_size = 1

# Set random seed for reproducibility
torch.manual_seed(0)

# Create attention head
attn = AttentionHead(d_model)

# Generate random input
x = torch.randn(batch_size, seq_length, d_model)

# Forward pass
with torch.no_grad():
    output = attn(x)

# Save weights and biases
np.savetxt('query_weights.txt', attn.query.weight.detach().numpy(), fmt='%.10f')
np.savetxt('query_bias.txt', attn.query.bias.detach().numpy(), fmt='%.10f')
np.savetxt('key_weights.txt', attn.key.weight.detach().numpy(), fmt='%.10f')
np.savetxt('key_bias.txt', attn.key.bias.detach().numpy(), fmt='%.10f')
np.savetxt('value_weights.txt', attn.value.weight.detach().numpy(), fmt='%.10f')
np.savetxt('value_bias.txt', attn.value.bias.detach().numpy(), fmt='%.10f')

# Save input and output
np.savetxt('input.txt', x.detach().numpy().reshape(-1, d_model), fmt='%.10f')
np.savetxt('output.txt', output.detach().numpy().reshape(-1, d_model), fmt='%.10f')

print("Files saved successfully.")