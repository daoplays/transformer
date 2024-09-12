from transformers import GPT2Tokenizer, GPT2Model
import torch

def read_weights_h5(file_path):
     with h5py.File(file_path, 'r') as f:
         # Print the structure of the file
         def print_structure(name, obj):
             print(name, obj)
         f.visititems(print_structure)
 
         # Example of reading a specific weight
         wte = np.array(f['/transformer/tfgp_t2lm_head_model/transformer/wte/weight:0'])
         print("Token embedding shape:", wte.shape)


# Usage
model_name = "/home/ltl/Documents/machine_learning/gpt2/"
input_string = "GPT2 is a model developed by OpenAI"


# Compare with Hugging Face implementation
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)

tokens = tokenizer.tokenize(input_string)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
input_ids = tokenizer.encode(input_string, return_tensors="pt")

# Compare results
print(f"Tokens: {tokens}")
print(f"Token IDs: {token_ids}")


token_embeddings = model.wte.weight[input_ids]

# Get the position embeddings
position_ids = torch.arange(0, input_ids.shape[-1]).unsqueeze(0)
position_embeddings = model.wpe(position_ids)

# Print shapes and a few values
print(f"Input IDs shape: {input_ids.shape}")
print(f"Token embeddings shape: {token_embeddings.shape}")
print(f"Position embeddings shape: {position_embeddings.shape}")

print("\nFirst few values of token embeddings:")
print(token_embeddings[0, 0, :10])

print("\nFirst few values of position embeddings:")
print(position_embeddings[0, 0, :10])

# If you want to get the combined embeddings (token + position)
combined_embeddings = token_embeddings + position_embeddings
print(f"\nCombined embeddings shape: {combined_embeddings.shape}")
print("First few values of combined embeddings:")
print(combined_embeddings[0, 0, :10])