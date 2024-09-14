from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import torch
import torch.nn as nn
import math

torch.set_default_tensor_type(torch.DoubleTensor)

class GPT2ModelWithIntermediates(GPT2Model):
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None):
        intermediates = {}
        
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]

        if position_ids is None:
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        
        intermediates['token_embeddings'] = inputs_embeds.detach()
        intermediates['position_embeddings'] = position_embeds.detach()
        
        hidden_states = inputs_embeds + position_embeds
        intermediates['combined_embeddings'] = hidden_states.detach()

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        for i, block in enumerate(self.h):
            
            
            residual = hidden_states
            hidden_states = block.ln_1(hidden_states)
            intermediates[f'layer_{i}_after_ln_1'] = hidden_states.detach()

            attn_outputs = block.attn(
                hidden_states,
                layer_past=None,
                attention_mask=attention_mask,
                head_mask=None,
                use_cache=False,
                output_attentions=True
            )
            attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
            intermediates[f'layer_{i}_attention_output'] = attn_output.detach()

            hidden_states = attn_output + residual
            intermediates[f'layer_{i}_after_attention'] = hidden_states.detach()

            residual = hidden_states
            hidden_states = block.ln_2(hidden_states)
            intermediates[f'layer_{i}_after_ln_2'] = hidden_states.detach()

            hidden_states = block.mlp(hidden_states)
            intermediates[f'layer_{i}_after_mlp'] = hidden_states.detach()

            hidden_states = residual + hidden_states

            intermediates[f'layer_{i}_exit_weights'] = hidden_states.detach()


        hidden_states = self.ln_f(hidden_states)
        intermediates['final_layer_norm'] = hidden_states.detach()

        hidden_states = hidden_states.view(*output_shape)

        # Compute logits
        lm_logits = torch.matmul(hidden_states, self.wte.weight.transpose(-1, -2))
        intermediates['logits'] = lm_logits.detach()

        return hidden_states, intermediates

# Usage
model_name = "/home/lindley/Documents/machine_learning/gpt2/"
input_string = "GPT2 is a model developed by OpenAI"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
my_model = GPT2ModelWithIntermediates.from_pretrained(model_name).double()
model = GPT2LMHeadModel.from_pretrained(model_name).double()

input_ids = tokenizer.encode(input_string, return_tensors="pt").to(torch.long)


# Forward pass
outputs, intermediates = my_model(input_ids)

# Print shapes and a few values for each intermediate state
for name, tensor in intermediates.items():
    print(f"\n{name} shape: {tensor.shape}")
    print(f"First few values of {name}:")
    print(tensor[0, 0, :10])


# Forward pass
outputs = model(input_ids)
logits = outputs.logits

# Get the logits for the last token
last_token_logits = logits[0, -1, :]

# Find the index of the token with the highest probability
next_token_id = torch.argmax(last_token_logits).item()

# Decode the token ID to get the actual word
next_word = tokenizer.decode([next_token_id])

print(f"Input: {input_string}")
print(f"Most likely next word: {next_word}")

# If you want to see the top 5 most likely next words:
top_5_token_ids = torch.topk(last_token_logits, k=5).indices.tolist()
top_5_words = [tokenizer.decode([token_id]) for token_id in top_5_token_ids]

print("\nTop 5 most likely next words:")
for i, word in enumerate(top_5_words, 1):
    print(f"{i}. {word}")