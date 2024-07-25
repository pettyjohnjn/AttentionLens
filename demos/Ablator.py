import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the tokenizer and model
model_name = "/grand/SuperBERT/aswathy/models/models--meta-llama--Meta-Llama-3-8B-Instruct"
#model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, config=config).to(device)

# Prepare input
input_text = "The cat sat on the mat and"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

inject_tensor = torch.randn(1, 1, 4096, device=device) * 1000
inject_tensor = None

inject_layer = 10
inject_head = 4

# Forward pass with tensor injection
outputs = model.generate(input_ids, max_new_tokens=20, num_return_sequences=1, inject_tensor=inject_tensor, inject_layer=inject_layer, inject_head=inject_head)
print(tokenizer.decode(outputs[0]))

# attention_layer = model.transformer.h[inject_layer].attn.head_out
# print(attention_layer[:,:,inject_head,:])
# print(attention_layer.shape)

# transformer_layers = model.model.layers
# attention_layer = transformer_layers[inject_layer].self_attn.head_out

# print(attention_layer.shape)

print(f'Unembed W: {model.lm_head.weight.shape}')
print(f'Unembed b: {model.lm_head.bias}')
print(f'n_head: {config.num_attention_heads}')
print(f'n_model: {config.hidden_size}')
print(f'd_vocab: {config.vocab_size}')


# print(model)
# print(config)