import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda:0",
    torch_dtype="auto",
    trust_remote_code=True,
)

# Create pipeline for text generation
generator = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    return_full_text=False,  # Only return the generated text, not the prompt
    max_new_tokens=50,  # Maximum number of tokens to generate
    do_sample=False,  # Disable sampling for deterministic results
)

prompt = "The capital of France is"

# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

# Get the output of the model before lm_head   
model_output = model.model(input_ids)

# Get the output of the lm head
lm_head_output = model.lm_head(model_output[0])

print(lm_head_output)
print(lm_head_output[0,-1])

token_id = lm_head_output[0,-1].argmax().item()
predicted_token = tokenizer.decode(token_id)

print("Predicted token:", predicted_token)