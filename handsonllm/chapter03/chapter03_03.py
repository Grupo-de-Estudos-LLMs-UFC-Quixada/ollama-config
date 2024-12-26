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

prompt = "Write a very long email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."

# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")

# Generate text using the model
generation_outputs = model.generate(
    input_ids,
    max_new_tokens=100,
    use_cache=False,
)
