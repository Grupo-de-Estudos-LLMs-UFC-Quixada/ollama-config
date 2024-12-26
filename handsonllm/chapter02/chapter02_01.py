from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct",
                                             device_map="cuda:0",
                                             torch_dtype="auto", 
                                             trust_remote_code=True,
                                             attn_implementation="flash_attention_2")

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened. <|assistant|>"

inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt")
attention_mask = (inputs['input_ids'] != tokenizer.pad_token_id).long()
input_ids = inputs.input_ids.to("cuda:0")
attention_mask = attention_mask.to("cuda:0")

generation_output = model.generate(
    input_ids = input_ids,
    attention_mask=attention_mask,
    max_new_tokens=20
)

print(tokenizer.decode(generation_output[0]))

print(input_ids)

print(generation_output)

print(attention_mask)

for id in input_ids[0]:
    print(id, tokenizer.decode([id]))

