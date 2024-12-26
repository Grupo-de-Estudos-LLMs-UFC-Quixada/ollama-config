from transformers import AutoModel, AutoTokenizer

# Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

# Load a language model
model = AutoModel.from_pretrained("microsoft/deberta-base")

# Tokenize the sentence
tokens = tokenizer("Hello world", return_tensors="pt")

# Process the tokens
output = model(**tokens)[0]

print(output.shape)

for token in tokens['input_ids'][0]:
    print(tokenizer.decode([token]))