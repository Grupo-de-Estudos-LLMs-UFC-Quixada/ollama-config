from transformers import pipeline

# Path to our HF model
model_path = "cardiffnlp/twitter-roberta-base-sentiment"

# Load model into pipeline
pipe = pipeline(
    model = model_path,
    tokenizer = model_path,
    top_k=None,
    device="cuda:0"
)
