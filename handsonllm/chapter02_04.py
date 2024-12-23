from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Convert the text to text embeddings
vector = model.encode("Best movie ever!!")
print(vector.shape)
