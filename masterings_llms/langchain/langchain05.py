import spacy
import spacy.cli
import numpy as np
import time

# Load the spaCy model
# spacy.cli.download("en_core_web_md")
nlp = spacy.load("en_core_web_md")

# Function to get the embedding and similarity
def calculate_similarity(text1, text2):
    # Convert texts to vectors
    vector1 = nlp(text1).vector
    vector2 = nlp(text2).vector

    # Calculate cosine similarity
    #similarity = vector1 @ vector2 / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    similarity = nlp(text1).similarity(nlp(text2))

    return similarity

# Input texts
text1 = "The quick brown fox jumps over the lazy dog."
text2 = "A quick brown dog leaps over a lazy fox."

start = time.time()

# Calculate similarity between texts
for i in range(100):
    similarity_score = calculate_similarity(text1, text2)
end = time.time()
print(f"Time taken: {end - start} seconds")
print(f"Similarity score: {similarity_score:.4f}")