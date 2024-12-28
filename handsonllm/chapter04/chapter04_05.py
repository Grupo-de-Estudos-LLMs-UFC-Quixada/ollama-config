from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def evaluate_performance(y_true, y_pred):
    """ Create and print the classification report """
    performance = classification_report(
        y_true, y_pred,
        target_names=["Negative Review", "Positive Review"],
        zero_division=1
    )
    print(performance)

# Load our data
data = load_dataset("rotten_tomatoes")

# Load model
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Convert text to embeddings
label_embeddings = model.encode(["A description of a review that is negative", "A description of a review that is positive"])
train_embeddings = model.encode(data['train']['text'], show_progress_bar=True)
test_embeddings = model.encode(data['test']['text'], show_progress_bar=True)

# Find the best matching label for each document
sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
y_pred = np.argmax(sim_matrix, axis=1)

# Evaluate performance
evaluate_performance(data["test"]["label"], y_pred)
