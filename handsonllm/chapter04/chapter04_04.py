from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

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
train_embeddings = model.encode(data['train']['text'], show_progress_bar=True)
test_embeddings = model.encode(data['test']['text'], show_progress_bar=True)

# Train a logistic regression on our train embeddings 
clf = LogisticRegression(random_state=42)
clf.fit(train_embeddings, data['train']['label'])

# Predict previously unseen instances
y_pred = clf.predict(test_embeddings)
evaluate_performance(data["test"]["label"], y_pred)
