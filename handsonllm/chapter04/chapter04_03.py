from datasets import load_dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset 
import numpy as np
from tqdm import tqdm
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

# Path to our HF model
model_path = "cardiffnlp/twitter-roberta-base-sentiment"

# Load model into pipeline
pipe = pipeline(
    model = model_path,
    tokenizer = model_path,
    top_k=None,
    device="cuda:0"
)

y_pred = []
for output in tqdm(pipe(KeyDataset(data["test"],  "text")), total=len(data["test"])):
    negative_score = output[0]["score"]
    positive_score = output[2]["score"]
    assignment = np.argmax([negative_score, positive_score])
    y_pred.append(assignment)

evaluate_performance(data["test"]["label"], y_pred)
