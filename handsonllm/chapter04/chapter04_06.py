from datasets import load_dataset
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset 
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm

def evaluate_performance(y_true, y_pred):
    """ Create and print the classification report """
    performance = classification_report(
        y_true, y_pred,
        target_names=["Negative Review", "Positive Review"],
        zero_division=1
    )
    print(performance)

# Load our model
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device="cuda:0"
)

# Load our data
data = load_dataset("rotten_tomatoes")

# Prepare our data
prompt = "Is the following sentence positive or negative? "
data = data.map(lambda example: {"t5": prompt + example['text']})

# Run inference
y_pred = []
for output in tqdm(pipe(KeyDataset(data["test"],  "t5")), total=len(data["test"])):
    text = output[0]["generated_text"]
    y_pred.append(0 if text == "negative" else 1)
    
# Evaluate performance
evaluate_performance(data["test"]["label"], y_pred)
