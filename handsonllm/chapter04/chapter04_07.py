import openai
from datasets import load_dataset
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

# Create client
client = openai.Client(api_key='your-api-key-here')

def chatgpt_generation(prompt, document, model="gpt-3.5-turbo-0125"):
    """ Generate an output based on a prompt and an input document """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt.replace("[DOCUMENT]", document)},
    ]
    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return chat_completion.choices[0].message.content

# Define a prompt template as a base
prompt = """Predict whether the following document is a positive or a negative movie review:
[DOCUMENT]

If it is a positive return 1 and if it is a negative return 0. Do not give any other answers.
"""

document = "unpretentious, charming, quirky, original"

chatgpt_generation(prompt, document)

predictions = [chatgpt_generation(prompt, doc) for doc in tqdm(data["test"]["text"])]

# Extract predictions
y_pred = [int (pred) for pred in predictions]

# Evaluate performance
evaluate_performance(data["test"]["label"], y_pred)
