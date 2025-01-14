from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired 
from bertopic.representation import MaximalMarginalRelevance
from bertopic.representation import TextGeneration
from copy import deepcopy
from transformers import pipeline   
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset from Hugging Face Datasets library
dataset = load_dataset('maartengr/arxiv_nlp')['train']

# Extract metadata
abstracts = dataset['Abstracts']
titles = dataset['Titles']

embedding_model = SentenceTransformer('thenlper/gte-small')  # Load pre-trained model
embeddings = embedding_model.encode(abstracts, show_progress_bar=True)  # Encode abstracts into embeddings

print(embeddings.shape)  # Output the shape of the embeddings array

# We reduce the input embeddings from 384 dimensions to 5 dimensions
umap_model = UMAP(n_components=5, min_dist=0.0, metric='cosine', random_state=42)
reduced_embeddings = umap_model.fit_transform(embeddings)

hdbscan_model = HDBSCAN(min_cluster_size=50, metric='euclidean', cluster_selection_method='eom').fit(reduced_embeddings)  # Fit the model to the reduced embeddings
clusters = hdbscan_model.labels_  # Get the labels of each point in the dataset

#print(len(set(clusters)))

# Print the first three documents in cluster 0
cluster = 0 
for index in np.where(clusters == cluster)[0][:3]:
    print(abstracts[index][:300] + "...\n")

# Reduce the 384 dimensions to 2 for visualization purposes
reduced_embeddings = UMAP (
    n_components=2, min_dist=0.0, metric='cosine', random_state=42
).fit_transform(embeddings)

# Create the dataframe
df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
df['title'] = titles
df['cluster'] = [str(c) for c in clusters]

# Select outliers and non-outliers (clusters)
to_plot = df[df.cluster != '-1']
outliers = df[df.cluster == '-1']

# Plot outliers and non-outliers separately
plt.scatter(outliers.x, outliers.y, alpha=0.05, s=2, c='grey')
plt.scatter(
    to_plot.x, to_plot.y, alpha=0.6, s=2, c=to_plot.cluster.astype(int), cmap="tab20b"
)
plt.axis('off')
plt.savefig("chapter05_01.png", dpi=300)

# Train our model with our previously defined model
topic_model = BERTopic (
    embedding_model=embedding_model, 
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True,
).fit(abstracts,embeddings)

#print(topic_model.get_topic_info())
#print(topic_model.get_topic(0))
#print(topic_model.find_topics("topic modeling"))
#print(topic_model.get_topic(22))

# Visualize the topics and their distribution in a word cloud
fig = topic_model.visualize_documents(
    titles,
    reduced_embeddings=reduced_embeddings,
    width=1200,
    hide_annotations=True
)

# Update fonts of legend for easier visualization
fig.update_layout(font=dict(size=16))

fig.write_html("chapter05_02.html")

# Save original representations
original_topics = deepcopy(topic_model.topic_representations_)

def topic_differences(model, original_topics, nr_topics=5):
    """Show the differences in topic representation between two models."""
    df = pd.DataFrame(columns=["Topic", "Original", "Updated"])
    for topic in range(nr_topics):
        # Extract top 5 words per topic per model
        og_words = " | ".join(list(zip(*original_topics[topic]))[0][:5])
        new_words = " | ".join(list(zip(*model.get_topic(topic)))[0][:5])
        df.loc[len(df)] = [topic, og_words, new_words]
    return df

# Update our topic representations using KeyBERTInspired
representation_model = KeyBERTInspired()
topic_model.update_topics(abstracts, representation_model=representation_model)

# Show topic differences
print(topic_differences(topic_model, original_topics))

# Update our topic representations using MaximalMarginalRelevance
representation_model = MaximalMarginalRelevance(diversity=0.2)
topic_model.update_topics(abstracts, representation_model=representation_model)

# Show topic differences
print(topic_differences(topic_model, original_topics))

prompt = """ I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.
[KEYWORDS]

Based on the documents and keywords, what is this topic about?"""

# Update our topic representations using Llama 3.2
#generator = pipeline("text2text-generation", model="meta-llama/Llama-3.2-3B")
#representation_model = TextGeneration(
#    generator, prompt=prompt, doc_length=50, tokenizer="whitespace"
#)
#topic_model.update_topics(abstracts, representation_model=representation_model) 

# Show topic differences
#print(topic_differences(topic_model, original_topics))

# Visualize topics and documents
fig = topic_model.visualize_document_datamap(
    titles,
    topics=list(range(20)),
    reduced_embeddings=reduced_embeddings,
    width=1200,
    label_font_size=11,
    label_wrap_width=20,
    use_medoids=True,
)

fig.write_html("topic_model.html")