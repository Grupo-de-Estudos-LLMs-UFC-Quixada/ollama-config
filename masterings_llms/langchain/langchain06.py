import spacy

# Load the spaCy model for embeddings
nlp = spacy.load("en_core_web_md")

def split_text_into_sentences(text):
    # Use spaCy to process the text and extract sentences
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences

def calculate_similarity(reference_sentence, sentences):
    # Calculate similarity between the reference sentence and each sentence
    similarities = []
    for sentence in sentences:
        similarity = nlp(sentence).similarity(nlp(reference_sentence))
        similarities.append((sentence,similarity))
    return similarities

def reorder_sentences_by_similarity(similarities):
    """Reorder sentences base on similarity scores."""
    return sorted(similarities, key=lambda x: x[1], reverse=True)

long_text = """
Natural language processing (NLP) is a field of artificial intelligence in which machines learn to understand and interpret human language. NLP involves the use of natural language understanding (NLU) and natural language generation (NLG). NLU refers to the ability of a machine to understand the meaning behind words, phrases, and sentences. NLG refers to the ability of a machine to generate human-like text based on input data.
"""

sentences = split_text_into_sentences(long_text)
reference_sentence = "NLP is a field of artificial intelligence."
similarities = calculate_similarity(reference_sentence, sentences)
sorted_similarities = reorder_sentences_by_similarity(similarities)

# Display results
for sentence, similarity in sorted_similarities:
    print(f"Similarity: {similarity:.2f}, Sentence: '{sentence}'")