# Import necessary libraries
import torch
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
from tokenizers import normalizers
from sklearn.metrics.pairwise import cosine_similarity

# Define function to load data and model
def load_data_and_model():
    # Load tokenizer and model for DistilBERT
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    # Load dataset containing proverbs and their meanings
    dataset = pd.read_csv("proverbs_and_meanings_data.csv", encoding='latin1')  # Bypass UTF-8 warning
    dataset = dataset.rename(columns={'Meaning ': 'Meaning'})  # Rename column for consistency

    # Extract meanings and proverbs from dataset
    meanings = dataset["Meaning"]
    proverbs = dataset["Proverbs"]

    # Normalize proverbs and meanings using tokenizers
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
    normalized_meanings = meanings.apply(tokenizer.normalizer.normalize_str).tolist()
    normalized_proverbs = proverbs.apply(tokenizer.normalizer.normalize_str).tolist()

    return tokenizer, model, normalized_meanings, normalized_proverbs

# Define function to compute embeddings for proverbs
def compute_proverb_embeddings(tokenizer, model, normalized_meanings):
    # Tokenize proverbs
    encoded_input = tokenizer(normalized_meanings, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform mean pooling to get embeddings
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    attention_mask = encoded_input['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    mean_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    return mean_embeddings

# Define function to save embeddings to a file
def save_embeddings(embeddings, filename):
    np.save(filename, embeddings.numpy())

# Define function to load embeddings from a file
def load_embeddings(filename):
    return torch.tensor(np.load(filename))

# Define function to compute similar proverbs based on user input
def compute_proverbs_similarity(user_input, tokenizer, model, loaded_embeddings, proverbs, meanings, k=5):
    # Normalize user input
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
    normalized_user_input = tokenizer.normalizer.normalize_str(user_input)
    user_encoded_input = tokenizer(normalized_user_input, padding=True, truncation=True, return_tensors='pt')

    # Compute embeddings for user input
    with torch.no_grad():
        model_output = model(**user_encoded_input)
    user_embeddings = compute_proverb_embeddings(tokenizer, model, [user_input])

    # Compute cosine similarities between user embeddings and loaded embeddings
    similarities = cosine_similarity(user_embeddings, loaded_embeddings)

    # Find nearest neighbors
    top_indices = similarities.argsort()[0][-k:][::-1]

    similar_proverbs = []

    for idx in top_indices:
        similar_proverbs.append(f'Proverbs: {proverbs[idx]}')

    return similar_proverbs

# Define function to compute similar meanings based on user input
def compute_meanings_similarity(user_input, tokenizer, model, loaded_embeddings, proverbs, meanings, k=5):
    # Normalize user input
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
    normalized_user_input = tokenizer.normalizer.normalize_str(user_input)
    user_encoded_input = tokenizer(normalized_user_input, padding=True, truncation=True, return_tensors='pt')

    # Compute embeddings for user input
    with torch.no_grad():
        model_output = model(**user_encoded_input)
    user_embeddings = compute_proverb_embeddings(tokenizer, model, [user_input])

    # Compute cosine similarities between user embeddings and loaded embeddings
    similarities = cosine_similarity(user_embeddings, loaded_embeddings)

    # Find nearest neighbors
    top_indices = similarities.argsort()[0][-k:][::-1]

    similar_meanings = []

    for idx in top_indices:
        similar_meanings.append(f"Meaning: {meanings[idx]}")

    return similar_meanings

if __name__ == '__main__':
    # Load data and model
    tokenizer, model, normalized_meanings, normalized_proverbs = load_data_and_model()

    # Compute embeddings for proverbs
    proverb_embeddings = compute_proverb_embeddings(tokenizer, model, normalized_meanings)

    # Save embeddings to file
    save_embeddings(proverb_embeddings, "proverb_embeddings.npy")

    # Load embeddings from file
    loaded_embeddings = load_embeddings("proverb_embeddings.npy")

    # Get user input
    user_input = input("Enter here: ")

    # Compute similar proverbs and meanings based on user input
    similar_proverbs = compute_proverbs_similarity(user_input, tokenizer, model, loaded_embeddings, normalized_proverbs, normalized_meanings)
    similar_meanings = compute_meanings_similarity(user_input, tokenizer, model, loaded_embeddings, normalized_proverbs, normalized_meanings)

    # Print similar proverbs and meanings
    for proverb in similar_proverbs:
        print(proverb)
    for meaning in similar_meanings:
        print(meaning)
