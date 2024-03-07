# Import necessary libraries
import torch
import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizer 
from transformers import DistilBertTokenizer, DistilBertModel
from proverbs_recommender import compute_proverbs_similarity, compute_meanings_similarity, load_embeddings, load_data_and_model
import streamlit as st
import random

# Load tokenizer and model for DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Load dataset containing proverbs and their meanings
dataset = pd.read_csv("proverbs_and_meanings_data.csv", encoding='latin1')  # Bypass UTF-8 warning
dataset = dataset.rename(columns={'Meaning ': 'Meaning'})  # Rename column for consistency
meanings = dataset["Meaning"]
proverbs = dataset["Proverbs"]

# Load precomputed embeddings for proverbs
loaded_embeddings = load_embeddings("proverb_embeddings.npy")

# Streamlit front-end

# Display header with a warning about program limitations
st.header("Learn African Proverbs")
st.subheader("Type a theme/mood. Watch the magic happen!")
st.warning("May reproduce incorrect results. More data is being added to improve performance.")

# Input field for user to enter a theme/mood
user_input = st.text_input("Enter a theme/mood here", placeholder="family, love, happiness, wisdom, stress")

# Store submitted proverbs in session state
if "proverb_list" not in st.session_state:
    st.session_state.proverb_list = []

# Button to submit user input and retrieve similar proverbs
if st.button("Submit"):
    similar_proverbs = compute_proverbs_similarity(user_input, tokenizer, model, loaded_embeddings, proverbs, meanings)
    st.session_state.proverb_list = list(similar_proverbs)
    st.write(random.choice(st.session_state.proverb_list))

# Button to display another randomly selected proverb
if st.button("See Another"):
    if st.session_state.proverb_list:
        random_proverb = random.choice(st.session_state.proverb_list)
        st.write(random_proverb)
        st.session_state.proverb_list.remove(random_proverb)
    else:
        st.write("No more proverbs to display")
