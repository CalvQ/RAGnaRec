"""
This file creates a set of files containing processed data from the Yelp Dataset.
Only needs to be run once to create all files. 

Files produced:
    clean_reviews.pkl       :   list of cleaned tokens per document
    clean_review_corpus.pkl :   bag-of-words list per document
"""

import pandas as pd
import numpy as np
import pickle as pkl
import tqdm

from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.corpora import Dictionary

# Load Yelp review data
print("Loading Yelp Data")
splits = {'train': 'yelp_review_full/train-00000-of-00001.parquet',
          'test': 'yelp_review_full/test-00000-of-00001.parquet'}
df_train = pd.read_parquet(
    "hf://datasets/Yelp/yelp_review_full/" + splits["train"])
print("Train Data Loaded")
df_test = pd.read_parquet(
    "hf://datasets/Yelp/yelp_review_full/" + splits["test"])
print("Test Data Loaded")

# Combine train/test frames
df_combined = pd.concat([df_train, df_test], axis=0, ignore_index=True)

# initialize tokenizer and lemmatizer
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()

# process each review
reviews = df_combined["text"].tolist()
for idx in tqdm.tqdm(range(len(reviews)), leave=False, desc="Review Tokenization"):
    # reviews[idx] = reviews[idx].lower()  # Convert to lowercase.
    # reviews[idx] = tokenizer.tokenize(reviews[idx])  # Split into words.
    review_sentence = reviews[idx].lower()  # Convert to lowercase
    review_tokens = tokenizer.tokenize(review_sentence)  # Split into words
    review_tokens_cleaned = [token for token in review_tokens if (
        not token.isnumeric() and len(token) > 1)]
    lemmatized_tokens = [lemmatizer.lemmatize(
        token) for token in review_tokens_cleaned]
    reviews[idx] = lemmatized_tokens
    # TODO: consider different lemmatizer/doing less preprocessing
    # concerned that it over-normalizes, but no lemmatization likely too noisy

print("Writing reviews to file")

# Save cleaned reviews in file
with open("clean_reviews.pkl", "wb") as clean_review_file:
    pkl.dump(reviews, clean_review_file)

print("Finished writing")

# Create Dictionary for vocabulary
dictionary = Dictionary(reviews)
dictionary.filter_extremes(no_below=20, no_above=0.5)

# Create bag-of-words for each review
corpus = []
for idx in tqdm.tqdm(range(len(reviews)), leave=False, desc="Corpus Creation"):
    corpus.append(dictionary.doc2bow(reviews[idx]))
    # [dictionary.doc2bow(doc) for doc in reviews]

print("Writing corpus to file")

# Save corpus in file
with open("clean_review_corpus.pkl", "wb") as clean_review_corpus:
    pkl.dump(corpus, clean_review_corpus)

print("Finished")
