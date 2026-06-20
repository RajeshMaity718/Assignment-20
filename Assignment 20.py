import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==================================================
# DATASET INFORMATION
# ==================================================

# Example Dataset:
# TMDB Movie Recommendation Dataset

# Kaggle Link:
# https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

# Required Columns:
# title
# overview

# ==================================================
# PART 1 : DATA PREPROCESSING
# ==================================================

# ==================================================
# Task 1 : Load Dataset
# ==================================================

df = pd.read_csv("movies.csv")

print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# Text column used

text_column = "overview"

# ==================================================
# Task 2 : Text Preprocessing
# ==================================================

stop_words = [
    "the","a","an","is","are","was","were",
    "and","or","of","to","in","on","for",
    "with","at","by","from"
]

def clean_text(text):

    if pd.isna(text):
        return ""

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z\s]',
        '',
        text
    )

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["clean_text"] = (
    df[text_column]
    .fillna("")
    .apply(clean_text)
)

print("\nCleaned Text")

print(
    df[
        ["title","clean_text"]
    ].head()
)

# ==================================================
# PART 2 : TEXT VECTORIZATION
# ==================================================

# ==================================================
# Task 3 : TF-IDF
# ==================================================

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

tfidf_matrix = vectorizer.fit_transform(
    df["clean_text"]
)

print("\nTF-IDF Matrix Shape")

print(tfidf_matrix.shape)

# ==================================================
# Task 4 : Cosine Similarity
# ==================================================

similarity_matrix = cosine_similarity(
    tfidf_matrix
)

print("\nSimilarity Matrix Shape")

print(similarity_matrix.shape)

print("""

Why Cosine Similarity?

1. Measures angle between vectors.
2. Works well with TF-IDF vectors.
3. Independent of document length.
4. Most commonly used in recommendation systems.

""")

# ==================================================
# PART 3 : RECOMMENDATION LOGIC
# ==================================================

# ==================================================
# Task 5 : Recommendation Function
# ==================================================

def recommend(item_name, top_n=5):

    item_name = item_name.lower()

    movie_indices = (
        df[
            df["title"]
            .str.lower() == item_name
        ]
        .index
    )

    if len(movie_indices) == 0:

        return ["Movie Not Found"]

    idx = movie_indices[0]

    similarity_scores = list(
        enumerate(
            similarity_matrix[idx]
        )
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[1:top_n+1]

    recommendations = []

    for i in similarity_scores:

        recommendations.append(
            df.iloc[i[0]]["title"]
        )

    return recommendations

# ==================================================
# TESTING
# ==================================================

print("\nRecommendations For Avatar")

print(
    recommend(
        "Avatar",
        5
    )
)

print("\nRecommendations For Titanic")

print(
    recommend(
        "Titanic",
        5
    )
)

print("\nRecommendations For Batman")

print(
    recommend(
        "Batman",
        5
    )
)

# ==================================================
# SAVE MODEL ARTIFACTS (OPTIONAL)
# ==================================================

print("\nRecommendation Engine Ready")
