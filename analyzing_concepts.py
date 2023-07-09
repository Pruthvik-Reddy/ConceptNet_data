import csv
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


data=pd.read_csv("original_metaphor_file.csv",delim_whitespace=True)

with open("glove.6B.50d.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
embeddings = {}
for line in lines:
    parts = line.split()
    word = parts[0]
    embedding = [float(value) for value in parts[1:]]
    embeddings[word] = embedding

def calculate_cosine_similarity_1(row):
    word1 = row["word_1"]
    word2 = row["word_2"]

    if word1 == "" or word2 == "":
        return 0.0

    embedding1 = embeddings.get(word1)
    embedding2 = embeddings.get(word2)

    if embedding1 is None or embedding2 is None:
        return 0.0

    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity

def calculate_cosine_similarity_2(row):
    word1 = row["word_2"]
    word2 = row["word_3"]

    if word1 == "" or word2 == "":
        return 0.0

    embedding1 = embeddings.get(word1)
    embedding2 = embeddings.get(word2)

    if embedding1 is None or embedding2 is None:
        return 0.0

    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity


data["CosineSimilarity_1"] = data.apply(calculate_cosine_similarity_1, axis=1)
data["CosineSimilarity_2"] = data.apply(calculate_cosine_similarity_2, axis=1)
data.to_excel("with_cosine_similarities.xlsx")