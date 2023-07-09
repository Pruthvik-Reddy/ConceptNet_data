import pandas as pd
import csv
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

data=pd.read_excel("with_cosine_similarities.xlsx")
with open("glove.6B.50d.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
embeddings = {}
for line in lines:
    parts = line.split()
    word = parts[0].lower()
    embedding = [float(value) for value in parts[1:]]
    embeddings[word] = embedding

def calculate_cosine_similarity_1(row):
    word1 = row["arg1"]
    word2 = row["arg2"]
    word1=word1.lower()
    word2=word2.lower()
    if word1 == "" or word2 == "":
        return 0.0

    embedding1 = embeddings.get(word1)
    embedding2 = embeddings.get(word2)

    if embedding1 is None or embedding2 is None:
        return 0.0

    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity

data["cos_sim_add"]=data["CosineSimilarity_1"]+data["CosineSimilarity_2"]
data["original_pair"]=data.apply(calculate_cosine_similarity_1, axis=1)
data.to_excel("cosine_similarities_analysis.xlsx")