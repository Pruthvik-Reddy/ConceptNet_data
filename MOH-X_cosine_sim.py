import pandas as pd
import csv
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


data=pd.read_excel("excel_files/MOH_X_2.xlsx")
with open("numberbatch-en-3.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
embeddings = {}
lines=lines[1:]
for line in lines:
    line = line.strip().split(" ")
    word = line[0].lower()
    word=word.split("/")
    flag=0
    if word[2]=="en":
        flag=1
    word=word[3]
    if flag:
        embedding = [float(value) for value in line[1:]]
        embeddings[word] = embedding
        #print(word)


with open("glove.6B.50d.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
embeddings2 = {}
for line in lines:
    parts = line.split()
    word = parts[0].lower()
    embedding= [float(value) for value in parts[1:]]
    embeddings2[word] = embedding


def calculate_cosine_similarity_1(row):
    word1 = row["arg2"]
    word2 = row["verb"]
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

def calculate_cosine_similarity_2(row):
    word1 = row["arg2"]
    word2 = row["verb"]
    word1=word1.lower()
    word2=word2.lower()
    if word1 == "" or word2 == "":
        return 0.0

    embedding1 = embeddings2.get(word1)
    embedding2 = embeddings2.get(word2)

    if embedding1 is None or embedding2 is None:
        return 0.0

    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity


data["second_pair_glove"]=data.apply(calculate_cosine_similarity_2, axis=1)
data["second_pair_numberbatch"]=data.apply(calculate_cosine_similarity_1, axis=1)

data.to_excel("excel_files/MOH_X_3.xlsx")
"""
avg_values = data.groupby('label')['second_pair_glove'].mean()
max_values = data.groupby('label')['second_pair_glove'].max()
min_values = data.groupby('label')['second_pair_glove'].min()
print("For Glove Embeddings")
print("Average values:\n", avg_values)
print("Highest values:\n", max_values)
print("Lowest values:\n", min_values)

avg_values = data.groupby('label')['second_pair_numberbatch'].mean()
max_values = data.groupby('label')['second_pair_numberbatch'].max()
min_values = data.groupby('label')['second_pair_numberbatch'].min()
print("For Numberbatch Embeddings")
print("Average values:\n", avg_values)
print("Highest values:\n", max_values)
print("Lowest values:\n", min_values)
"""
print("For Glove Embeddings :")
X=data["second_pair_glove"]
Y=data["label"]
X=np.reshape(X,[-1,1])
Y=np.reshape(Y,[-1,1])
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

y_train=np.ravel(y_train)
y_test=np.ravel(y_test)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("F1 score : ",f1)
print("Precision : ",precision)
print("Recall : ",recall)


print("For Numberbatch Embeddings :")
X=data["second_pair_numberbatch"]
Y=data["label"]
X=np.reshape(X,[-1,1])
Y=np.reshape(Y,[-1,1])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
y_train=np.ravel(y_train)
y_test=np.ravel(y_test)


model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("F1 score : ",f1)
print("Precision : ",precision)
print("Recall : ",recall)

