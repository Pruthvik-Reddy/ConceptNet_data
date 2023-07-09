import pandas as pd

data=pd.read_excel("with_cosine_similarities.xlsx")

data["cos_sim_add"]=data["CosineSimilarity_1"]+data["CosineSimilarity_2"]
data.to_excel("cosine_similarities_analysis.xlsx")