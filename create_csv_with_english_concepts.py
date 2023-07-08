import csv
import pandas as pd

data=pd.read_csv("conceptnet-assertions.csv",header=None,delimeter="\t")

data.columns=["edge","start_node","end_node","other"]
data = pd.read_csv('conceptnet-assertions.csv', delimiter='\t', header=None)

# Filter rows based on the second column
filtered_data = data[data.iloc[:, 1].str.split('/').str[1] == 'en']

filtered_data = filtered_data[filtered_data.iloc[:, 2].str.split('/').str[1] == 'en']
filtered_data.to_csv("english_assertions.csv")