import csv
import pandas as pd

data=pd.read_csv("english_assertions.csv",header=None,delim_whitespace=True)

data.columns=["relation","start_node","end_node"]
filtered_data = data[data.iloc[:, 1] == 'car']

filtered_data_2 = data[data.iloc[:, 2] == 'car']
filtered_data.to_csv("english_assertions.csv")
combined_df = pd.concat([filtered_data, filtered_data_2], axis=0)
combined_df.to_csv("test_car.csv")