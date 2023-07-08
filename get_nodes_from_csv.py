import csv
import pandas as pd

data=pd.read_csv("conceptnet-assertions.csv",header=None,delimeter="\t")

data.columns=["edge","start_node","end_node","other"]

