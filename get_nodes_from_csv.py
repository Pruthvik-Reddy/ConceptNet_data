import csv
import pandas as pd

data=pd.read_csv("english_assertions.csv",header=None,delim_whitespace=True)

data.columns=["relation","start_node","end_node"]




csv_file = 'MOH-X_formatted_svo_cleaned.csv'
metaphoric_pairs=[]
literal_pairs=[]

with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if row[5] == "0" or row[5]==0:
            literal_pairs.append((row[0],row[2]))
        else:
            metaphoric_pairs.append((row[0],row[2]))


positive_metaphoric_pairs_count=0
negative_metaphoric_pairs_count=0


for i in range(len(metaphoric_pairs)):
  concept1=metaphoric_pairs[i][0]
  concept2=metaphoric_pairs[i][1]

  filtered_data = data[data.iloc[:, 1] == concept1]
  filtered_data_2 = data[data.iloc[:, 2] == concept1]
  combined_df = pd.concat([filtered_data, filtered_data_2], axis=0)
  end_exists = data["start_node"].str.contains(concept2, case=False) | data["end_node"].str.contains(concept2, case=False)
  if end_exists:
     positive_metaphoric_pairs_count+=1
     print("Positive : ", concept1,concept2)
  else:
    negative_metaphoric_pairs_count+=1
    print("Negative : ", concept1,concept2)
    

    

print("The fraction of pairs containing edges is : ",positive_metaphoric_pairs_count/len(metaphoric_pairs))
print("The fraction of pairs not containing edges is : ",negative_metaphoric_pairs_count/len(metaphoric_pairs))


positive_literal_pairs_count=0
negative_literal_pairs_count=0
for i in range(len(literal_pairs)):
  concept1=literal_pairs[i][0]
  concept2=literal_pairs[i][1]

  filtered_data = data[data.iloc[:, 1] == concept1]
  filtered_data_2 = data[data.iloc[:, 2] == concept1]
  combined_df = pd.concat([filtered_data, filtered_data_2], axis=0)
  end_exists = data["start_node"].str.contains(concept2, case=False) | data["end_node"].str.contains(concept2, case=False)
  if end_exists.any():
     positive_literal_pairs_count+=1
     print("Positive : ", concept1,concept2)
  else:
    negative_literal_pairs_count+=1
    print("Negative : ", concept1,concept2)
print("The fraction of pairs containing edges is : ",positive_literal_pairs_count/len(literal_pairs))
print("The fraction of pairs not containing edges is : ",negative_literal_pairs_count/len(literal_pairs))

