import requests
from sklearn.metrics.pairwise import cosine_similarity


def get_all_edges_of_a_concept(concept,language="en"):
    all_edges=[]
    uri='http://api.conceptnet.io/c/'+language+"/"+concept
    obj = requests.get(uri).json()


    edges=obj["edges"]
    for i in edges:
        all_edges.append(i)



    while "nextPage" in obj["view"]:
        nextpage=obj["view"]["nextPage"]
        next_uri="http://api.conceptnet.io"+nextpage
        obj=requests.get(next_uri).json()
        edges=obj["edges"]
        for i in edges:
            all_edges.append(i)
        if "nextPage" in obj["view"]:
            nextpage=obj["view"]["nextPage"]

    return all_edges


edges=get_all_edges_of_a_concept("graduation")
related_nodes=[]
for edge in edges:
  lang1=edge["start"]["@id"].split("/")[2]
  lang2=edge["end"]["@id"].split("/")[2]
  if lang1=="en" and lang2=="en" :
    if edge["start"]["label"]=="graduation":
      related_nodes.append(edge["end"]["label"])
      #print(edge["end"]["label"],edge["rel"]["label"])
    else:
      related_nodes.append(edge["start"]["label"])
      #print(edge["start"]["label"],edge["rel"]["label"])
stop_words = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 
              'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with']
stop_words=set(stop_words)

potential_words=[]
for i in range(len(related_nodes)):
   word=related_nodes[i]
   words=word.split(" ")
   for j in words:
      if j not in stop_words:
         potential_words.append(j)
   potential_words.append(word)

with open("glove.6B.50d.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
embeddings = {}
for line in lines:
    parts = line.split()
    word = parts[0].lower()
    embedding = [float(value) for value in parts[1:]]
    embeddings[word] = embedding

start_node_embedding = embeddings["graduation"]
end_node_embedding = embeddings["journey"]

concepts_cosine_sim = {}
for i in range(len(potential_words)):
    word = potential_words[i]
    words = word.split(" ")
    if len(words) == 1:
        if word in embeddings:
            concepts_cosine_sim[word] = cosine_similarity([embeddings[word]], [end_node_embedding])[0][0]
    
    else:
        embedding_sum = [0] * len(end_node_embedding)
        for j in range(len(words)):
            if words[j] in embeddings:
               #embedding_sum+=embeddings[words[j]]
                embedding_sum = [a + b for a, b in zip(embedding_sum, embeddings[words[j]])]
        embedding_sum = [val / len(words) for val in embedding_sum]
        concepts_cosine_sim[word] = cosine_similarity([embedding_sum], [end_node_embedding])[0][0]

sorted_items = sorted(concepts_cosine_sim.items(), key=lambda x: x[1], reverse=True)
top_k_items = sorted_items[:6]
print(top_k_items)
