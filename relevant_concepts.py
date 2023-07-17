import requests
from sklearn.metrics.pairwise import cosine_similarity


def get_all_edges_of_a_concept(concept,language="en"):
    all_edges=[]
    uri='http://api.conceptnet.io/c/'+language+"/"+concept
    obj = requests.get(uri).json()


    edges=obj["edges"]
    for i in edges:
        all_edges.append(i)

    if "view" not in obj:
        return all_edges

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


#Get Glove Embeddings
with open("glove.6B.50d.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()
glove_embeddings = {}
for line in lines:
    parts = line.split()
    word = parts[0].lower()
    embedding = [float(value) for value in parts[1:]]
    glove_embeddings[word] = embedding


def return_sorted_properties_with_end_embedding_only(start_node,end_node):
    edges=get_all_edges_of_a_concept(start_node)
    related_nodes=[]
    for edge in edges:
        lang1=edge["start"]["@id"].split("/")[2]
        lang2=edge["end"]["@id"].split("/")[2]
        if lang1=="en" and lang2=="en" :
            if edge["start"]["label"]=="life":
                related_nodes.append(edge["end"]["label"])
            else:
                related_nodes.append(edge["start"]["label"])
            
    stop_words = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 
                'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with']
    stop_words=set(stop_words)

    potential_words=[]
    broken_words=[]

    for i in range(len(related_nodes)):
        word=related_nodes[i]
        words=word.split(" ")
        for j in words:
            if j not in stop_words:
                broken_words.append(j)
        potential_words.append(word)
    
    broken_words=list(set(broken_words))
    potential_words=list(set(potential_words))

    start_node_embedding = glove_embeddings[start_node]
    end_node_embedding = glove_embeddings[end_node]
    potential_words=potential_words+broken_words

    concepts_cosine_sim = {}
    for i in range(len(potential_words)):
        word = potential_words[i]
        words = word.split(" ")
        if len(words) == 1:
            if word in glove_embeddings and word!=start_node and word!=end_node:
                concepts_cosine_sim[word] = cosine_similarity([glove_embeddings[word]], [end_node_embedding])[0][0]
        
        else:
            embedding_sum = [0] * len(end_node_embedding)
            for j in range(len(words)):
                if words[j] in glove_embeddings and words[j]!=start_node and words[j]!=end_node:
                
                    embedding_sum = [a + b for a, b in zip(embedding_sum, glove_embeddings[words[j]])]
            embedding_sum = [val / len(words) for val in embedding_sum]
            concepts_cosine_sim[word] = cosine_similarity([embedding_sum], [end_node_embedding])[0][0]

    sorted_items = sorted(concepts_cosine_sim.items(), key=lambda x: x[1], reverse=True)
    top_k_items = sorted_items[:10]
    print("Top Items by comparing only End_node_Embedding ")
    print(top_k_items)


def return_sorted_properties_with_start_embedding_only(start_node,end_node):
    edges=get_all_edges_of_a_concept(start_node)
    related_nodes=[]
    for edge in edges:
        lang1=edge["start"]["@id"].split("/")[2]
        lang2=edge["end"]["@id"].split("/")[2]
        if lang1=="en" and lang2=="en" :
            if edge["start"]["label"]=="life":
                related_nodes.append(edge["end"]["label"])
            else:
                related_nodes.append(edge["start"]["label"])
            
    stop_words = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 
                'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with']
    stop_words=set(stop_words)

    potential_words=[]
    broken_words=[]

    for i in range(len(related_nodes)):
        word=related_nodes[i]
        words=word.split(" ")
        for j in words:
            if j not in stop_words:
                broken_words.append(j)
        potential_words.append(word)
    
    broken_words=list(set(broken_words))
    potential_words=list(set(potential_words))

    start_node_embedding = glove_embeddings[start_node]
    end_node_embedding = glove_embeddings[end_node]
    potential_words=potential_words+broken_words

    concepts_cosine_sim = {}
    for i in range(len(potential_words)):
        word = potential_words[i]
        words = word.split(" ")
        if len(words) == 1:
            if word in glove_embeddings and word!=start_node and word!=end_node:
                concepts_cosine_sim[word] = cosine_similarity([glove_embeddings[word]], [start_node_embedding])[0][0]
        
        else:
            embedding_sum = [0] * len(start_node_embedding)
            for j in range(len(words)):
                if words[j] in glove_embeddings and words[j]!=start_node and words[j]!=end_node:
                
                    embedding_sum = [a + b for a, b in zip(embedding_sum, glove_embeddings[words[j]])]
            embedding_sum = [val / len(words) for val in embedding_sum]
            concepts_cosine_sim[word] = cosine_similarity([embedding_sum], [start_node_embedding])[0][0]

    sorted_items = sorted(concepts_cosine_sim.items(), key=lambda x: x[1], reverse=True)
    top_k_items = sorted_items[:10]
    print("Top Items by comparing only Start_node_Embedding ")
    print(top_k_items)




def return_sorted_properties_with_start_and_end_embedding(start_node,end_node):
    edges=get_all_edges_of_a_concept(start_node)
    related_nodes=[]
    for edge in edges:
        lang1=edge["start"]["@id"].split("/")[2]
        lang2=edge["end"]["@id"].split("/")[2]
        if lang1=="en" and lang2=="en" :
            if edge["start"]["label"]=="life":
                related_nodes.append(edge["end"]["label"])
            else:
                related_nodes.append(edge["start"]["label"])
            
    stop_words = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 
                'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with']
    stop_words=set(stop_words)

    potential_words=[]
    broken_words=[]

    for i in range(len(related_nodes)):
        word=related_nodes[i]
        words=word.split(" ")
        for j in words:
            if j not in stop_words:
                broken_words.append(j)
        potential_words.append(word)
    
    broken_words=list(set(broken_words))
    potential_words=list(set(potential_words))

    start_node_embedding = glove_embeddings[start_node]
    end_node_embedding = glove_embeddings[end_node]
    end_node_embedding=[a + b for a, b in zip(start_node_embedding, end_node_embedding)]

    potential_words=potential_words+broken_words

    concepts_cosine_sim = {}
    for i in range(len(potential_words)):
        word = potential_words[i]
        words = word.split(" ")
        if len(words) == 1:
            if word in glove_embeddings and word!=start_node and word!=end_node:
                concepts_cosine_sim[word] = cosine_similarity([glove_embeddings[word]], [end_node_embedding])[0][0]
        
        else:
            embedding_sum = [0] * len(end_node_embedding)
            for j in range(len(words)):
                if words[j] in glove_embeddings and words[j]!=start_node and words[j]!=end_node:
                
                    embedding_sum = [a + b for a, b in zip(embedding_sum, glove_embeddings[words[j]])]
            embedding_sum = [val / len(words) for val in embedding_sum]
            concepts_cosine_sim[word] = cosine_similarity([embedding_sum], [end_node_embedding])[0][0]

    sorted_items = sorted(concepts_cosine_sim.items(), key=lambda x: x[1], reverse=True)
    top_k_items = sorted_items[:10]
    print("Top Items by comparing both Start and End_node_Embedding ")
    print(top_k_items)




#Accommodate the results
print("Results for Accomodate the results")
return_sorted_properties_with_end_embedding_only("accommodate","results")
return_sorted_properties_with_start_embedding_only("accommodate","results")
return_sorted_properties_with_start_and_end_embedding("accommodate","results")


print()
print()
print()
print()
#Life is a journey
print("Results for Life is a journey")
return_sorted_properties_with_end_embedding_only("life","journey")
return_sorted_properties_with_start_embedding_only("life","journey")
return_sorted_properties_with_start_and_end_embedding("life","journey")

print()
print()
print()
print()

#Life in the camp drain him ( Life, drain )
print("Results for Life in the camp drained him")
return_sorted_properties_with_end_embedding_only("life","drain")
return_sorted_properties_with_start_embedding_only("life","drain")
return_sorted_properties_with_start_and_end_embedding("life","drain")

print()
print()
print()
print()
#The boss exploded when he heard the resignation of his secretary ( Boss, Explode)
print("Results for Boss exploded")
return_sorted_properties_with_end_embedding_only("boss","explode")
return_sorted_properties_with_start_embedding_only("boss","explode")
return_sorted_properties_with_start_and_end_embedding("boss","explode")



