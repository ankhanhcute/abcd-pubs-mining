import numpy as np 
import pandas as pd
import os 
import sys 
import argparse
from openai import OpenAI

MODEL = "text-embedding-3-small"
VECTOR_PATH = "data/dict_embeddings.npy"
METADATA_PATH = "data/dict_embeddings_meta.csv"

vectors = np.load(VECTOR_PATH)
metadata = pd.read_csv(METADATA_PATH)

print("Vector shape:", vectors.shape)
print("Metadata shape:", metadata.shape)

print(
    metadata[
        [
            "name",
            "label",
            "domain"
        ]
    ].head(10).to_string(index=False)
)
if len(vectors) != len(metadata):
    raise ValueError(
        "The number of vectors does not match",
        "the number of metadata rows."
    )
    
parser = argparse.ArgumentParser(
    description="Search ABCD dictionary embeddings."
)
parser.add_argument(
    "--query",
    required=True,
    help="Raw IV/DV text to search for"
)
parser.add_argument(
    "--top-k",
    type=int, 
    default=5, 
    help="Number of matches to return"
)

args = parser.parse_args()
query = args.query 
top_k = args.top_k

def embed_query(client, query):
    response= client.embeddings.create(
        model=MODEL,
        input=query
    )
    query_vector = np.array(
        response.data[0].embedding, 
        dtype=np.float32
        )
    return query_vector
if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("ERROR: OEPNAI_API_KEY is not set")
    
client = OpenAI()
query_vector = embed_query(
    client, query
)
    
print("Query:", query)
print("Query vector shape:", query_vector.shape)
#use cosine similarity because we want to know the direction of meaning, 
#so the angle between vectors

#calculate cosine length, the magnitude of the query vector
query_length = np.linalg.norm(query_vector)

dictionary_lengths = np.linalg.norm(\
    vectors, 
    axis=1) #calc one length for each row

dot_products = vectors @ query_vector
similarities = dot_products / (
    dictionary_lengths * query_length
)
"""
dot product
----------------------------------
dictionary length × query length
So, 
dot_products = [
    0.45,  # dictionary variable 1 compared with query
    0.72,  # dictionary variable 2 compared with query
    0.31,  # dictionary variable 3 compared with query
    ...
]

"""

print("Similarity shape:", similarities.shape)
print("Similarity scores:", similarities)

top_indices = np.argsort(similarities)[::-1][:top_k]
print(f"top indexes", top_indices)
#argsort return indexes ordered from the smallest to the largest score.
#::-1 make largest first 
for rank, index in enumerate(top_indices, start=1):
    row = metadata.iloc[index] #retrieve the corresponding metadata row 
    score = similarities[index] #row similarity score
    
    print(f"\nRank {rank}")
    print(f"Index: {index}")
    print(f"Score: {score:.4f}")
    print(f"Name: {row['name']}")
    print(f"Label: {row['label']}")
    print(f"Domain: {row['domain']}")
    
