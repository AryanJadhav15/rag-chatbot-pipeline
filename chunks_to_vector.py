import os
import requests
import json
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def create_embedding(text):
    r = requests.post("http://localhost:11434/api/embeddings",json={"model":"bge-m3","prompt":text})
    
    embedding = r.json()["embedding"]
    return embedding

my_dict = []
json_files = os.listdir("jsons")
json_files.sort()
for file in json_files:
    with open(f"jsons/{file}") as f:
        content = json.load(f)
    for chunk in content["chunks"]:
        chunk['embeddings'] = create_embedding(chunk['text'])
        my_dict.append(chunk)
    print(f"Worked on {file}")
    
df = pd.DataFrame.from_records(my_dict)
# Saving the dataframe
joblib.dump(df,"embeddings_dataframe.joblib")

