import pandas as pd
import numpy as np
import joblib
import requests
from sklearn.metrics.pairwise import cosine_similarity

def create_embedding(text):
    r = requests.post("http://localhost:11434/api/embeddings",json={"model":"bge-m3","prompt":text})
    
    embedding = r.json()["embedding"]
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate",json = {"model":"llama3.2","prompt":prompt,"stream":False})
    
    response = r.json()
    return response
    
df = joblib.load("embeddings_dataframe.joblib")

question = input("Ask Your Question - ")
question_embedding = create_embedding(question)

similarities = cosine_similarity(np.vstack(df['embeddings']),[question_embedding]).flatten()

top_results = 5
max_idx = similarities.argsort()[::-1][:top_results]

new_df = df.loc[max_idx]

prompt = f'''I am providing you with some chucks which are all related to python Object Oriented Programming  chunks contain video title, video number, start time in seconds, end time in seconds, the text at that time (these are for only your reference and you should not give the chuks as ou):

{new_df[["video_number","video_name","start","end","text"]].to_json(orient = "records")}
---------------------------------
"{question}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course
'''

with open("prompt.txt" ,"w") as f :
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt","w") as f:
    f.write(response)
    
# for idx,rows in new_df.iterrows():
#     print(idx,rows["video_number"],rows["video_name"],rows["start"],rows["end"],rows["text"])
    