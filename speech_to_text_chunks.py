import whisper
import json
import os

files = os.listdir("audios")
model = whisper.load_model('large-v2')
for file in files:
    result = model.transcribe(audio = f"audios/{file}",language = "hi",task = "translate")
    file_num = file.split("_")[0]
    file_name = file.split("_")[1].split(".")[0]
    chunks = []
    for segment in result["segments"]:
        chunks.append({"video_number":file_num,"video_name":file_name,"id":segment["id"],"start":segment["start"],"end":segment["end"],"text":segment["text"]})
        
        chunk_with_metadata = {"chunks":chunks,"text":result["text"]}
    
    with open(f"jsons/{file}.json" , "w") as f:
        json.dump(chunk_with_metadata , f)
    print(f"Created {file}.json Successfully".center(100,"*"))
        
