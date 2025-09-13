# Converts video file to mp3 file
import os
import subprocess

files = os.listdir("videos")
for file in files:
    if file == ".DS_Store":
        continue
    else:
        file_name = file.split(" ｜ ")[0]
        file_number = file.split(" [")[0].split(" #")[-1]
        subprocess.run(["ffmpeg","-i",f"videos/{file}",f"audios/{file_number}_{file_name}.mp3"])
    