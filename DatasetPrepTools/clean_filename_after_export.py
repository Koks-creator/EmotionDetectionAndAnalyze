from glob import glob
from pathlib import Path
import os


folder = r"C:\Users\table\PycharmProjects\MojeCos2\emotion_recognition\train_data\labels\val"

for file_path in glob(f"{folder}/*.txt"):
    try:
        if "classes" in file_path:
            continue
        p = Path(file_path)
        file_name, parent = p.name, p.parent
        new_file_name = file_name.split("-")[1]
        new_path = rf"{parent}\{new_file_name}"
        os.rename(file_path, new_path)
        # break
    except IndexError:
        print(file_path)