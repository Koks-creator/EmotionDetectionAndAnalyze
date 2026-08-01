import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from glob import glob
import shutil
from collections import Counter

from config import Config


image_filenames_map = {}
for file_path in glob(f"{Config.VAL_IMAGES_FOLDER}/*.*"):
    img_filename = Path(file_path).name
    img_filename_no_ext = os.path.splitext(img_filename)[0]
    image_filenames_map[img_filename_no_ext] = img_filename


for file_path in glob(f"{Config.VAL_LABELS_FOLDER}/*.txt"):
    counter = Counter()
    with open(file_path) as f:
        content = f.read().strip().split("\n")
    
    for row in content:
        class_id = row.split()[0]
        counter[class_id] += 1
    
    if counter["1"] > 3 or counter["4"] > 1 or counter["6"] > 3 or counter["7"] > 3:
        label_filename = Path(file_path).name
        filename_noext = label_filename.replace(".txt", "")
        # print(Config.VAL_IMAGES_FOLDER / image_filenames_map[filename_noext],
        #          Config.TRAIN_IMAGES_FOLDER / image_filenames_map[filename_noext])
        # input("xd")
        shutil.move(Config.VAL_LABELS_FOLDER / label_filename, Config.TRAIN_LABELS_FOLDER / label_filename)
        shutil.move(Config.VAL_IMAGES_FOLDER / image_filenames_map[filename_noext],
                 Config.TRAIN_IMAGES_FOLDER / image_filenames_map[filename_noext])

    # break