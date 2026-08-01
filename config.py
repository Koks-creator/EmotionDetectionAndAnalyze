import json
import logging
import os
from pathlib import Path
from typing import Union

#  $env:DATA_UPLOAD_MAX_NUMBER_FILES="10000" 

class Config:
    # Overall
    ROOT_PATH: str = Path(__file__).resolve().parent

    # Folders
    RAW_DATA_PATH: Path = ROOT_PATH / "RawData"
    CLEANED_DATA_PATH: Path = ROOT_PATH / "DataCleaned"
    TRAIN_IMAGES_FOLDER: Path = ROOT_PATH / "train_data" / "images"/ "train"
    VAL_IMAGES_FOLDER: Path = ROOT_PATH / "train_data" / "images" / "val"
    TRAIN_LABELS_FOLDER: Path = ROOT_PATH / "train_data" / "labels" / "train"
    VAL_LABELS_FOLDER: Path = ROOT_PATH / "train_data" / "labels" / "val"
    TEST_DATA_FOLDER: Path = ROOT_PATH / "TestData"
    VIDEOS_FOLDER: Path =  ROOT_PATH / "videos"

    # SORT
    SORT_MAX_AGE: int = 50
    SORT_MIN_HITS: int = 1
    SORT_IOU_THRESHOLD: float = .3

    # SAHI MODEL PARAMS
    USE_SAHI: bool = False
    SAHI_CONF_THRESH: float = .2
    SAHI_SLICE_HEIGHT: int = 480
    SAHI_SLICE_WIDTH: int = 480
    SAHI_OVERLAP_HEIGHT_RATIO: float = 0.2
    SAHI_OVERLAP_WIDTH_RATIO: float = 0.2

    # YOLO Model
    YOLO_MODELS_FOLDER_PATH: Path = ROOT_PATH / "models_yolo"
    YOLO_MODEL_FOLDER: Path = YOLO_MODELS_FOLDER_PATH / "model_small"
    YOLO_MODEL_NAME: str = "best.pt"
    YOLO_CLASSES_FILE: Path = YOLO_MODEL_FOLDER / "classes.txt"
    YOLO_DEVICE: str = "cpu"
    YOLO_IOU: float = .2
    YOLO_CONF_THRESH: float = .2
    YOLO_AUGMENT: bool = True
    YOLO_AGNOSTIC_NMS: bool = True

    # CNN Model
    CNN_MODELS_FOLDER_PATH: Path = ROOT_PATH / "models_cnn"
    CNN_MODEL_FOLDER: Path = CNN_MODELS_FOLDER_PATH / "model1"
    CNN_MODEL_NAME: str = "emotion_vgg16_ferplus.keras"
    CNN_MODEL_META_NAME: str = "emotion_vgg16_ferplus_meta.json"

    # Dataset filtering
    MIN_HEIGHT: int = 150
    MIN_WIDTH: int = 150
    MAX_HEIGHT: int = 5000
    MAX_WIDTH: int = 5000
    ALLOWED_EXTENSIONS: tuple = (".jpg", ".png", ".jpeg")

    # LOGGER
    CLI_LOG_LEVEL: int = logging.DEBUG
    FILE_LOG_LEVEL: int = logging.DEBUG
