import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from config import Config
from custom_logger import CustomLogger
from tensorflow import keras
from tensorflow.keras import layers

logger = CustomLogger(
    logger_log_level=Config.CLI_LOG_LEVEL,
    file_handler_log_level=Config.FILE_LOG_LEVEL,
    log_file_name=Config.LOGS_PATH
).create_logger()


@keras.utils.register_keras_serializable()
class RandomDownscale(layers.Layer):
    def __init__(self, min_px=40, max_px=112, **kw):
        super().__init__(**kw)
        self.min_px, self.max_px = min_px, max_px

    def call(self, x, training=None):
        if not training:
            return x
        s = tf.random.uniform([], self.min_px, self.max_px, dtype=tf.int32)
        small = tf.image.resize(x, (s, s), method="area")
        return tf.image.resize(small, tf.shape(x)[1:3], method="bilinear")

    def get_config(self):
        return {**super().get_config(), "min_px": self.min_px, "max_px": self.max_px}


@keras.utils.register_keras_serializable()
class VGGPreprocess(layers.Layer):
    """
    The Caffe convention expected by VGG16 weights from ImageNet: BGR, 0-255, mean subtracted.

    No std - this is NOT torchvision-style normalisation.

    Subtracting the mean raises the entire input by ~120, so the first convolution
    receives a constant offset that its biases do not compensate for. VGG16 does not have BatchNorm,
    so nothing centres this along the way - hence greater sensitivity than in MobileNetV2.

    Three different numbers, because ImageNet is not colour-neutral.
    With duplicated greyscale values, inverting the channels yields nothing - only the mean itself matters.
    """
    MEAN = [103.939, 116.779, 123.68] # ImageNet was trained on centred input, and VGG16 doesn’t have BN,
                                      # so it can’t compensate for this on its own. 
                                      # Without it, the first convolution operates on a point it has never seen before

    def call(self, x):
        x = x[..., ::-1]
        return x - tf.constant(self.MEAN, dtype=x.dtype)


@dataclass
class EmotionClassification:
    model_path: Path
    meta_path: Path

    def __post_init__(self) -> None:
        self.model = keras.models.load_model(self.model_path)

        with open(self.meta_path) as f:
            self.meta = json.load(f)
        self.classes = self.meta["classes"]

        self.in_h, self.in_w = self.model.input_shape[1:3]

    def predict_face_bulk(self, imgs: list[np.ndarray]) -> list[tuple[str, float]]:
        if not imgs:
            return []

        batch = np.empty((len(imgs), self.in_h, self.in_w), dtype="float32")
        for i, img in enumerate(imgs):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            batch[i] = cv2.resize(gray, (self.in_w, self.in_h),
                                interpolation=cv2.INTER_AREA)

        ps = self.model(batch, training=False).numpy()
        return [(self.classes[int(p.argmax())], float(p.max())) for p in ps]

    def predict_face(self, img: np.ndarray) -> tuple[str, float]:
        return self.predict_face_bulk([img])[0]

    # def predict_face(self, img: np.ndarray) -> tuple[str, float]:
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     x = cv2.resize(gray, (self.in_w, self.in_h)).astype("float32")
    #     p = self.model.predict(np.array([x]), verbose=0)[0]
        
    #     return self.classes[int(p.argmax())], float(p.max())

    # def predict_face_bulk(self, imgs: list):
    #     grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
    #     x = [cv2.resize(g, (self.in_w, self.in_h)).astype("float32") for g in grays]

    #     ps = self.model.predict(np.array(x), verbose=0)
    #     return [(self.classes[int(p.argmax())], float(p.max())) for p in ps]


if __name__ == "__main__":
    ec = EmotionClassification(
         model_path=r"C:\Users\table\PycharmProjects\MojeCos2\emotion_recognition\models_cnn\model1\emotion_vgg16_ferplus.keras",
         meta_path=r"C:\Users\table\PycharmProjects\MojeCos2\emotion_recognition\models_cnn\model1\emotion_vgg16_ferplus_meta.json"
    )

    img = cv2.imread(r"C:\Users\table\PycharmProjects\MojeCos2\gowienko\Screenshot_6.png")

    print(ec.predict_face(img=img))
