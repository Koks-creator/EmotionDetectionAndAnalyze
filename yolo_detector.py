from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from time import time

import cv2
import numpy as np
from config import Config
from custom_decorators import log_call, timeit
from custom_logger import CustomLogger
from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.prediction import PredictionResult
from sahi.utils.cv import visualize_object_predictions
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

logger = CustomLogger(logger_log_level=Config.CLI_LOG_LEVEL,
                      file_handler_log_level=Config.FILE_LOG_LEVEL,
                      log_file_name=fr"{Config.ROOT_PATH}/logs/yolo_logs.log"
                      ).create_logger()


@dataclass
class YoloDetector:
    model_path: str | Path
    classes_path: str| Path
    device: str = "cpu"  # Domyślnie "auto", może być też "cpu", "cuda:0" itp.

    @timeit(logger=logger)
    def __post_init__(self) -> None:
        logger.info(
            "Initing detector:\n"
            f"{self.model_path=}\n"
            f"{self.classes_path=}\n"
            f"{self.device=}\n"
        )
        self.model = YOLO(self.model_path)
        logger.info("Model loaded")
        self.sahi_model = None

        with open(self.classes_path) as f:
            self.classes_list = f.read().strip().split("\n")
            logger.info(f"{self.classes_list=}")
        
        self.colors = {key: tuple(int(x) for x in np.random.randint(0, 255, 3)) for key in self.classes_list }
        logger.info(f"{self.colors=}")
    
    @log_call(logger=logger, log_params=["conf", "iou", "augment", "agnostic_nms"], hide_res=True)
    @timeit(logger=logger)
    def detect(self, images: list[np.array], conf: float = .1, iou: float = .35, augment: bool = True, 
               agnostic_nms: bool = True) -> list[tuple]:
        
        results = self.model.predict(
            source=images,
            conf=conf,
            iou=iou,
            augment=augment,
            device=self.device,
            agnostic_nms=agnostic_nms,
        )
        return [(r.boxes, r.plot()) for r in results]
    
    @log_call(logger=logger, log_params=["conf", "slice_height", "slice_width", "overlap_height_ratio", "overlap_width_ratio"], hide_res=True)
    @timeit(logger=logger)
    def detect_with_sahi(self, images: list[np.ndarray], conf: float = 0.2, 
                         slice_height: int = 256, slice_width: int = 256, 
                         overlap_height_ratio: float = 0.2, overlap_width_ratio: float = 0.2) -> list[tuple]:
        """
        slice_height, slice_width = the larger the better detection of smaller objects but longer processing time
        overlap_height_ratio, overlap_width_ratio = slice overlay 
        """
        if self.sahi_model is None:
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=self.model_path,
                confidence_threshold=conf,
                device=self.device  # Dodany parametr device
            )
        
        results = []
        for img in images:
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)

            sahi_result = get_sliced_prediction(
                image=pil_img,
                detection_model=self.sahi_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
            )
            vis_img = visualize_object_predictions(
                image=rgb_img,
                object_prediction_list=sahi_result.object_prediction_list,
                rect_th=2,
                text_size=0.5,
                text_th=2
            )
            
            vis_img = cv2.cvtColor(np.array(vis_img["image"]), cv2.COLOR_RGB2BGR)

            results.append((sahi_result, np.array(vis_img)))
            
        return results
    
    @log_call(logger=logger, log_params=[""], hide_res=True)
    @timeit(logger=logger)
    def yield_sahi_data(self, sahi_result: PredictionResult) -> Generator:
        """
        Converts SAHI results to regular YOLO data format
        """
        
        for object_prediction in sahi_result.object_prediction_list:
            bbox = object_prediction.bbox.to_xyxy()
            cls_id = object_prediction.category.id
            conf = object_prediction.score.value
            class_name = object_prediction.category.name
            
            x1, y1, x2, y2 = np.array(bbox).astype(int)
            # x2, y2 = x1 + w, y1 + h
            yield cls_id, class_name, conf, (x1, y1, x2, y2)
    
    @log_call(logger=logger, log_params=[""], hide_res=True)
    @timeit(logger=logger)
    def yield_data(self, bbox: Boxes) -> Generator:
        for box in bbox:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            conf = float(box.conf[0])
            yield cls_id, class_name, conf,  (x1, y1, x2, y2)
    

if __name__ == "__main__":
    @dataclass(frozen=True)
    class FrameDetectionData:
        original_frame: np.ndarray
        detection_frame: np.ndarray
        detections_data: Generator

    def get_center(bbox: list[int, int, int, int]) -> list[int, int]:
        return bbox[0] + (abs(bbox[2]-bbox[0])//2), bbox[1] + (abs(bbox[3]-bbox[1])//2)
    
    yolo_predictor = YoloDetector(
        model_path=r"C:\Users\table\PycharmProjects\MojeCos2\emotion_recognition\models\model1\best.pt",
        classes_path=r"C:\Users\table\PycharmProjects\MojeCos2\emotion_recognition\models\classes.txt"
    )

    # for img_path in glob(r"C:\Users\table\PycharmProjects\MojeCos2\emotion_recognition\TestData\*.*"):
    # img = cv2.imread(r"C:\Users\table\PycharmProjects\MojeCos2\emotion_recognition\test_data\images\ffhq_1023_png_jpg.rf.773445fdeb82226802837a6f470c2a27.jpg")
    # res = yolo_predictor.detect(images=[img])

    # detection_results, detection_frames = map(list, zip(*res)) if res else ([], [])
    # detection_generators = [yolo_predictor.yield_data(bbox=detection_res) for detection_res in detection_results]
    # print(next(detection_generators[0]))
    # cv2.imshow("res", detection_frames[0])
    # cv2.waitKey(0)

    # yolo_predictor2 = YoloPredictor(model_path="model_tuned.pt")

    cap = cv2.VideoCapture(0)
    p_time = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        res, res_img = yolo_predictor.detect(images=[frame])[0]

        key = cv2.waitKey(1)
        if key == 27:
            break
        # cv2.imshow("Wykryte uszkodzenia drogi", res[0][1])
        # print(res2[0][1])
        c_time = time()
        fps = int(1 / (c_time - p_time))
        p_time = c_time
        cv2.putText(res_img, f"FPS: {fps}", (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.4, (100, 0, 255), 2)
        alpha = 0.8
        # final_img = cv2.addWeighted(res_img, alpha, overlay, 1 - alpha, 0)
        cv2.imshow("res", res_img)
    cv2.destroyAllWindows()
    cap.release()