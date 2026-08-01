from collections import deque
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from config import Config
from emotion_classificator import EmotionClassification
from emotion_plots import save_plots
from sort_tracker import Sort
from timeline_tracker import EmotionTimeline
from yolo_detector import YoloDetector


@dataclass
class FrameDetectionData:
    original_frame: np.ndarray
    detection_frame: np.ndarray
    detections_data: Generator

@dataclass
class EmotionAnalyze:
    model_folder: Path = Config.YOLO_MODEL_FOLDER
    model_name: str = Config.YOLO_MODEL_NAME
    classes_path: Path = Config.YOLO_CLASSES_FILE
    device: str = Config.YOLO_DEVICE
    cnn_model_folder: Path = Config.CNN_MODEL_FOLDER
    cnn_model_name: str = Config.CNN_MODEL_NAME
    cnn_meta_name: str = Config.CNN_MODEL_META_NAME
    sort_max_age: int = Config.SORT_MAX_AGE
    sort_min_hits: int = Config.SORT_MIN_HITS
    sort_iou_thr: float = Config.SORT_IOU_THRESHOLD

    def __post_init__(self) -> None:
        self.model_path = self.model_folder / self.model_name
        self.yolo_detector = YoloDetector(
            model_path=self.model_path,
            classes_path=self.classes_path,
            device=self.device
        )
        self.sort_alg = Sort(
            max_age=self.sort_max_age,
            min_hits=self.sort_min_hits,
            iou_threshold=self.sort_iou_thr
        )
        self.emotion_classificator = EmotionClassification(
            model_path=self.cnn_model_folder / self.cnn_model_name,
            meta_path=self.cnn_model_folder / self.cnn_meta_name
        )

        with open(self.classes_path) as f:
            self.class_names = f.read().split("\n")

    def yolo_detect(self,
                    images: list[np.ndarray],
                    conf: float = .35,
                    iou: float = .1,
                    augment: bool = True,
                    agnostic_nms: bool = True,
                    use_sahi: bool = False,
                    sahi_conf: float = 0.2, 
                    sahi_slice_height: int = 256, 
                    sahi_slice_width: int = 256, 
                    sahi_overlap_height_ratio: float = 0.2, 
                    sahi_overlap_width_ratio: float = 0.2
        ) -> tuple[list[Generator], list[np.ndarray],  list[np.ndarray]]:

        if use_sahi:
            res = self.yolo_detector.detect_with_sahi(
                images=images,
                conf=sahi_conf, 
                slice_height=sahi_slice_height, 
                slice_width=sahi_slice_width, 
                overlap_height_ratio=sahi_overlap_height_ratio,
                overlap_width_ratio=sahi_overlap_width_ratio
            )
            detection_results, detection_frames = map(list, zip(*res)) if res else ([], [])
            detection_generators = [self.yolo_detector.yield_sahi_data(sahi_result=detection_res)
                                    for detection_res in detection_results]
        else:
            res = self.yolo_detector.detect(images=images,
                                            conf=conf,
                                            iou=iou,
                                            augment=augment,
                                            agnostic_nms=agnostic_nms)
            detection_results, detection_frames = map(list, zip(*res)) if res else ([], [])
            detection_generators = [self.yolo_detector.yield_data(bbox=detection_res) for detection_res in detection_results]

        return detection_generators, detection_frames, images

    def normalize_yolo_predictions(self, detection_generators: list[Generator],
                                   detection_frames: list[np.ndarray],
                                   images: list[np.ndarray]
                                   ) -> list[FrameDetectionData]:
        result = []
        for detection_generator, detection_frame, frame in zip(detection_generators, detection_frames, images):
            result.append(
                FrameDetectionData(
                    original_frame=frame,
                    detection_frame=detection_frame,
                    detections_data=detection_generator
                )
            )
        
        return result

    def set_object_ids(self, detections: list[FrameDetectionData]) -> list[np.ndarray]:
        total_track_data = []
        for detection_obj in detections:
            detection_gen = detection_obj.detections_data
            track_data = []
            for detection in detection_gen:
                class_id, _, conf, x1, y1, x2, y2 = *detection[:3], *detection[3]
                track_data.append([x1, y1, x2, y2, conf, class_id])
            
            updated_tracks = self.sort_alg.update(track_data).astype(float)
            total_track_data.append(updated_tracks)
        
        return total_track_data

    def draw_bbox(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, class_name: str, obj_id: int) -> None:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 50, 80), 2, 1)
        cv2.putText(frame, f"{class_name} ID: {obj_id}", (x1, y1-5), cv2.FONT_HERSHEY_PLAIN, 1.4, (200, 50, 80), 2)
    
    def run(self,
            video_input: int | Path,
            timeline_smoothing_seconds: float = .3,
            timeline_min_conf: float = .4,
            timeline_min_duration: float = .4,
            timeline_max_gap: float | None = None,
            yolo_conf: float = .35,
            yolo_iou: float = .1,
            yolo_augment: bool = True,
            yolo_agnostic_nms: bool = True,
            use_sahi: bool = False,
            sahi_conf: float = 0.2, 
            sahi_slice_height: int = 256, 
            sahi_slice_width: int = 256, 
            sahi_overlap_height_ratio: float = 0.2, 
            sahi_overlap_width_ratio: float = 0.2,
            save_raport: bool = True,
            plot_out_dir: str = "plots",
            resize_shape: None | tuple[int, int] = (1280, 720)            
            ) -> dict:
        
        cap = cv2.VideoCapture(video_input)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        is_stream = cap.get(cv2.CAP_PROP_FRAME_COUNT) <= 0

        timeline = EmotionTimeline(
            smoothing_seconds=timeline_smoothing_seconds,
            min_conf=timeline_min_conf,
            min_duration=timeline_min_duration,
            max_gap=timeline_max_gap,
        )

        frame_idx = 0
        timestamp = 0.0
        start = perf_counter()
        p_time = start

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            timestamp = perf_counter() - start if is_stream else frame_idx / fps
            frame_idx += 1
            clean_frame = frame.copy()

            detection_generators, detection_frames, images = self.yolo_detect(
                images=[frame],
                conf=yolo_conf,
                iou=yolo_iou,
                augment=yolo_augment,
                agnostic_nms=yolo_agnostic_nms,
                use_sahi=use_sahi,
                sahi_conf=sahi_conf, 
                sahi_slice_height=sahi_slice_height, 
                sahi_slice_width=sahi_slice_width, 
                sahi_overlap_height_ratio=sahi_overlap_height_ratio, 
                sahi_overlap_width_ratio=sahi_overlap_width_ratio
            )
            detections = self.normalize_yolo_predictions(
                detection_generators=detection_generators,
                detection_frames=detection_frames,
                images=images
            )
            total_track_data = self.set_object_ids(detections=detections)

            cropped_images = []
            boxes = []
            for object_track_data in total_track_data:
                for track_data in object_track_data:
                    x1, y1, x2, y2, _, obj_id, class_id, _ = track_data
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    obj_id, class_id = int(obj_id), int(class_id)

                    h, w = clean_frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    cropped = clean_frame[y1:y2, x1:x2]
                    cropped_images.append(cropped)
                    boxes.append({
                        "bbox": (x1, y1, x2, y2),
                        "face_id": obj_id
                    })

            res = self.emotion_classificator.predict_face_bulk(imgs=cropped_images)
            for c, r in zip(boxes, res, strict=True): # it should always be the same len
                emotion, em_conf = r
                # print(c)
                obj_id = c["face_id"]
                x1, y1, x2, y2 = c["bbox"]
                stable = timeline.update(obj_id, emotion, float(em_conf), timestamp)

                self.draw_bbox(frame=frame, x1=x1, y1=y1, x2=x2, y2=y2,
                            class_name=f"{stable} ({em_conf:.2f})", obj_id=obj_id)

            c_time = perf_counter()
            fps_ = int(1 / (c_time - p_time))
            p_time = c_time

            cv2.putText(frame, f"FPS: {fps_}", (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.4, (100, 0, 255), 2)
            if resize_shape:
                frame = cv2.resize(frame, resize_shape)

            cv2.imshow("res", frame)

            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

        timeline.finalize(timestamp)
        raport = timeline.to_dict()
        if save_raport:
            save_plots(raport, plot_out_dir, video_duration=timestamp)
        return raport


if __name__ == "__main__":
    vid = Config.VIDEOS_FOLDER / "8379044-hd_1920_1080_25fps.mp4"
    ea = EmotionAnalyze()
    print(ea.run(video_input=vid))
    
