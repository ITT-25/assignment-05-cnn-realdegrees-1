from typing import Optional, Tuple
import cv2
import numpy as np
from src.hand_detector import HandDetector
from src.gesture_model import GestureModel
from src.gesture_queue import GestureQueue
from src.media_controller import MediaController

class FrameProcessor:
    def __init__(self, hand_detector: HandDetector, gesture_model: GestureModel, gesture_queue: GestureQueue, media_controller: MediaController, gesture_actions, color_channels):
        self.hand_detector = hand_detector
        self.gesture_model = gesture_model
        self.gesture_queue = gesture_queue
        self.media_controller = media_controller
        self.gesture_actions = gesture_actions
        self.color_channels = color_channels

    def process(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int, str, float]]]:
        if not bbox:
            return frame
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        hand_img = frame[y:y+h, x:x+w]
        if hand_img.size == 0:
            return frame
        preprocessed_img = self.gesture_model.preprocess_image(hand_img)
        prediction = self.gesture_model.get_prediction(preprocessed_img)
        if prediction is None:
            return frame
        label, confidence = prediction
        if label is None:
            return frame
        display_img = cv2.resize(preprocessed_img, (w, h))
        if self.color_channels == 1:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
        frame[y:y+h, x:x+w] = display_img
        return frame, (x, y, w, h, label, confidence)
