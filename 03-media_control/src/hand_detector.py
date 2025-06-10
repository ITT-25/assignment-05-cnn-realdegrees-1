import cv2
import numpy as np
from typing import Optional, Tuple
from mediapipe.python.solutions.hands import Hands


class HandDetector:
    def __init__(self):
        self.hands = Hands(
            static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def detect_hand_bbox(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        # Convert to rgb for media pipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to find hands
        results = self.hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return None

        # Get the first detected hand's landmarks and form a bbox around them
        height, width = frame.shape[:2]
        landmarks = results.multi_hand_landmarks[0].landmark
        points = np.array([[int(landmark.x * width), int(landmark.y * height)] for landmark in landmarks])
        x_min, y_min, w, h = cv2.boundingRect(points)
        padding_w = int(w * 0.15)
        padding_h = int(h * 0.15)
        x_min = max(0, x_min - padding_w)
        y_min = max(0, y_min - padding_h)
        w = min(width - x_min, w + 2 * padding_w)
        h = min(height - y_min, h + 2 * padding_h)

        # Calculate uniform margin
        margin_percent = 0.10
        margin_size = int(min(w, h) * margin_percent)

        # Apply uniform margin to all sides of bbox
        x_min = max(0, x_min - margin_size)
        y_min = max(0, y_min - margin_size)
        w = min(width - x_min, w + 2 * margin_size)
        h = min(height - y_min, h + 2 * margin_size)

        return (x_min, y_min, w, h)
