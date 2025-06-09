import cv2
import os
import logging
import click
from pynput.keyboard import Key
from src.gesture_model import GestureModel
from src.hand_detector import HandDetector
from src.cooldown_timer import CooldownTimer
from src.gesture_queue import GestureQueue

# Disable tensorflow and mediapipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
logging.getLogger('absl').setLevel(logging.ERROR)

# Configuration
IMG_SIZE = 64
COLOR_CHANNELS = 1
SIZE = (IMG_SIZE, IMG_SIZE)
TRAINING_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gesture_dataset_sample")
GESTURE_ACTIONS = {
    "stop": Key.media_stop,
    "fist": Key.media_play_pause,
    "peace": Key.media_volume_down,
    "two_up": Key.media_volume_up
}

gesture_queue = GestureQueue(maxlen=50, threshold=0.8)
cooldown_timer = CooldownTimer(cooldown=2.0)
gesture_model = GestureModel(SIZE, COLOR_CHANNELS, TRAINING_DATA_PATH, GESTURE_ACTIONS)
hand_detector = HandDetector()

@click.command()
@click.option('--video-id', '-c', default=0, help='ID of the webcam you want to use', type=int, show_default=True)
def main(video_id: int) -> None:
    print(f"Starting webcam capture with camera ID: {video_id}")
    cap = cv2.VideoCapture(video_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print(f"Error: Could not open camera with ID {video_id}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect hand bounding box
        bbox = hand_detector.detect_hand_bbox(frame)
        
        # If no hand is detected, clear the gesture queue
        if not bbox:
            gesture_queue.clear()
        
        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
