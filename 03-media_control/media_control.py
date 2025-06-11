import cv2
import os
import click
from pynput.keyboard import Key
from src.gesture_model import GestureModel
from src.hand_detector import HandDetector
from src.cooldown_timer import CooldownTimer
from src.gesture_queue import GestureQueue
from src.media_controller import MediaController
from src.frame_processor import FrameProcessor
from src.ui_drawer import UIDrawer

# Configuration
COLOR_CHANNELS = 1
TRAINING_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gesture_dataset_sample")
GESTURE_ACTIONS = {
    "stop": Key.media_stop,
    "fist": Key.media_play_pause,
    "peace": Key.media_volume_down,
    "two_up": Key.media_volume_up,
}

gesture_queue = GestureQueue(maxlen=50, threshold=0.8)
cooldown_timer = CooldownTimer(cooldown=2.0)
gesture_model = GestureModel(COLOR_CHANNELS, TRAINING_DATA_PATH, GESTURE_ACTIONS)
hand_detector = HandDetector()
media_controller = MediaController(GESTURE_ACTIONS, cooldown_timer, gesture_queue)
frame_processor = FrameProcessor(
    hand_detector, gesture_model, gesture_queue, media_controller, GESTURE_ACTIONS, COLOR_CHANNELS
)
ui_drawer = UIDrawer(gesture_queue, GESTURE_ACTIONS, cooldown_timer)


@click.command()
@click.option("--video-id", "-c", default=0, help="ID of the webcam you want to use", type=int, show_default=True)
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

        # Process the frame and get predictions
        result = frame_processor.process(frame, bbox)
        if isinstance(result, tuple):
            frame, (x, y, w, h, label, confidence) = result
            ui_drawer.draw_label_and_progress(frame, x, y, w, h, label, confidence)
            media_controller.update_state()
            if not media_controller.in_cooldown() and confidence > 80:
                gesture_queue.append(label)
                if gesture_queue.is_full_and_consistent(label) and label in GESTURE_ACTIONS:
                    media_controller.handle_prediction(label, gesture_queue.queue)

        ui_drawer.draw_cooldown_bar(frame)
        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
