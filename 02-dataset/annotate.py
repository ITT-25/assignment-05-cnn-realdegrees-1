import sys
import os
import cv2
import json
import uuid
import shutil
import argparse
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

class AnnotationTool:
    def __init__(self, image_paths: List[str], gesture_names: List[str], output_path: str, no_rename: bool = False) -> None:
        self.image_paths: List[str] = image_paths
        self.gesture_names: List[str] = gesture_names
        self.output_path: str = output_path
        self.annotations: Dict[str, Dict[str, List]] = {}
        self.current_image_index: int = 0
        self.rect_start: Optional[Tuple[int, int]] = None
        self.rect_end: Optional[Tuple[int, int]] = None
        self.drawing: bool = False
        self.no_rename: bool = no_rename
        self.image_path: str = ""
        self.default_label: str = ""
        self.image: np.ndarray = None
        self.clone: np.ndarray = None
        self.h: int = 0
        self.w: int = 0

        cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Image", self.mouse_callback)
        self.load_next_image()
        self.run()

    def run(self) -> None:
        while self.current_image_index < len(self.image_paths):
            img_display = self.image.copy()
            if self.rect_start and self.rect_end:
                cv2.rectangle(img_display, self.rect_start, self.rect_end, (0, 255, 0), 2)
            cv2.putText(img_display, f"Gesture: {self.default_label}  |  Press 's' to save  |  ESC to exit",
                        (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Image", img_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.save_annotation()
            elif key == 27:  # ESC
                self.finish()
                break

    def load_next_image(self) -> None:
        if self.current_image_index >= len(self.image_paths):
            self.finish()
            return

        self.image_path = self.image_paths[self.current_image_index]
        self.default_label = self.gesture_names[self.current_image_index]
        self.image = cv2.imread(self.image_path)
        self.clone = self.image.copy()
        self.h, self.w = self.image.shape[:2]
        self.rect_start = self.rect_end = None
        self.drawing = False

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.rect_start = (x, y)
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.rect_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.rect_end = (x, y)
            self.drawing = False

    def save_annotation(self) -> None:
        if not self.rect_start or not self.rect_end:
            print("Please draw a rectangle first.")
            return

        label = self.default_label

        x1, y1 = self.rect_start
        x2, y2 = self.rect_end
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        bbox = [x1 / self.w, y1 / self.h, (x2 - x1) / self.w, (y2 - y1) / self.h]

        annotation_id = str(uuid.uuid4())
        self.annotations[annotation_id] = {
            "bboxes": [bbox],
            "labels": [label]
        }

        ext = os.path.splitext(self.image_path)[1]
        source_dir = os.path.dirname(self.image_path)
        uuid_path = os.path.join(source_dir, f"{annotation_id}{ext}")

        if self.no_rename:
            shutil.copy2(self.image_path, uuid_path)
        else:
            shutil.move(self.image_path, uuid_path)

        print(f"Saved annotation for {os.path.basename(self.image_path)} as {annotation_id}{ext}")
        self.current_image_index += 1
        self.load_next_image()

    def finish(self) -> None:
        with open(os.path.join(self.output_path, "_annotations.json"), 'w') as f:
            json.dump(self.annotations, f, indent=4)
        print("All images annotated. Annotations saved to _annotations.json")
        cv2.destroyAllWindows()
        sys.exit(0)


def collect_image_paths_and_labels(folder: str) -> Tuple[List[str], List[str]]:
    image_paths = []
    gesture_names = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".jpg"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, folder)
                parts = rel_path.split(os.sep)
                if len(parts) > 1:
                    gesture_name = parts[0]
                else:
                    gesture_name = os.path.splitext(file)[0]
                image_paths.append(full_path)
                gesture_names.append(gesture_name)
    return image_paths, gesture_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gesture Annotation Tool")
    parser.add_argument("folder", help="Target folder containing images")
    parser.add_argument("--no-rename", action="store_true", help="Keep original file and copy to UUID")

    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print("Provided path is not a directory")
        sys.exit(1)

    images, gestures = collect_image_paths_and_labels(args.folder)
    if not images:
        print("No .jpg images found in folder.")
        sys.exit(1)

    AnnotationTool(images, gestures, args.folder, no_rename=args.no_rename)
