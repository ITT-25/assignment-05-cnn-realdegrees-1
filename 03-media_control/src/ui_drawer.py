from typing import Dict
import cv2
from src.cooldown_timer import CooldownTimer
from src.gesture_queue import GestureQueue
import numpy as np
from pynput.keyboard import Key

class UIDrawer:
    def __init__(self, gesture_queue: GestureQueue, gesture_actions: Dict[str, Key], cooldown_timer: CooldownTimer):
        self.gesture_queue = gesture_queue
        self.gesture_actions = gesture_actions
        self.cooldown_timer = cooldown_timer

    def draw_cooldown_bar(self, frame: np.ndarray) -> None:
        cooldown = self.cooldown_timer.get_progress()
        bar_width = int(frame.shape[1] * ((self.cooldown_timer.cooldown - cooldown) / self.cooldown_timer.cooldown))
        bar_height = 20
        bar_color = (0, 0, 255) if cooldown < self.cooldown_timer.cooldown else (0, 255, 0)
        cv2.rectangle(frame, (0, frame.shape[0] - bar_height), (bar_width, frame.shape[0]), bar_color, -1)
        cv2.rectangle(frame, (0, frame.shape[0] - bar_height), (frame.shape[1], frame.shape[0]), (255,255,255), 2)
        cv2.putText(frame, f"Cooldown: {max(0, self.cooldown_timer.cooldown-cooldown):.1f}s", (10, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        if self.cooldown_timer.last_action_label:
            action_text = f"Last action: {self.gesture_actions[self.cooldown_timer.last_action_label]} ({self.cooldown_timer.last_action_label})"
            cv2.putText(frame, action_text, (10, frame.shape[0] - bar_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
