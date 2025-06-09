import time
from pynput.keyboard import Controller
from src.cooldown_timer import CooldownTimer
from src.gesture_queue import GestureQueue

class MediaController:
    def __init__(self, gesture_actions, cooldown_timer: CooldownTimer, gesture_queue: GestureQueue):
        self.gesture_actions = gesture_actions
        self.cooldown_timer = cooldown_timer
        self.gesture_queue = gesture_queue
        self.keyboard = Controller()
        self.cooldown_was_active = False

    def handle_prediction(self, pred_label: str, gesture_queue: GestureQueue):
        now = time.time()
        COOLDOWN = 3.0
        if len(gesture_queue) == gesture_queue.maxlen and all(g == pred_label for g in gesture_queue):
            if pred_label in self.gesture_actions and (now - self.cooldown_timer.last_action_time >= COOLDOWN):
                key = self.gesture_actions[pred_label]
                self.keyboard.press(key)
                self.keyboard.release(key)
                self.cooldown_timer.last_action_time = now
                self.cooldown_timer.last_action_label = pred_label
                self.cooldown_timer.trigger(pred_label)
                self.gesture_queue.clear()
                print(f"Gesture '{pred_label}' recognized and action performed: {key}")

    def update_state(self):
        if self.cooldown_timer.ready():
            if self.cooldown_was_active:
                self.gesture_queue.clear()
                self.cooldown_was_active = False
        else:
            self.cooldown_was_active = True

    def in_cooldown(self):
        return not self.cooldown_timer.ready()
