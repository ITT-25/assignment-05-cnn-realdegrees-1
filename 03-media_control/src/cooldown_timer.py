import time


class CooldownTimer:
    def __init__(self, cooldown: float):
        self.cooldown = cooldown
        self.last_action_time = 0.0
        self.last_action_label = ''

    def ready(self) -> bool:
        return (time.time() - self.last_action_time) >= self.cooldown

    def trigger(self, label: str):
        self.last_action_time = time.time()
        self.last_action_label = label

    def get_remaining(self) -> float:
        return max(0.0, self.cooldown - (time.time() - self.last_action_time))

    def get_progress(self) -> float:
        return min(self.cooldown, max(0.0, time.time() - self.last_action_time))
