from collections import deque
from typing import Deque, Dict


class GestureQueue:
    def __init__(self, maxlen: int, threshold: float = 1.0):
        self.queue: Deque[str] = deque(maxlen=maxlen)
        self.threshold: float = threshold

    def append(self, label: str):
        self.queue.append(label)

    def progress(self, label: str) -> float:
        if not self.queue:
            return 0.0
        return min(1.0, len([g for g in self.queue if g == label]) / (self.queue.maxlen * self.threshold))

    def is_full_and_consistent(self, label: str) -> bool:
        return len(self.queue) == self.queue.maxlen and self.progress(label) >= 1.0

    def distribution(self) -> Dict[str, float]:
        if not self.queue:
            return {}
        return {label: self.progress(label) for label in set(self.queue)}

    def clear(self):
        self.queue.clear()

    def __len__(self) -> int:
        return len(self.queue)
