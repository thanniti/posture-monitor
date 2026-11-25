from collections import deque


class PostureMonitorState:
    """Turns a stream of per-frame (timestamp, label) predictions into a smoothed
    label plus a debounced bad-posture flag."""

    def __init__(self, window_seconds, debounce_seconds):
        self.window_seconds = window_seconds
        self.debounce_seconds = debounce_seconds
        self.history = deque()  # (timestamp, label)
        self.bad_posture_since = None

    def update(self, timestamp, label):
        self.history.append((timestamp, label))
        cutoff = timestamp - self.window_seconds
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

        labels_in_window = [l for _, l in self.history]
        smoothed = max(set(labels_in_window), key=labels_in_window.count)

        if smoothed == "CorrectSitting":
            self.bad_posture_since = None
            return smoothed, False

        if self.bad_posture_since is None:
            self.bad_posture_since = timestamp

        flagged = (timestamp - self.bad_posture_since) >= self.debounce_seconds
        return smoothed, flagged
