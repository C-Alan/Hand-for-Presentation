
import csv
import os
import time
from dataclasses import dataclass
from typing import List, Optional

from gestures.temporal_engine import EngineOutput


@dataclass
class TransitionRecord:
    timestamp: float
    from_state: str
    to_state: str
    triggered_gesture: str
    confidence: float
    frames_in_state: int
    quality: float


class GestureTransitionManager:
    """
    Enregistre l'historique des transitions et des événements de gestes dynamiques.
    Exporte vers un fichier CSV pour l'analyse post-session de présentation.
    """

    def __init__(self, log_path: Optional[str] = None, max_history: int = 500):
        self.log_path = log_path
        self.max_history = max_history

        self._history: List[TransitionRecord] = []
        self._prev_state: str = "IDLE"
        self._frame_count: int = 0
        self._state_since: int = 0

        if log_path:
            dir_path = os.path.dirname(log_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "from_state", "to_state",
                    "triggered_gesture", "confidence",
                    "frames_in_state", "quality",
                ])

    def update(self, output: EngineOutput) -> EngineOutput:
        """Reçoit un EngineOutput, enregistre les transitions et événements, retourne output inchangé."""
        self._frame_count += 1

        state_changed = output.active_state != self._prev_state
        gesture_fired = output.triggered_gesture is not None

        if state_changed or gesture_fired:
            record = TransitionRecord(
                timestamp=time.monotonic(),
                from_state=self._prev_state,
                to_state=output.active_state,
                triggered_gesture=output.triggered_gesture or output.raw_gesture,
                confidence=round(output.confidence, 3),
                frames_in_state=self._frame_count - self._state_since,
                quality=round(output.quality, 3),
            )

            if len(self._history) >= self.max_history:
                self._history.pop(0)
            self._history.append(record)

            if self.log_path:
                self._append_csv(record)

            self._prev_state = output.active_state
            self._state_since = self._frame_count

        return output

    def _append_csv(self, record: TransitionRecord):
        try:
            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    f"{record.timestamp:.3f}",
                    record.from_state,
                    record.to_state,
                    record.triggered_gesture,
                    f"{record.confidence:.3f}",
                    record.frames_in_state,
                    f"{record.quality:.3f}",
                ])
        except OSError:
            pass

    @property
    def history(self) -> List[TransitionRecord]:
        return list(self._history)

    def reset(self):
        self._prev_state = "IDLE"
        self._state_since = self._frame_count
