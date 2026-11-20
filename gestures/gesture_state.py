
from typing import Optional
from gestures.temporal_engine import TemporalGestureEngine, EngineOutput
from gestures.transition_manager import GestureTransitionManager
from gestures.landmark_processor import ProcessedHand


class GestureStateManager:
    """
    Façade publique d'état pour les gestes dynamiques de présentation.
    Délègue à TemporalGestureEngine + GestureTransitionManager.
    """

    def __init__(
        self,
        log_path: Optional[str] = None,
        window_duration: float   = 0.45,
        cooldown_duration: float = 0.55,
        require_center_unlock: bool = True,
    ):
        self._engine = TemporalGestureEngine(
            window_duration=window_duration,
            cooldown_duration=cooldown_duration,
            require_center_unlock=require_center_unlock,
        )
        self._transitions = GestureTransitionManager(log_path=log_path)

        self.current_state = "IDLE"
        self.is_paused     = False
        self._prev_state   = "IDLE"

    def update(
        self,
        hand: Optional[ProcessedHand] = None,
        quality: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> dict:
        """
        Met à jour l'état temporel et enregistre les transitions.
        Retourne un dictionnaire de résumé.
        """
        output: EngineOutput = self._engine.update(hand, quality, timestamp)
        output = self._transitions.update(output)

        self.current_state = output.active_state
        changed = output.active_state != self._prev_state
        self._prev_state = output.active_state

        return {
            "state":             output.active_state,
            "triggered_gesture": output.triggered_gesture,
            "last_gesture":      output.last_gesture,
            "changed":           changed,
            "confidence":        output.confidence,
            "cooldown_progress": output.cooldown_progress,
            "speed":             output.speed,
            "direction":         output.direction,
            "output":            output,
        }

    def update_full(
        self,
        hand: Optional[ProcessedHand] = None,
        quality: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> EngineOutput:
        """
        Met à jour et retourne l'EngineOutput complet.
        """
        output: EngineOutput = self._engine.update(hand, quality, timestamp)
        output = self._transitions.update(output)

        self.current_state = output.active_state
        self._prev_state   = output.active_state
        return output

    def reset(self):
        self._engine.reset()
        self._transitions.reset()
        self.current_state = "IDLE"
        self.is_paused     = False
        self._prev_state   = "IDLE"
