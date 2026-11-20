
from typing import List, Optional, Tuple
from gestures.landmark_processor import LandmarkProcessor, ProcessedHand
from gestures.gesture_classifier import GestureClassifier, GestureScore, TrajectoryPoint


class GestureDetector:
    """
    Façade de détection pour les gestes dynamiques de présentation.
    Délègue la normalisation à LandmarkProcessor et l'analyse cinématique à GestureClassifier.
    """

    def __init__(self, classifier: Optional[GestureClassifier] = None):
        self._processor  = LandmarkProcessor(smoothing_alpha=0.35)
        self._classifier = classifier or GestureClassifier()

    def process_hand(
        self,
        landmarks,
        world_landmarks=None,
        image_aspect: float = 1.0,
    ) -> ProcessedHand:
        """Normalise et filtre les landmarks de la main."""
        return self._processor.process(landmarks, world_landmarks, image_aspect)

    def detect_gesture(
        self,
        landmarks,
        world_landmarks=None,
        image_aspect: float = 1.0,
        trajectory: Optional[List[TrajectoryPoint]] = None,
    ) -> str:
        """Retourne le nom du meilleur geste dynamique détecté ou UNKNOWN."""
        hand = self._processor.process(landmarks, world_landmarks, image_aspect)
        score = self._classifier.best(hand, trajectory)
        return score.gesture

    def classify_all(
        self,
        landmarks,
        world_landmarks=None,
        image_aspect: float = 1.0,
        trajectory: Optional[List[TrajectoryPoint]] = None,
    ) -> Tuple[List[GestureScore], ProcessedHand]:
        """Retourne les scores de tous les gestes dynamiques et la main traitée."""
        hand = self._processor.process(landmarks, world_landmarks, image_aspect)
        scores = self._classifier.classify(hand, trajectory)
        return scores, hand

    def reset(self):
        self._processor.reset()
