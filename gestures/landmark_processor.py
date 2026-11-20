
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from gestures.finger_state_analyzer import FingerStateAnalyzer, FingerAnalysis


@dataclass
class ProcessedHand:
    """Résultat normalisé et filtré d'une main détectée en coordonnées écran/caméra."""
    points: List[Tuple[float, float, float]]        # 21 points en coordonnées caméra normalisées [0..1]
    palm_width: float                               # Largeur de la paume normalisée [0..1]
    quality: float                                  # 0.0 (perdu) → 1.0 (parfait)
    timestamp: float
    has_world: bool = False
    world_points: Optional[List[Tuple[float, float, float]]] = None  # Coordonnées métriques locales optionnelles
    is_palm_open: bool = True                       # True si la paume est ouverte (précondition requise)
    openness: float = 1.0                           # Score d'ouverture global [0.0 → 1.0]
    finger_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def palm_center(self) -> Tuple[float, float, float]:
        """
        Centre représentatif de la paume en coordonnées image [0..1] (landmark 9, base du majeur).
        Permet de suivre le déplacement réel de la main à travers le champ de la caméra.
        """
        if self.points and len(self.points) > 9:
            return self.points[9]
        return (0.5, 0.5, 0.0)


class LandmarkProcessor:
    """
    Normalise, corrige et filtre les repères de la main pour le suivi cinématique.

    Points essentiels :
    - Les coordonnées principales (points) sont STRICTEMENT conservées en espace caméra [0..1]
      pour que les balayages spatiaux (gauche, droite, haut, bas) soient correctement mesurés.
    - Évalue l'ouverture de la paume (doigts déployés vs poing fermé) comme précondition de balayage.
    - Le lissage EMA élimine les micro-tremblements sans introduire de latence.
    """

    def __init__(self, smoothing_alpha: float = 0.40):
        self.alpha = smoothing_alpha
        self._prev_points: Optional[List[Tuple]] = None
        self._analyzer = FingerStateAnalyzer()

    # =========================================
    # CONVERSION EN COORDONNÉES ÉCRAN
    # =========================================

    @staticmethod
    def _extract_camera_points(landmarks) -> List[Tuple[float, float, float]]:
        """Extrait les 21 points dans l'espace image normalisé [0.0 → 1.0]."""
        return [(p.x, p.y, getattr(p, "z", 0.0)) for p in landmarks]

    # =========================================
    # LISSAGE TEMPOREL EMA
    # =========================================

    def _smooth(self, points: List[Tuple]) -> List[Tuple]:
        if self._prev_points is None or len(self._prev_points) != len(points):
            return points
        a = self.alpha
        return [
            (
                self._prev_points[i][0] * (1.0 - a) + points[i][0] * a,
                self._prev_points[i][1] * (1.0 - a) + points[i][1] * a,
                self._prev_points[i][2] * (1.0 - a) + points[i][2] * a,
            )
            for i in range(len(points))
        ]

    # =========================================
    # QUALITÉ DU SUIVI
    # =========================================

    @staticmethod
    def _compute_quality(palm_width: float) -> float:
        if palm_width < 0.02:
            return 0.0
        if palm_width < 0.05:
            return palm_width / 0.05
        return 1.0

    # =========================================
    # ÉVALUATION DE L'OUVERTURE DE LA PAUME
    # =========================================

    @staticmethod
    def _evaluate_openness(fa: FingerAnalysis) -> Tuple[bool, float]:
        """
        Détermine si la paume est ouverte.
        Exige au moins 3 doigts nettement étendus (index, majeur, annulaire, auriculaire)
        et un score moyen d'ouverture >= 0.48.
        Rejette formellement les poings fermés, pointages à un doigt, ou mains repliées.
        """
        primary = [fa.scores.get(f, 0.0) for f in ("index", "middle", "ring", "pinky")]
        extended_count = sum(1 for s in primary if s >= 0.48)
        avg_score = sum(primary) / 4.0

        is_open = (extended_count >= 3) and (avg_score >= 0.48)
        return is_open, avg_score

    # =========================================
    # TRAITEMENT PRINCIPAL
    # =========================================

    def process(
        self,
        landmarks,
        world_landmarks=None,
        image_aspect: float = 1.0,
    ) -> ProcessedHand:

        # Extraire les coordonnées caméra de l'image
        raw_screen = self._extract_camera_points(landmarks)
        smoothed_screen = self._smooth(raw_screen)

        # Calculer la largeur de paume (distance MCP index 5 -> MCP auriculaire 17)
        p5 = smoothed_screen[5]
        p17 = smoothed_screen[17]
        palm_width = math.hypot(p5[0] - p17[0], p5[1] - p17[1])

        quality = self._compute_quality(palm_width)
        self._prev_points = smoothed_screen

        # Analyse d'extension des doigts pour l'ouverture de paume
        fa = self._analyzer.analyze(smoothed_screen, palm_width, quality)
        is_open, openness = self._evaluate_openness(fa)

        world_pts = None
        if world_landmarks is not None:
            world_pts = [(p.x, p.y, p.z) for p in world_landmarks]

        return ProcessedHand(
            points=smoothed_screen,
            palm_width=palm_width,
            quality=quality,
            timestamp=time.monotonic(),
            has_world=(world_landmarks is not None),
            world_points=world_pts,
            is_palm_open=is_open,
            openness=openness,
            finger_scores=fa.scores,
        )

    def reset(self):
        self._prev_points = None
