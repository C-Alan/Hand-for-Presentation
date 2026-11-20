
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from gestures.landmark_processor import ProcessedHand


@dataclass
class TrajectoryPoint:
    """Point d'échantillonnage temporel pour l'analyse dynamique."""
    x: float
    y: float
    z: float
    timestamp: float
    palm_width: float
    is_palm_open: bool = True       # Indique si la paume était ouverte à cet instant
    openness: float = 1.0           # Degré d'ouverture [0.0 → 1.0]


@dataclass
class GestureScore:
    gesture: str
    confidence: float               # 0.0 → 1.0
    features: Dict = field(default_factory=dict)


# Gestes dynamiques pour la présentation de document
GESTURES = ("SWIPE_LEFT", "SWIPE_RIGHT", "SWIPE_UP", "SWIPE_DOWN")


class GestureClassifier:
    """
    Classifieur de gestes dynamiques pour le contrôle de présentation de documents.

    Analyse la cinématique sur fenêtre glissante avec précondition de paume ouverte :
    - Précondition obligatoire : La main doit être ouverte (doigts déployés).
    - Déplacement net et vitesse vectorielle.
    - Linéarité de trajectoire (efficacité de déplacement).
    - Dominance directionnelle orthogonale (rejet des diagonales ambiguës).
    """

    MIN_CONFIDENCE = 0.50

    def __init__(
        self,
        min_displacement: float  = 0.08,   # Déplacement minimal normalisé (~8% de l'image)
        min_velocity: float      = 0.28,   # Vitesse minimale (unités d'écran / seconde)
        min_efficiency: float    = 0.55,   # Ratio distance nette / longueur totale du chemin
        directional_ratio: float = 1.15,   # Dominance de l'axe principal (assoupli pour les balayages horizontaux naturels)
        min_confidence: float    = 0.50,
        require_open_palm: bool  = True,   # Exiger la paume ouverte pour valider un balayage
    ):
        self.min_displacement  = min_displacement
        self.min_velocity      = min_velocity
        self.min_efficiency    = min_efficiency
        self.directional_ratio = directional_ratio
        self.min_confidence   = min_confidence
        self.require_open_palm = require_open_palm

    # =========================================
    # ANALYSE CINÉMATIQUE DE TRAJECTOIRE
    # =========================================

    def analyze_trajectory(self, trajectory: List[TrajectoryPoint]) -> Dict:
        """
        Calcule les caractéristiques cinématiques d'une série temporelle de points.
        Retourne un dictionnaire de caractéristiques.
        """
        if not trajectory or len(trajectory) < 2:
            return {
                "valid": False, "dx": 0.0, "dy": 0.0, "dt": 0.0,
                "distance": 0.0, "speed": 0.0, "efficiency": 0.0,
                "vx": 0.0, "vy": 0.0, "ratio_h": 1.0, "ratio_v": 1.0,
                "is_open": False,
            }

        start = trajectory[0]
        end   = trajectory[-1]

        dt = end.timestamp - start.timestamp
        if dt <= 1e-4:
            return {
                "valid": False, "dx": 0.0, "dy": 0.0, "dt": dt,
                "distance": 0.0, "speed": 0.0, "efficiency": 0.0,
                "vx": 0.0, "vy": 0.0, "ratio_h": 1.0, "ratio_v": 1.0,
                "is_open": False,
            }

        dx = end.x - start.x
        dy = end.y - start.y
        distance = math.hypot(dx, dy)

        # Longueur cumulée du chemin (pour évaluer la rectitude)
        path_length = 0.0
        for i in range(1, len(trajectory)):
            path_length += math.hypot(
                trajectory[i].x - trajectory[i - 1].x,
                trajectory[i].y - trajectory[i - 1].y,
            )

        efficiency = distance / max(1e-6, path_length)
        speed = distance / dt
        vx = dx / dt
        vy = dy / dt

        abs_dx = abs(dx)
        abs_dy = abs(dy)
        ratio_h = abs_dx / max(1e-4, abs_dy)
        ratio_v = abs_dy / max(1e-4, abs_dx)

        open_ratio = sum(1 for p in trajectory if p.is_palm_open) / len(trajectory)

        return {
            "valid": True,
            "dx": dx,
            "dy": dy,
            "dt": dt,
            "distance": distance,
            "speed": speed,
            "efficiency": efficiency,
            "vx": vx,
            "vy": vy,
            "ratio_h": ratio_h,
            "ratio_v": ratio_v,
            "path_length": path_length,
            "palm_width": end.palm_width,
            "open_ratio": open_ratio,
            "is_end_open": end.is_palm_open,
        }

    # =========================================
    # ÉVALUATION DES GESTES DYNAMIQUES
    # =========================================

    def classify_trajectory(self, trajectory: List[TrajectoryPoint]) -> List[GestureScore]:
        """
        Évalue la probabilité de chaque balayage dynamique sur la trajectoire fournie.
        """
        feat = self.analyze_trajectory(trajectory)
        if not feat["valid"]:
            return [GestureScore("UNKNOWN", 0.0, {"reason": "insufficient_trajectory"})]

        # ── 1. Précondition stricte : Paume Ouverte ───────────────
        # Empêche tout déclenchement si la main est fermée (poing, repos, stylo en main)
        if self.require_open_palm:
            if feat["open_ratio"] < 0.50:
                return [GestureScore("UNKNOWN", 0.0, {**feat, "reason": "palm_not_open"})]

        distance   = feat["distance"]
        speed      = feat["speed"]
        efficiency = feat["efficiency"]
        dx         = feat["dx"]
        dy         = feat["dy"]
        ratio_h    = feat["ratio_h"]
        ratio_v    = feat["ratio_v"]
        dt         = feat["dt"]

        # Critères d'exclusion cinématique :
        # - Mouvement trop court (< min_displacement)
        # - Vitesse trop lente (< min_velocity)
        # - Trajectoire trop erratique (< min_efficiency)
        # - Durée trop longue (> 0.70 s, ce n'est plus un balayage vif)
        if (
            distance < self.min_displacement
            or speed < self.min_velocity
            or efficiency < self.min_efficiency
            or dt > 0.70
        ):
            return [GestureScore("UNKNOWN", 0.0, feat)]

        scores: List[GestureScore] = []

        # ── 2. Balayage Horizontal (SWIPE_LEFT / SWIPE_RIGHT) ──
        if ratio_h >= self.directional_ratio:
            f_dist = min(1.0, distance / (self.min_displacement * 1.4))
            f_spd  = min(1.0, speed / (self.min_velocity * 1.4))
            f_eff  = min(1.0, efficiency / 0.75)
            f_dir  = min(1.0, ratio_h / (self.directional_ratio * 1.3))

            conf = (f_dist * 0.35 + f_spd * 0.35 + f_eff * 0.15 + f_dir * 0.15)
            conf = max(0.0, min(1.0, conf))

            if dx < 0:
                scores.append(GestureScore("SWIPE_LEFT", conf, feat))
                scores.append(GestureScore("SWIPE_RIGHT", 0.0, feat))
            else:
                scores.append(GestureScore("SWIPE_RIGHT", conf, feat))
                scores.append(GestureScore("SWIPE_LEFT", 0.0, feat))

            scores.append(GestureScore("SWIPE_UP", 0.0, feat))
            scores.append(GestureScore("SWIPE_DOWN", 0.0, feat))

        # ── 3. Balayage Vertical (SWIPE_UP / SWIPE_DOWN) ───────
        elif ratio_v >= self.directional_ratio:
            f_dist = min(1.0, distance / (self.min_displacement * 1.4))
            f_spd  = min(1.0, speed / (self.min_velocity * 1.4))
            f_eff  = min(1.0, efficiency / 0.75)
            f_dir  = min(1.0, ratio_v / (self.directional_ratio * 1.3))

            conf = (f_dist * 0.35 + f_spd * 0.35 + f_eff * 0.15 + f_dir * 0.15)
            conf = max(0.0, min(1.0, conf))

            if dy < 0:
                scores.append(GestureScore("SWIPE_UP", conf, feat))
                scores.append(GestureScore("SWIPE_DOWN", 0.0, feat))
            else:
                scores.append(GestureScore("SWIPE_DOWN", conf, feat))
                scores.append(GestureScore("SWIPE_UP", 0.0, feat))

            scores.append(GestureScore("SWIPE_LEFT", 0.0, feat))
            scores.append(GestureScore("SWIPE_RIGHT", 0.0, feat))

        # ── 4. Diagonale ou direction ambiguë ─────────────────
        else:
            return [GestureScore("UNKNOWN", 0.0, {**feat, "reason": "ambiguous_direction"})]

        scores.sort(key=lambda s: s.confidence, reverse=True)
        return scores

    # =========================================
    # INTERFACE DE CLASSIFICATION
    # =========================================

    def classify(
        self,
        hand: ProcessedHand,
        trajectory: Optional[List[TrajectoryPoint]] = None,
    ) -> List[GestureScore]:
        if hand.quality < 0.20:
            return [GestureScore("UNKNOWN", 0.0, {"reason": "low_quality"})]

        if not trajectory:
            return [GestureScore("UNKNOWN", 0.0, {"reason": "no_trajectory"})]

        return self.classify_trajectory(trajectory)

    def best(
        self,
        hand: ProcessedHand,
        trajectory: Optional[List[TrajectoryPoint]] = None,
    ) -> GestureScore:
        scores = self.classify(hand, trajectory)
        if scores and scores[0].confidence >= self.min_confidence:
            return scores[0]
        return GestureScore("UNKNOWN", 0.0, scores[0].features if scores else {})
