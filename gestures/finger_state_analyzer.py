
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple


class FingerState(Enum):
    EXTENDED = "EXTENDED"   # Doigt tendu
    BENT     = "BENT"       # Demi-plié (état intermédiaire de transition)
    CURLED   = "CURLED"     # Doigt replié
    UNKNOWN  = "UNKNOWN"    # Indéterminé (qualité insuffisante)


@dataclass
class FingerAnalysis:
    states: Dict[str, FingerState]   # État discret par doigt
    scores: Dict[str, float]         # Score d'extension [0.0 → 1.0]


class FingerStateAnalyzer:
    """
    Détermine l'état de chaque doigt avec trois niveaux : EXTENDED, BENT, CURLED.

    Avantages par rapport à l'ancien booléen :
    - Le niveau BENT permet de modéliser les configurations intermédiaires
      (doigt en train de se tendre ou de se replier).
    - Le score continu permet au GestureClassifier de nuancer sa confiance.
    - Fonctionne côté paume et côté dos de la main.
    """

    # Seuils de classification à partir du score continu
    EXTENDED_THRESHOLD = 0.65
    BENT_THRESHOLD     = 0.30

    # =========================================
    # UTILITAIRES GÉOMÉTRIQUES
    # =========================================

    @staticmethod
    def _distance(a: Tuple, b: Tuple) -> float:
        return math.sqrt(
            (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
        )

    @staticmethod
    def _sub(a: Tuple, b: Tuple) -> Tuple:
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    @classmethod
    def _dot(cls, a: Tuple, b: Tuple) -> float:
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @classmethod
    def _cosine(cls, a: Tuple, b: Tuple) -> float:
        na = math.sqrt(cls._dot(a, a))
        nb = math.sqrt(cls._dot(b, b))
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return cls._dot(a, b) / (na * nb)

    # =========================================
    # SCORE D'EXTENSION D'UN DOIGT (INDEX→AURICULAIRE)
    # =========================================

    @classmethod
    def _finger_score(
        cls,
        points: List[Tuple],
        mcp_idx: int,
        pip_idx: int,
        tip_idx: int,
    ) -> float:
        """
        Score continu [0.0 → 1.0] d'extension pour l'index, le majeur,
        l'annulaire ou l'auriculaire.

        Combinaison de trois critères :
        1. Ratio de distance tip/wrist vs pip/wrist (longueur perçue).
        2. Ratio de distance tip/mcp vs pip/mcp (déploiement depuis la jointure).
        3. Cosinus de l'angle de flexion au PIP (rectitude du doigt).
        """
        wrist = points[0]
        mcp   = points[mcp_idx]
        pip   = points[pip_idx]
        tip   = points[tip_idx]

        tip_w = cls._distance(tip, wrist)
        pip_w = cls._distance(pip, wrist)
        tip_m = cls._distance(tip, mcp)
        pip_m = cls._distance(pip, mcp)

        # Score basé sur la distance au poignet
        if pip_w < 1e-6:
            dist_wrist_score = 0.0
        else:
            # 1.0 → bien tendu, 0.0 → replié
            dist_wrist_score = (tip_w / pip_w - 0.80) / 0.45
            dist_wrist_score = max(0.0, min(1.0, dist_wrist_score))

        # Score basé sur la distance à la MCP
        if pip_m < 1e-6:
            dist_mcp_score = 0.0
        else:
            dist_mcp_score = (tip_m / pip_m - 1.10) / 0.40
            dist_mcp_score = max(0.0, min(1.0, dist_mcp_score))

        # Score basé sur l'angle de flexion au PIP
        bend_cos = cls._cosine(cls._sub(pip, mcp), cls._sub(tip, pip))
        cos_score = (bend_cos - 0.05) / 0.85
        cos_score = max(0.0, min(1.0, cos_score))

        return 0.35 * dist_wrist_score + 0.30 * dist_mcp_score + 0.35 * cos_score

    # =========================================
    # SCORE D'EXTENSION DU POUCE
    # =========================================

    @classmethod
    def _thumb_score(cls, points: List[Tuple], palm_width: float) -> float:
        """
        Score continu [0.0 → 1.0] d'extension du pouce vers l'extérieur de la paume.
        Fonctionne côté paume et côté dos.

        Critères :
        - Décalage latéral (axe auriculaire→index).
        - Éloignement depuis la MCP de l'auriculaire.
        - Rectitude (cosinus au niveau de l'IP).
        """
        if palm_width < 1e-6:
            return 0.0

        mcp  = points[2]
        ip   = points[3]
        tip  = points[4]
        idx5 = points[5]   # MCP de l'index
        pk17 = points[17]  # MCP de l'auriculaire

        # Axe latéral de la paume (auriculaire → index)
        across    = cls._sub(idx5, pk17)
        across_sq = cls._dot(across, across)
        if across_sq < 1e-9:
            return 0.0

        lateral = cls._dot(cls._sub(tip, idx5), across) / across_sq
        lateral_score = (lateral - 0.08) / 0.35
        lateral_score = max(0.0, min(1.0, lateral_score))

        # Éloignement depuis la MCP de l'auriculaire
        tip_pk = cls._distance(tip, pk17)
        ip_pk  = cls._distance(ip,  pk17)
        mcp_pk = cls._distance(mcp, pk17)
        if ip_pk < 1e-6 or mcp_pk < 1e-6:
            ext_score = 0.0
        else:
            ext_score = min(
                (tip_pk / ip_pk  - 0.78) / 0.30,
                (tip_pk / mcp_pk - 0.75) / 0.30,
            )
            ext_score = max(0.0, min(1.0, ext_score))

        # Rectitude
        cos_score = (cls._cosine(cls._sub(ip, mcp), cls._sub(tip, ip)) + 0.2) / 1.2
        cos_score = max(0.0, min(1.0, cos_score))

        return 0.40 * lateral_score + 0.40 * ext_score + 0.20 * cos_score

    # =========================================
    # CLASSIFICATION À PARTIR DU SCORE
    # =========================================

    @classmethod
    def _to_state(cls, score: float) -> FingerState:
        if score >= cls.EXTENDED_THRESHOLD:
            return FingerState.EXTENDED
        elif score >= cls.BENT_THRESHOLD:
            return FingerState.BENT
        else:
            return FingerState.CURLED

    # =========================================
    # ANALYSE PRINCIPALE
    # =========================================

    # Mapping des doigts : nom → (MCP, PIP, TIP)
    _FINGERS = {
        "index":  (5,  6,  8),
        "middle": (9,  10, 12),
        "ring":   (13, 14, 16),
        "pinky":  (17, 18, 20),
    }

    def analyze(
        self,
        points: List[Tuple],
        palm_width: float,
        quality: float = 1.0,
    ) -> FingerAnalysis:

        scores: Dict[str, float] = {}
        states: Dict[str, FingerState] = {}

        if quality < 0.15 or palm_width < 1e-4:
            for name in ("thumb", "index", "middle", "ring", "pinky"):
                scores[name] = 0.0
                states[name] = FingerState.UNKNOWN
            return FingerAnalysis(states=states, scores=scores)

        # Pouce
        ts = self._thumb_score(points, palm_width)
        scores["thumb"] = ts
        states["thumb"] = self._to_state(ts)

        # Autres doigts
        for name, (mcp, pip, tip) in self._FINGERS.items():
            s = self._finger_score(points, mcp, pip, tip)
            scores[name] = s
            states[name] = self._to_state(s)

        return FingerAnalysis(states=states, scores=scores)

