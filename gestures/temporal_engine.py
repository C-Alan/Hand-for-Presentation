
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from gestures.gesture_classifier import (
    GestureClassifier,
    GestureScore,
    TrajectoryPoint,
    GESTURES,
)
from gestures.landmark_processor import ProcessedHand


# =====================================================
# SORTIE DU MOTEUR DYNAMIQUE
# =====================================================

@dataclass
class EngineOutput:
    active_state: str                  # "IDLE", "SWIPING", "TRIGGERED", "COOLDOWN", "BLOCKED", "LOST"
    triggered_gesture: Optional[str]   # Événement unique ("SWIPE_LEFT", etc.) ou None
    last_gesture: Optional[str]        # Dernier geste validé pour affichage HUD
    confidence: float                  # Confiance du geste (0.0 → 1.0)
    cooldown_progress: float           # 0.0 (début cooldown) → 1.0 (prêt)
    speed: float                       # Vitesse instantanée (unités/s)
    velocity: Tuple[float, float]      # (vx, vy)
    direction: Optional[str]           # "LEFT", "RIGHT", "UP", "DOWN"
    trajectory: List[Tuple[float, float]]  # Points récents normalisés pour le HUD
    quality: float                     # Qualité du tracking
    raw_gesture: str                   # Nom du geste brut détecté
    is_palm_open: bool = True          # True si la paume est ouverte
    openness: float = 1.0              # Score d'ouverture [0.0 → 1.0]
    is_locked: bool = False            # True si le système est verrouillé (attente paume ouverte au centre)
    is_in_center: bool = False         # True si la main est dans la boîte centrale
    center_box: Tuple[float, float, float, float] = (0.20, 0.15, 0.80, 0.85)
    is_transitioning: bool = False     # Compatibilité
    is_armed: bool = True              # True si le geste est armé (main stabilisée au repos)


# =====================================================
# MOTEUR TEMPOREL POUR GESTES DYNAMIQUES
# =====================================================

class TemporalGestureEngine:
    """
    Moteur temporel pour la reconnaissance de gestes dynamiques (présentation de document).

    Règles de sécurité :
    1. VERROUILLAGE SYSTÉMATIQUE À CHAQUE ENTRÉE/SORTIE DE CHAMP :
       Dès que la main quitte le champ de vision ou y entre, toute détection est BLOQUÉE.
    2. DÉVERROUILLAGE PAR PAUME OUVERTE AU CENTRE :
       Tant que la paume n'est pas présentée OUVERTE au CENTRE du champ visuel,
       tout balayage est strictement neutralisé.
    3. Scan cinématique multi-fenêtres pour une exécution réactive une fois déverrouillé.
    4. Filtrage anti-rebond pour éliminer le mouvement de retour de la main.
    """

    def __init__(
        self,
        classifier: Optional[GestureClassifier] = None,
        window_duration: float   = 0.50,   # Durée maximale de la fenêtre d'analyse (secondes)
        cooldown_duration: float = 0.45,   # Période réfractaire post-déclenchement (secondes)
        rest_speed_threshold: float = 0.30,# Vitesse sous laquelle la main est considérée au repos
        lost_tolerance_frames: int  = 10,  # Tolérance aux pertes de tracking (10 frames = ~330ms)
        require_center_unlock: bool = True,# Exiger la paume ouverte au centre après chaque entrée/sortie
        center_box: Tuple[float, float, float, float] = (0.20, 0.15, 0.80, 0.85), # Boîte centrale ergonomique élargie
        center_hold_frames: int  = 2,      # Frames stables requises au centre pour déverrouiller
    ):
        self.classifier = classifier or GestureClassifier()
        self.window_duration = window_duration
        self.cooldown_duration = cooldown_duration
        self.rest_speed_threshold = rest_speed_threshold
        self.lost_tolerance_frames = lost_tolerance_frames
        self.require_center_unlock = require_center_unlock
        self.center_box = center_box
        self.center_hold_frames = center_hold_frames

        self._trajectory: Deque[TrajectoryPoint] = deque(maxlen=60)
        self._last_trigger_time: float = -1.0
        self._last_gesture: Optional[str] = None
        self._armed: bool = True
        self._locked: bool = require_center_unlock  # Verrouillé au départ
        self._center_frames: int = 0
        self._lost_frames: int = 0
        self.is_paused: bool = False

    def unlock(self):
        """Déverrouille manuellement le système (utile pour les tests et overrides)."""
        self._locked = False
        self._armed = True
        self._center_frames = self.center_hold_frames
        self._trajectory.clear()

    def lock(self):
        """Verrouille le système."""
        self._locked = True
        self._armed = False
        self._center_frames = 0
        self._trajectory.clear()

    # =========================================
    # GESTION DU BUFFER DE TRAJECTOIRE
    # =========================================

    def _prune_trajectory(self, current_time: float):
        cutoff = current_time - self.window_duration
        while self._trajectory and self._trajectory[0].timestamp < cutoff:
            self._trajectory.popleft()

    # =========================================
    # RECHERCHE MULTI-FENÊTRE DU GESTE
    # =========================================

    def _scan_for_swipe(self, traj_list: List[TrajectoryPoint]) -> Tuple[Optional[str], float]:
        n = len(traj_list)
        if n < 3:
            return None, 0.0

        best_gesture = None
        best_conf = 0.0

        for start_idx in range(n - 2):
            sub = traj_list[start_idx:]
            if len(sub) < 3:
                continue
            dt = sub[-1].timestamp - sub[0].timestamp
            if dt < 0.07 or dt > 0.45:
                continue

            scores = self.classifier.classify_trajectory(sub)
            if scores and scores[0].confidence > best_conf:
                best_conf = scores[0].confidence
                best_gesture = scores[0].gesture

        return best_gesture, best_conf

    # =========================================
    # MISE À JOUR PRINCIPALE
    # =========================================

    def update(
        self,
        hand: Optional[ProcessedHand] = None,
        quality: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> EngineOutput:
        now = timestamp if timestamp is not None else time.monotonic()

        # ── 1. Gestion de la sortie du champ de vision ─────────
        if hand is None or quality < 0.15:
            self._lost_frames += 1
            if self._lost_frames > self.lost_tolerance_frames:
                self._trajectory.clear()
                # La main a quitté le champ : VERROUILLAGE OBLIGATOIRE
                if self.require_center_unlock:
                    self._locked = True
                self._center_frames = 0
                self._armed = False

            cd_progress = 1.0
            if self._last_trigger_time > 0:
                elapsed = now - self._last_trigger_time
                cd_progress = min(1.0, elapsed / self.cooldown_duration)

            return EngineOutput(
                active_state="LOST",
                triggered_gesture=None,
                last_gesture=self._last_gesture,
                confidence=0.0,
                cooldown_progress=cd_progress,
                speed=0.0,
                velocity=(0.0, 0.0),
                direction=None,
                trajectory=[],
                quality=0.0,
                raw_gesture="UNKNOWN",
                is_palm_open=False,
                openness=0.0,
                is_locked=self._locked,
                is_in_center=False,
                center_box=self.center_box,
                is_transitioning=False,
                is_armed=False,
            )

        self._lost_frames = 0
        cx, cy, cz = hand.palm_center

        # Vérifier si la main est dans la zone centrale de déverrouillage
        xmin, ymin, xmax, ymax = self.center_box
        in_center = (xmin <= cx <= xmax) and (ymin <= cy <= ymax)

        # ── 2. Gestion du Verrou de Sécurité (Entrée / Sortie de champ) ─
        if self._locked and self.require_center_unlock:
            # Pour déverrouiller : la paume doit être OUVERTE et située au CENTRE
            if hand.is_palm_open and in_center:
                self._center_frames += 1
                if self._center_frames >= self.center_hold_frames:
                    # DÉVERROUILLAGE RÉUSSI !
                    self._locked = False
                    self._armed = False  # Déverrouillé mais NON armé tant que la main bouge encore
                    self._trajectory.clear()
            else:
                self._center_frames = 0

            # Tant que le système est verrouillé, AUCUNE détection de geste n'est admise
            if self._locked:
                return EngineOutput(
                    active_state="BLOCKED",
                    triggered_gesture=None,
                    last_gesture=self._last_gesture,
                    confidence=0.0,
                    cooldown_progress=1.0,
                    speed=0.0,
                    velocity=(0.0, 0.0),
                    direction=None,
                    trajectory=[(cx, cy)],
                    quality=hand.quality,
                    raw_gesture="UNKNOWN",
                    is_palm_open=hand.is_palm_open,
                    openness=hand.openness,
                    is_locked=True,
                    is_in_center=in_center,
                    center_box=self.center_box,
                    is_transitioning=False,
                    is_armed=False,
                )

        # ── 3. Système Déverrouillé : Enregistrement cinématique ───
        self._trajectory.append(
            TrajectoryPoint(
                x=cx,
                y=cy,
                z=cz,
                timestamp=now,
                palm_width=hand.palm_width,
                is_palm_open=hand.is_palm_open,
                openness=hand.openness,
            )
        )
        self._prune_trajectory(now)

        traj_list = list(self._trajectory)
        feat = self.classifier.analyze_trajectory(traj_list)

        speed = feat["speed"]
        vx = feat["vx"]
        vy = feat["vy"]
        trajectory_2d = [(p.x, p.y) for p in traj_list]

        direction = None
        if speed > 0.18:
            if abs(vx) > abs(vy) * 1.2:
                direction = "LEFT" if vx < 0 else "RIGHT"
            elif abs(vy) > abs(vx) * 1.2:
                direction = "UP" if vy < 0 else "DOWN"

        # ── 4. Cooldown & Réarmement ───────────────────────────
        in_cooldown = False
        cd_progress = 1.0
        if self._last_trigger_time > 0:
            elapsed = now - self._last_trigger_time
            if elapsed < self.cooldown_duration:
                in_cooldown = True
                cd_progress = max(0.0, min(1.0, elapsed / self.cooldown_duration))
            else:
                cd_progress = 1.0

        if not in_cooldown:
            if not self._armed:
                # Armer UNIQUEMENT si la vitesse mesurée est valide (>= 2 points) ET inférieure au seuil de repos
                if feat.get("valid", False) and speed < self.rest_speed_threshold:
                    self._armed = True
        else:
            self._armed = False

        # ── 5. Évaluation du geste dynamique ──────────────────
        triggered_gesture: Optional[str] = None
        detected_gesture, detected_conf = self._scan_for_swipe(traj_list)
        raw_gesture = detected_gesture or "UNKNOWN"

        if detected_gesture and detected_conf >= self.classifier.min_confidence:
            if self._armed and not in_cooldown:
                triggered_gesture = detected_gesture
                self._last_gesture = detected_gesture
                self._last_trigger_time = now
                self._armed = False
                in_cooldown = True
                cd_progress = 0.0
                self._trajectory.clear()

        # ── 6. Détermination de l'état actif ──────────────────
        if triggered_gesture is not None:
            active_state = "TRIGGERED"
        elif in_cooldown:
            active_state = "COOLDOWN"
        elif not hand.is_palm_open:
            active_state = "CLOSED"
        elif speed > self.classifier.min_velocity * 0.7:
            active_state = "SWIPING"
        else:
            active_state = "IDLE"

        return EngineOutput(
            active_state=active_state,
            triggered_gesture=triggered_gesture,
            last_gesture=self._last_gesture,
            confidence=detected_conf,
            cooldown_progress=cd_progress,
            speed=speed,
            velocity=(vx, vy),
            direction=direction,
            trajectory=trajectory_2d,
            quality=hand.quality,
            raw_gesture=raw_gesture,
            is_palm_open=hand.is_palm_open,
            openness=hand.openness,
            is_locked=False,
            is_in_center=in_center,
            center_box=self.center_box,
            is_transitioning=in_cooldown,
            is_armed=self._armed,
        )

    def reset(self):
        self._trajectory.clear()
        self._last_trigger_time = -1.0
        self._last_gesture = None
        self._armed = True
        self._locked = self.require_center_unlock
        self._center_frames = 0
        self._lost_frames = 0
        self.is_paused = False
