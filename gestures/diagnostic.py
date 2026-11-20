
import math
import time
from typing import List, Optional, Tuple
import cv2

from gestures.temporal_engine import EngineOutput


class DiagnosticOverlay:
    """
    Affichage temps réel pour le contrôle de présentation de documents par gestes dynamiques.

    Affiche :
    - Boîte centrale d'activation : guide visuel indiquant où ouvrir la paume pour déverrouiller.
    - Trajectoire cinématique (traînée temporelle avec dégradé d'opacité).
    - Détection de la paume ouverte (précondition visuelle en temps réel).
    - Vecteur de vitesse et jauge cinématique.
    - État du moteur (BLOQUÉ, PRÊT, BALAYAGE, DÉCLENCHÉ, COOLDOWN).
    - Bannière de confirmation de l'action de présentation.
    - Guide des gestes de présentation au bas de l'écran.
    """

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    ACTION_LABELS = {
        "SWIPE_RIGHT": ">> DIAPOSITIVE SUIVANTE [Flèche Droite]",
        "SWIPE_LEFT":  "<< DIAPOSITIVE PRÉCÉDENTE [Flèche Gauche]",
        "SWIPE_UP":    "^^ LANCER DIAPORAMA [F5]",
        "SWIPE_DOWN":  "vv QUITTER DIAPORAMA [Échap]",
    }

    ACTION_COLORS = {
        "SWIPE_RIGHT": (0, 255, 128),   # Vert émeraude
        "SWIPE_LEFT":  (255, 180, 0),   # Bleu cyan / azur
        "SWIPE_UP":    (0, 230, 255),   # Jaune doré
        "SWIPE_DOWN":  (80, 80, 255),   # Rouge / corail
    }

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._last_banner_gesture: Optional[str] = None
        self._last_banner_time: float = 0.0
        self._banner_duration: float = 1.4

    # =========================================
    # UTILITAIRES DE RENDU TEXTE
    # =========================================

    def _text(
        self,
        frame,
        text: str,
        pos: Tuple[int, int],
        scale: float = 0.55,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 1,
    ):
        x, y = pos
        cv2.putText(frame, text, (x, y), self.FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), self.FONT, scale, color, thickness, cv2.LINE_AA)

    # =========================================
    # ZONE CENTRALE D'ACTIVATION / DÉVERROUILLAGE
    # =========================================

    def draw_center_activation_zone(self, frame, output: EngineOutput):
        """Dessine la boîte centrale indiquant la zone d'armement par paume ouverte."""
        h, w = frame.shape[:2]
        xmin, ymin, xmax, ymax = output.center_box

        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)

        if output.is_locked:
            # Système verrouillé : cadre d'invitation visible
            if output.is_in_center and output.is_palm_open:
                box_color = (0, 255, 120)  # Main ouverte bien positionnée au centre
                prompt = "[DÉVERROUILLAGE EN COURS...]"
            elif output.is_in_center:
                box_color = (0, 165, 255)  # Au centre mais paume non ouverte
                prompt = "[OUVREZ LA PAUME ICI POUR DÉVERROUILLER]"
            else:
                box_color = (60, 130, 255)  # En attente d'entrée au centre
                prompt = "[PRÉSENTEZ LA PAUME OUVERTE POUR ACTIVER]"

            # Rectangle semi-transparent
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)
            cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)

            # Texte au-dessus du cadre
            t_size = cv2.getTextSize(prompt, self.FONT, 0.48, 1)[0]
            tx = max(10, x1 + (x2 - x1 - t_size[0]) // 2)
            ty = max(25, y1 - 10)
            self._text(frame, prompt, (tx, ty), scale=0.48, color=box_color, thickness=2)
        else:
            # Système déverrouillé : repères discrets et texte discret
            corner_len = 20
            color = (0, 200, 100)
            # Coin supérieur gauche
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 1)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 1)
            # Coin supérieur droit
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 1)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 1)
            # Coin inférieur gauche
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 1)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 1)
            # Coin inférieur droit
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 1)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 1)

    # =========================================
    # TRACÉ DE LA TRAJECTOIRE CINÉMATIQUE
    # =========================================

    def draw_trajectory_trail(
        self,
        frame,
        trajectory: List[Tuple[float, float]],
        active_state: str,
        is_palm_open: bool = True,
        is_locked: bool = False,
    ):
        if not trajectory or len(trajectory) < 2:
            return

        h, w = frame.shape[:2]
        pts_px = [(int(p[0] * w), int(p[1] * h)) for p in trajectory]

        n = len(pts_px)
        for i in range(1, n):
            alpha = i / float(n)
            if is_locked:
                color = (int(40 + alpha * 80), int(60 + alpha * 80), int(120 + alpha * 60))
            elif is_palm_open:
                color = (int(60 + alpha * 195), int(120 + alpha * 135), int(255 - alpha * 55))
            else:
                color = (int(40 + alpha * 100), int(100 + alpha * 65), int(200 + alpha * 55))
            cv2.line(frame, pts_px[i - 1], pts_px[i], color, max(1, int(1 + alpha * 3)), cv2.LINE_AA)

        last_pt = pts_px[-1]
        if is_locked:
            center_color = (120, 120, 160)
        else:
            center_color = (0, 255, 120) if is_palm_open else (0, 140, 255)
        cv2.circle(frame, last_pt, 8, center_color, -1, cv2.LINE_AA)
        cv2.circle(frame, last_pt, 12, (255, 255, 255), 2, cv2.LINE_AA)

    # =========================================
    # BANNIÈRE DE DÉCLENCHEMENT DE COMMANDE
    # =========================================

    def _draw_action_banner(self, frame, now: float):
        if not self._last_banner_gesture:
            return

        elapsed = now - self._last_banner_time
        if elapsed > self._banner_duration:
            return

        h, w = frame.shape[:2]
        gesture = self._last_banner_gesture
        label = self.ACTION_LABELS.get(gesture, gesture)
        color = self.ACTION_COLORS.get(gesture, (0, 255, 0))

        fade = 1.0 - (elapsed / self._banner_duration)
        box_alpha = min(0.85, fade)

        box_w = 640
        box_h = 60
        bx = (w - box_w) // 2
        by = 65

        overlay = frame.copy()
        cv2.rectangle(overlay, (bx, by), (bx + box_w, by + box_h), (25, 25, 30), -1)
        cv2.rectangle(overlay, (bx, by), (bx + box_w, by + box_h), color, 3)
        cv2.addWeighted(overlay, box_alpha, frame, 1.0 - box_alpha, 0, frame)

        font_scale = 0.75
        text_size = cv2.getTextSize(label, self.FONT, font_scale, 2)[0]
        tx = bx + (box_w - text_size[0]) // 2
        ty = by + (box_h + text_size[1]) // 2
        self._text(frame, label, (tx, ty), scale=font_scale, color=color, thickness=2)

    # =========================================
    # RENDU DU DASHBOARD HUD
    # =========================================

    def draw(
        self,
        frame,
        output: EngineOutput,
        fps: float,
        hand_count: int,
    ):
        if not self.enabled:
            return frame

        now = time.monotonic()
        h, w = frame.shape[:2]

        if output.triggered_gesture:
            self._last_banner_gesture = output.triggered_gesture
            self._last_banner_time = now

        # ── 1. Boîte centrale d'activation ──────────────────────
        self.draw_center_activation_zone(frame, output)

        # ── 2. Traînée cinématique ──────────────────────────────
        self.draw_trajectory_trail(
            frame,
            output.trajectory,
            output.active_state,
            is_palm_open=output.is_palm_open,
            is_locked=output.is_locked,
        )

        # ── 3. Panneau latéral d'état (HUD haut-gauche) ─────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (15, 15), (380, 245), (20, 20, 25), -1)
        cv2.rectangle(overlay, (15, 15), (380, 245), (70, 70, 80), 1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        self._text(frame, "MODE PRÉSENTATION", (25, 42), scale=0.62, color=(0, 230, 255), thickness=2)

        q_color = (0, 220, 100) if output.quality >= 0.50 else (0, 160, 255)
        self._text(frame, f"FPS : {fps:.1f}   Mains : {hand_count}", (25, 68), scale=0.48, color=(200, 200, 200))
        self._text(frame, f"Qualite suivi : {output.quality:.2f}", (25, 90), scale=0.48, color=q_color)

        # État de la Paume (Ouverte vs Fermée)
        if output.is_palm_open:
            p_color = (0, 255, 120)
            p_label = f"Paume : OUVERTE ({int(output.openness * 100)}%)"
        else:
            p_color = (0, 165, 255)
            p_label = f"Paume : FERMEE ({int(output.openness * 100)}% - Bloque)"
        self._text(frame, p_label, (25, 114), scale=0.48, color=p_color, thickness=2)

        # État du système
        state_str = output.active_state
        state_color = (180, 180, 180)
        state_label = "EN ATTENTE"

        if output.is_locked:
            state_color = (0, 140, 255)
            state_label = "BLOQUÉ (OUVRIR PAUME POUR ACTIVER)"
        elif state_str == "TRIGGERED":
            state_color = (0, 255, 120)
            state_label = f"ACTION : {output.triggered_gesture}"
        elif state_str == "SWIPING":
            state_color = (0, 220, 255)
            state_label = f"BALAYAGE ({output.direction or '...'})"
        elif state_str == "CLOSED":
            state_color = (0, 165, 255)
            state_label = "PAUME NON OUVERTE"
        elif state_str == "COOLDOWN":
            state_color = (0, 140, 255)
            state_label = "RECUPERATION"
        elif state_str == "LOST":
            state_color = (120, 120, 120)
            state_label = "MAIN HORS DU CHAMP"
        elif state_str == "IDLE":
            if not getattr(output, "is_armed", True):
                state_color = (0, 200, 255)
                state_label = "STABILISATION (MARQUEZ UN TEMPS D'ARRÊT)"
            else:
                state_color = (100, 255, 100)
                state_label = "ARMÉ (BALAYEZ LIBREMENT)"

        self._text(frame, f"Etat : {state_label}", (25, 138), scale=0.46, color=state_color, thickness=2)

        # Jauge de vitesse
        self._text(frame, f"Vitesse : {output.speed:.2f} u/s", (25, 164), scale=0.48, color=(220, 220, 220))
        speed_ratio = min(1.0, output.speed / 1.20)
        cv2.rectangle(frame, (25, 172), (220, 180), (50, 50, 60), -1)
        spd_fill = int(195 * speed_ratio)
        cv2.rectangle(frame, (25, 172), (25 + spd_fill, 180), (0, 200, 255), -1)

        # Barre de cooldown (anti-rebond)
        self._text(frame, "Disponibilite :", (25, 204), scale=0.48, color=(180, 180, 180))
        cv2.rectangle(frame, (25, 212), (220, 220), (50, 50, 60), -1)
        cd_fill = int(195 * max(0.0, min(1.0, output.cooldown_progress)))
        cd_color = (0, 220, 100) if output.cooldown_progress >= 0.99 else (0, 140, 255)
        cv2.rectangle(frame, (25, 212), (25 + cd_fill, 220), cd_color, -1)

        # ── 4. Bannière d'action au centre ─────────────────────
        self._draw_action_banner(frame, now)

        # ── 5. Guide des commandes en bas de l'écran ────────────
        bar_h = 36
        overlay_bot = frame.copy()
        cv2.rectangle(overlay_bot, (0, h - bar_h), (w, h), (15, 15, 20), -1)
        cv2.addWeighted(overlay_bot, 0.85, frame, 0.15, 0, frame)

        if output.is_locked:
            guide_text = "BLOCAGE ACTIF : Placez votre paume ouverte au CENTRE de l'image pour activer la detection"
            g_color = (0, 160, 255)
        else:
            guide_text = (
                "[Actif]  --> Droite : Diapo Suivante  |  <-- Gauche : Diapo Precedente  |  "
                "^^ Haut : F5  |  vv Bas : Echap"
            )
            g_color = (210, 210, 210)

        t_size = cv2.getTextSize(guide_text, self.FONT, 0.44, 1)[0]
        tx = max(10, (w - t_size[0]) // 2)
        self._text(frame, guide_text, (tx, h - 12), scale=0.44, color=g_color, thickness=1)

        return frame

    def draw_index_cursor(self, frame, landmarks, state: str, is_transitioning: bool):
        if not landmarks or len(landmarks) < 10:
            return frame
        h, w = frame.shape[:2]
        palm = landmarks[9]
        cx, cy = int(palm.x * w), int(palm.y * h)
        cv2.circle(frame, (cx, cy), 10, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
        return frame
