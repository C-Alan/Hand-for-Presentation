
import time
from typing import Dict, List, Optional
import pyautogui


class InputController:
    """
    Contrôleur d'entrées pour la présentation de documents.

    Transforme les gestes dynamiques détectés en raccourcis clavier de présentation :
    - SWIPE_RIGHT  : Diapositive / Page suivante (Touche Right ou PageDown)
    - SWIPE_LEFT   : Diapositive / Page précédente (Touche Left ou PageUp)
    - SWIPE_UP     : Démarrer diaporama / Plein écran (Touche F5)
    - SWIPE_DOWN   : Quitter diaporama / Plein écran (Touche Escape)
    """

    def __init__(
        self,
        key_next: str             = "right",
        key_prev: str             = "left",
        key_fullscreen: str       = "f5",
        key_exit: str             = "escape",
        enabled: bool             = True,
        simulate_only: bool       = False,  # True pour tests unitaires sans frappe réelle
    ):
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE    = 0.02

        self.key_next       = key_next
        self.key_prev       = key_prev
        self.key_fullscreen = key_fullscreen
        self.key_exit       = key_exit
        self.enabled        = enabled
        self.simulate_only  = simulate_only

        # Mapping geste → touche clavier (Balayage Droite = Suivante, Balayage Gauche = Précédente)
        self.key_mapping = {
            "SWIPE_RIGHT": self.key_next,
            "SWIPE_LEFT":  self.key_prev,
            "SWIPE_UP":    self.key_fullscreen,
            "SWIPE_DOWN":  self.key_exit,
        }

        # Descriptions explicites pour l'interface et les logs
        self.action_descriptions = {
            "SWIPE_RIGHT": "Diapositive Suivante (Page Suivante)",
            "SWIPE_LEFT":  "Diapositive Précédente (Page Précédente)",
            "SWIPE_UP":    "Lancer Diaporama / Plein écran",
            "SWIPE_DOWN":  "Quitter Diaporama / Plein écran",
        }

        # Historique et métriques
        self.history: List[Dict] = []
        self.action_counts: Dict[str, int] = {g: 0 for g in self.key_mapping}
        self.last_action: Optional[str] = None
        self.last_action_time: float = 0.0

    # =========================================
    # EXÉCUTION D'UNE COMMANDE
    # =========================================

    def trigger(self, gesture: str) -> bool:
        """
        Déclenche l'action clavier associée au geste dynamique reçu.
        """
        if not self.enabled:
            return False

        if gesture not in self.key_mapping:
            return False

        key = self.key_mapping[gesture]
        desc = self.action_descriptions.get(gesture, gesture)

        if not self.simulate_only:
            try:
                pyautogui.press(key)
            except Exception as err:
                print(f"[InputController] Erreur frappe touche {key} : {err}")
                return False

        now = time.monotonic()
        self.last_action = gesture
        self.last_action_time = now
        self.action_counts[gesture] = self.action_counts.get(gesture, 0) + 1

        record = {
            "timestamp": now,
            "gesture": gesture,
            "key": key,
            "description": desc,
        }
        self.history.append(record)
        if len(self.history) > 100:
            self.history.pop(0)

        print(f"[Présentation] >> {gesture} déclenché -> Touche '{key}' ({desc})")
        return True

    def process(
        self,
        state: Optional[str] = None,
        triggered_gesture: Optional[str] = None,
        **kwargs,
    ):
        """
        Traite la sortie du moteur temporel. Déclenche l'action si triggered_gesture est renseigné.
        Accepte aussi un objet EngineOutput complet.
        """
        if hasattr(state, "triggered_gesture"):
            triggered_gesture = state.triggered_gesture

        if triggered_gesture:
            self.trigger(triggered_gesture)

    # =========================================
    # GESTION DU CYCLE DE VIE
    # =========================================

    def reset(self):
        self.last_action = None
        self.last_action_time = 0.0

    def close(self):
        self.reset()
