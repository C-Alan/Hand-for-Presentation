
import unittest
import time
from typing import List, Tuple

from gestures.landmark_processor import ProcessedHand
from gestures.gesture_classifier import (
    GestureClassifier,
    GestureScore,
    TrajectoryPoint,
    GESTURES,
)
from gestures.temporal_engine import TemporalGestureEngine, EngineOutput
from gestures.gesture_state import GestureStateManager
from gestures.input_controller import InputController


def make_dummy_hand(
    x: float = 0.5,
    y: float = 0.5,
    z: float = 0.0,
    palm_width: float = 0.12,
    quality: float = 1.0,
    timestamp: float = 0.0,
    is_palm_open: bool = True,
    openness: float = 1.0,
) -> ProcessedHand:
    """Génère une ProcessedHand factice dont le landmark 9 (centre de paume) est à (x, y, z)."""
    points = [(x, y, z) for _ in range(21)]
    # Landmark 9 est la référence palm_center
    points[9] = (x, y, z)
    return ProcessedHand(
        points=points,
        palm_width=palm_width,
        quality=quality,
        timestamp=timestamp,
        has_world=False,
        is_palm_open=is_palm_open,
        openness=openness,
    )


class TestDynamicGestures(unittest.TestCase):
    """
    Suite de tests validant l'adaptation pour les gestes dynamiques (présentation de document)
    et confirmant la suppression complète des anciennes poses statiques.
    """

    # =========================================================
    # 1. VÉRIFICATION DE LA SUPPRESSION DES ANCIENNES POSES STATIQUES
    # =========================================================

    def test_static_poses_completely_removed(self):
        """Vérifie que les anciennes poses (INDEX, PINCH, SCROLL, FIST, OPEN_PALM) sont supprimées."""
        old_poses = {"INDEX", "PINCH", "SCROLL", "FIST", "OPEN_PALM"}
        for old in old_poses:
            self.assertNotIn(
                old,
                GESTURES,
                f"L'ancienne pose statique '{old}' ne doit plus figurer dans GESTURES.",
            )

        # Les gestes actifs doivent être les 4 balayages dynamiques
        expected = {"SWIPE_LEFT", "SWIPE_RIGHT", "SWIPE_UP", "SWIPE_DOWN"}
        self.assertEqual(set(GESTURES), expected)

    def test_camera_space_preservation_over_world_landmarks(self):
        """Vérifie que le centre de la paume suit l'espace caméra et n'est pas figé par world_landmarks."""
        from gestures.landmark_processor import LandmarkProcessor

        class DummyLM:
            def __init__(self, x, y, z=0.0):
                self.x, self.y, self.z = x, y, z

        lp = LandmarkProcessor()
        # Repères caméra déplacés à (0.25, 0.50)
        cam_lms = [DummyLM(0.25, 0.50) for _ in range(21)]
        # Repères mondiaux centrés sur la main à (0.01, -0.05)
        world_lms = [DummyLM(0.01, -0.05) for _ in range(21)]

        hand = lp.process(cam_lms, world_landmarks=world_lms)
        self.assertAlmostEqual(hand.palm_center[0], 0.25, places=3)
        self.assertAlmostEqual(hand.palm_center[1], 0.50, places=3)

    # =========================================================
    # 2. DÉTECTION DES 4 BALAYAGES DYNAMIQUES (SWIPES)
    # =========================================================

    def test_swipe_left_detection(self):
        """Balayage rapide vers la gauche -> SWIPE_LEFT."""
        engine = TemporalGestureEngine(cooldown_duration=0.5, require_center_unlock=False)
        # Déplacement de x=0.65 à x=0.35 en 0.20s (dx = -0.30, vx = -1.5)
        t0 = 100.0
        triggered = []
        for i in range(7):
            t = t0 + i * 0.033
            x = 0.65 - (i / 6.0) * 0.30
            y = 0.50 + (i % 2) * 0.005  # Légère variation neutre en Y
            hand = make_dummy_hand(x=x, y=y, timestamp=t)
            out = engine.update(hand, quality=1.0, timestamp=t)
            if out.triggered_gesture:
                triggered.append(out.triggered_gesture)

        self.assertEqual(triggered, ["SWIPE_LEFT"], "SWIPE_LEFT doit être déclenché exactement une fois.")
        self.assertEqual(engine._last_gesture, "SWIPE_LEFT")

    def test_swipe_right_detection(self):
        """Balayage rapide vers la droite -> SWIPE_RIGHT."""
        engine = TemporalGestureEngine(cooldown_duration=0.5, require_center_unlock=False)
        # Déplacement de x=0.35 à x=0.65 en 0.20s (dx = +0.30, vx = +1.5)
        t0 = 100.0
        triggered = []
        for i in range(7):
            t = t0 + i * 0.033
            x = 0.35 + (i / 6.0) * 0.30
            y = 0.50
            hand = make_dummy_hand(x=x, y=y, timestamp=t)
            out = engine.update(hand, quality=1.0, timestamp=t)
            if out.triggered_gesture:
                triggered.append(out.triggered_gesture)

        self.assertEqual(triggered, ["SWIPE_RIGHT"], "SWIPE_RIGHT doit être déclenché exactement une fois.")
        self.assertEqual(engine._last_gesture, "SWIPE_RIGHT")

    def test_swipe_up_detection(self):
        """Balayage rapide vers le haut -> SWIPE_UP."""
        engine = TemporalGestureEngine(cooldown_duration=0.5, require_center_unlock=False)
        # Déplacement de y=0.65 à y=0.35 en 0.20s (dy = -0.30 vers le haut)
        t0 = 100.0
        triggered = []
        for i in range(7):
            t = t0 + i * 0.033
            x = 0.50
            y = 0.65 - (i / 6.0) * 0.30
            hand = make_dummy_hand(x=x, y=y, timestamp=t)
            out = engine.update(hand, quality=1.0, timestamp=t)
            if out.triggered_gesture:
                triggered.append(out.triggered_gesture)

        self.assertEqual(triggered, ["SWIPE_UP"], "SWIPE_UP doit être déclenché exactement une fois.")
        self.assertEqual(engine._last_gesture, "SWIPE_UP")

    def test_swipe_down_detection(self):
        """Balayage rapide vers le bas -> SWIPE_DOWN."""
        engine = TemporalGestureEngine(cooldown_duration=0.5, require_center_unlock=False)
        # Déplacement de y=0.35 à y=0.65 en 0.20s (dy = +0.30 vers le bas)
        t0 = 100.0
        triggered = []
        for i in range(7):
            t = t0 + i * 0.033
            x = 0.50
            y = 0.35 + (i / 6.0) * 0.30
            hand = make_dummy_hand(x=x, y=y, timestamp=t)
            out = engine.update(hand, quality=1.0, timestamp=t)
            if out.triggered_gesture:
                triggered.append(out.triggered_gesture)

        self.assertEqual(triggered, ["SWIPE_DOWN"], "SWIPE_DOWN doit être déclenché exactement une fois.")
        self.assertEqual(engine._last_gesture, "SWIPE_DOWN")

    # =========================================================
    # 3. REJET DES MOUVEMENTS PARASITES ET AMBIGUS
    # =========================================================

    def test_slow_drift_rejected(self):
        """Un déplacement lent (dérive) ne doit jamais déclencher de balayage."""
        engine = TemporalGestureEngine(require_center_unlock=False)
        # Même amplitude dx=0.30 mais étalée sur 2.0 secondes (vitesse = 0.15 u/s < 0.35)
        t0 = 100.0
        triggered = []
        for i in range(20):
            t = t0 + i * 0.10
            x = 0.65 - (i / 19.0) * 0.30
            hand = make_dummy_hand(x=x, y=0.50, timestamp=t)
            out = engine.update(hand, quality=1.0, timestamp=t)
            if out.triggered_gesture:
                triggered.append(out.triggered_gesture)

        self.assertEqual(len(triggered), 0, "Un déplacement lent ne doit pas déclencher de geste.")

    def test_micro_movements_rejected(self):
        """Les micro-tremblements de la main (< min_displacement) doivent être ignorés."""
        engine = TemporalGestureEngine(require_center_unlock=False)
        t0 = 100.0
        triggered = []
        for i in range(10):
            t = t0 + i * 0.033
            # Oscillation de seulement 0.02
            x = 0.50 + (0.02 if i % 2 == 0 else -0.02)
            hand = make_dummy_hand(x=x, y=0.50, timestamp=t)
            out = engine.update(hand, quality=1.0, timestamp=t)
            if out.triggered_gesture:
                triggered.append(out.triggered_gesture)

        self.assertEqual(len(triggered), 0, "Les micro-mouvements ne doivent rien déclencher.")

    def test_diagonal_ambiguity_rejected(self):
        """Un mouvement en diagonale sans axe dominant (dx ≈ dy) doit être rejeté."""
        classifier = GestureClassifier()
        # Mouvement à 45°
        t0 = 100.0
        traj = [
            TrajectoryPoint(
                x=0.30 + i * 0.05,
                y=0.30 + i * 0.05,
                z=0.0,
                timestamp=t0 + i * 0.04,
                palm_width=0.12,
            )
            for i in range(6)
        ]
        scores = classifier.classify_trajectory(traj)
        self.assertEqual(scores[0].gesture, "UNKNOWN")
        self.assertEqual(scores[0].confidence, 0.0)

    def test_closed_palm_movement_rejected(self):
        """Un mouvement rapide effectué avec la paume fermée (poing) ne doit JAMAIS déclencher de geste."""
        engine = TemporalGestureEngine(cooldown_duration=0.5, require_center_unlock=False)
        t0 = 100.0
        triggered = []
        for i in range(7):
            t = t0 + i * 0.033
            x = 0.65 - (i / 6.0) * 0.30
            # is_palm_open = False (main fermée / poing)
            hand = make_dummy_hand(x=x, y=0.50, timestamp=t, is_palm_open=False, openness=0.15)
            out = engine.update(hand, quality=1.0, timestamp=t)
            if out.triggered_gesture:
                triggered.append(out.triggered_gesture)

        self.assertEqual(
            triggered,
            [],
            "Un balayage avec la paume fermée doit être formellement rejeté.",
        )

    def test_open_palm_movement_accepted(self):
        """Le même balayage rapide avec la paume ouverte doit déclencher le geste."""
        engine = TemporalGestureEngine(cooldown_duration=0.5, require_center_unlock=False)
        t0 = 100.0
        triggered = []
        for i in range(7):
            t = t0 + i * 0.033
            x = 0.65 - (i / 6.0) * 0.30
            # is_palm_open = True (paume déployée)
            hand = make_dummy_hand(x=x, y=0.50, timestamp=t, is_palm_open=True, openness=1.0)
            out = engine.update(hand, quality=1.0, timestamp=t)
            if out.triggered_gesture:
                triggered.append(out.triggered_gesture)

        self.assertEqual(
            triggered,
            ["SWIPE_LEFT"],
            "Un balayage avec la paume ouverte doit déclencher SWIPE_LEFT.",
        )

    # =========================================================
    # 4. SÉCURITÉ ANTI-REBOND : REJET DU MOUVEMENT DE RETOUR
    # =========================================================

    def test_return_movement_blocked_by_cooldown_and_rearm(self):
        """
        Scénario réel critique :
        1. L'utilisateur balaye à gauche -> SWIPE_LEFT déclenché.
        2. L'utilisateur ramène sa main au centre vers la droite (mouvement retour).
        -> SWIPE_RIGHT NE DOIT PAS être déclenché lors du retour !
        """
        engine = TemporalGestureEngine(cooldown_duration=0.55, rest_speed_threshold=0.25, require_center_unlock=False)

        # 1. Balayage Gauche (x=0.65 -> x=0.35)
        t = 100.0
        triggered_first = []
        for i in range(7):
            t += 0.033
            x = 0.65 - (i / 6.0) * 0.30
            hand = make_dummy_hand(x=x, y=0.50, timestamp=t)
            out = engine.update(hand, quality=1.0, timestamp=t)
            if out.triggered_gesture:
                triggered_first.append(out.triggered_gesture)

        self.assertEqual(triggered_first, ["SWIPE_LEFT"])

        # 2. Mouvement retour immédiat vers la droite (x=0.35 -> x=0.65)
        return_triggered = []
        for i in range(7):
            t += 0.033
            x = 0.35 + (i / 6.0) * 0.30
            hand = make_dummy_hand(x=x, y=0.50, timestamp=t)
            out = engine.update(hand, quality=1.0, timestamp=t)
            if out.triggered_gesture:
                return_triggered.append(out.triggered_gesture)

        self.assertEqual(
            return_triggered,
            [],
            "Le mouvement de retour de la main ne doit jamais déclencher de geste inversé !",
        )

        # 3. Main au repos après le cooldown
        t += 0.60
        hand_rest = make_dummy_hand(x=0.50, y=0.50, timestamp=t)
        rest_out = engine.update(hand_rest, quality=1.0, timestamp=t)
        self.assertIn(rest_out.active_state, ("IDLE", "SWIPING"))
        self.assertIsNone(rest_out.triggered_gesture)

    # =========================================================
    # 5. GESTION DES PERTES DE SUIVI (TRACKING LOSS)
    # =========================================================

    def test_lost_tracking_clears_trajectory_and_prevents_jump(self):
        """Si la main disparaît puis réapparaît à un autre endroit, aucun saut ne doit être pris pour un swipe."""
        engine = TemporalGestureEngine()

        # Main à gauche
        t = 100.0
        hand1 = make_dummy_hand(x=0.20, y=0.50, timestamp=t)
        engine.update(hand1, quality=1.0, timestamp=t)

        # Perte de main pendant 6 frames (> lost_tolerance_frames)
        for _ in range(6):
            t += 0.033
            out_lost = engine.update(None, quality=0.0, timestamp=t)
            self.assertEqual(out_lost.active_state, "LOST")

        # Réapparition soudaine à droite (x=0.80)
        t += 0.033
        hand2 = make_dummy_hand(x=0.80, y=0.50, timestamp=t)
        out_return = engine.update(hand2, quality=1.0, timestamp=t)

        self.assertIsNone(out_return.triggered_gesture)
        self.assertNotEqual(out_return.active_state, "TRIGGERED")

    # =========================================================
    # 6. CONTRÔLEUR D'ENTRÉES DE PRÉSENTATION (INPUT CONTROLLER)
    # =========================================================

    def test_input_controller_presentation_keys(self):
        """Vérifie le mapping des touches de présentation dans InputController."""
        controller = InputController(simulate_only=True)

        self.assertEqual(controller.key_mapping["SWIPE_RIGHT"], "right")
        self.assertEqual(controller.key_mapping["SWIPE_LEFT"], "left")
        self.assertEqual(controller.key_mapping["SWIPE_UP"], "f5")
        self.assertEqual(controller.key_mapping["SWIPE_DOWN"], "escape")

        # Test déclenchement
        res_right = controller.trigger("SWIPE_RIGHT")
        self.assertTrue(res_right)
        self.assertEqual(controller.action_counts["SWIPE_RIGHT"], 1)
        self.assertEqual(controller.last_action, "SWIPE_RIGHT")

        res_left = controller.trigger("SWIPE_LEFT")
        self.assertTrue(res_left)
        self.assertEqual(controller.action_counts["SWIPE_LEFT"], 1)

        res_up = controller.trigger("SWIPE_UP")
        self.assertTrue(res_up)
        self.assertEqual(controller.action_counts["SWIPE_UP"], 1)

        res_down = controller.trigger("SWIPE_DOWN")
        self.assertTrue(res_down)
        self.assertEqual(controller.action_counts["SWIPE_DOWN"], 1)

        # Geste inconnu ou ancienne pose -> False
        self.assertFalse(controller.trigger("INDEX"))
        self.assertFalse(controller.trigger("PINCH"))
        self.assertFalse(controller.trigger("UNKNOWN"))

    # =========================================================
    # 7. INTÉGRATION DE BOUT EN BOUT VIA GESTURE STATE MANAGER
    # =========================================================

    def test_end_to_end_pipeline(self):
        """Test d'intégration complet via la façade GestureStateManager."""
        mgr = GestureStateManager(cooldown_duration=0.5, require_center_unlock=False)

        t = 200.0
        # Exécuter un balayage vers le haut (dy < 0)
        triggered = []
        for i in range(7):
            t += 0.033
            y = 0.70 - (i / 6.0) * 0.35
            hand = make_dummy_hand(x=0.50, y=y, timestamp=t)
            output = mgr.update_full(hand, quality=1.0, timestamp=t)
            if output.triggered_gesture:
                triggered.append(output.triggered_gesture)

        self.assertEqual(triggered, ["SWIPE_UP"])
        self.assertEqual(mgr._engine._last_gesture, "SWIPE_UP")

    # =========================================================
    # 8. SÉCURITÉ DE VERROUILLAGE ENTRÉE/SORTIE DE CHAMP & DÉVERROUILLAGE CENTRAL
    # =========================================================

    def test_hand_entry_is_blocked_until_center_open_palm(self):
        """À l'entrée dans le champ, la détection est bloquée tant que la paume n'est pas ouverte au centre."""
        engine = TemporalGestureEngine(require_center_unlock=True)
        t = 100.0

        # 1. La main entre par le côté gauche (x=0.15) avec paume ouverte
        hand_edge = make_dummy_hand(x=0.15, y=0.50, timestamp=t, is_palm_open=True)
        out1 = engine.update(hand_edge, quality=1.0, timestamp=t)
        self.assertEqual(out1.active_state, "BLOCKED")
        self.assertTrue(out1.is_locked)

        # 2. Balayage rapide sur le côté sans être passé au centre
        for i in range(5):
            t += 0.033
            x = 0.15 + i * 0.05
            h = make_dummy_hand(x=x, y=0.50, timestamp=t, is_palm_open=True)
            out = engine.update(h, quality=1.0, timestamp=t)
            self.assertIsNone(out.triggered_gesture, "Aucun geste ne doit être émis tant que le centre n'a pas déverrouillé.")

    def test_closed_hand_in_center_does_not_unlock(self):
        """Une main fermée (poing) au centre ne doit PAS déverrouiller la détection."""
        engine = TemporalGestureEngine(require_center_unlock=True)
        t = 100.0

        # Poing fermé au centre (x=0.50, y=0.50, is_palm_open=False)
        for _ in range(5):
            t += 0.033
            hand_fist = make_dummy_hand(x=0.50, y=0.50, timestamp=t, is_palm_open=False, openness=0.10)
            out = engine.update(hand_fist, quality=1.0, timestamp=t)
            self.assertEqual(out.active_state, "BLOCKED")
            self.assertTrue(out.is_locked)

    def test_open_palm_in_center_unlocks_and_allows_swipes(self):
        """Présenter la paume ouverte au centre pendant 3 frames déverrouille le système."""
        engine = TemporalGestureEngine(require_center_unlock=True)
        t = 100.0

        # 1. Paume ouverte au centre pendant 3 frames
        for _ in range(3):
            t += 0.033
            hand_center = make_dummy_hand(x=0.50, y=0.50, timestamp=t, is_palm_open=True, openness=1.0)
            out = engine.update(hand_center, quality=1.0, timestamp=t)

        self.assertFalse(out.is_locked)
        self.assertEqual(out.active_state, "IDLE")

        # 2. Maintenant un balayage vers la droite doit déclencher SWIPE_RIGHT
        triggered = []
        for i in range(7):
            t += 0.033
            x = 0.40 + (i / 6.0) * 0.30
            h = make_dummy_hand(x=x, y=0.50, timestamp=t, is_palm_open=True)
            out_swipe = engine.update(h, quality=1.0, timestamp=t)
            if out_swipe.triggered_gesture:
                triggered.append(out_swipe.triggered_gesture)

        self.assertEqual(triggered, ["SWIPE_RIGHT"])

    def test_hand_leaving_and_reentering_relocks_engine(self):
        """Lorsque la main quitte le champ, le système se reverrouille et exige une réactivation centrale."""
        engine = TemporalGestureEngine(require_center_unlock=True)
        t = 100.0

        # Déverrouillage initial au centre
        for _ in range(3):
            t += 0.033
            h = make_dummy_hand(x=0.50, y=0.50, timestamp=t, is_palm_open=True)
            engine.update(h, quality=1.0, timestamp=t)
        self.assertFalse(engine._locked)

        # La main quitte le champ pendant 12 frames (> lost_tolerance_frames)
        for _ in range(12):
            t += 0.033
            out_lost = engine.update(None, quality=0.0, timestamp=t)
            self.assertEqual(out_lost.active_state, "LOST")

        # La main revient sur le bord : le système DOIT être reverrouillé
        t += 0.033
        hand_reentry = make_dummy_hand(x=0.10, y=0.50, timestamp=t, is_palm_open=True)
        out_reentry = engine.update(hand_reentry, quality=1.0, timestamp=t)
        self.assertTrue(out_reentry.is_locked)
        self.assertEqual(out_reentry.active_state, "BLOCKED")

    def test_hand_raising_motion_does_not_trigger_false_swipe_up(self):
        """Lever la main du bas vers le centre pour déverrouiller ne doit PAS déclencher de faux SWIPE_UP (F5)."""
        engine = TemporalGestureEngine(require_center_unlock=True)
        t = 100.0

        triggered = []
        # La main monte du bas (y=0.85 -> y=0.50) vers le centre
        for i in range(5):
            t += 0.033
            y = 0.85 - (i / 4.0) * 0.35
            h = make_dummy_hand(x=0.50, y=y, timestamp=t, is_palm_open=True)
            out = engine.update(h, quality=1.0, timestamp=t)
            if out.triggered_gesture:
                triggered.append(out.triggered_gesture)

        self.assertEqual(
            triggered,
            [],
            "Le mouvement de montée de la main vers le centre ne doit jamais déclencher de SWIPE_UP !",
        )
        self.assertFalse(engine._locked)


if __name__ == "__main__":
    unittest.main()
