import cv2
import mediapipe as mp
import os
import threading
import time

import numpy as np

from gestures.gesture_detector import GestureDetector
from gestures.gesture_state    import GestureStateManager
from gestures.input_controller import InputController
from gestures.diagnostic       import DiagnosticOverlay

# =====================================================
# CONFIGURATION
# =====================================================

CAMERA_INDEX = 0
MODEL_PATH   = "hand_landmarker.task"
NUM_HANDS    = 1               # 1 main suivie à la fois
WINDOW_NAME  = "Hand Control — Presentation de Document"
LOG_DIR      = "logs"

# =====================================================
# MEDIAPIPE
# =====================================================

BaseOptions          = mp.tasks.BaseOptions
HandLandmarker       = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode    = mp.tasks.vision.RunningMode

latest_result = None
result_lock   = threading.Lock()


def on_result(result, output_image, timestamp_ms):
    global latest_result
    with result_lock:
        latest_result = result


# =====================================================
# BOUCLE PRINCIPALE
# =====================================================

def main():
    global latest_result

    if not os.path.exists(MODEL_PATH):
        print(f"Erreur : modèle introuvable : {MODEL_PATH}")
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "transitions.csv")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=on_result,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector    = GestureDetector()
    state_mgr   = GestureStateManager(log_path=log_path)
    controller  = InputController(
        key_next="right",
        key_prev="left",
        key_fullscreen="f5",
        key_exit="escape",
    )
    overlay     = DiagnosticOverlay(enabled=True)

    prev_time  = time.perf_counter()

    print("==================================================")
    print("  HAND CONTROL — MODE PRÉSENTATION DE DOCUMENT   ")
    print("==================================================")
    print("Gestes dynamiques actifs :")
    print("  - SWIPE_RIGHT  : Diapositive Suivante  [Right]")
    print("  - SWIPE_LEFT   : Diapositive Précédente [Left]")
    print("  - SWIPE_UP     : Démarrer Diaporama   [F5]")
    print("  - SWIPE_DOWN   : Quitter Diaporama    [Échap]")
    print("--------------------------------------------------")
    print("Appuie sur Q ou ESC pour quitter.")
    print(f"Événements enregistrés dans : {log_path}")
    print("==================================================")

    try:
        with HandLandmarker.create_from_options(options) as landmarker:

            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Erreur de lecture webcam.")
                    break

                # Effet miroir naturel (selfie)
                frame = cv2.flip(frame, 1)
                h, w  = frame.shape[:2]

                rgb = np.ascontiguousarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
                mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = time.perf_counter_ns() // 1_000_000

                landmarker.detect_async(mp_image, timestamp_ms)

                with result_lock:
                    current_result = latest_result

                # ── Calcul FPS ──────────────────────────
                now      = time.perf_counter()
                fps      = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                # ── Traitement des mains ─────────────────
                hand_count = 0
                engine_out = None

                if current_result is not None:
                    hand_count = len(current_result.hand_landmarks)

                    if hand_count > 0:
                        landmarks = current_result.hand_landmarks[0]
                        world_lms = (
                            current_result.hand_world_landmarks[0]
                            if current_result.hand_world_landmarks
                            else None
                        )

                        # Traitement et normalisation de la main
                        hand = detector.process_hand(
                            landmarks,
                            world_landmarks=world_lms,
                            image_aspect=w / h,
                        )

                        # Mise à jour du moteur temporel
                        engine_out = state_mgr.update_full(hand, hand.quality)

                        # Exécution des actions de présentation
                        if engine_out.triggered_gesture:
                            controller.process(engine_out)

                # ── Perte de main ───────────────────────
                if hand_count == 0:
                    engine_out = state_mgr.update_full(None, 0.0)

                # ── Diagnostic overlay ──────────────────
                if engine_out is not None:
                    overlay.draw(frame, engine_out, fps, hand_count)

                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    except Exception as err:
        print(f"Erreur : {err}")
        import traceback
        traceback.print_exc()

    finally:
        controller.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Hand Control arrêté.")
        print(f"Transitions enregistrées dans : {log_path}")


if __name__ == "__main__":
    main()
