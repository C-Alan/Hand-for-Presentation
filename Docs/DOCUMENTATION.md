# Documentation Technique — Hand Control (Mode Présentation de Document)

## 1. Présentation du projet

**Hand Control** est une application Python permettant de piloter des présentations de documents (PowerPoint, PDF, Google Slides, Keynote, visionneuses d'images) sans contact, par reconnaissance de **gestes dynamiques spatio-temporels de la main avec précondition de paume ouverte** via une webcam standard.

### Caractéristiques principales
* **Zéro contact physique** : Contrôle fluide du diaporama à distance (1 à 3 mètres de la webcam).
* **Gestes dynamiques naturels** : Remplacement des anciennes poses statiques de souris par des balayages intuitifs (`SWIPE_LEFT`, `SWIPE_RIGHT`, `SWIPE_UP`, `SWIPE_DOWN`).
* **Verrou de sécurité « Paume Ouverte »** : Seuls les gestes effectués avec la paume ouverte déployée sont pris en compte. Les mouvements de bras avec poing fermé, main repliée ou objet en main (stylo, télécommande, tasse) sont systématiquement ignorés.
* **Verrou d'Entrée / Sortie de Champ (`BLOCKED`)** : Chaque fois que la main entre ou réentre dans le champ de la caméra, la détection est bloquée. Elle n'est déverrouillée qu'en présentant sa paume ouverte au centre du champ de vision.
* **Protection Anti-Rebond** : Rejet formel du mouvement de retour de la main grâce à un cooldown temporel et une exigence de retour au repos.
* **Affichage tête haute (HUD) en temps réel** : Traînée cinématique de la main, indicateur d'ouverture de la paume, jauge de vitesse instantanée, barre de disponibilité et bannières visuelles grand format.

---

## 2. Prérequis et lancement

### Prérequis système
* **Système d'exploitation** : Windows 10 / 11, macOS ou Linux.
* **Webcam** : 720p ou 1080p standard (30 FPS recommandés).
* **Environnement Python** : `env\` contenant `mediapipe`, `opencv-python`, `numpy`, `pyautogui`.
* **Modèle d'inférence** : `hand_landmarker.task` placé à la racine du projet.

### Commandes utiles

* **Démarrer l'application** :
  ```powershell
  .\env\Scripts\python.exe main.py
  ```
  Appuyer sur la touche **Q** ou **Échap (`ESC`)** pour quitter la fenêtre de capture.

* **Lancer l'ensemble des tests automatisés (20 tests unitaires)** :
  ```powershell
  .\env\Scripts\python.exe -m unittest discover -s tests -v
  ```

---

## 3. Architecture modulaire du système

```text
Webcam (OpenCV)
   │
   ▼
MediaPipe HandLandmarker (LIVE_STREAM)
   │  landmarks (21 repères caméra normalisés [0.0 → 1.0])
   ▼
GestureDetector (façade de traitement)
   ├── LandmarkProcessor       : Coordonnées caméra [0..1], lissage EMA, largeur de paume,
   │                             qualité stable et évaluation morphologique d'ouverture
   │                             via FingerStateAnalyzer
   └── GestureClassifier       : Analyse cinématique (vitesse, déplacement, ratio directionnel, rectitude)
                                 + VERROU MORPHOLOGIQUE : Rejet si la paume n'est pas ouverte
   │
   ▼  EngineOutput
GestureStateManager (façade d'état)
   ├── TemporalGestureEngine   : Buffer glissant de trajectoire (0.50s), scan multi-fenêtres
   │                             (80–400 ms), cooldown anti-rebond (0.45s) et réarmement au repos
   └── GestureTransitionManager: Enregistrement et journalisation CSV (logs/transitions.csv)
   │
   ▼  Événements clavier de présentation
InputController (Raccourcis)  : Touches clavier Windows (Right, Left, F5, Escape) avec mode simulation
DiagnosticOverlay (HUD)       : Traînée lumineuse de la main, état paume ouverte/fermée, jauge cinématique
```

### Rôle des modules (`gestures/`)

| Fichier | Rôle technique |
|---|---|
| `main.py` | Orchestration générale : capture webcam OpenCV, inférence asynchrone MediaPipe, injection miroir et affichage HUD. |
| `gestures/landmark_processor.py` | Extraction des coordonnées caméra $[0..1]$, lissage EMA, calcul de `palm_width` et évaluation continue de l'ouverture de la paume (`is_palm_open`). |
| `gestures/finger_state_analyzer.py` | Analyse d'extension des 5 doigts (scores continus $[0..1]$ et états `EXTENDED`, `BENT`, `CURLED`) invariante paume/dos. |
| `gestures/gesture_classifier.py` | Analyse cinématique des trajectoires : calcul du déplacement net, de la vitesse vectorielle, de la rectitude et vérification stricte de la paume ouverte. |
| `gestures/temporal_engine.py` | Moteur temporel gérant la file glissante, le scan multi-fenêtres pour détection instantanée, le cooldown anti-rebond et le réarmement au repos. |
| `gestures/input_controller.py` | Traduction des gestes confirmés en raccourcis clavier de présentation (`Right`, `Left`, `F5`, `Escape`) avec historique et compteur d'actions. |
| `gestures/diagnostic.py` | Overlay HUD : traînée cinématique avec dégradé de couleur, indicateur d'état paume, jauge de vitesse, barre de cooldown et bannières grand format. |
| `gestures/transition_manager.py` | Historique en mémoire et export CSV (`logs/transitions.csv`) des transitions d'états et gestes déclenchés. |
| `gestures/gesture_detector.py` | Façade reliant le processeur de repères et le classifieur cinématique. |
| `gestures/gesture_state.py` | Façade reliant le moteur temporel et le gestionnaire de transition. |

---

## 4. Catalogue des gestes dynamiques de présentation

| Geste Dynamique | Condition morphologique | Trajectoire cinématique | Touche clavier | Action de présentation |
|---|---|---|:---:|---|
| **`SWIPE_RIGHT`** | **Paume ouverte** | Balayage horizontal vif vers la **droite** ($\Delta x > +0.08$, $\|\Delta x\| \ge 1.25 \|\Delta y\|$) | **`Right`** (ou `PageDown`) | **Diapositive Suivante** / Page suivante (avancer) |
| **`SWIPE_LEFT`** | **Paume ouverte** | Balayage horizontal vif vers la **gauche** ($\Delta x < -0.08$, $\|\Delta x\| \ge 1.25 \|\Delta y\|$) | **`Left`** (ou `PageUp`) | **Diapositive Précédente** / Page précédente (reculer) |
| **`SWIPE_UP`** | **Paume ouverte** | Balayage vertical vif vers le **haut** ($\Delta y < -0.08$, $\|\Delta y\| \ge 1.25 \|\Delta x\|$) | **`F5`** | **Lancer Diaporama** / Mode Plein écran |
| **`SWIPE_DOWN`** | **Paume ouverte** | Balayage vertical vif vers le **bas** ($\Delta y > +0.08$, $\|\Delta y\| \ge 1.25 \|\Delta x\|$) | **`Escape`** | **Quitter Diaporama** / Sortie du plein écran |

---

## 5. Spécifications des algorithmes et sécurités

### A. Ancrage en coordonnées caméra $[0.0 \rightarrow 1.0]$
Contrairement aux coordonnées locales `world_landmarks` (dont l'origine est fixée au centre de la main et qui restent immobiles à $(0, 0, 0)$ lors des translations), le système utilise les **coordonnées d'image normalisées** de MediaPipe :
* $x \in [0.0, 1.0]$ de gauche à droite de la caméra.
* $y \in [0.0, 1.0]$ du haut vers le bas de la caméra.
* Le point de suivi cinématique est le **landmark 9** (base du majeur, centre géométrique stable de la paume).

### B. Précondition stricte de Paume Ouverte (`require_open_palm`)
Pour que la trajectoire soit prise en compte :
1. Au moins 3 doigts parmi l'index, le majeur, l'annulaire et l'auriculaire doivent être déployés ($\text{score} \ge 0.48$).
2. La moyenne d'extension de ces 4 doigts doit être $\ge 0.48$.
3. La paume doit être ouverte sur au moins $60\,\%$ de la trajectoire et **strictement ouverte sur le point d'impact final**.
*Si la main est fermée en poing, ou si l'utilisateur tient un objet en main, le mouvement est neutralisé avec le motif `reason: palm_not_open`.*

### C. Détection cinématique multi-fenêtres
Pour éviter que l'immobilité préalable de la main ne dilue la vitesse moyenne d'un balayage, le moteur scanne en temps réel des sous-fenêtres temporelles de $80\text{ ms}$ à $400\text{ ms}$ (3 à 12 images) se terminant à la frame actuelle :
* **Déplacement net minimal** : $\text{distance} \ge 0.08$ (8 % de l'écran).
* **Vitesse minimale** : $v \ge 0.28\text{ unité/s}$.
* **Dominance orthogonale** : $\frac{|\Delta x|}{|\Delta y|} \ge 1.25$ (balayage horizontal) ou $\frac{|\Delta y|}{|\Delta x|} \ge 1.25$ (balayage vertical).
* **Linéarité de trajectoire** : $\text{Efficacité} = \frac{\text{Distance nette}}{\text{Longueur de trajectoire}} \ge 0.55$.

### D. Protection Anti-Rebond (Rejet du mouvement retour)
Lorsqu'un utilisateur balaye vers la gauche, sa main revient naturellement vers le centre (mouvement physique vers la droite). Pour interdire tout déclenchement parasite :
1. Le buffer de trajectoire est **immédiatement vidé** dès qu'un balayage est validé.
2. Le système entre en **`COOLDOWN` pendant $0.45\text{ seconde}$** : aucun geste ne peut être émis.
3. À l'issue du cooldown, le réarmement exige que la vitesse de la main retombe sous le seuil de repos ($< 0.30\text{ u/s}$).

### E. Verrou d'Entrée / Sortie de Champ & Déverrouillage Central
Pour éliminer les faux gestes involontaires lors de l'entrée de la main dans le champ de la caméra ou après une absence :
1. **Verrouillage systématique (`BLOCKED`)** : À l'entrée ou réentrée de la main dans le champ (ou démarrage de l'app), l'état est verrouillé. Aucune analyse de swipe n'est effectuée.
2. **Déverrouillage par Présentation Centrale** :
   * La main doit se positionner dans la boîte centrale de l'écran : $x \in [0.28, 0.72]$, $y \in [0.22, 0.78]$.
   * La paume doit y être maintenue **OUVERTE** pendant au moins 3 frames consécutives ($\approx 100\text{ ms}$).
   * Dès déverrouillage, le système passe en `IDLE` et autorise les swipes.
   * Si la main quitte le champ (> 6 frames d'absence), le système se re-verrouille automatiquement.

---

## 6. Personnalisation des raccourcis et paramètres

Les touches et comportements peuvent être configurés dans [gestures/input_controller.py](file:///c:/Users/LOU/Documents/CDAR/Hand/App_P/gestures/input_controller.py) ou lors de l'instanciation dans `main.py` :

```python
controller = InputController(
    key_next="right",         # Diapositive suivante (ex: "pagedown", "space")
    key_prev="left",          # Diapositive précédente (ex: "pageup")
    key_fullscreen="f5",      # Lancer diaporama (ex: "f11", "ctrl+l")
    key_exit="escape",        # Quitter plein écran
    simulate_only=False,      # True pour tests sans émission réelle de frappe
)
```

Principaux paramètres de détection dans `TemporalGestureEngine` et `GestureClassifier` :
* `min_displacement` (défaut : `0.08`) : amplitude minimale requise du geste.
* `min_velocity` (défaut : `0.28`) : vivacité minimale du geste.
* `cooldown_duration` (défaut : `0.45`) : durée de blocage post-action.
* `rest_speed_threshold` (défaut : `0.30`) : vitesse de repos pour le réarmement.
* `require_center_unlock` (défaut : `True`) : verrouillage obligatoire à l'entrée/sortie de champ.
* `center_box` (défaut : `(0.20, 0.15, 0.80, 0.85)`) : zone de visée centrale élargie et ergonomique.
* `directional_ratio` (défaut : `1.15`) : ratio orthogonal assoupli pour les balayages horizontaux naturels.

---

## 7. Validation et couverture des tests automatisés

La suite de tests unitaires dans [tests/test_dynamic_gestures.py](file:///c:/Users/LOU/Documents/CDAR/Hand/App_P/tests/test_dynamic_gestures.py) valide **20 scénarios critiques** :

1. `test_static_poses_completely_removed` : Vérifie l'absence des 5 anciennes poses statiques (`INDEX`, `PINCH`, `SCROLL`, `FIST`, `OPEN_PALM`).
2. `test_camera_space_preservation_over_world_landmarks` : Vérifie que le centre de la main est bien suivi en coordonnées caméra et n'est pas figé par `world_landmarks`.
3. `test_swipe_left_detection` : Validation du balayage gauche (`SWIPE_LEFT`).
4. `test_swipe_right_detection` : Validation du balayage droite (`SWIPE_RIGHT`).
5. `test_swipe_up_detection` : Validation du balayage haut (`SWIPE_UP`).
6. `test_swipe_down_detection` : Validation du balayage bas (`SWIPE_DOWN`).
7. `test_closed_palm_movement_rejected` : Rejet formel d'un déplacement rapide effectué avec la main fermée (poing).
8. `test_open_palm_movement_accepted` : Acceptation du balayage rapide avec la paume ouverte.
9. `test_slow_drift_rejected` : Rejet des dérives lentes du bras ($v < 0.28$).
10. `test_micro_movements_rejected` : Rejet des tremblements et micro-mouvements ($d < 0.08$).
11. `test_diagonal_ambiguity_rejected` : Rejet des diagonales sans axe dominant.
12. `test_return_movement_blocked_by_cooldown_and_rearm` : Validation de la protection anti-rebond sur le mouvement retour.
13. `test_lost_tracking_clears_trajectory_and_prevents_jump` : Réinitialisation du buffer lors d'une sortie/réentrée de champ.
14. `test_hand_entry_is_blocked_until_center_open_palm` : Vérifie que l'entrée de la main dans le champ est bloquée tant que le centre n'a pas été activé.
15. `test_closed_hand_in_center_does_not_unlock` : Vérifie qu'un poing fermé au centre ne déverrouille jamais la détection.
16. `test_open_palm_in_center_unlocks_and_allows_swipes` : Vérifie le déverrouillage effectif après présentation de la paume ouverte au centre.
17. `test_hand_leaving_and_reentering_relocks_engine` : Vérifie le re-verrouillage automatique dès que la main sort du champ de vision.
18. `test_hand_raising_motion_does_not_trigger_false_swipe_up` : Vérifie que la montée de la main vers le centre pour déverrouiller ne déclenche pas de faux SWIPE_UP (F5).
19. `test_input_controller_presentation_keys` : Contrôle des touches associées aux gestes.
20. `test_end_to_end_pipeline` : Intégration complète de bout en bout via `GestureStateManager`.
