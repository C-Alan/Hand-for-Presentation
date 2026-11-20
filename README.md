# Hand Control — Contrôle de Présentation par Gestes de la Main

Application Python de contrôle sans contact de présentations de documents (PowerPoint, PDF, Google Slides, Keynote, visionneuses d'images) par reconnaissance de **gestes dynamiques de la main avec précondition de paume ouverte** via une webcam standard.

---

## 🚀 Démarrage rapide

### 1. Lancement de l'application
```powershell
.\env\Scripts\python.exe main.py
```
*Appuyez sur la touche **Q** ou **Échap (`ESC`)** pour quitter la fenêtre.*

### 2. Exécution des tests unitaires automatisés
```powershell
.\env\Scripts\python.exe -m unittest discover -s tests -v
```

---

## ✋ Commandes et Gestes Dynamiques

| Geste Dynamique | Condition | Mouvement physique | Touche clavier | Action de présentation |
|---|---|---|:---:|---|
| **`SWIPE_RIGHT`** | **Paume ouverte** | Balayage horizontal vif vers la **droite** | `Right` (ou `PageDown`) | **Diapositive Suivante** / Page suivante (avancer) |
| **`SWIPE_LEFT`** | **Paume ouverte** | Balayage horizontal vif vers la **gauche** | `Left` (ou `PageUp`) | **Diapositive Précédente** / Page précédente (reculer) |
| **`SWIPE_UP`** | **Paume ouverte** | Balayage vertical vif vers le **haut** | `F5` | **Lancer le Diaporama** / Plein écran |
| **`SWIPE_DOWN`** | **Paume ouverte** | Balayage vertical vif vers le **bas** | `Escape` | **Quitter le Diaporama** / Sortie plein écran |

> **Sécurité « Paume Ouverte »** : Les balayages ne sont pris en compte que si votre main est **ouverte face à la caméra** (doigts déployés). Si votre main est fermée en poing, ou si vous tenez un objet en main (stylo, tasse, télécommande), tout balayage involontaire est automatiquement neutralisé.
>
> **Sécurité d'Entrée/Sortie (Verrou & Armement)** : Dès que la main entre ou réentre dans le champ de la webcam, la détection est **bloquée** (`BLOCKED`). Pour l'activer, il suffit de présenter sa **paume ouverte** dans la zone de visée pendant 2 images. Une fois la main stabilisée au repos, le système s'arme et vous permet de balayer **librement sur tout l'écran**.

---

## 📂 Structure du projet

```text
App_P/
├── Docs/
│   ├── DOCUMENTATION.md        # Documentation technique complète et architecture
│   ├── REGLES_DE_TRANSITION.md # Spécification de la machine à états et cinématique
│   └── Catalogue_Gestes_Hand_Control.pdf # Référentiel initial de conception
├── gestures/
│   ├── landmark_processor.py   # Normalisation en espace caméra [0..1] & ouverture de paume
│   ├── finger_state_analyzer.py# Extension continue des doigts (pouce, index, majeur, etc.)
│   ├── gesture_classifier.py   # Analyse cinématique des balayages et verrou paume ouverte
│   ├── temporal_engine.py      # Moteur temporel, scan multi-fenêtres & cooldown anti-rebond
│   ├── input_controller.py     # Émission des touches de présentation Windows (PyAutoGUI)
│   ├── diagnostic.py           # HUD temps réel (traînée cinématique, jauge, bannières)
│   ├── transition_manager.py   # Journalisation CSV (logs/transitions.csv)
│   ├── gesture_detector.py     # Façade de détection
│   └── gesture_state.py        # Façade d'état
├── tests/
│   └── test_dynamic_gestures.py# Suite de 20 tests unitaires validant l'ensemble du pipeline
├── logs/
│   └── transitions.csv         # Journal d'événements et transitions
├── hand_landmarker.task        # Modèle d'inférence MediaPipe Hand Landmarker
├── main.py                     # Script principal (capture webcam OpenCV & boucle temps réel)
└── README.md                   # Ce fichier
```

---

## 📖 Documentation détaillée

* Consultez [Docs/DOCUMENTATION.md](file:///c:/Users/LOU/Documents/CDAR/Hand/App_P/Docs/DOCUMENTATION.md) pour les détails complets de conception, d'architecture et de paramétrage.
* Consultez [Docs/REGLES_DE_TRANSITION.md](file:///c:/Users/LOU/Documents/CDAR/Hand/App_P/Docs/REGLES_DE_TRANSITION.md) pour les formules cinématiques, diagrammes Mermaid de la machine à états et sécurités anti-rebond.
