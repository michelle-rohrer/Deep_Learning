# Deep Learning Challenge: Emotionserkennung mit CNN (FER-2013)

Deep-Learning-Projekt zur Klassifikation von Gesichtsausdrücken mit Convolutional Neural Networks (CNN) auf Basis von FER-2013.

## Quickstart

```bash
# 1) Environment
conda create -n deep_learning python=3.11
conda activate deep_learning

# 2) Dependencies
pip install -r requirements.txt

# 3) Notebook starten
jupyter notebook main.ipynb
```

Weitere Experimente (Training/Hyperparameter-Tuning) sind in `training.ipynb` dokumentiert.

## Projektübersicht

**Aufgabe:** Klassifikation von Gesichtsausdrücken in 7 Emotionen:
- 0: Angry (Wütend)
- 1: Disgust (Ekel)
- 2: Fear (Angst)
- 3: Happy (Glücklich)
- 4: Sad (Traurig)
- 5: Surprise (Überrascht)
- 6: Neutral (Neutral)

**Datensatz:** FER-2013 (48×48 Pixel, Graustufen)
- In diesem Repo liegt der Datensatz bereits unter `data/` im Format `train/<klasse>/...` und `test/<klasse>/...`.
- Die konkreten Bildzahlen entsprechen dem aktuellen Stand im Ordner `data/` (siehe “Projektstruktur”).

### Optional: Datensatz neu herunterladen

Falls `data/` bei dir fehlt oder du den Datensatz neu beziehen willst:
- **Manuell**: FER-2013 aus der Quelle deiner Wahl laden und als Ordnerstruktur `data/train/<klasse>/...` und `data/test/<klasse>/...` ablegen.
- **Per KaggleHub (Beispiel)**: In `src/data_load.py` ist ein Download-Beispiel hinterlegt, das den lokalen Download-Pfad ausgibt.

## Setup

### Option 1: Conda Environment
```bash
conda create -n deep_learning python=3.11
conda activate deep_learning
pip install -r requirements.txt
```

### Option 2: Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Optional: Weights & Biases (wandb)

Wenn du Experiment-Tracking mit `wandb` nutzen willst, richte die Authentifizierung ein (z.B. via Environment-Variable oder `.env`).

Beispiel `.env` (nicht committen):

```bash
WANDB_API_KEY=dein_api_key
```

## Projektstruktur
```
Deep_Learning/
├── data/
│   ├── test/              # Test-Dataset (7.178 Bilder)
│   │   ├── angry/         # 958 Bilder
│   │   ├── disgust/       # 6 Bilder
│   │   ├── fear/          # 1.024 Bilder
│   │   ├── happy/         # 1.774 Bilder
│   │   ├── neutral/       # 1.233 Bilder
│   │   ├── sad/           # 1.247 Bilder
│   │   └── surprise/      # 831 Bilder
│   └── train/             # Training-Dataset (28.592 Bilder)
│       ├── angry/         # 3.995 Bilder
│       ├── disgust/       # 319 Bilder
│       ├── fear/          # 4.097 Bilder
│       ├── happy/         # 7.215 Bilder
│       ├── neutral/       # 4.965 Bilder
│       ├── sad/           # 4.830 Bilder
│       └── surprise/      # 3.171 Bilder
├── models/
│   ├── baseline_model.pth   # Trainiertes Basismodell
│   └── final_best_model.pth # Bestes Modell (aktueller Stand)
├── plots/
│   ├── baseline_confusion_matrix.png      # Konfusionsmatrix
│   ├── baseline_training_curves.png       # Lernkurven
│   ├── hyperparameter_comparison.png      # Hyperparameter-Tuning
│   └── cross_validation_results.png       # Cross-Validation
├── results/
│   └── cross_validation_results.json      # CV-Ergebnisse
├── src/
│   ├── data_load.py       # (optional) Dataset-Download (KaggleHub)
│   ├── evaluation.py      # Modell-Evaluation
│   ├── experiments.py     # Experiment-Utilities
│   ├── model.py           # CNN-Architektur
│   ├── plots.py           # Visualisierung
│   └── test_train.py      # Training-Funktionen
├── setup_config.py        # Device/Worker-Setup (MPS/CUDA/CPU)
├── task/
│   └── SGDS_DEL_MC.pdf   # Aufgabenstellung
├── wandb/                # (optional) Experiment-Logs (Weights & Biases)
├── main.ipynb            # Hauptnotebook für Analyse
├── training.ipynb        # Training und Hyperparameter-Tuning
├── requirements.txt      # Python-Abhängigkeiten
├── .gitignore
└── README.md
```

## Ausführung

### Hauptanalyse starten
```bash
jupyter notebook main.ipynb
```

### Training / Hyperparameter-Tuning

- Für Training und Tuning: `training.ipynb` öffnen und ausführen.
- Vortrainierte Modelle liegen unter `models/`.

## Autor

**Michelle Rohrer**  
Deep Learning Challenge – Fachhochschule Nordwestschweiz (FHNW), Data Science & Artificial Intelligence

## Lizenz

Dieses Projekt wurde im Rahmen einer Hochschul-Challenge erstellt.