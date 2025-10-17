# Deep Learning Challenge: Emotionserkennung mit CNN

Ein Deep Learning Projekt zur Klassifikation von Gesichtsausdrücken mit Convolutional Neural Networks (CNN) auf dem FER-2013 Datensatz.

## Projektübersicht

**Aufgabe:** Klassifikation von Gesichtsausdrücken in 7 Emotionen:
- 0: Angry (Wütend)
- 1: Disgust (Ekel)  
- 2: Fear (Angst)
- 3: Happy (Glücklich)
- 4: Sad (Traurig)
- 5: Surprise (Überrascht)
- 6: Neutral (Neutral)

**Datensatz:** FER-2013 (Facial Expression Recognition 2013)
- **Training:** 28.709 Bilder
- **Test:** 3.589 Bilder
- **Auflösung:** 48x48 Pixel Graustufenbilder

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

## Projektstruktur
```
project/
├── data/
│   ├── test/              # Test-Dataset (7.178 Bilder)
│   │   ├── angry/         # 958 Bilder
│   │   ├── disgust/       # 111 Bilder
│   │   ├── fear/          # 1.024 Bilder
│   │   ├── happy/         # 1.774 Bilder
│   │   ├── neutral/       # 1.233 Bilder
│   │   ├── sad/           # 1.247 Bilder
│   │   └── surprise/      # 831 Bilder
│   └── train/             # Training-Dataset (28.709 Bilder)
│       ├── angry/         # 3.995 Bilder
│       ├── disgust/       # 436 Bilder
│       ├── fear/          # 4.097 Bilder
│       ├── happy/         # 7.215 Bilder
│       ├── neutral/       # 4.965 Bilder
│       ├── sad/           # 4.830 Bilder
│       └── surprise/      # 3.171 Bilder
├── models/
│   └── baseline_model.pth # Trainiertes Basismodell
├── plots/
│   ├── baseline_confusion_matrix.png      # Konfusionsmatrix
│   ├── baseline_training_curves.png       # Lernkurven
│   ├── hyperparameter_comparison.png      # Hyperparameter-Tuning
│   └── cross_validation_results.png       # Cross-Validation
├── results/
│   └── cross_validation_results.json      # CV-Ergebnisse
├── src/
│   ├── data_load.py       # Datenladung und Preprocessing
│   ├── evaluation.py      # Modell-Evaluation
│   ├── model.py           # CNN-Architektur
│   ├── plots.py           # Visualisierung
│   └── test_train.py      # Training-Funktionen
├── task/
│   └── SGDS_DEL_MC.pdf   # Aufgabenstellung
├── wandb/                # Experiment Tracking (Weights & Biases)
├── main.ipynb            # Hauptnotebook für Analyse
├── training.ipynb        # Training und Hyperparameter-Tuning
├── requirements.txt      # Python-Abhängigkeiten
├── .env                  # Umgebungsvariablen (WanDB Key)
├── .gitignore
└── README.md
```

## Ausführung

### Hauptanalyse starten
```bash
jupyter notebook main.ipynb
```

## Autor

**Michelle Rohrer**  
Deep Learning Challenge - Fachhochachule Nordwestschweiz Data Science & Artificial Intelligence

## Lizenz

Dieses Projekt wurde im Rahmen einer universitären Deep Learning Challenge erstellt.