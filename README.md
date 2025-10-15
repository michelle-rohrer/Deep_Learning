# Analyse Gesichtsausdrücke

## Setup
### Option 1: Conda Environment
conda create -n NAME_OF_ENVIRONMENT python=3.11
conda activate NAME_OF_ENVIRONMENT

### Option 2: Virtual Environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Projektstruktur
project/
├── data/
│   ├── test/   
│   └── train/
├── doc/
│   └── data_load.py
├── src/
│   └── evaluation.py
│   └── model.py
│   └── plots.py
│   └── test_train.py
├── main.ipynb
├── traiining.ipynb
├── environment.yml
├── requirements.txt
├── .env
├── .gitignore
└── README.md