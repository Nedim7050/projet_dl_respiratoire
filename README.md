# Projet Deep Learning Respiratoire

Ce projet permet de classifier des images pulmonaires (COVID19, NORMAL, PNEUMONIA) à l’aide d’un modèle de deep learning et d’une interface Streamlit.

## Structure du projet

- `app/app_streamlit.py` : Application web Streamlit pour l’inférence.
- `src/train_model.py` : Script d’entraînement du modèle.
- `best_model.h5` : Modèle entraîné (fourni, <100 Mo).
- `class_indices.json` : Mapping des classes.
- `requirements.txt` : Dépendances Python.
- `data/` : (Non inclus dans le repo) Placez ici vos images d’entraînement/test.

## Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/Nedim7050/projet_dl_respiratoire.git
cd projet_dl_respiratoire
```
2. Installez les dépendances :
```bash
pip install -r requirements.txt
```
3. Ajoutez vos données dans le dossier `data/` (structure : `data/train/COVID19/`, `data/train/NORMAL/`, etc.).

## Utilisation

### Lancer l’application Streamlit
```bash
streamlit run app/app_streamlit.py
```

### (Optionnel) Réentraîner le modèle
```bash
python src/train_model.py
```
