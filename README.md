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

## Déploiement Streamlit Cloud
1. Poussez ce repo sur GitHub (déjà fait).
2. Rendez-vous sur https://share.streamlit.io/ et connectez votre compte GitHub.
3. Sélectionnez ce repo et le fichier `app/app_streamlit.py`.
4. Ajoutez un secret ou un stockage externe pour les données si besoin.

## Remarques
- Les images/données ne sont pas versionnées (voir `.gitignore`).
- Le modèle fourni (`best_model.h5`) est prêt à l’emploi.
- Pour toute question, ouvrez une issue sur GitHub.
