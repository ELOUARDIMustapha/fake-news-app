# Fake News Detection - PFE Master

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ELOUARDIMustapha/fake-news-app/blob/main/mon_projet.ipynb)

Projet de fin d'études (Master) : détection automatique des fausses informations (*fake news*) dans des articles de presse en anglais, à l'aide du traitement automatique du langage (NLP), du Machine Learning et du Deep Learning.

## Problématique

La diffusion rapide de fausses informations sur Internet et les réseaux sociaux rend leur vérification manuelle impossible à grande échelle. L'objectif est de construire un modèle capable de classer un article comme **FAKE** ou **REAL** à partir de son seul texte, puis de le rendre utilisable via une interface web.

## Dataset

Dataset **Fake and Real News** (ISOT), composé de deux fichiers :

| Fichier    | Contenu                         | Label |
|------------|---------------------------------|-------|
| `Fake.csv` | Articles de fausses informations | `0`   |
| `True.csv` | Articles réels (Reuters)         | `1`   |

- 44 898 articles au total, dont 6 251 doublons supprimés et les textes de moins de 10 caractères retirés, soit environ **38 600 articles** après nettoyage.
- Seule la colonne `text` est utilisée (`title`, `subject` et `date` sont supprimées).

Le dataset n'est pas inclus dans ce dépôt. Le notebook l'attend sous forme d'archive `dataset_fake_news.zip` (contenant `Fake.csv` et `True.csv`) dans Google Drive, au chemin `MyDrive/mon_projet/`.

## Démarche

Le notebook est organisé en sections numérotées :

0. **Configuration** : tous les paramètres (chemins, taille du vocabulaire, nombre d'époques…) sont regroupés dans une seule cellule.
1. **Chargement** des données depuis Google Drive, suppression des doublons et des textes vides.
2. **Analyse exploratoire** : répartition des classes, longueur des textes, vérification du biais « (Reuters) ».
3. **Prétraitement** dans un fichier `preprocessing.py`, utilisé à la fois pour l'entraînement et par l'application :
   suppression du préfixe de source, minuscules, liens, chiffres, ponctuation, *stopwords*, lemmatisation (WordNet).
4. **Visualisation du vocabulaire** : mots les plus fréquents et nuage de mots, pour chaque classe.
5. **Séparation train / test** (80 / 20, stratifiée) : un seul découpage, commun aux deux modèles.
6. **Modèle 1 : TF-IDF (unigrammes + bigrammes) + LinearSVC**, avec les mots les plus discriminants.
7. **Modèle 2 : LSTM** (Keras) : Embedding → LSTM → GlobalMaxPooling → Dense → sortie sigmoïde, avec validation et *early stopping*.
8. **Comparaison** des deux modèles (accuracy, F1).
9. **Test** sur un nouveau texte.
10. **Sauvegarde** des modèles dans Google Drive.
11. **Interface web** Streamlit, accessible via ngrok.

## Résultats

Évaluation sur le jeu de test (20 % des articles, découpage stratifié), après suppression du préfixe `VILLE (Reuters) -` :

| Modèle               | Accuracy | F1-score (macro) |
|----------------------|----------|------------------|
| **TF-IDF + LinearSVC** | **99,02 %** | **99,00 %**  |
| LSTM                 | 98,25 %  | 98,23 %          |

**Le modèle TF-IDF + LinearSVC obtient les meilleurs résultats.** Il est aussi beaucoup plus rapide à entraîner que le LSTM.

### Pourquoi supprimer le préfixe « (Reuters) » ?

Dans ce dataset, presque tous les articles réels commencent par `VILLE (Reuters) -`, ce qui n'est presque jamais le cas des articles faux. Sans traitement, un modèle peut apprendre à reconnaître **la source** plutôt que **le contenu**. Ce préfixe est donc retiré avant l'entraînement (paramètre `REMOVE_SOURCE_PREFIX = True`).

| Version | Biais Reuters | TF-IDF + LinearSVC | LSTM |
|---|---|---|---|
| Première version | présent | 99,28 % | 98,86 % |
| Version actuelle | **retiré** | 99,02 % | 98,25 % |

Les scores restent supérieurs à 98 % après suppression du biais : les modèles s'appuient bien sur le contenu des articles. Leurs performances sur des articles venant d'autres sources ou d'autres périodes restent toutefois à vérifier.

## Exécution

1. Ouvrir le notebook dans Colab avec le badge **Open in Colab** ci-dessus.
2. Placer `dataset_fake_news.zip` dans `MyDrive/mon_projet/` sur Google Drive.
3. Activer le GPU (*Exécution → Modifier le type d'exécution → GPU*), puis lancer *Exécution → Tout exécuter*.
4. Pour l'interface Streamlit :
   - créer un compte gratuit sur [ngrok](https://ngrok.com) et récupérer un *authtoken* ;
   - dans Colab, l'ajouter dans les **Secrets** (icône 🔑) sous le nom `NGROK_AUTHTOKEN` ;
   - exécuter les cellules de la section **11. Interface Streamlit** : un lien public vers l'application s'affiche.

Pour une installation locale :

```bash
pip install -r requirements.txt
```

## Structure du dépôt

```
.
├── mon_projet.ipynb          # Notebook complet : données, prétraitement, modèles, interface
├── requirements.txt          # Dépendances Python
└── README.md
```

Les fichiers générés par le notebook (`preprocessing.py`, `app.py`) et les modèles sauvegardés dans Google Drive (`models/`) ne sont pas versionnés.

## Technologies

Python · pandas · NLTK · scikit-learn · TensorFlow / Keras · matplotlib · seaborn · WordCloud · Streamlit · ngrok

## Auteur

**Mustapha El Ouardi**, projet de fin d'études, Master.
