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

1. **Importation** des données depuis Google Drive.
2. **Prétraitement**
   - suppression des doublons et des textes trop courts ;
   - analyse exploratoire : répartition des classes, longueur des textes, mots les plus fréquents, nuage de mots ;
   - nettoyage : minuscules, suppression des liens, des chiffres, de la ponctuation et des *stopwords* ;
   - tokenisation (NLTK) et lemmatisation (WordNet).
3. **Modélisation**
   - **TF-IDF + LinearSVC** dans un pipeline scikit-learn ;
   - **LSTM** (Keras) : Embedding (100) → LSTM (150) → GlobalMaxPooling → Dense (64) → Dense (2), avec Dropout 0,5.
4. **Exploration** de représentations complémentaires : TF-IDF limité à 5 000 mots et Word2Vec (gensim).
5. **Interface** web avec **Streamlit**, exposée publiquement via **ngrok**.

## Résultats

Évaluation sur un jeu de test de 12 749 articles (33 % des données) :

| Modèle               | Accuracy | F1-score (macro) |
|----------------------|----------|------------------|
| TF-IDF + LinearSVC   | 99,28 %  | 0,99             |
| LSTM (5 époques)     | 98,86 %  | 0,99             |

Matrice de confusion du modèle TF-IDF + LinearSVC :

|                  | Prédit FAKE | Prédit REAL |
|------------------|-------------|-------------|
| **Réel FAKE**    | 5 714       | 60          |
| **Réel REAL**    | 32          | 6 943       |

**Limite connue :** dans ce dataset, la plupart des articles réels commencent par `WASHINGTON (Reuters) -`. Le modèle peut donc en partie apprendre à reconnaître le style ou la source de l'article plutôt que la véracité de son contenu. Les scores très élevés doivent être interprétés avec prudence, et les performances seront probablement plus faibles sur des articles provenant d'autres sources.

## Exécution

1. Ouvrir le notebook dans Colab avec le badge **Open in Colab** ci-dessus.
2. Placer `dataset_fake_news.zip` dans `MyDrive/mon_projet/` sur Google Drive.
3. Exécuter les cellules dans l'ordre (un GPU est recommandé pour le LSTM : *Exécution → Modifier le type d'exécution*).
4. Pour l'interface Streamlit :
   - créer un compte gratuit sur [ngrok](https://ngrok.com) et récupérer un *authtoken* ;
   - dans Colab, l'ajouter dans les **Secrets** (icône 🔑) sous le nom `NGROK_AUTHTOKEN` ;
   - exécuter les cellules de la section **Interface** : un lien public vers l'application s'affiche.

Pour une installation locale :

```bash
pip install -r requirements.txt
```

## Structure du dépôt

```
.
├── mon_projet.ipynb   # Notebook complet : données, prétraitement, modèles, interface
├── requirements.txt   # Dépendances Python
└── README.md
```

Les fichiers générés (`model_lstm.h5`, `tokenizer.pkl`, `app.py`) ne sont pas versionnés.

## Technologies

Python · pandas · NLTK · scikit-learn · TensorFlow / Keras · gensim · matplotlib · seaborn · WordCloud · Streamlit · ngrok

## Auteur

**Mustapha El Ouardi**, projet de fin d'études, Master.
