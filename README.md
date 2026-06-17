# Projet de Fin de Module — Deep Learning

**EMSI Casablanca** — Filière Informatique, 4IIR-IAD — Année universitaire 2025–2026

Conception, implémentation, comparaison et analyse critique de modèles de deep learning pour données tabulaires, images et séquences.

**Auteur :** EL FATIMI Oumnia

---

## Description

Ce projet est un examen final intégrateur couvrant trois familles d'architectures de deep learning sous PyTorch, chacune appliquée à un type de données différent et à un dataset réel :

| Partie | Architecture | Dataset | Tâche |
|---|---|---|---|
| I | MLP | Wine Quality (1599 vins) | Classification binaire (bon / mauvais vin) |
| II | CNN (LeNet-5) | MNIST (70 000 images) | Classification de chiffres manuscrits |
| III | RNN / LSTM / GRU / Seq2Seq | Tatoeba eng-fra (95 204 paires) | Modélisation de séquences & traduction |

---

## Structure du dépôt

```
.
├── Partie1_MLP.ipynb              # MLP — données tabulaires
├── Partie2_CNN.ipynb              # CNN — vision par ordinateur
├── Partie3_RNN.ipynb              # RNN/LSTM/GRU/Seq2Seq — séquences
├── Rapport_Projet_DeepLearning_Oumnia.docx   # Rapport complet
└── README.md
```

---

## Environnement et exécution

Les notebooks sont conçus pour **Google Colab** avec accélérateur GPU (Tesla T4).

1. Ouvrir le notebook souhaité sur [colab.research.google.com](https://colab.research.google.com)
2. Activer le GPU : `Exécution` → `Modifier le type d'exécution` → `T4 GPU`
3. Exécuter les cellules **dans l'ordre**, de haut en bas (`Exécution` → `Tout exécuter`)

### Dépendances principales
`torch`, `torchvision`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn` — toutes préinstallées sur Colab.

---

## Partie I — MLP (Wine Quality)

- **Données :** 1599 vins, 11 variables physico-chimiques, aucune valeur manquante. Cible binarisée (qualité ≥ 6 = bon vin) : 855 bons / 744 mauvais.
- **Split :** 1119 train / 240 validation / 240 test, standardisation (StandardScaler).
- **Modèles :** MLP `nn.Sequential` et MLP en classe personnalisée (architectures identiques, 2 914 paramètres), initialisation Xavier.
- **Entraînement :** 100 époques, Adam (lr=0.001), CrossEntropyLoss, Dropout 0.3, sauvegarde du meilleur modèle.

**Résultats (jeu de test) :**

| Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|
| 0.7667 | 0.7937 | 0.7692 | 0.7812 |

---

## Partie II — CNN (MNIST)

- **Données :** 60 000 images d'entraînement / 10 000 de test (28×28, niveaux de gris).
- **Implémentation manuelle :** corrélation croisée 2D, max-pooling et average-pooling (validées identiques à PyTorch).
- **Modèle :** LeNet-5 (61 706 paramètres), entraîné 10 époques, Adam (lr=0.001).

**Résultats (jeu de test, 10 000 images) :**

| Modèle | Paramètres | Accuracy |
|---|---|---|
| MLP simple (784-256-128-10) | ≈ 235 000 | 0.9771 |
| **CNN LeNet-5** | **61 706** | **0.9841** |

Le CNN surpasse le MLP avec environ 4 fois moins de paramètres, confirmant l'intérêt du partage des poids et de la localité des filtres sur des données image.

---

## Partie III — RNN / LSTM / GRU / Seq2Seq (Traduction EN→FR)

- **Données :** corpus Tatoeba normalisé et filtré (95 204 paires, phrases < 10 mots), vocabulaires de 10 026 mots (EN) et 16 815 mots (FR).
- **Comparaison RNN/LSTM/GRU** (modèle de langage, 5 époques, gradient clipping) :

| Architecture | Perplexité finale | Temps d'entraînement |
|---|---|---|
| RNN simple | 10.79 | 54.2 s |
| LSTM | 9.79 | 57.3 s |
| **GRU** | **9.21** | 56.6 s |

- **Système Seq2Seq** (encodeur-décodeur GRU, teacher forcing 0.5) : décodage glouton et beam search testés. **BLEU score moyen (50 exemples) : 0.3211**.

> ⚠️ **Limite connue :** l'entraînement Seq2Seq enregistré dans le notebook affiche une perte `NaN` sur les 10 époques (probable cas limite de `CrossEntropyLoss` avec `ignore_index` sur des positions entièrement composées de PAD). Le détail et les pistes de correction sont documentés dans le rapport, section 3.5 (Analyse critique).

---

## Synthèse des résultats

| Partie | Métrique principale | Résultat |
|---|---|---|
| I — MLP | F1-Score | 0.7812 |
| II — CNN | Accuracy | 0.9841 |
| III — Seq2Seq | BLEU | 0.3211 |

Pour l'analyse théorique complète, les justifications méthodologiques et la discussion critique de chaque partie, voir **`Rapport_Projet_DeepLearning_Oumnia.docx`**.

---

## Licence

Projet académique réalisé dans le cadre du module Deep Learning — EMSI Casablanca, 2025–2026. Travail strictement individuel.
