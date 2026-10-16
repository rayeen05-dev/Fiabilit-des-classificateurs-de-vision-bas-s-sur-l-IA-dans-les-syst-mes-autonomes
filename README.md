# Fiabilité des classificateurs de vision basés sur l'IA dans les systèmes autonomes

> Rapport de stage d'été — Institut Supérieur d'Informatique et de Multimédia de Sfax (ISIMS)  
> Réalisé au Centre de Recherche en Numérique de Sfax (CRNS)  
> **Auteur :** Oudali Rayen  
> **Encadrante :** Mme. Maissa Elleuch  
> **Année universitaire :** 2025 – 2026

---

## À propos du projet

Ce projet étudie la **fiabilité formelle de classificateurs de vision par IA** dans le contexte des systèmes autonomes (véhicules, drones, surveillance). Plutôt que de mesurer une précision moyenne sur un jeu de données, l'objectif est de **prouver mathématiquement** des propriétés de robustesse, ou d'en quantifier précisément les limites.

La question centrale est :

> *Dans quelle mesure un réseau de neurones de vision peut-il être certifié robuste face à des perturbations adversariales, et quelles sont les limites pratiques des outils de vérification formelle actuels ?*

Les expérimentations portent sur deux domaines :
- **CIFAR-10** — classification d'images naturelles (10 classes : avion, voiture, oiseau, etc.)
- **GTSRB** — reconnaissance de panneaux routiers (10 classes, résolution 32×32×3)

---

## Concepts clés

| Terme | Définition |
|---|---|
| **Vérification formelle** | Démonstration mathématique qu'une propriété est vraie pour *tout* un domaine d'entrées, pas seulement sur des exemples de test |
| **Propriété de robustesse** | « Aucune perturbation bornée par ε (norme L∞) ne peut changer la décision du modèle » |
| **UNSAT** | Aucun contre-exemple n'existe dans le domaine : robustesse **prouvée** |
| **SAT** | Un témoin de violation a été trouvé : vulnérabilité **détectée** |
| **TIMEOUT** | Le solveur n'a pas pu conclure dans le temps imparti : résultat **indéterminé** |
| **Exemple adversarial** | Perturbation imperceptible à l'œil nu qui induit une erreur de classification |
| **Marge softmax** | Différence entre les deux premiers logits — indicateur de confiance du modèle |

---

## Outils de vérification utilisés

### Marabou
Solveur SMT basé sur l'algorithme **Reluplex**. Il prouve formellement l'absence d'exemples adversariaux (UNSAT) ou fournit un témoin de violation (SAT). Optimisé pour les preuves UNSAT via branch-and-bound.

### α,β-CROWN
Combine une **attaque PGD rapide** (falsification) et une **vérification par relaxations linéaires** pour les preuves d'absence. Nettement plus rapide que Marabou sur les instances SAT (jusqu'à ×153 dans les expériences).

### ONNX Runtime
Utilisé pour la **réinférence en précision float32** des témoins SAT, étape obligatoire pour valider qu'un SAT formel correspond bien à une erreur de classification réelle.

---

## Architecture du pipeline

Le pipeline est entièrement automatisé et composé des étapes suivantes :

```
collect()
    └─► Sélection des images correctement classifiées

generate_vnnlib()
    └─► Encodage de la propriété L∞ pairwise → fichier .vnnlib
        (une propriété par classe adversariale, soit 9 fichiers par image par ε)

_worker()
    └─► Exécution de Marabou / α,β-CROWN en sous-processus isolé
        → Verdict (SAT / UNSAT / TIMEOUT) + temps + témoin

verif()
    └─► Orchestration des 9 propriétés pairwise × toutes les valeurs de ε
        → CSV structuré des résultats

analysis.py
    └─► Courbes de robustesse, extraction des témoins SAT,
        réinjection float32 via ONNXRuntime

plot_heatmap.py
    └─► Visualisation : image originale / image adversariale / carte |x - x_adv|

collect_margin_spectrum.py
    └─► Échantillonnage d'images CIFAR-10 selon un spectre de marges softmax
```

### Formulation mathématique

Pour une image `x` correctement classifiée comme `c`, le domaine de perturbation est :

```
Xε = { x' | ‖x' − x‖∞ ≤ ε }
```

La propriété pairwise vérifie si un `x'` dans ce domaine peut produire `score(t) > score(c)` pour une classe cible `t ≠ c`.

---

## Structure du dépôt

```
neural-verification-robustness/
│
├── README.md
├── requirements.txt
│
├── pipeline/
│   ├── collect.py               # Sélection automatique des images à certifier
│   ├── generate_vnnlib.py       # Génération des fichiers de propriétés .vnnlib
│   ├── worker.py                # Appel isolé du solveur (Marabou / α,β-CROWN)
│   └── verif.py                 # Orchestration complète par image et par ε
│
├── analysis/
│   ├── analysis.py              # Courbes, corrélations, réinjection des témoins SAT
│   ├── plot_heatmap.py          # Heatmaps des perturbations adversariales
│   └── collect_margin_spectrum.py  # Échantillonnage par spectre de marge softmax
│
├── models/
│   ├── cifar10_fc.onnx          # Modèle FC substitut (3072→512→256→128→10)
│   ├── cifar10_small.onnx       # Modèle officiel VNN-COMP
│   └── gtsrb-10x2.onnx         # Réseau convolutionnel ReLU GTSRB (32×32×3, 10 classes)
│
├── data/
│   ├── cifar10/                 # Images PNG (ex. 2402_truck.png, 8235_horse.png)
│   └── gtsrb/                   # Tableaux NumPy au format pickle
│
├── properties/
│   ├── cifar10/                 # Fichiers .vnnlib générés pour CIFAR-10
│   └── gtsrb/                   # Fichiers .vnnlib générés pour GTSRB
│
├── witnesses/
│   ├── *.npy                    # Témoins SAT (tableaux d'entrées adversariales)
│   └── *.png                    # Témoins SAT (images adversariales)
│
└── results/
    ├── results.csv
    ├── consolidation_results.csv
    ├── semantic_proximity_results.csv
    └── robustness_summary.csv
```

---

## Principaux résultats

### Certification formelle
L'image `2402_truck.png` (CIFAR-10) a été certifiée robuste jusqu'à **ε\* = 0,008** (norme L∞) : toutes les 9 propriétés pairwise sont UNSAT à ce rayon. Il s'agit d'une **preuve mathématique**, pas d'un résultat statistique.

### Comparaison Marabou / α,β-CROWN

| Situation | Outil recommandé | Gain observé |
|---|---|---|
| Trouver rapidement un contre-exemple (SAT) | α,β-CROWN (PGD) | Jusqu'à ×153 plus rapide |
| Certifier l'absence de contre-exemple (UNSAT) | Marabou | Preuve formelle garantie |
| Propriété très difficile | Les deux atteignent leurs limites | TIMEOUT dans les deux cas |

### Corrélation marge softmax / rayon certifié

| Domaine | R² | n |
|---|---|---|
| CIFAR-10 | 0,967 | 6 images exploitables |
| GTSRB | 0,991 | 9 images |

Une marge softmax élevée est un **bon indicateur** pour prioriser les images à certifier, sans être un prédicteur formel du rayon certifié.

### Phénomène SAT/float32
Un verdict SAT formel ne garantit pas une erreur de classification dans l'exécution réelle. La propriété pairwise (`score(cible) > score(origine)`) est plus faible que la notion opérationnelle d'erreur (`argmax = cible`). **Toute certification doit inclure une étape de réinjection float32 via ONNXRuntime.**

### Proximité sémantique
Aucune relation robuste entre le taux de confusion inter-classes et la difficulté de vérification formelle n'a été établie. La difficulté de vérification est une **propriété locale de l'image**, pas une propriété globale de la paire de classes.

---

## Progression expérimentale

Les travaux ont suivi une montée en complexité contrôlée en six phases :

```
Réseau jouet (valeur absolue)
    └─► Validation des verdicts SAT / UNSAT / TIMEOUT sur Marabou

MNIST
    └─► Premier pipeline image réel — modification 7→8 observée à ε = 0,10

CIFAR-10 (modèle FC substitut)
    └─► Validation du pipeline complet sur modèle compatible

CIFAR-10 (campagnes de robustesse)
    └─► Profils de robustesse, comparaison Marabou / α,β-CROWN,
        analyse SAT/float32, étude sémantique, corrélation marge/rayon

GTSRB
    └─► Généralisation du pipeline, propriété de domination complète,
        confirmation des phénomènes observés sur CIFAR-10
```

---

## Notes importantes

- **TIMEOUT ≠ vulnérabilité.** Un timeout signifie que le solveur n'a pas pu conclure dans les ressources disponibles. Ce n'est ni une preuve de robustesse ni une preuve de vulnérabilité.
- **Le modèle FC (`cifar10_fc.onnx`) est un outil de validation**, pas un modèle scientifique comparable aux benchmarks VNN-COMP. Il a été entraîné uniquement pour valider le pipeline avant d'utiliser `cifar10_small.onnx`.
- **Les échantillons sont de petite taille** (6 à 9 images). Les corrélations observées (R² = 0,967 / 0,991) constituent des signaux expérimentaux forts mais ne constituent pas des lois générales.

---

## Références

| # | Référence |
|---|---|
| [1] | VNN-COMP — https://www.vnncompetition.org |
| [2] | Marabou — https://github.com/NeuralNetworkVerification/Marabou |
| [3] | Katz et al., *Reluplex*, CAV 2017 — https://doi.org/10.1007/978-3-319-63387-9_13 |
| [4] | ONNX — https://onnx.ai |
| [5] | VNN-LIB — https://www.vnnlib.org |
| [6] | α,β-CROWN — https://github.com/Verified-Intelligence/alpha-beta-CROWN |
| [7] | ONNX Runtime — https://onnxruntime.ai |
| [8] | CIFAR-10 — https://www.cs.toronto.edu/~kriz/cifar.html |
| [9] | GTSRB — Stallkamp et al., IJCNN 2012 — https://doi.org/10.1109/IJCNN.2012.6252500 |

---

*Stage effectué au CRNS — Technopole de Sfax, Cité El Ons, route de Tunis.*
