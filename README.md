# CNN vs Transfer Learning — Cats vs Dogs  

## Objectif  
Comparer deux approches sur le dataset **Cats vs Dogs** :  
1. Un **CNN entraîné from scratch**.  
2. Un modèle en **transfert d’apprentissage (ResNet18)**, avec backbone gelé puis éventuellement fine-tuné.  

L’évaluation porte sur :  
- **Convergence** (rapidité, stabilité, surapprentissage éventuel).  
- **Performance** (Accuracy, Précision, Recall, **F1**).  
- **Robustesse** (matrices de confusion, analyse des erreurs typiques).  

---

## Environnement  

### Option pip  
```bash
# Créer l’environnement virtuel
python -m venv .venv

# Activer l’environnement (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

>  La détection du périphérique (CUDA / MPS / CPU) est automatique dans le code.
Dans notre cas, l’entraînement s’est déroulé sur CPU, car c’était le seul disponible dans l’environnement local (VSCode). Cela a rendu l’entraînement relativement long.

Nous avons également testé sur Google Colab avec GPU activé. Cependant, l’accès gratuit étant limité, Colab nous a notifié des restrictions d’utilisation du GPU après un certain temps, ce qui a interrompu les expériences.

---

## Organisation des données  
Placement localement le dataset **( les données ne seront pas versionner sur GitHub)** :  

```
cnn_cats_dos_MariemeFAYE/
└─ Cat_dog_data/
   ├─ train/
   │  ├─ cats/
   │  └─ dogs/
   └─ test/
      ├─ cats/
      └─ dogs/

```

Le split validation est généré automatiquement à partir de `train/`.  

---

## Expériences  

Nous avons réalisé **4 expériences distinctes** :  
- **Expérience A – CNN from Scratch** : entraînement avec Adam et SGD.  
- **Expérience B – Transfer Learning (ResNet18)** : entraînement avec Adam et SGD.  

Chaque exécution renvoie un historique d’entraînement et sauvegarde automatiquement le meilleur modèle.

Les expériences se lancent via le notebook **notebook.ipynb** (ou en script).  

### Expérience A — CNN from scratch  
- Hyperparamètres : `batch_size`, `lr`, `epochs`, `optimizer`, `dropout`, **BN**, `scheduler`.  
- Exemple de config par défaut : Optimiseur **Adam** + scheduler **Cosine**.  
- Option bonus : **OneCycleLR** (`sched_name="onecycle"`).  

### Expérience B — Transfer Learning (ResNet18)  
- Backbone : **ResNet18_Weights.DEFAULT**.  
- Mode **frozen** : backbone gelé (`fine_tune=False`), seule la tête est adaptée :  
  `Dropout → Linear → BN → ReLU → Dropout → Linear`.  
- LR conseillé (frozen) : `1e-4` à `1e-3` (≈ `lr/10` vs scratch).  
- Option fine-tuning : déverrouiller `layer4` avec un LR plus petit (`1e-5` à `5e-5`).  

---

## Évaluation & Checkpoints  

Les modèles sauvegardent automatiquement leurs meilleurs checkpoints dans `checkpoints/` :  
- CNN scratch : `checkpoints/scratch_adam_cosine_best.pth`  
- TL ResNet18 (frozen) : `checkpoints/tl_resnet18_frozen_adam_step_best.pth`  

L’évaluation **test set** recharge le checkpoint et calcule :  
- `Accuracy / Precision / Recall / F1`.  
- Matrices de confusion + visualisation des erreurs typiques.  

---

## Résultats 

### Figures (générées automatiquement → `figures/`) :  
- `figures/scratch_adam_cosine_*.png` → courbes loss, accuracy, precision, recall, F1.  
- `figures/tl_resnet18_frozen_adam_step_*.png`.  
- `figures/cm_scratch.png`, `figures/cm_tl.png`.  

### Tableau comparatif :  
### Validation (meilleure époque)

| Expérience | Optimiseur | Val Loss | Val Acc | Val Prec | Val Rec |
|------------|------------|---------:|--------:|---------:|--------:|
| Scratch    | Adam       | 0.589    | 0.698   | 0.799    | 0.530   |
| Scratch    | SGD        | 0.669    | 0.573   | 0.930    | 0.160   |
| Transfer   | Adam       | 0.056    | 0.980   | 0.975    | 0.985   |
| Transfer   | SGD        | 0.055    | 0.979   | 0.977    | 0.981   |

### Test final

| Expérience (best) | Test Loss | Test Acc | Test Prec | Test Rec |
|-------------------|----------:|---------:|----------:|---------:|
| Scratch (Adam)    | ~0.60     | ~0.66    | ~0.83     | ~0.40    |
| Transfer (Adam)   | ~0.06     | ~0.98    | ~0.97     | ~0.98    |

---

## Comparaison Scratch vs Transfer

Nous avons comparé deux approches (CNN from scratch vs Transfer Learning avec ResNet18) et deux optimiseurs (Adam, SGD).  
Chaque expérience a été menée sur 8 époques, avec suivi de la loss, accuracy, précision et rappel.

---

### A. Optimiseur **Adam**

![Courbes Adam](a3239c10-4d7c-4375-9a10-8bf1e97d5b28.png)

- **Scratch** : convergence lente, accuracy plafonnant autour de 0.70, rappel limité (~0.50).  
- **Transfer** : convergence immédiate, accuracy ~0.97 dès la 1ʳᵉ époque, précision/recall ≈ 1.0.  

 Adam s’avère très efficace pour stabiliser et accélérer l’entraînement, surtout en transfert learning.

---

### B. Optimiseur **SGD**

![Courbes SGD](31d209a4-9ca7-4912-89bb-14a59d5de986.png)

- **Scratch** : difficulté à apprendre, accuracy instable autour de 0.55–0.60, rappel très faible (<0.25).  
- **Transfer** : performance nettement meilleure (accuracy ~0.98, précision et rappel >0.95).  
- Cependant, la convergence est plus lente et moins stable qu’avec Adam.

---

### Tableau récapitulatif (Test set)

| Expérience                  | Optimiseur | Test Loss | Test Acc | Test Précision | Test Recall |
|-----------------------------|------------|-----------|----------|----------------|-------------|
| **A — Scratch**             | Adam       | ~0.60     | ~0.66    | ~0.83          | ~0.40       |
| **A — Scratch**             | SGD        | ~0.67     | ~0.57    | ~0.85–0.90     | ~0.15–0.20  |
| **B — Transfer (ResNet18)** | Adam       | ~0.06     | ~0.98    | ~0.97–0.99     | ~0.97–0.98  |
| **B — Transfer (ResNet18)** | SGD        | ~0.05–0.07| ~0.97–0.98| ~0.96–0.98    | ~0.97–0.98  |

---

### Matrices de confusion

#### A. CNN From Scratch
![Confusion Matrix Scratch](153816f1-d465-4cc2-91a1-b87579b2955e.png)

- Les **chats** sont relativement bien reconnus (1153/1250).  
- Mais beaucoup de **chiens** sont mal classés comme chats (755 erreurs).  
- Cela explique le **recall limité (~0.50)**.

#### B. Transfer Learning (ResNet18)
![Confusion Matrix Transfer](81b185c4-3cc9-4ad7-ac60-626532c67261.png)

- Diagonale quasi parfaite : 1228/1250 pour chaque classe.  
- Seulement 22 erreurs par classe.  
- Cela confirme les excellentes performances observées (>97% acc, précision et recall ~0.98).

---

### Analyse comparative

- **Impact du transfert learning** :  
  Sur les deux optimiseurs, le transfert learning domine largement : pertes très faibles, métriques quasi parfaites dès les premières époques.  

- **Impact de l’optimiseur** :  
  Adam est plus stable et rapide à converger, surtout pour le modèle from scratch.  
  Avec SGD, le scratch reste inefficace, alors que le transfert learning parvient tout de même à d’excellents résultats.  

- **Robustesse** :  
  Le transfert learning montre une grande robustesse, tandis que le modèle from scratch souffre d’overfitting partiel et d’un rappel limité, indiquant des difficultés à détecter correctement toutes les classes.

---

### Conclusion générale

- **CNN from scratch** : peut apprendre, mais reste limité (~70% acc, recall <0.55).  
- **Transfer learning (ResNet18)** : offre une convergence ultra-rapide et des performances supérieures (>97% acc, précision/recall ~1.0).  
- **Optimiseur** : Adam > SGD en stabilité et en efficacité, surtout pour un entraînement from scratch.
  
---

## Limites & pistes d’amélioration

### Limites observées
- **Modèle from scratch** :  
  - Convergence lente et performances limitées (accuracy <70%, recall souvent faible).  
  - Forte sensibilité au choix de l’optimiseur (SGD → résultats médiocres).  
  - Risque d’overfitting si l’entraînement est prolongé, faute de données suffisantes.

- **Transfert learning** :  
  - Très performant mais dépendant de la disponibilité de modèles pré-entraînés (ResNet18 ici).  
  - Temps d’entraînement plus long sur certaines configurations (SGD sans scheduler adapté).  
  - Moins flexible pour explorer des architectures originales.

- **Jeu de données** :  
  - Corpus Cats vs Dogs limité en diversité (conditions d’éclairage, angles de vue).  
  - Absence de classes supplémentaires, ce qui limite l’évaluation de la généralisation.  

### Pistes d’amélioration
1. **Augmentation de données** :  
   Introduire des transformations plus variées (rotations, recadrages aléatoires, bruit gaussien) pour accroître la robustesse.

2. **Validation plus rigoureuse** :  
   Utiliser un split train/val clair et/ou la validation croisée pour mieux estimer la généralisation.

3. **Optimisation avancée** :  
   Tester d’autres optimiseurs (AdamW, RMSprop) et scheduler plus sophistiqués (CosineAnnealingLR, OneCycleLR).

4. **Fine-tuning du backbone** :  
   Dégeler progressivement certaines couches du ResNet18 pour améliorer encore la précision.

5. **Enrichissement du dataset** :  
   Expérimenter sur des bases plus complexes (ImageNet subsets, CIFAR-100, etc.) pour comparer la capacité à généraliser.

6. **Suivi expérimental** :  
   Mettre en place TensorBoard ou Weights & Biases pour une journalisation complète (courbes interactives, suivi des hyperparamètres).

7. **Exploration d’architectures modernes** :  
   Tester des architectures plus récentes comme EfficientNet, DenseNet ou MobileNetV3 qui offrent un bon compromis précision/vitesse.


