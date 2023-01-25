# HMER Project

## Résumé de la démarche

J'ai commencé par créer un gros dataset d'images d'équations manuscrites sur un fond de papier. Le but était d'avoir suffisamment d'images pour pouvoir évaluer notre modèle, sans avoir à créer le dataset à la main (ce qui serait très fastidieux notamment pour la labellisation).

J'ai pris le dataset d'équations manuscrites CROHME_train_2011. Il est conçu pour du on-line OCR, donc le format des images est en .inkml (pas très pratique). J'ai utilisé le Git repository [ThomasLech/CROHME_extractor](https://github.com/ThomasLech/CROHME_extractor) pour convertir les fichiers .inkml en .png, et j'ai ajouté plusieurs fonds différents. J'ai aussi enregistré chaque label de chaque équation. Cf le fichier zip que je t'ai envoyé : 
    - data/CROHME_train_2011_PNG/ contient les images des équations en .png, avec fond blanc
    - data/CROHME_train_2011_background/ contient les images des équations en .png, avec différents fonds de papiers 
    - data/CROHME_train_2011_labels/ contient les labels de chaque équation
    - data/all_classes.txt contient tous les labels des symboles présents dans ces équations

On commence par entraîner un classifieur XGBoost à reconnaître des symboles manuscrits isolés. On l'entraîne sur le dataset de symboles isolés présent dans le fichier `data/isolated_symbols/train.pickle`. Malheureusement, il s'agit d'un dataset différent du CROHME_train_2011 utilisé avec les équations entières (on ne pouvait pas utiliser le CROHME_train_2011 pour cette étape car les caractères individuels ne sont pas labellisés ; on a que le label de l'équation entière et c'est assez galère de remonter aux labels de chaque caractère). 
    - `data/isolated_symbols/train.pickle` contient les images de symboles manuscrits isolés sous forme d'array ainsi que leur label. 
    - La correspondance entre l'encoding One Hot utilisé dans ce fichier et le label correspondant est donné dans le fichier `data/isolated_symbols/one_hot_classes.txt`

Les features utilisées dans le XGBoost pour décrire chaque symbole sont : 
    - La cosine similarity entre le symbole et chaque symbole de référence (listés dans data/all_classes.txt)
    - La similarité entre la signature du symbole et la signature de chaque symbole de référence (listés dans data.all_classes.txt)

Pour l'instant, le lancement du fichier `xgb.py` exécute : 
- L'entraînement du XGBoost sur le dataset des symboles isolés ; ATTENTION, le calcul des features est super lent, du coup là dans le code (ligne 38) j'ai tronqué le dataset à 50 samples pour voir si le code tourne, mais du coup ça donne pas de résultats probant. Il faudrait lancer ça sur un serveur avec plus de puissance pour entraîner sur plus de samples.
- La prédiction des caractères d'une des images du CROHME_train_2011 dataset.

Pour l'instant je n'ai pas pu vérifier la performance du modèle car les features sont très longues à calculer, donc je n'ai pas pu réaliser l'entraînement sur beaucoup de donneés.

## A faire dans la suite : 

1. Entraîner le XGBoost sur un bon nombre de symboles --> lancer sur un serveur plus puissant ? Optimiser le code de création des features pour que ça tourne plus vite ?

2. Faire nos prédictions des caractères sur un certain nombre d'équations et évaluer le résultat

3. Créer un script qui à partir des caractères isolés renvoie un code latex

=> Je ne pense pas qu'on ait le temps de faire tout ça + le rapport d'ici mardi. Donc je pense qu'on pourrait éventuellement se limiter à la reconnaissance des caractères séparés de l'équation, sans faire la partie traduction en latex.
