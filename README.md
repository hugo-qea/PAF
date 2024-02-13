# PAF 2022 - Ma première IA jouant à un jeu vidéo

Juin 2022 - Aristide LALOUX, Hugo QUENIAT, Mohamed Ali SRIR

Résumé de ce que nous avons fait :

**1) CartPole :**


-> Nous avons implémenté un agent pour jouer au CartPole à l'aide de l'algorithme du Q-learning. On remarque une convergence rapide de l'algorithme depuis une machine personnelle. L'agent dépasse largement les performances d'un humain. Le jeu est résolu et le bâton reste à l'équilibre vertical en permanence. 

-> Dans le dossier CartPole vous trouverez le notebook avec les commentaires explicatifs ainsi que les sources. Il faut lancer dans l'ordre les différentes cellules pour faire fonctionner le code. Vous trouverez également à la fin du fichier de quoi réaliser des graphiques de convergence de Q(s,a) (au centre de la matrice) pour des exploration-rates et learning-rates différents. 


**2) Space-Invaders Atari 2600 :**

-> Nous avons implémenté deux agents pour jouer à Space-Invaders Atari 2600 à l'aide d'un algorithme de Deep-Q Learning (DQN). Nous avons essayé deux réseaux de neurones convulotifs (CNN). A terme, il faudrait que pour une image du jeu en entrée, le CNN renvoie l'action optimale a effectuer.

-> Le premier CNN testé est le suivant : 

Image en entrée: (210, 160, 3).
32 filtres de Convolution2D de taille (8,8) et de pas (4,4) avec une fonction "relu" en sortie.
64 filtres de Convolution2D taille (4,4) et de pas (2,2) avec une fonction "relu" en sortie.
64 filtres de Convolution2D taille (3,3) et de pas (1,1) avec une fonction "relu" en sortie.
Une couche Flatten.
Dense de sortie 512 avec une fonction "relu" en sortie.
Dense de sortie 256 avec une fonction "relu" en sortie.
Dense de sortie 6 avec une fonction "linéaire" en sortie.
Sortie : 1 des 6 actions du jeu. 

Nous avons utilisé un agent entraîné à 1 millions de frame/steps. On observe bien que l'agent joue mieux qu'un agent aléatoire. Cependant, il reste bien loin des performances d'un humain. 

Après discussion avec Stéphane Lathuilière, nous avons essayé un autre CNN, moins profond. 

-> Second CNN testé :

Image en entrée: (210, 160, 3).
16 filtres de Convolution2D de taille (8,8) et de pas (4,4) avec une fonction "relu" en sortie.
32 filtres de Convolution2D de taille (4,4) et de pas (2,2) avec une fonction "relu" en sortie.
Dense de sortie 6 avec une fonction "linéaire" en sortie. 

-> Conclusion :

Après un entraînement moins long que le premier agent, ce nouveau réseau fonctionne mieux. L'agent joue mieux que l'agent aléatoire et joue mieux que le premier agent. Cependant, il reste lui aussi loin des performances d'un humain. 


**3) Mario Bros sur NES :**

-> Nous avons implémenté deux algorithmes pour faire jouer un agent à Super Mario Bros sur NES : Le DDQN (Double Deep Q Learning) et le PPO (Proximal Policy Optimization).

-> Pour le DDQN :
32 filtres de Convolution2D de taille (8,8) et de pas (4,4) avec une fonction "relu" en sortie.
64 filtres de Convolution2D de taille (4,4) et de pas (2,2) avec une fonction "relu" en sortie.
32 filtres de Convolution2D de taille (3,3) et de pas (1,1) avec une fonction "relu" en sortie.
Une couche Flatten
Dense de taille de sortie 512 avec une fonction "relu" en sortie.
Dense de taille de sortie 7. Comme les 7 actions.

-> Le PPO est un code entièrement implémenté par la bibliothèque stable_baselines et qui a donc permis d'effectuer une comparaison.

-> Conclusion :
Le DDQN entraîne l'agent très lentement par rapport au PPO. Nous remarquons que les résultats du PPO sont bien meilleurs que ceux du DDQN. En effet, avec le DDQN l'agent passe parfois les 3 premiers obstacles tandis que pour le PPO c'est quasiment systématique. Avec le PPO, l'agent réussit même parfois à terminer le niveau entier. 


**4) Flappy Bird :**

Nous avons deux codes : 

Le premier qui fonctionne très bien est un code entièrement trouvé sur https://github.com/kyokin78/rl-flappybird. L'agent s'entraîne en quelques heures et atteint des performances bien supérieures de celles d'un humain (même expert du jeu). Cependant le code est from scratch, difficile à comprendre et à priori peu intéressant dans le cadre d'un TP. 

Pour pallier à cela, nous avons écrit un autre code en nous inspirant du Q Learning sur le CartPole. Il est sous la forme d'un fichier .ipynb et le code est bien plus hospitalier pour faire un TP. Cependant, après training de l'agent,  les performances restent médiocres. L'agent arrive à dépasser péniblement quelques tuyaux, mais il n'y a rien de très impressionant. Nous avons passé du temps à réfléchir à d'où pouvait venir le problème et nous pensons que la difficulté provient de la discrétisation de l'espace. Un état est caractérisé par DeltaX, la distance entre l'agent et le prochain tuyau horizontalement, et DeltaY, la distance entre l'agent et la prochaine ouverture verticalement. Nous avons écrit une fonction bucketize qui permet de discrétiser l'espace DeltaX et DeltaY mais nous ne savons pas bien en combien il faudrait découper l'espace de jeu. En effet, d'un côté une découpe trop précise entrainera un trop grand nombre d'états et donc une matrice Q trop grande à optimiser pour le Q learning. De l'autre côté, une découpe trop grossière de l'espace de jeu entrainera trop peu d'états connus par l'agent pour jouer convenablement. 
Nous avons testé plusieurs valeurs de discrétisation mais sans succès. L'agent n'arrive qu'à dépasser quelques tuyaux.

Nous avons entrainé deux agents avec des reward-shaping distincts : 
    - Les sauvegardes 'onehour.pt' et 'twohours.pt' correspondent à un reward-shaping uniquement de survie
    - La sauvegarde 'onemilliongames.pt' correpond à la survie et aller à droite.
