# Tolk_test

## Description du logiciel développé

Le projet est conçu pour permettre aux utilisateurs (clients) d'entraîner les modèles de classification pré-entraînés (classification des intentions du chatbot) avec un ensemble de données personnalisé avec une API, qui a été construite pour permettre aux utilisateurs de sélectionner le modèle pré-entraîné (chatbot) basé sur le domaine, de télécharger leur nouvel ensemble de données et d'interagir avec le chatbot nouvellement affiné. L'organigramme ci-dessous illustre un aperçu du processus du projet.  

![flowchart](https://github.com/ianastafeva/Tolk_test/blob/8f6ee04043d5e35d373d4cde858b9a8efde844f3/Chatbot_API_flowchart.png?raw=true)

Il convient de mentionner que le bloc de prétraitement du projet est seulement envisagé et non mis en œuvre parce qu'il y a une difficulté de PNL présente ici. La difficulté consiste à restructurer le nouveau jeu de données (surtout s'il n'y a pas de règles strictes sur le formatage du jeu de données) de la même manière que le jeu de données original utilisé pour construire le modèle sélectionné par l'utilisateur. Ce que je veux dire par là, c'est que, par exemple, si le jeu de données de l'utilisateur est du texte brut, comment extraire les informations pertinentes telles que l'intention, les modèles et les réponses de ce texte. Une solution consiste à développer un modèle NLP capable de reconnaître les similitudes communes dans l'ensemble de données, de diviser l'ensemble de données en classes (intentions) sur la base de la similitude, puis de restructurer chaque classe en modèles et réponses. 

Cette difficulté n'est pas seulement spécifique aux modèles de ce projet, elle est également présente dans des modèles plus avancés tels que RASA NLU où l'ensemble des étiquettes (intentions, actions, entités et créneaux) pour lesquelles le modèle de base est entraîné doit être exactement le même que celui présent dans les données d'entraînement utilisées pour le réglage fin.

Enfin, lorsque l'on teste le modèle affiné (chatbot) via l'API, on peut remarquer que les réponses sont étranges, ce qui est dû au fait que notre intégration du chatbot avec l'API n'est pas précise. Par conséquent, l'utilisateur peut tester localement le chatbot nouvellement ajusté, comme décrit ci-dessous.


## Configuration
Clonez ou téléchargez les fichiers du projet sur le périphérique local et décompressez-les.

### L'utilisateur a ananconda python dans son appareil
1) Ouvrir termainal (Mac, Linux) ou anaconda Prompt en tant qu'administrateur (windows)

2) Déplacez-vous vers le répertoire contenant les fichiers du projet dézippé en utilisant la commande 'cd'

3) Créez un nouvel environnement virtuel avec la commande suivante :

   >conda create -n chatbot_api python=3.8.5

4) Activez l'environnement avec la commande suivante :

   >conda activate chatbot_api

5) Installez les paquets python nécessaires avec la commande suivante :

   >pip install -r requirements.txt

6) Téléchargez les fichiers nltk nécessaires par les commandes suivantes :
   >python
   
   >import nltk 
   
   >nltk.download()
 
   Dans la fenêtre GUI qui s'ouvre, il suffit de cliquer sur le bouton 'Download' pour télécharger tous les corpus.
   
7) Une fois le téléchargement terminé, fermez l'interface graphique et quittez l'environnement python par la commande suivante :
   >quit() 

L'utilisateur est maintenant prêt à exécuter les codes des projets.

### L'utilisateur n'a pas de python anaconda sur son appareil.
1) Installez miniconda pour Python 3.8 en suivant les instructions (basées sur le système) du lien ci-dessous :

   https://docs.conda.io/en/latest/miniconda.html

2) Ouvrir termainal (Mac, Linux) ou anaconda prompt (Miniconda3) en tant qu'administrateur (windows)

3) Déplacez-vous vers le répertoire contenant les fichiers du projet dézippé en utilisant la commande 'cd'

4) Créez un nouvel environnement virtuel avec la commande suivante :

   >conda create -n chatbot_api python=3.8.5

5) Activez l'environnement avec la commande suivante :

   >conda activate chatbot_api

6) Installez les paquets python nécessaires avec la commande suivante :

   >pip install -r requirements.txt

7) Téléchargez les fichiers nltk nécessaires par les commandes suivantes :
   >python
   
   >import nltk 
   
   >nltk.download()
 
   Dans la fenêtre GUI qui s'ouvre, il suffit de cliquer sur le bouton 'Download' pour télécharger tous les corpus.
   
8) Une fois le téléchargement terminé, fermez l'interface graphique et quittez l'environnement python par la commande suivante :
   >quit() 

L'utilisateur est maintenant prêt à exécuter les codes des projets.

## Exécuter
Pour exécuter les codes et l'API, il suffit à l'utilisateur d'exécuter la commande suivante :
>python API_Chatbot.py

Copiez ensuite le lien affiché dans le terminal et collez-le dans un navigateur Web. Le lien doit être celui-ci (http://localhost:5000/) ou quelque chose comme ça.

Enfin, dans le navigateur Web, l'utilisateur doit télécharger un ensemble de données de réglage fin (pour l'instant, il doit s'agir d'un des fichiers json du dossier "databases"), sélectionner le type de données (pour l'instant, il s'agit de json) et le domaine du modèle (chatbot), puis cliquer sur "submit". 

L'utilisateur peut interagir avec le chatbot nouvellement affiné via l'API. Cependant, comme expliqué ci-dessus, en raison de problèmes d'intégration, les réponses du chatbot n'ont pas de sens.

## Testez les chatbots localement
Pour tester les chatbots nouvellement réglés, l'utilisateur doit quitter l'API dans le terminal (CTRL+C) et exécuter l'une des commandes suivantes (une pour chaque domaine de chatbot) :
>python chat_dialog_test.py

ou 
>python chat_anecdote_test.py

ou
>python chat_voyage_test.py

Même en effectuant des tests au niveau local, les réponses peuvent encore sembler étranges, et ce en raison de la faible qualité des modèles eux-mêmes. 
