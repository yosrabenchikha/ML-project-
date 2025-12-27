
🎯 Qu'est-ce que c'est ?
Un moteur de recommandation musicale qui analyse 232,725 chansons Spotify pour vous suggérer des morceaux similaires à ceux que vous aimez.
Exemple : Vous aimez "Bohemian Rhapsody" ? Le système vous recommande "Don't Stop Me Now", "We Will Rock You" et d'autres chansons rock classiques.

✨ Fonctionnalités Principales
🎵 1. Recommandations Intelligentes

Recherche par nom de chanson
Filtrage par artiste
Découverte de nouveaux artistes similaires
Recommandations basées sur 3 critères :

Audio : énergie, danceability, acoustique...
Genre : rock, pop, jazz...
Artiste : style musical



📊 2. Analyse Avancée

Visualisation des profils musicaux
Graphiques de similarité (t-SNE)
Comparaison des recommandations
Explication des suggestions

⚙️ 3. Personnalisation

Ajuster l'importance de chaque critère
Exclure certains artistes
Contrôler le nombre de recommandations

Dataset : https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db
Fichier : SpotifyFeatures.csv
🚀 Ce qui rend ce projet unique
💡 Approche Hybride Multi-Niveaux
Contrairement aux systèmes simples qui utilisent une seule méthode, ce projet combine 3 techniques avancées :
1. Encodage Intelligent des Genres
27 genres musicaux → Clustering → 5-8 super-genres
        ↓
    PCA (2-3 dimensions)
        ↓
    Encodage numérique
Avantage : Capture les relations entre genres (ex: rock et metal sont proches)
2. Encodage des Artistes
Pour chaque artiste → Calcul du profil audio moyen
        ↓
    PCA (3 dimensions)
        ↓
    Représentation du style musical
Avantage : Identifie le "son" caractéristique de chaque artiste
3. Distance Pondérée Personnalisable
Distance Finale = 
    0.7 × Distance_Audio +      (caractéristiques sonores)
    0.2 × Distance_Genre +      (style musical)
    0.1 × Distance_Artiste      (signature de l'artiste)
Avantage : Vous contrôlez ce qui compte le plus pour vous


### **Pipeline en 5 Étapes**
```
1️⃣ NETTOYAGE
   └─→ Suppression doublons, gestion NaN
   
2️⃣ ENCODAGE
   ├─→ Genre : PCA + Clustering
   ├─→ Artiste : Profil audio moyen
   └─→ Normalisation StandardScaler
   
3️⃣ RÉDUCTION DIMENSIONNALITÉ
   └─→ PCA : 30 → 15 composantes (72.5% variance)
   
4️⃣ CLUSTERING
   └─→ K-means : clustering de chansons similaires
   
5️⃣ RECOMMANDATION
   └─→ Distance hybride pondérée
```

### **Technologies Utilisées**
- **Scikit-learn** : K-means, PCA, t-SNE
- **Pandas/NumPy** : Manipulation de données
- **Matplotlib/Seaborn** : Visualisations
