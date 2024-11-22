import pandas as pd
from sklearn.model_selection import train_test_split

# Charger les données depuis le fichier CSV raw
df = pd.read_csv('data/raw_data/raw.csv')

# Définir les features (X) et la cible (y)
# Ici, 'silica_concentrate' est la variable cible et se trouve dans la dernière colonne
X = df.drop(columns=['silica_concentrate'])
y = df['silica_concentrate']

# Split des données en ensembles d'entraînement et de test (80% entraînement, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sauvegarder les datasets d'entraînement et de test
X_train.to_csv('data/processed_data/X_train.csv', index=False)
X_test.to_csv('data/processed_data/X_test.csv', index=False)
y_train.to_csv('data/processed_data/y_train.csv', index=False)
y_test.to_csv('data/processed_data/y_test.csv', index=False)

print("Les données ont été divisées et enregistrées avec succès.")
