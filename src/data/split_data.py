import pandas as pd
from sklearn.model_selection import train_test_split

# Charger les données
df = pd.read_csv('data/raw/raw.csv')

# Définir les features et la cible
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sauvegarder les datasets
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)
