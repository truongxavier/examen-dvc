import pandas as pd
from sklearn.linear_model import Ridge
import joblib

# Charger les données normalisées
X_train = pd.read_csv('data/processed/X_train_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv')

# Charger les meilleurs paramètres
best_params = joblib.load('models/best_params.pkl')

# Entraîner le modèle
model = Ridge(**best_params)
model.fit(X_train, y_train)

# Sauvegarder le modèle entraîné
joblib.dump(model, 'models/trained_model.joblib')
