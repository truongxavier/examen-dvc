import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

# Charger le modèle et les données
model = joblib.load('models/trained_model.joblib')
X_test = pd.read_csv('data/processed/X_test_scaled.csv')
y_test = pd.read_csv('data/processed/y_test.csv')

# Prédictions
y_pred = model.predict(X_test)

# Calcul des métriques
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Sauvegarder les métriques
metrics = {'mse': mse, 'r2': r2}
with open('metrics/scores.json', 'w') as f:
    json.dump(metrics, f)

# Sauvegarder les prédictions
pd.DataFrame({'y_test': y_test, 'y_pred': y_pred}).to_csv('data/processed/predictions.csv', index=False)
