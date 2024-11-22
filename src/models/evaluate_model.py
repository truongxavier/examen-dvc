import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

# Charger le modèle et les données
model = joblib.load('models/trained_model.joblib')
X_test = pd.read_csv('data/processed_data/X_test_scaled.csv')
y_test = pd.read_csv('data/processed_data/y_test.csv')

# Convertir y_test en un tableau unidimensionnel si nécessaire
if y_test.ndim > 1:
    y_test = y_test.squeeze()

# Prédictions du modèle
y_pred = model.predict(X_test)

# Vérifier que y_pred est unidimensionnel
if len(y_pred.shape) > 1:
    y_pred = y_pred.squeeze()

# Calcul des métriques
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Sauvegarder les métriques dans un fichier JSON
metrics = {'mse': mse, 'r2': r2}
with open('metrics/scores.json', 'w') as f:
    json.dump(metrics, f)

# Sauvegarder les prédictions dans un DataFrame
predictions_df = pd.DataFrame({
    'y_test': y_test,
    'y_pred': y_pred
})

# Sauvegarder les prédictions dans un fichier CSV
predictions_df.to_csv('data/processed_data/predictions.csv', index=False)

print("Évaluation du modèle réussie. Les métriques ont été enregistrées et les prédictions ont été sauvegardées.")
