import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import joblib

# Charger les données normalisées
X_train = pd.read_csv('data/processed/X_train_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv')

# Définir le modèle et les paramètres à tester
model = Ridge()
parameters = {'alpha': [0.1, 1.0, 10.0]}

# GridSearch
grid_search = GridSearchCV(model, parameters, cv=5)
grid_search.fit(X_train, y_train)

# Sauvegarder les meilleurs paramètres
joblib.dump(grid_search.best_params_, 'models/best_params.pkl')
