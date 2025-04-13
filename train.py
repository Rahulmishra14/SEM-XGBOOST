import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib
import warnings
import json

warnings.filterwarnings("ignore")

# Load data
data = pd.read_csv("./Data Set/energydata_complete.csv")
data.dropna(inplace=True)

# Selected features with importance >= 0.03
selected_features = [
    'T9', 'RH_out', 'T8', 'Press_mm_hg', 'RH_6', 'T5', 'T3',
    'T7', 'RH_8', 'RH_7', 'RH_5', 'Windspeed', 'T4', 'RH_2', 'RH_9'
]

# Define target
X = data[selected_features]
y = data['Appliances']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline with scaler + model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', XGBRegressor(objective='reg:squarederror', random_state=42))
])

# Hyperparameter search space
param_dist = {
    'model__n_estimators': [100, 200, 300, 400],
    'model__max_depth': [3, 5, 7, 10],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__subsample': [0.6, 0.8, 1.0],
    'model__colsample_bytree': [0.6, 0.8, 1.0],
    'model__gamma': [0, 0.1, 0.3, 0.5]
}

# Randomized Search
search = RandomizedSearchCV(
    pipeline, param_distributions=param_dist,
    n_iter=30, scoring='r2', cv=5,
    verbose=1, n_jobs=-1, random_state=42
)

search.fit(X_train, y_train)

# Evaluate
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
print("\n✅ Best R^2 on test set:", round(r2_score(y_test, y_pred), 4))
print("📉 MAE:", round(mean_absolute_error(y_test, y_pred), 2))
print("🎯 Best Parameters:", search.best_params_)

# Save model
joblib.dump(best_model, "./Models/tuned_appliance_energy_model.pkl")
print("✅ Model saved!")

# Save list of features used during training
with open('./Models/model_features.json', 'w') as f:
    json.dump(selected_features, f)
print("🧾 Trained feature names saved!")
