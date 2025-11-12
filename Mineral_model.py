import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import json

print("Loading and cleaning data...")
master_df = pd.read_csv('master_mineral_data.csv')

if 'Total' in master_df.columns:
    master_df.drop(columns=['Total'], inplace=True, errors='ignore')
if 'Source_File' in master_df.columns:
    master_df.drop(columns=['Source_File'], inplace=True, errors='ignore')

print("Data loaded successfully.")
print("Data shape:", master_df.shape)

print("\nSplitting data into training and hold-out test sets...")
data_for_training, holdout_test_set = train_test_split(
    master_df, 
    test_size=0.15, 
    random_state=42, 
    stratify=master_df['Deposit_Type']
)

holdout_test_set.to_csv('holdout_test_data.csv', index=False)

print("Data for Training:", data_for_training.shape)
print("Hold-out Test Set:", holdout_test_set.shape)
print("✅ Hold-out test data saved to 'holdout_test_data.csv'")


print("\nPreparing training data (X and y)...")

X = data_for_training.drop('Deposit_Type', axis=1)
y_text = data_for_training['Deposit_Type']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

print("Text labels:", label_encoder.classes_)
print("Encoded labels:", np.unique(y))

print("\nBuilding the Scikit-learn pipeline...")
imputer = SimpleImputer(strategy='constant', fill_value=0)

rf_classifier = RandomForestClassifier(random_state=42)

model_pipeline = Pipeline(steps=[
    ('imputer', imputer),
    ('classifier', rf_classifier)
])
print("Pipeline created successfully with ZERO imputation strategy.")

param_grid = {
    'classifier__n_estimators': [100, 150],
    'classifier__max_depth': [5, 10, 15],
    'classifier__min_samples_split': [5, 10],
    'classifier__min_samples_leaf': [2, 4]
}
print("\nParameter grid for tuning defined.")

grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

print("Starting Grid Search... (This may take a few minutes)")
grid_search.fit(X, y)

print("\n--- Grid Search Complete ---")
print("Best Parameters Found:", grid_search.best_params_)
print("Best Cross-Validated Accuracy:", grid_search.best_score_)

print("\nCalculating final metrics and saving assets...")

final_model = grid_search.best_estimator_

validation_score = grid_search.best_score_
training_score = final_model.score(X, y)
overfitting_gap = training_score - validation_score

print(f"\nCross-Validation Accuracy (on unseen folds): {validation_score:.4f}")
print(f"Full Training Set Accuracy (on seen data):   {training_score:.4f}")
print(f"Overfitting Gap:                              {overfitting_gap:.4f}")

model_features = X.columns.tolist()

model_performance = {
    "cross_validation_accuracy": validation_score,
    "full_training_set_accuracy": training_score,
    "overfitting_gap": overfitting_gap,
    "best_parameters": grid_search.best_params_
}

joblib.dump(final_model, 'mineral_deposit_classifier_sklearn.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(model_features, 'model_features.pkl')
with open('model_performance.json', 'w') as f:
    json.dump(model_performance, f, indent=4)

print("\nAll assets saved successfully:")
print("- mineral_deposit_classifier_sklearn.pkl")
print("- label_encoder.pkl")
print("- model_features.pkl")
print("- model_performance.json")