# --- src/model_training.py ---
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # To save/load models and encoders
import os

# Ensure the 'models' directory exists
def ensure_models_dir(path='models/'):
    if not os.path.exists(path):
        os.makedirs(path)

def split_data(X, Y, test_size=0.2, random_state=42):
    """Splits data into training and testing sets."""
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=test_size, random_state=random_state, stratify=Y)
    return xtrain, xtest, ytrain, ytest

def train_xgboost_model(xtrain, ytrain_encoded):
    """Trains an XGBoost Classifier."""
    xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)
    xgb_model.fit(xtrain, ytrain_encoded)
    return xgb_model

def tune_xgboost_hyperparameters(xtrain, ytrain_encoded, save_best_model=False, model_path='models/best_salary_predictor_xgb_model.pkl'):
    """
    Performs GridSearchCV for XGBoost hyperparameter tuning.
    Returns the best estimator.
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0]
    }

    grid_search = GridSearchCV(estimator=XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42),
                               param_grid=param_grid,
                               cv=3,
                               n_jobs=-1,
                               verbose=2,
                               scoring='accuracy')

    print("Starting GridSearchCV for XGBoost...")
    grid_search.fit(xtrain, ytrain_encoded)

    print("\nBest parameters found:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    best_model = grid_search.best_estimator_

    if save_best_model:
        ensure_models_dir()
        joblib.dump(best_model, model_path)
        print(f"Best XGBoost model saved to {model_path}")
        
    return best_model

def evaluate_model(model, xtest, ytest, income_label_encoder=None):
    """Evaluates the trained model and prints performance metrics."""
    ypred_encoded = model.predict(xtest)
    
    # Decode predictions if an encoder is provided
    if income_label_encoder:
        ypred = income_label_encoder.inverse_transform(ypred_encoded)
        print("\nClassification Report (decoded labels):")
        print(classification_report(ytest, ypred))
        print("\nConfusion Matrix (decoded labels):")
        print(confusion_matrix(ytest, ypred))
        accuracy = accuracy_score(ytest, ypred)
    else:
        # If no income_label_encoder, assume ytest is already encoded or use raw predictions
        accuracy = accuracy_score(ytest, ypred_encoded)
        print("\nClassification Report (raw encoded labels):")
        print(classification_report(ytest, ypred_encoded))
        print("\nConfusion Matrix (raw encoded labels):")
        print(confusion_matrix(ytest, ypred_encoded))
        
    print(f"\nAccuracy Score: {accuracy:.4f}")

    return accuracy