import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Create necessary directories
Path("models").mkdir(exist_ok=True)
Path("datasets").mkdir(exist_ok=True)

def load_dataset(dataset_name):
    """Load a dataset from the datasets directory"""
    try:
        df = pd.read_csv(Path("datasets") / dataset_name)
        return df
    except Exception as e:
        raise Exception(f"Error loading dataset: {str(e)}")

def split_data(df, target_column, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(model_type, X_train, y_train, params=None):
    """Train a machine learning model"""
    models = {
        "Logistic Regression": LogisticRegression,
        "Random Forest": RandomForestClassifier,
        "SVM": SVC,
        "XGBoost": XGBClassifier
    }
    
    if model_type not in models:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model_class = models[model_type]
    model = model_class(**params) if params else model_class()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1": f1_score(y_test, y_pred, average='weighted'),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None
    }
    
    cm = confusion_matrix(y_test, y_pred)
    
    return metrics, cm

def save_model(model, model_name):
    """Save a trained model to disk"""
    model_path = Path("models") / f"{model_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return model_path

def load_model(model_name):
    """Load a saved model from disk"""
    model_path = Path("models") / f"{model_name}.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def get_feature_importance(model, feature_names):
    """Get feature importance from a model"""
    if hasattr(model, 'feature_importances_'):
        return dict(zip(feature_names, model.feature_importances_))
    elif hasattr(model, 'coef_'):
        return dict(zip(feature_names, model.coef_[0]))
    else:
        return None

def generate_shap_plot(model, X, feature_names):
    """Generate SHAP values plot"""
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    return plt.gcf()