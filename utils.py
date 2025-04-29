import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from pathlib import Path
import numpy as np

MODELS = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
    "SVM": SVC,
    "XGBoost": XGBClassifier
}

def preprocess_data(df, target_column):
    """
    Preprocess the data by encoding categorical columns, handling missing values,
    and removing unwanted columns (e.g., CustomerID).
    
    Args:
        df: DataFrame containing the data
        target_column: The target column to exclude from the DataFrame
    
    Returns:
        df: DataFrame with categorical columns encoded, missing values handled,
            and unwanted columns removed
    """
    # Make a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Drop the 'CustomerID' column if it exists
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'], errors='ignore')
    
    # Check if target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")

    # Handle missing values - replace with appropriate method per column type
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            # For numeric columns, fill with median
            df[col] = df[col].fillna(df[col].median())
        else:
            # For categorical columns, fill with most frequent value
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    # Convert categorical columns to one-hot encoding
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df

def train_and_save_model(df, target_column, model_name, test_size=0.2, random_state=42):
    """
    Train a model and save it to disk.

    Args:
        df: DataFrame containing the data
        target_column: Name of the target column
        model_name: Name of the model to train
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        tuple: (accuracy, model_path, y_test, y_pred, feature_importance)
    """
    try:
        # Preprocess data to handle categorical variables
        df = preprocess_data(df, target_column)

        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Initialize and train model
        model_class = MODELS[model_name]
        model = model_class()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Extract feature importance (if available)
        feature_importance = None
        if model_name in ["Random Forest", "XGBoost"]:
            feature_importance = model.feature_importances_
        elif model_name == "Logistic Regression":
            # For logistic regression, use coefficients as feature importance
            if hasattr(model, 'coef_'):
                feature_importance = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)

        # Save model and feature names
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model_filename = f"models/{model_name.replace(' ', '_')}_{random_state}.pkl"

        # Create a dictionary containing both the model and feature names
        model_data = {
            'model': model,
            'feature_names': X.columns.tolist(),
            'target_column': target_column,
            'feature_importance': feature_importance
        }

        joblib.dump(model_data, model_filename)

        return accuracy, model_filename, y_test, y_pred, feature_importance
    
    except Exception as e:
        raise ValueError(f"Error during model training: {str(e)}")

def load_model(model_path):
    """Load a trained model from disk."""
    try:
        model_data = joblib.load(model_path)
        return model_data['model']
    except Exception as e:
        raise ValueError(f"Error loading model: {str(e)}")

def predict(model, df):
    """Make predictions using a trained model."""
    try:
        return model.predict(df)
    except Exception as e:
        raise ValueError(f"Error during prediction: {str(e)}")

def get_model_info(model_path):
    """Get basic information about a trained model."""
    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        feature_names = model_data['feature_names']
        target_column = model_data['target_column']
        feature_importance = model_data.get('feature_importance', None)

        return {
            "type": type(model).__name__,
            "parameters": model.get_params(),
            "feature_names": feature_names,
            "target_column": target_column,
            "feature_importance": feature_importance
        }
    except Exception as e:
        raise ValueError(f"Error getting model info: {str(e)}")