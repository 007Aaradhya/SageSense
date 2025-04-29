import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import sys
# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils

def show():
    st.title("🎯 Train Machine Learning Models")
    
    # Get available datasets
    datasets_dir = Path("datasets")
    available_datasets = list(datasets_dir.glob("*.csv"))
    
    if not available_datasets:
        st.warning("No datasets available. Please upload a dataset first in the Dataset Management page.")
        return
    
    # Dataset selection
    selected_dataset = st.selectbox(
        "Select Dataset",
        options=[f.name for f in available_datasets],
        format_func=lambda x: x.replace(".csv", "").replace("_", " ").title()
    )
    
    if selected_dataset:
        # Load the selected dataset
        try:
            df = pd.read_csv(datasets_dir / selected_dataset)
            
            # Display dataset info
            st.subheader("Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Rows:** {df.shape[0]}")
            with col2:
                st.write(f"**Columns:** {df.shape[1]}")
            with col3:
                missing_values = df.isnull().sum().sum()
                st.write(f"**Missing Values:** {missing_values}")
                
            # Show dataset sample
            with st.expander("Dataset Preview"):
                st.dataframe(df.head(), use_container_width=True)
                
                # Data types and missing values info
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Data Types:**")
                    st.write(df.dtypes)
                with col2:
                    st.write("**Missing Values:**")
                    st.write(df.isnull().sum())
            
            # Target selection
            st.subheader("1. Select Target Variable")
            target_column = st.selectbox("Select Target Column", df.columns)
            
            if target_column:
                # Check if task is classification or regression
                unique_values = df[target_column].nunique()
                is_categorical = df[target_column].dtype == 'object' or unique_values < 10
                
                task_type = "Classification" if is_categorical else "Regression"
                st.info(f"Detected task type: **{task_type}**")
                
                # Feature selection
                st.subheader("2. Select Features")
                
                # Automatically exclude non-numeric columns if too many columns
                if df.shape[1] > 15:
                    default_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                    default_features = [f for f in default_features if f != target_column]
                else:
                    default_features = [c for c in df.columns if c != target_column]
                
                selected_features = st.multiselect(
                    "Select Features",
                    options=[c for c in df.columns if c != target_column],
                    default=default_features
                )
                
                if not selected_features:
                    st.warning("Please select at least one feature to train the model.")
                    return
                
                # Handle missing values
                st.subheader("3. Handle Missing Values")
                cols_with_missing = df[selected_features + [target_column]].columns[df[selected_features + [target_column]].isnull().any()].tolist()
                
                if cols_with_missing:
                    st.warning(f"Missing values found in: {', '.join(cols_with_missing)}")
                    
                    missing_strategy = st.selectbox(
                        "Strategy for handling missing values",
                        options=["Drop rows with missing values", "Fill missing values (numeric: mean, categorical: mode)"]
                    )
                    
                    # Apply missing value strategy
                    if missing_strategy == "Drop rows with missing values":
                        df_clean = df.dropna(subset=selected_features + [target_column])
                        st.write(f"Rows after dropping missing values: {df_clean.shape[0]} (removed {df.shape[0] - df_clean.shape[0]} rows)")
                    else:
                        df_clean = df.copy()
                        for col in cols_with_missing:
                            if df[col].dtype in ['int64', 'float64']:
                                df_clean[col].fillna(df[col].mean(), inplace=True)
                            else:
                                df_clean[col].fillna(df[col].mode()[0], inplace=True)
                        st.write("Missing values filled with mean/mode")
                else:
                    df_clean = df.copy()
                    st.success("No missing values in the selected features and target.")
                
                # Train-test split
                st.subheader("4. Train-Test Split")
                test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
                random_state = st.number_input("Random State", 0, 999, 42)
                
                # Model selection
                st.subheader("5. Select Model")
                
                if task_type == "Classification":
                    models = {
                        "Random Forest": RandomForestClassifier,
                        "Logistic Regression": LogisticRegression,
                        "Support Vector Machine": SVC,
                        "K-Nearest Neighbors": KNeighborsClassifier,
                        "Decision Tree": DecisionTreeClassifier,
                        "Gradient Boosting": GradientBoostingClassifier
                    }
                else:
                    # Placeholder for regression models
                    st.error("Regression tasks are currently being developed. Please select a classification task.")
                    return
                
                selected_model = st.selectbox("Select ML Algorithm", list(models.keys()))
                
                # Model specific hyperparameters
                st.subheader("6. Model Hyperparameters")
                
                hyperparams = {}
                
                if selected_model == "Random Forest":
                    hyperparams['n_estimators'] = st.slider("Number of Trees", 10, 300, 100)
                    hyperparams['max_depth'] = st.slider("Maximum Depth", 1, 20, 10)
                    hyperparams['min_samples_split'] = st.slider("Minimum Samples Split", 2, 20, 2)
                    hyperparams['random_state'] = random_state
                
                elif selected_model == "Logistic Regression":
                    hyperparams['C'] = st.slider("Regularization Strength", 0.01, 10.0, 1.0)
                    hyperparams['max_iter'] = st.slider("Maximum Iterations", 100, 1000, 100)
                    hyperparams['random_state'] = random_state
                
                elif selected_model == "Support Vector Machine":
                    hyperparams['C'] = st.slider("Regularization Strength", 0.01, 10.0, 1.0)
                    hyperparams['kernel'] = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
                    hyperparams['random_state'] = random_state
                
                elif selected_model == "K-Nearest Neighbors":
                    hyperparams['n_neighbors'] = st.slider("Number of Neighbors", 1, 20, 5)
                    hyperparams['weights'] = st.selectbox("Weight Function", ["uniform", "distance"])
                
                elif selected_model == "Decision Tree":
                    hyperparams['max_depth'] = st.slider("Maximum Depth", 1, 20, 10)
                    hyperparams['min_samples_split'] = st.slider("Minimum Samples Split", 2, 20, 2)
                    hyperparams['random_state'] = random_state
                
                elif selected_model == "Gradient Boosting":
                    hyperparams['n_estimators'] = st.slider("Number of Estimators", 10, 300, 100)
                    hyperparams['learning_rate'] = st.slider("Learning Rate", 0.01, 1.0, 0.1)
                    hyperparams['max_depth'] = st.slider("Maximum Depth", 1, 10, 3)
                    hyperparams['random_state'] = random_state
                
                # Train model button
                if st.button("Train Model"):
                    with st.spinner(f"Training {selected_model}..."):
                        try:
                            # Prepare data
                            X = df_clean[selected_features]
                            y = df_clean[target_column]
                            
                            # Encode categorical variables
                            categorical_cols = X.select_dtypes(include=['object']).columns
                            
                            if len(categorical_cols) > 0:
                                st.info(f"Encoding categorical features: {', '.join(categorical_cols)}")
                                for col in categorical_cols:
                                    le = LabelEncoder()
                                    X[col] = le.fit_transform(X[col].astype(str))
                            
                            # Encode target if needed
                            if y.dtype == 'object':
                                le = LabelEncoder()
                                y = le.fit_transform(y)
                                class_names = le.classes_
                                st.info(f"Target classes: {', '.join(class_names)}")
                            else:
                                class_names = sorted(y.unique())
                            
                            # Split data
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=random_state
                            )
                            
                            st.write(f"Training set size: {X_train.shape[0]} samples")
                            st.write(f"Test set size: {X_test.shape[0]} samples")
                            
                            # Scale features
                            scaler = StandardScaler()
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_test_scaled = scaler.transform(X_test)
                            
                            # Initialize and train model
                            model = models[selected_model](**hyperparams)
                            model.fit(X_train_scaled, y_train)
                            
                            # Make predictions
                            y_pred = model.predict(X_test_scaled)
                            
                            # Calculate metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                            
                            # Display results
                            st.success(f"Model trained successfully! Accuracy: {accuracy:.4f}")
                            
                            # Create metrics tabs
                            metrics_tab, conf_matrix_tab, class_report_tab = st.tabs(["Metrics", "Confusion Matrix", "Classification Report"])
                            
                            with metrics_tab:
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("Accuracy", f"{accuracy:.4f}")
                                col2.metric("Precision", f"{precision:.4f}")
                                col3.metric("Recall", f"{recall:.4f}")
                                col4.metric("F1 Score", f"{f1:.4f}")
                            
                            with conf_matrix_tab:
                                cm = confusion_matrix(y_test, y_pred)
                                fig, ax = plt.subplots(figsize=(10, 8))
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                                ax.set_xlabel('Predicted')
                                ax.set_ylabel('Actual')
                                ax.set_title('Confusion Matrix')
                                st.pyplot(fig)
                            
                            with class_report_tab:
                                report = classification_report(y_test, y_pred, output_dict=True)
                                report_df = pd.DataFrame(report).transpose()
                                st.dataframe(report_df)
                            
                            # Feature importance (if available)
                            if hasattr(model, 'feature_importances_'):
                                st.subheader("Feature Importance")
                                feature_imp = pd.DataFrame({
                                    'Feature': X.columns,
                                    'Importance': model.feature_importances_
                                }).sort_values('Importance', ascending=False)
                                
                                fig = px.bar(
                                    feature_imp, x='Importance', y='Feature', 
                                    orientation='h', title='Feature Importance'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Save model
                            model_name = st.text_input("Model name for saving", f"{selected_model.replace(' ', '_').lower()}_{target_column.lower()}")
                            
                            if st.button("Save Model"):
                                # Create models directory if it doesn't exist
                                Path("models").mkdir(exist_ok=True)
                                
                                # Save model
                                model_path = Path("models") / f"{model_name}.pkl"
                                
                                # Create model info
                                model_info = {
                                    'name': model_name,
                                    'type': selected_model,
                                    'dataset': selected_dataset,
                                    'target': target_column,
                                    'features': selected_features,
                                    'metrics': {
                                        'accuracy': float(accuracy),
                                        'precision': float(precision),
                                        'recall': float(recall),
                                        'f1': float(f1)
                                    },
                                    'hyperparameters': hyperparams,
                                    'scaler': True  # Indicates that a scaler was used
                                }
                                
                                # Save model, scaler and info
                                with open(model_path, 'wb') as f:
                                    pickle.dump({
                                        'model': model,
                                        'scaler': scaler,
                                        'features': selected_features,
                                        'info': model_info
                                    }, f)
                                
                                st.success(f"Model saved as {model_path}")
                        
                        except Exception as e:
                            st.error(f"Error training model: {str(e)}")
                
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")

if __name__ == "__main__":
    show()