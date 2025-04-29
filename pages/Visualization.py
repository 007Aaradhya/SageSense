import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from utils import get_model_info
import os
from pathlib import Path
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from scipy import stats
import joblib

def show():
    st.title("Model Visualization & Monitoring")

    # Get available models and datasets
    models_dir = Path("models")
    datasets_dir = Path("datasets")

    if not models_dir.exists() or not any(models_dir.glob("*.pkl")):
        st.warning("No trained models found. Please train a model first.")
        return

    if not datasets_dir.exists() or not any(datasets_dir.glob("*.csv")):
        st.warning("No datasets available. Please upload a dataset first.")
        return

    # Sidebar for model and dataset selection
    with st.sidebar:
        st.subheader("Select Model & Dataset")
        model_files = [f for f in models_dir.glob("*.pkl")]
        selected_model = st.selectbox(
            "Select Model",
            options=[f.name for f in model_files],
            format_func=lambda x: x.replace(".pkl", "").replace("_", " ").title()
        )

        dataset_files = [f for f in datasets_dir.glob("*.csv")]
        selected_dataset = st.selectbox(
            "Select Dataset",
            options=[f.name for f in dataset_files],
            format_func=lambda x: x.replace(".csv", "").title()
        )

    if selected_model and selected_dataset:
        try:
            # Load model and dataset
            model_path = models_dir / selected_model
            model_data = joblib.load(model_path)
            model = model_data['model']
            feature_names = model_data['feature_names']
            target_column = model_data['target_column']

            df = pd.read_csv(datasets_dir / selected_dataset)

            # Get model information
            model_info = get_model_info(model_path)

            # Main content
            st.subheader("Model Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Model Type:", model_info["type"])
                st.write("Parameters:", model_info["parameters"])
            with col2:
                st.write("Dataset Shape:", df.shape)
                st.write("Features:", len(feature_names))

            # Feature Importance
            if model_info.get("feature_importance") is not None:
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model_info["feature_importance"]
                }).sort_values('Importance', ascending=False)

                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                            color='Importance', color_continuous_scale='Blues')
                st.plotly_chart(fig)

            # Check if target column exists in dataset
            if target_column in df.columns:
                # Data Quality Analysis
                st.subheader("Data Quality Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Missing Values:")
                    missing_values = df.isnull().sum()
                    fig = px.bar(x=missing_values.index, y=missing_values.values)
                    st.plotly_chart(fig)
                with col2:
                    st.write("Data Types:")
                    st.write(df.dtypes)

                # Prepare data for model analysis
                # Preprocess data to handle issues
                try:
                    # Handle categorical columns
                    categorical_cols = df.select_dtypes(include=['object']).columns
                    if len(categorical_cols) > 0:
                        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
                    
                    # Handle missing values
                    for col in df.columns:
                        if df[col].dtype in ['int64', 'float64']:
                            df[col] = df[col].fillna(df[col].median())
                        else:
                            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
                    
                    # Select only the feature columns known to the model
                    X = df[feature_names] if all(col in df.columns for col in feature_names) else None
                    if X is not None and target_column in df.columns:
                        y_true = df[target_column]
                        
                        # Make predictions
                        y_pred = model.predict(X)
                        
                        # Model Performance Metrics
                        st.subheader("Model Performance Metrics")
                        
                        # Classification Report
                        report = classification_report(y_true, y_pred)
                        st.text(report)
                        
                        # Confusion Matrix
                        cm = confusion_matrix(y_true, y_pred)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap="Blues")
                        plt.ylabel('Actual')
                        plt.xlabel('Predicted')
                        st.pyplot(fig)
                        
                        # ROC Curve (if applicable)
                        try:
                            if hasattr(model, 'predict_proba'):
                                y_pred_proba = model.predict_proba(X)[:, 1]
                                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                                roc_auc = auc(fpr, tpr)
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
                                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random'))
                                fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
                                st.plotly_chart(fig)
                        except Exception as e:
                            st.warning(f"Could not generate ROC curve: {str(e)}")
                    else:
                        st.warning("Cannot generate model metrics: Feature columns or target column not found in dataset")
                
                except Exception as e:
                    st.error(f"Error in data preprocessing: {str(e)}")
            else:
                st.warning(f"Target column '{target_column}' not found in selected dataset")

            # Feature Distributions
            st.subheader("Feature Distributions")
            # Select features for visualization
            numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            selected_features = st.multiselect("Select features to visualize", numeric_features, 
                                              default=numeric_features[:min(3, len(numeric_features))])
            
            if selected_features:
                for column in selected_features:
                    fig = px.histogram(df, x=column, marginal="box", title=f"Distribution of {column}")
                    st.plotly_chart(fig)
                    
                # Correlation heatmap
                st.subheader("Feature Correlation Matrix")
                corr_features = st.multiselect("Select features for correlation matrix", 
                                              numeric_features, 
                                              default=numeric_features[:min(6, len(numeric_features))])
                if corr_features:
                    corr_matrix = df[corr_features].corr()
                    fig = px.imshow(corr_matrix, color_continuous_scale="RdBu_r")
                    st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error processing model and data: {str(e)}")
            st.error("Please ensure the model and dataset are compatible.")