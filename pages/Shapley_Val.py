import streamlit as st
import pandas as pd
import os
import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Set Streamlit Page Configuration
st.set_page_config(page_title="Shapley Values", layout="wide")
st.title("SHAP Values: Model's Interpretability & Accuracy")

# Directories for models and datasets
model_dir = "models"
dataset_dir = "datasets"

# Ensure directories exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(dataset_dir, exist_ok=True)

# Get available models and datasets
available_models = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
available_datasets = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]

# Sidebar selections
selected_model = st.sidebar.selectbox("Select a trained model", ["None"] + available_models)
selected_dataset = st.sidebar.selectbox("Select a dataset", ["None"] + available_datasets)

if selected_model == "None" or selected_dataset == "None":
    st.warning("⚠️ Please select a trained model and dataset.")
    st.stop()

# Load dataset
df = pd.read_csv(os.path.join(dataset_dir, selected_dataset))

# Ensure dataset is not empty
if df.empty:
    st.error("⚠️ The selected dataset is empty. Please choose another dataset.")
    st.stop()

# Data Preprocessing
st.write("### Data Preprocessing")
st.write("🔍 Ensuring numeric columns are correctly formatted...")

# Convert numerical columns to proper format
for col in df.columns:
    if df[col].dtype == "O":  # If column is object type
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to numeric
        except Exception as e:
            st.warning(f"⚠️ Could not convert column '{col}' to numeric: {e}")

# Fill missing values with median (can be changed to mean or another method)
df.fillna(df.median(numeric_only=True), inplace=True)

# Display cleaned data
st.write(df.head())

# Select Target Variable
target = st.selectbox("Select Target Variable", df.columns)

# Ensure target is not empty or categorical
if df[target].dtype not in [np.int64, np.float64]:
    st.warning("⚠️ The selected target variable is not numeric. Choose a numerical target.")
    st.stop()

# Prepare Features
X = df.drop(columns=[target])
X = pd.get_dummies(X, drop_first=True)  # Encode categorical variables

# Load model
model_path = os.path.join(model_dir, selected_model)
try:
    model = joblib.load(model_path)
    st.success(f"✅ Model **{selected_model}** loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# Ensure features match the trained model
missing_cols = set(model.feature_names_in_) - set(X.columns)
for col in missing_cols:
    X[col] = 0  # Add missing columns with default values

X = X[model.feature_names_in_]  # Reorder columns

# SHAP Explanation
st.write("### SHAP Values: Feature Contributions to Predictions")

try:
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
except Exception as e:
    st.error(f"❌ SHAP computation failed: {e}")
    st.stop()

# Waterfall Plot for First Prediction
st.write("#### Feature Contribution for First Data Point")
try:
    fig, ax = plt.subplots()
    shap.waterfall_plot(shap_values[0])
    st.pyplot(fig)
except ValueError as e:
    if "Image size" in str(e):
        st.error("⚠️ SHAP is trying to generate an extremely large plot. Matplotlib has a size limit (2^23 pixels), and your plot exceeds this. Try reducing the number of features or selecting a different plot.")
    else:
        st.error(f"⚠️ Unexpected error: {e}")

# Summary Plot for Feature Importance
st.write("#### Feature Importance Across All Predictions")
try:
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig)
except ValueError as e:
    if "Image size" in str(e):
        st.error("⚠️ SHAP is trying to generate an extremely large plot. Matplotlib has a size limit (2^23 pixels), and your plot exceeds this. Try reducing the number of features or selecting a different plot.")
    else:
        st.error(f"⚠️ Unexpected error: {e}")

# SHAP Dependence Plot
feature_for_dependence = st.selectbox("Select Feature for Dependence Plot", X.columns)
try:
    fig_dependence = shap.plots.scatter(shap_values[:, feature_for_dependence])
    st.pyplot(fig_dependence)
except ValueError as e:
    if "Image size" in str(e):
        st.error("⚠️ SHAP is trying to generate an extremely large plot. Matplotlib has a size limit (2^23 pixels), and your plot exceeds this. Try reducing the number of features or selecting a different plot.")
    else:
        st.error(f"⚠️ Unexpected error: {e}")

# Accuracy Calculation
st.write("### Model Accuracy on Dataset")
y_true = df[target]
y_pred = model.predict(X)
accuracy = np.mean(y_true == y_pred)

st.success(f"✅ Model Accuracy: **{accuracy:.2%}**")