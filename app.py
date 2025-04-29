import streamlit as st
import os
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(page_title="SageSense", layout="wide")

st.title("SageeSense")

st.sidebar.header("📂 Upload/Select a Dataset")

# Ensure directories exist
dataset_dir = "datasets"
model_dir = "models"
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# List available datasets
available_datasets = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]

# File uploader for dataset
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Dataset selection from stored datasets
selected_dataset = st.sidebar.selectbox("Or select a pre-stored dataset", ["None"] + available_datasets)

# Clear selection button
if st.sidebar.button("Clear Selection"):
    uploaded_file = None
    selected_dataset = "None"
    st.session_state.selected_dataset = None
    st.experimental_rerun()

# Load dataset
df = None

if uploaded_file:
    try:
        # Save uploaded file to datasets folder
        dataset_path = os.path.join(dataset_dir, uploaded_file.name)
        with open(dataset_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        df = pd.read_csv(dataset_path, encoding="utf-8")
        st.session_state.selected_dataset = uploaded_file.name
        st.success(f"✅ Uploaded dataset **{uploaded_file.name}** saved and loaded successfully!")

    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")

elif selected_dataset != "None":
    try:
        dataset_path = os.path.join(dataset_dir, selected_dataset)
        df = pd.read_csv(dataset_path, encoding="utf-8")
        st.session_state.selected_dataset = selected_dataset
        st.success(f"✅ Pre-stored dataset **{selected_dataset}** loaded successfully!")

    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")

# Display dataset preview
if df is not None:
    st.write("### 📊 Dataset Preview")
    st.dataframe(df, height=400)

    # Save dataset info
    if st.button("💾 Save Dataset Info"):
        dataset_info = {
            "filename": st.session_state.selected_dataset,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "columns_list": df.columns.tolist()
        }
        dataset_info_path = os.path.join(dataset_dir, "dataset_info.pkl")
        joblib.dump(dataset_info, dataset_info_path)
        st.success("✅ Dataset info saved successfully!")


# Navigate to other pages
st.markdown("### 🔍 Use left sidebar to navigate to other pages.")