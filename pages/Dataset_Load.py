import streamlit as st
import pandas as pd
import os
import numpy as np
from pathlib import Path
import requests
from io import StringIO
import plotly.express as px

def load_sample_dataset(dataset_name):
    """Load a sample dataset with enhanced error handling."""
    datasets = {
        "Titanic": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "Iris": "https://raw.githubusercontent.com/uci-ml/datasets/main/iris/iris.csv",
        "Wine Quality": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "Heart Disease": "https://raw.githubusercontent.com/ronitgavaskar/datasets/master/Heart.csv"
    }
    
    iris_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    
    try:
        response = requests.get(datasets[dataset_name])
        response.raise_for_status()  # Raise HTTPError for bad responses
        
        if dataset_name == "Iris":
            df = pd.read_csv(StringIO(response.text), header=None, names=iris_columns)
        elif dataset_name == "Wine Quality":
            df = pd.read_csv(StringIO(response.text), sep=';')
        else:
            df = pd.read_csv(StringIO(response.text))
            
        return df
    
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download dataset: {str(e)}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        st.error("Received empty dataset")
        return pd.DataFrame()

def show():
    st.title("📊 Dataset Management")
    Path("datasets").mkdir(exist_ok=True)
    
    tab1, tab2 = st.tabs(["Upload Dataset", "Sample Datasets"])
    
    with tab1:
        st.subheader("Upload Your Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="file_uploader")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Generate default name if not provided
                default_name = uploaded_file.name if uploaded_file.name.endswith('.csv') else f"{uploaded_file.name}.csv"
                save_name = st.text_input("Save dataset as:", default_name)
                
                overwrite = st.checkbox("Overwrite existing file?")
                if st.button("Save Uploaded Dataset", key="save_uploaded"):
                    save_path = Path("datasets") / save_name
                    
                    if save_path.exists() and not overwrite:
                        st.error(f"File {save_name} exists! Enable overwrite.")
                    else:
                        df.to_csv(save_path, index=False)
                        st.success(f"Dataset saved as {save_path}")
                        st.write(f"Size: {save_path.stat().st_size/1024:.2f} KB")
                
                show_dataset_stats(df)
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab2:
        st.subheader("Sample Datasets")
        sample_dataset = st.selectbox(
            "Select a sample dataset",
            ["Titanic", "Iris", "Wine Quality", "Heart Disease"],
            key="sample_select"
        )
        
        descriptions = {
            "Titanic": "Passenger data from the Titanic shipwreck (1309 rows, 14 cols)",
            "Iris": "Measurements of iris flowers (150 rows, 5 cols)",
            "Wine Quality": "Physicochemical properties of red wines (1599 rows, 12 cols)",
            "Heart Disease": "Clinical heart disease data (303 rows, 14 cols)"
        }
        
        st.markdown(f"**{sample_dataset}**: {descriptions[sample_dataset]}")
        
        if st.button("Load Sample Dataset", key="load_sample"):
            with st.spinner(f"Loading {sample_dataset}..."):
                df = load_sample_dataset(sample_dataset)
                
                if not df.empty:
                    st.success(f"Successfully loaded {sample_dataset} dataset!")
                    st.dataframe(df.head())
                    
                    save_name = f"{sample_dataset.lower().replace(' ', '_')}.csv"
                    save_path = Path("datasets") / save_name
                    
                    if st.button(f"Save {sample_dataset} Dataset", key=f"save_{sample_dataset}"):
                        if save_path.exists() and not st.checkbox("Overwrite existing file?", key=f"overwrite_{sample_dataset}"):
                            st.error("File exists! Enable overwrite.")
                        else:
                            df.to_csv(save_path, index=False)
                            st.success(f"Saved as {save_name}")
                    
                    show_dataset_stats(df)

    show_available_datasets()

def show_dataset_stats(df):
    """Display dataset statistics and visualizations."""
    st.subheader("Dataset Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])
    with col2:
        st.metric("Missing Values", df.isnull().sum().sum())
        st.metric("Duplicates", df.duplicated().sum())
    
    # Show data types and memory usage
    st.subheader("Data Types")
    dtype_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes,
        'Unique': df.nunique(),
        'Missing': df.isnull().sum()
    })
    st.dataframe(dtype_info, use_container_width=True)
    
    # Visualizations
    st.subheader("Quick Visualizations")
    viz_col = st.selectbox("Select column to visualize", df.columns)
    
    if pd.api.types.is_numeric_dtype(df[viz_col]):
        fig = px.histogram(df, x=viz_col, title=f"Distribution of {viz_col}")
    else:
        fig = px.bar(df[viz_col].value_counts(), title=f"Count of {viz_col}")
    st.plotly_chart(fig)

def show_available_datasets():
    """Show and manage existing datasets."""
    st.subheader("Available Datasets")
    datasets = list(Path("datasets").glob("*.csv"))
    
    if not datasets:
        st.info("No datasets available yet.")
        return
    
    selected = st.selectbox(
        "Select a dataset to view", 
        [d.name for d in datasets],
        key="dataset_select"
    )
    
    if selected:
        df = pd.read_csv(Path("datasets") / selected)
        st.dataframe(df.head())
        
        if st.button(f"Delete {selected}", key=f"delete_{selected}"):
            (Path("datasets") / selected).unlink()
            st.success(f"Deleted {selected}")
            st.experimental_rerun()

if __name__ == "__main__":
    show()