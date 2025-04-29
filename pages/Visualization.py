import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils import load_dataset

def show():
    st.title("📈 Data Visualization & Analysis")
    
    # Get available datasets
    datasets_dir = Path("datasets")
    available_datasets = list(datasets_dir.glob("*.csv"))
    
    if not available_datasets:
        st.warning("No datasets available. Please upload a dataset first.")
        return
    
    # Dataset selection
    selected_dataset = st.selectbox(
        "Select Dataset",
        options=[f.name for f in available_datasets],
        format_func=lambda x: x.replace(".csv", "").replace("_", " ").title()
    )
    
    if selected_dataset:
        try:
            df = load_dataset(selected_dataset)
            
            # Display dataset info
            st.subheader("Dataset Overview")
            st.dataframe(df.head())
            
            # Visualization options
            st.subheader("Visualization Options")
            viz_type = st.selectbox(
                "Choose Visualization Type",
                ["Distribution Plot", "Scatter Plot", "Correlation Matrix", "Box Plot", "Pair Plot"]
            )
            
            if viz_type == "Distribution Plot":
                column = st.selectbox("Select Column", df.columns)
                fig = px.histogram(df, x=column, title=f"Distribution of {column}")
                st.plotly_chart(fig)
                
            elif viz_type == "Scatter Plot":
                col1, col2 = st.columns(2)
                with col1:
                    x_axis = st.selectbox("X Axis", df.columns)
                with col2:
                    y_axis = st.selectbox("Y Axis", df.columns)
                
                color_col = st.selectbox("Color By", [None] + list(df.columns))
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col)
                st.plotly_chart(fig)
                
            elif viz_type == "Correlation Matrix":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) < 2:
                    st.warning("Need at least 2 numeric columns for correlation matrix")
                else:
                    corr = df[numeric_cols].corr()
                    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig)
                    
            elif viz_type == "Box Plot":
                column = st.selectbox("Select Column", df.select_dtypes(include=[np.number]).columns)
                fig = px.box(df, y=column)
                st.plotly_chart(fig)
                
            elif viz_type == "Pair Plot":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) < 2:
                    st.warning("Need at least 2 numeric columns for pair plot")
                else:
                    selected_cols = st.multiselect("Select Columns", numeric_cols, default=numeric_cols[:3])
                    if len(selected_cols) >= 2:
                        fig = px.scatter_matrix(df[selected_cols])
                        st.plotly_chart(fig)
                    else:
                        st.warning("Select at least 2 columns")
            
            # Statistical analysis
            st.subheader("Statistical Analysis")
            if st.checkbox("Show Descriptive Statistics"):
                st.dataframe(df.describe())
                
            if st.checkbox("Show Missing Values Analysis"):
                missing = df.isnull().sum().reset_index()
                missing.columns = ['Column', 'Missing Count']
                missing = missing[missing['Missing Count'] > 0]
                if not missing.empty:
                    st.dataframe(missing)
                    fig = px.bar(missing, x='Column', y='Missing Count')
                    st.plotly_chart(fig)
                else:
                    st.success("No missing values found")
                    
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")

if __name__ == "__main__":
    show()