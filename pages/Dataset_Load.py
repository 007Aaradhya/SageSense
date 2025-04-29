import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np

st.set_page_config(page_title="Data Exploration", layout="wide")
st.title("Dataset Visualization")

# Ensure dataset selection persists
if "selected_dataset" not in st.session_state or not st.session_state.selected_dataset:
    st.error("⚠️ No dataset selected! Please choose one on the landing page.")
    st.stop()

dataset_path = os.path.join("datasets", st.session_state.selected_dataset)
df = pd.read_csv(dataset_path)

st.success(f"✅ Loaded dataset: **{st.session_state.selected_dataset}**")

# Dataset Preview
st.write("### 📜 Dataset Preview")
st.dataframe(df.head())

# Data Statistics
st.write("### 📊 Dataset Summary")
st.write(df.describe())

# Missing Values
st.write("### ❗ Missing Values in Dataset")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_data = pd.DataFrame({"Missing Values": missing_values, "Percentage": missing_percentage})
missing_data = missing_data[missing_data["Missing Values"] > 0]

if missing_data.empty:
    st.success("✅ No missing values found in the dataset!")
else:
    st.write(missing_data)

# Visualization Options
st.write("### 📈 Data Visualizations")
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

if numeric_cols:
    selected_column = st.selectbox("Select a column for histogram:", numeric_cols)
    
    # Histogram
    fig_hist = px.histogram(df, x=selected_column, nbins=30, title=f"📊 Distribution of {selected_column}")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Box Plot (For Outliers)
    st.write("### 📦 Box Plot (Outlier Detection)")
    fig_box = px.box(df, y=selected_column, title=f"📦 Box Plot of {selected_column}")
    st.plotly_chart(fig_box, use_container_width=True)

    # Pair Plot (Feature Relationships)
    st.write("### 🔗 Pair Plot (Feature Relationships)")
    if len(numeric_cols) > 5:
        st.warning("⚠️ Too many numeric columns! Select a smaller subset for pair plots.")
    else:
        fig_pair = px.scatter_matrix(df, dimensions=numeric_cols, title="🔗 Pair Plot")
        st.plotly_chart(fig_pair, use_container_width=True)

    # Correlation Heatmap
    st.write("### 🔥 Correlation Heatmap")
    correlation_matrix = df[numeric_cols].corr()

    if len(numeric_cols) > 10:
        st.warning("⚠️ Too many numeric columns! Select a smaller subset for correlation analysis.")
    else:
        fig_heatmap = ff.create_annotated_heatmap(
            z=correlation_matrix.values,
            x=list(correlation_matrix.columns),
            y=list(correlation_matrix.index),
            annotation_text=np.round(correlation_matrix.values, 2),
            colorscale="Blues",
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

else:
    st.warning("⚠️ No numeric columns available for visualization.")