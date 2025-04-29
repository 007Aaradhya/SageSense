import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ks_2samp, ttest_ind, chi2_contingency
import joblib

st.set_page_config(page_title="Evidential AI", layout="wide")
st.title("Evidential AI")

# Ensure dataset selection persists across pages
if "selected_dataset" not in st.session_state:
    st.session_state.selected_dataset = None

if not st.session_state.selected_dataset:
    st.error("⚠️ No dataset selected! Please choose one on the landing page.")
    st.stop()

dataset_path = os.path.join("datasets", st.session_state.selected_dataset)
df = pd.read_csv(dataset_path)

st.success(f"✅ Loaded dataset: **{st.session_state.selected_dataset}**")


# -------------------------------
# Model Selection: Use Pre-Trained Model
# -------------------------------
import streamlit as st
import joblib
import os

st.write("### 📜 Dataset Preview")
st.dataframe(df.head())

# ✅ Define numeric & categorical columns
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# -------------------------------
# 🔄 Concept Drift Detection
# -------------------------------
st.write("### 🔄 Concept Drift Detection")

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["year"] = df["timestamp"].dt.year

    if numeric_cols:
        selected_col = st.selectbox("Select a feature for drift analysis:", numeric_cols)

        # Get recent & past data
        recent_data = df[df["year"] == df["year"].max()]
        past_data = df[df["year"] == df["year"].min()]

        # Kolmogorov-Smirnov Test for Concept Drift
        stat, p_value = ks_2samp(recent_data[selected_col], past_data[selected_col])
        st.write(f"📉 **{selected_col} Drift Test**: p-value = {p_value:.4f}")

        if p_value < 0.05:
            st.error(f"⚠️ Significant drift detected in {selected_col}!")
        else:
            st.success(f"✅ No significant drift in {selected_col}.")

        # 📊 Plot Feature Distributions Over Time
        fig_drift = px.line(df, x="timestamp", y=selected_col, title=f"Trend of {selected_col} Over Time", markers=True)
        st.plotly_chart(fig_drift, use_container_width=True)
else:
    st.warning("⚠️ No 'timestamp' column found. Concept drift requires a time-based dataset.")

# -------------------------------
# 📊 Data Drift Analysis
# -------------------------------
st.write("### 📊 Data Drift Analysis")

if numeric_cols:
    selected_col_drift = st.selectbox("Select a feature for data drift visualization:", numeric_cols)

    # 📊 Histogram
    fig_hist = px.histogram(df, x=selected_col_drift, title=f"Data Distribution for {selected_col_drift}", nbins=30)
    st.plotly_chart(fig_hist, use_container_width=True)

    # 📦 Box Plot
    fig_box = px.box(df, y=selected_col_drift, title=f"Box Plot for {selected_col_drift}")
    st.plotly_chart(fig_box, use_container_width=True)

    # 🎻 Violin Plot
    fig_violin = px.violin(df, y=selected_col_drift, box=True, points="all", title=f"Violin Plot of {selected_col_drift}")
    st.plotly_chart(fig_violin, use_container_width=True)

# -------------------------------
# 🧪 Hypothesis Testing
# -------------------------------
st.write("### 🧪 Hypothesis Testing")

if len(numeric_cols) > 1:
    col1, col2 = st.selectbox("Select two numeric columns:", [(numeric_cols[i], numeric_cols[i+1]) for i in range(len(numeric_cols)-1)])

    # Perform T-test
    stat, p_val = ttest_ind(df[col1], df[col2])
    st.write(f"📊 **T-test for {col1} and {col2}**: p-value = {p_val:.4f}")

    if p_val < 0.05:
        st.error("🚨 Significant difference found!")
    else:
        st.success("✅ No significant difference.")

# -------------------------------
# 🔢 Chi-Square Test for Categorical Data
# -------------------------------
st.write("### 🔢 Chi-Square Test for Categorical Data")

if len(categorical_cols) >= 2:
    cat_col1, cat_col2 = st.selectbox("Select two categorical columns:", [(categorical_cols[i], categorical_cols[i+1]) for i in range(len(categorical_cols)-1)])

    contingency_table = pd.crosstab(df[cat_col1], df[cat_col2])
    chi2, chi_p, _, _ = chi2_contingency(contingency_table)

    st.write(f"🔢 **Chi-Square Test for {cat_col1} and {cat_col2}**: p-value = {chi_p:.4f}")

    if chi_p < 0.05:
        st.error("🚨 Significant relationship found between the variables!")
    else:
        st.success("✅ No significant relationship.")

# -------------------------------
# 📈 Model Drift Analysis
# -------------------------------
st.write("### 📈 Model Drift Analysis")

model_accuracy_file = os.path.join("datasets", "model_accuracy.csv")

if os.path.exists(model_accuracy_file):
    accuracy_df = pd.read_csv(model_accuracy_file)

    if "date" in accuracy_df.columns and "accuracy" in accuracy_df.columns:
        # Plot Model Accuracy Over Time
        fig_model_drift = px.line(accuracy_df, x="date", y="accuracy", title="Model Accuracy Over Time", markers=True)
        st.plotly_chart(fig_model_drift, use_container_width=True)

        # Check for significant model drift
        if len(accuracy_df) > 1:
            latest_accuracy = accuracy_df["accuracy"].iloc[-1]
            previous_accuracy = accuracy_df["accuracy"].iloc[-2]
            accuracy_drop = previous_accuracy - latest_accuracy

            if accuracy_drop > 0.05:  # More than 5% drop in accuracy
                st.error(f"⚠️ Model drift detected! Accuracy dropped from {previous_accuracy:.2f} to {latest_accuracy:.2f}.")
            else:
                st.success(f"✅ Model accuracy stable at {latest_accuracy:.2f}.")
    else:
        st.warning("⚠️ The model accuracy file is missing required columns ('date' and 'accuracy').")
else:
    st.warning("⚠️ No model accuracy data found. Train & save accuracy data for drift analysis.")