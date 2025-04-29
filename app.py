import streamlit as st
import os
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="SageSense - ML App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create necessary directories if they don't exist
Path("models").mkdir(exist_ok=True)
Path("datasets").mkdir(exist_ok=True)

# Main page content
st.title("🧠 SageSense")
st.subheader("A comprehensive platform for machine learning model development")

st.markdown("""
### Welcome to SageSense!

SageSense is a powerful web application for training, analyzing, and deploying machine learning models with an intuitive user interface.

**Getting Started:**
1. Use the sidebar to navigate between different pages
2. Start by uploading or selecting a dataset in the Dataset Load page
3. Train your models with customizable parameters
4. Make predictions on new data
5. Analyze model performance with interactive visualizations

**Features:**
- Support for multiple ML algorithms
- Comprehensive model performance metrics
- Interactive visualizations
- Dataset management
- Prediction generation
""")

# Show a quick guide
st.subheader("Quick Guide")
col1, col2, col3 = st.columns(3)

with col1:
    st.info("**📊 Dataset Management**\n- Upload and store datasets\n- Load sample datasets\n- Easy dataset selection")

with col2:
    st.info("**🎯 Model Training**\n- Multiple ML algorithms\n- Customizable parameters\n- Performance metrics")

with col3:
    st.info("**📈 Visualization & Analysis**\n- Model performance metrics\n- Feature importance\n- Interactive visualizations")

# Recent activity section
st.subheader("Recent Activity")

# Check for recent datasets
datasets_dir = Path("datasets")
recent_datasets = list(datasets_dir.glob("*.csv"))
if recent_datasets:
    st.write(f"📁 Recent datasets: {', '.join([ds.name for ds in recent_datasets[:3]])}")
else:
    st.write("📁 No datasets available. Upload a dataset to get started.")

# Check for recent models
models_dir = Path("models")
recent_models = list(models_dir.glob("*.pkl"))
if recent_models:
    st.write(f"💾 Recent models: {', '.join([m.name for m in recent_models[:3]])}")
else:
    st.write("💾 No models available. Train a model to see it here.")

# Footer
st.markdown("---")
st.markdown("**SageSense** - Developed with ❤️ using Streamlit")