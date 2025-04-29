import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from utils import load_model

def show():
    st.title("🔮 Make Predictions")
    
    # Check for available models
    models_dir = Path("models")
    available_models = list(models_dir.glob("*.pkl"))
    
    if not available_models:
        st.warning("No trained models available. Please train a model first.")
        return
    
    # Model selection
    selected_model = st.selectbox(
        "Select Model",
        options=[f.name for f in available_models],
        format_func=lambda x: x.replace(".pkl", "").replace("_", " ").title()
    )
    
    if selected_model:
        try:
            # Load the model
            model = load_model(selected_model.replace(".pkl", ""))
            
            # Prediction options
            st.subheader("Prediction Options")
            prediction_type = st.radio(
                "Choose prediction input method",
                ["Upload CSV", "Manual Input"]
            )
            
            if prediction_type == "Upload CSV":
                uploaded_file = st.file_uploader("Upload CSV for prediction", type=["csv"])
                if uploaded_file:
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.dataframe(df.head())
                        
                        if st.button("Predict"):
                            predictions = model.predict(df)
                            df['Prediction'] = predictions
                            
                            st.subheader("Prediction Results")
                            st.dataframe(df)
                            
                            # Download results
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "Download Predictions",
                                csv,
                                "predictions.csv",
                                "text/csv"
                            )
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
            
            else:  # Manual Input
                st.subheader("Enter Feature Values")
                # Create input fields based on expected features
                # This would need to be adapted based on your model's requirements
                input_data = {}
                for i in range(5):  # Example for 5 features
                    input_data[f"feature_{i}"] = st.number_input(f"Feature {i}")
                
                if st.button("Predict"):
                    try:
                        # Convert input to DataFrame
                        input_df = pd.DataFrame([input_data])
                        prediction = model.predict(input_df)
                        st.success(f"Prediction: {prediction[0]}")
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

if __name__ == "__main__":
    show()