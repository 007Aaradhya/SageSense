import streamlit as st
import pandas as pd
import os
import plotly.express as px
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Model Training & Prediction", layout="wide")
st.title("Model Training & Model Prediction")

# Ensure dataset selection persists across pages
if "selected_dataset" not in st.session_state:
    st.session_state.selected_dataset = None

if not st.session_state.selected_dataset:
    st.error("⚠️ No dataset selected! Please choose one on the landing page.")
    st.stop()

dataset_path = os.path.join("datasets", st.session_state.selected_dataset)
df = pd.read_csv(dataset_path)

st.success(f"✅ Loaded dataset: **{st.session_state.selected_dataset}**")

st.write("### 📜 Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# 🎯 Target Variable Selection
# -------------------------------
target = st.selectbox("🎯 Select Target Variable", df.columns)

# Detect if the target is categorical or continuous
if df[target].dtype == "object" or df[target].nunique() <= 10:
    problem_type = "classification"
    st.info("📌 Detected **Classification Problem** (Discrete Target Values)")
else:
    problem_type = "regression"
    st.info("📌 Detected **Regression Problem** (Continuous Target Values)")

# -------------------------------
# 🔢 Model Selection
# -------------------------------
if problem_type == "classification":
    model_choice = st.selectbox("🤖 Select a Classification Model", ["Random Forest", "Decision Tree", "Support Vector Machine"])
else:
    model_choice = st.selectbox("🤖 Select a Regression Model", ["Random Forest Regressor", "Decision Tree Regressor", "Linear Regression"])

# Define model path before training
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "latest_trained_model.pkl")

# -------------------------------
# 🛠️ Prepare Data (Define X Globally)
# -------------------------------
X = df.drop(columns=[target])
y = df[target]

# Handling non-numeric data
X = pd.get_dummies(X, drop_first=True)

# -------------------------------
# 🚀 Model Training
# -------------------------------
if st.button("🚀 Train Model"):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select Model Based on Problem Type
    if problem_type == "classification":
        if model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
        else:
            model = SVC(kernel="linear", probability=True, random_state=42)
    else:
        if model_choice == "Random Forest Regressor":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_choice == "Decision Tree Regressor":
            model = DecisionTreeRegressor(random_state=42)
        else:
            model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Model evaluation
    if problem_type == "classification":
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model trained with **{accuracy:.2%}** accuracy!")
    else:
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.success(f"✅ Model trained! **R² Score: {r2:.2f}** | **MAE: {mae:.2f}** | **MSE: {mse:.2f}")

    # Save model
    joblib.dump(model, model_path)
    st.success(f"✅ Model saved successfully!")

    # Feature Importance (Only for Tree-Based Models)
    if model_choice in ["Random Forest", "Decision Tree", "Random Forest Regressor", "Decision Tree Regressor"]:
        importance_df = pd.DataFrame({"Feature": X_train.columns, "Importance": model.feature_importances_})
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        fig = px.bar(importance_df, x="Importance", y="Feature", orientation="h", title="🔍 Feature Importance")
        st.plotly_chart(fig, use_container_width=True)

    # Download model
    with open(model_path, "rb") as f:
        st.download_button("📥 Download Trained Model", f, file_name="trained_model.pkl")

# -------------------------------
# 🔮 Model Prediction
# -------------------------------
st.write("### 🔍 Make Predictions")

# Load trained model
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success(f"✅ Loaded latest trained model")

    # Get user input for prediction
    st.write("📌 **Enter feature values for prediction:**")
    input_features = {}
    for feature in X.columns:
        input_features[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_features])

    if st.button("🔍 Predict"):
        prediction = model.predict(input_df)[0]
        st.write(f"🎯 **Predicted Value: {prediction}**")

else:
    st.warning("⚠️ No trained model found. Train a model first.")