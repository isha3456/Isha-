import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, accuracy_score

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="EcoAI Waste Segregation", page_icon="♻️", layout="wide")

st.title("♻️ EcoAI: Smart Waste Segregation System")
st.caption("AI-Based Waste Segregation and Recycling Recommendation System")

# -------------------------------
# Load Data & Preprocess
# -------------------------------
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("waste_segregation_dataset_10000_records.csv")

    df_clean = df.drop_duplicates().copy().ffill()

    # Drop identifiers and data leakage features ('Prediction' directly predicts Waste_Type)
    drop_cols = [c for c in ["Waste_ID", "Prediction"] if c in df_clean.columns]
    df_clean_features = df_clean.drop(columns=drop_cols)

    # Label Encoding for categorical features
    df_encoded = df_clean_features.copy()
    label_encoders = {}
    
    categorical_cols = df_encoded.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    return df_clean, df_encoded, label_encoders

df, df_encoded, label_encoders = load_and_preprocess_data()

# -------------------------------
# Model Training (Global Cache)
# -------------------------------
@st.cache_resource
def train_all_models(df_encoded):
    # 1. Linear Regression (Predict Weight from Sensor, Image Size, and Moisture)
    X_lr = df_encoded[["Sensor_Value", "Image_Size_KB", "Moisture_Level"]]
    y_lr = df_encoded["Weight_kg"]
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)
    
    lr = LinearRegression()
    lr.fit(X_train_lr, y_train_lr)
    lr_score = r2_score(y_test_lr, lr.predict(X_test_lr))

    # 2. Logistic Regression (Predict Waste_Type)
    X_clf = df_encoded.drop("Waste_Type", axis=1)
    y_clf = df_encoded["Waste_Type"]
    
    scaler = StandardScaler()
    X_clf_scaled = pd.DataFrame(scaler.fit_transform(X_clf), columns=X_clf.columns)
    
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf_scaled, y_clf, test_size=0.2, random_state=42)

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train_clf, y_train_clf)
    log_accuracy = accuracy_score(y_test_clf, log_model.predict(X_test_clf))

    return {
        "lr": lr, "lr_score": lr_score,
        "log_model": log_model, "log_accuracy": log_accuracy,
        "X_clf": X_clf, "scaler": scaler
    }

models = train_all_models(df_encoded)

# Helper function to render user inputs using human-readable text dropdowns
def render_user_input_form(key_prefix):
    user_inputs = {}
    col1, col2 = st.columns(2)

    feature_cols = [c for c in models["X_clf"].columns if c != "Waste_Type"]

    for idx, feature in enumerate(feature_cols):
        target_col = col1 if idx % 2 == 0 else col2
        
        if feature in label_encoders:
            options = list(df[feature].unique())
            selected_text = target_col.selectbox(f"Select {feature}", options=options, key=f"{key_prefix}_{feature}")
            user_inputs[feature] = label_encoders[feature].transform([selected_text])[0]
        else:
            default_val = float(df[feature].mean())
            user_inputs[feature] = target_col.number_input(f"Enter {feature}", value=round(default_val, 2), key=f"{key_prefix}_{feature}")

    return pd.DataFrame([user_inputs])

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("Navigation")
option = st.sidebar.selectbox(
    "Select Page",
    [
        "Home", 
        "Dataset", 
        "Data Cleaning", 
        "EDA", 
        "Linear Regression", 
        "Logistic Regression", 
        "K-Means", 
        "AI Recommendation", 
        "Model Comparison", 
        "Final Prediction"
    ]
)

# -------------------------------
# Page Implementations
# -------------------------------
if option == "Home":
    st.header("Project Objective")
    st.info("""
    This project uses Machine Learning to:
    * **Predict Waste Weight** based on physical properties and sensor measurements.
    * **Predict Waste Category** using Logistic Regression.
    * **Group Similar Waste** using K-Means clustering.
    * **Recommend Recycling Methods** based on waste properties.
    """)
    col1, col2 = st.columns(2)
    col1.metric("Total Records", len(df))
    col2.metric("Total Features", len(df.columns))

elif option == "Dataset":
    st.header("Dataset Overview")
    st.dataframe(df, use_container_width=True)

elif option == "Data Cleaning":
    st.header("🧹 Data Cleaning")
    st.success("Data Cleaning Completed Successfully!")
    st.dataframe(df_encoded.head(), use_container_width=True)

elif option == "EDA":
    st.header("📊 Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Waste Category Count")
        fig, ax = plt.subplots()
        df["Waste_Type"].value_counts().plot(kind="bar", ax=ax, color="#2ecc71")
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        st.subheader("Color Distribution")
        fig, ax = plt.subplots()
        df["Color"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

elif option == "Linear Regression":
    st.header("📈 Linear Regression")
    st.metric(label="R² Score", value=f"{models['lr_score']:.4f}")
    
    sensor_val = st.slider("Sensor Value", min_value=0, max_value=100, value=50)
    img_size = st.number_input("Image Size (KB)", min_value=100, max_value=10000, value=2000)
    moisture_str = st.selectbox("Moisture Level", options=list(df["Moisture_Level"].unique()))
    
    moisture_enc = label_encoders["Moisture_Level"].transform([moisture_str])[0]

    if st.button("Predict Weight"):
        input_data = pd.DataFrame([[sensor_val, img_size, moisture_enc]], columns=["Sensor_Value", "Image_Size_KB", "Moisture_Level"])
        result = models["lr"].predict(input_data)
        st.success(f"Predicted Weight: **{result[0]:.2f} kg**")

elif option == "Logistic Regression":
    st.header("🤖 Logistic Regression")
    st.metric(label="Accuracy", value=f"{models['log_accuracy']*100:.2f}%")

    st.subheader("Predict Waste Category")
    input_df = render_user_input_form("log")

    if st.button("Predict Category", key="log_button"):
        scaled_input = models["scaler"].transform(input_df)
        prediction = models["log_model"].predict(scaled_input)
        pred_label = label_encoders["Waste_Type"].inverse_transform(prediction)[0]
        st.success(f"Predicted Waste Type: **{pred_label}**")

elif option == "K-Means":
    st.header("🔵 K-Means Clustering")
    X = df_encoded[["Weight_kg", "Sensor_Value"]]
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    df_encoded["Cluster"] = kmeans.fit_predict(X)
    st.dataframe(df_encoded[["Weight_kg", "Sensor_Value", "Cluster"]].head(), use_container_width=True)

elif option == "AI Recommendation":
    st.header("♻️ AI Recycling Recommendation")
    selected_waste = st.selectbox("Select Waste Category", options=list(df["Waste_Type"].unique()))
    recommendations = {
        "Plastic": "Recycle into bottles, containers, or composite materials.",
        "Paper": "Send to paper mills for repulping.",
        "Glass": "Send to glass cullet facilities for melting.",
        "Metal": "Send to scrap facilities for smelting.",
        "Organic": "Divert to composting or anaerobic digestion facilities.",
        "Textile": "Send to textile shredding or upcycling plants.",
        "E-Waste": "Process at specialized electronic waste recovery facilities."
    }
    if st.button("Get Recommendation"):
        st.success(f"**Action Plan:** {recommendations.get(selected_waste, 'Process normally.')}")

elif option == "Model Comparison":
    st.header("🏆 Model Summary")
    summary = pd.DataFrame({
        "Model": ["Linear Regression", "Logistic Regression"],
        "Type": ["Regression (Weight)", "Classification (Waste Type)"],
        "Performance Metric": ["R² Score", "Accuracy Score"],
        "Value": [f"{models['lr_score']:.4f}", f"{models['log_accuracy'] * 100:.2f}%"]
    })
    st.dataframe(summary, use_container_width=True)

elif option == "Final Prediction":
    st.header("🔮 Final Prediction & Strategy")
    input_df = render_user_input_form("final")

    if st.button("Generate Strategy"):
        scaled_input = models["scaler"].transform(input_df)
        prediction = models["log_model"].predict(scaled_input)
        pred_label = label_encoders["Waste_Type"].inverse_transform(prediction)[0]
        
        routes = {
            "Plastic": "Recycle into commercial grade containers.",
            "Paper": "Process at regional paper pulping station.",
            "Glass": "Route to local glass recycling center.",
            "Metal": "Route to industrial scrap processing.",
            "Organic": "Direct to local municipal composting facility.",
            "Textile": "Route to textile fiber recovery facility.",
            "E-Waste": "Route to certified electronic waste recovery plant."
        }
        
        st.success(f"### Predicted Waste Category: **{pred_label}**")
        st.info(f"**Recommended Action:** {routes.get(pred_label, 'Process through standard municipal recycling.')}")
        st.balloons()