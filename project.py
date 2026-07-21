import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# Title
st.title(" ♻ Smart Garbage Waste Segregation System")

# Load Dataset
df = pd.read_csv("Garbage_Waste_Segregation_Dataset_1000.csv")

# Display Dataset
st.subheader("Dataset")
st.dataframe(df)

# Shape
st.subheader("Dataset Shape")
st.write("Rows:", df.shape[0])
st.write("Columns:", df.shape[1])

# Column Names
st.subheader("Columns")
st.write(df.columns)

# Missing Values
st.subheader("Missing Values")
st.write(df.isnull().sum())

# Duplicate Records
st.subheader("Duplicate Records")
st.write(df.duplicated().sum())

# Data Types
st.subheader("Data Types")
st.write(df.dtypes)

# Label Encoding
le = LabelEncoder()

text_columns = df.select_dtypes(include="object").columns

for col in text_columns:
    df[col] = le.fit_transform(df[col])

st.subheader("Encoded Dataset")
st.dataframe(df.head())

# Feature Selection
X = df.drop("Category", axis=1)
y = df["Category"]

st.subheader("Input Features")
st.write(X.head())

st.subheader("Target")
st.write(y.head())

st.subheader("Exploratory Data Analysis")

# Bar Chart
st.write("### Waste Category Count")
fig, ax = plt.subplots()
df["Category"].value_counts().plot(kind="bar", ax=ax)
st.pyplot(fig)

# Histogram
st.write("### Weight Distribution")
fig, ax = plt.subplots()
ax.hist(df["Weight_kg"], bins=10)
ax.set_xlabel("Weight")
ax.set_ylabel("Count")
st.pyplot(fig)

# Scatter Plot
st.write("### Weight vs Moisture")
fig, ax = plt.subplots()
ax.scatter(df["Weight_kg"], df["Moisture_%"])
ax.set_xlabel("Weight")
ax.set_ylabel("Moisture")
st.pyplot(fig)

#LinearRegression
st.subheader("Linear Regression")

# Features and Target
X = df[["Moisture_%"]]
y = df["Weight_kg"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# R2 Score
score = r2_score(y_test, y_pred)

st.write("R² Score :", round(score, 2))

# User Prediction
st.subheader("Predict Weight")

moisture3 = st.number_input("Enter Moisture (%)", min_value=0.0, key="lr_moisture")

if st.button("Predict Weight"):
    weight = model.predict([[moisture3]])
    st.success(f"Predicted Weight: {weight[0]:.2f} kg")

#LogisticRegression

st.subheader("Logistic Regression")

# Features and Target
X = df.drop("Category", axis=1)
y = df["Category"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Model
lr_model = LogisticRegression(max_iter=1000)

# Train Model
lr_model.fit(X_train, y_train)

# Prediction
y_pred = lr_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

st.write("Accuracy :", round(accuracy * 100, 2), "%")

# User Prediction
st.subheader("Predict Waste Category")

weight1 = st.number_input("Enter Weight (kg)", 0.0,key="log_weight")
moisture1 = st.number_input("Enter Moisture (%)", 0.0, key="log_moisture")

if st.button("Predict using Logistic Regression"):
    sample = [[1, 0, 0, weight1, moisture1, 1, 1, 0]]
    result = lr_model.predict(sample)
    st.success(f"Predicted Category : {result[0]}")

#RandomForestClassifier

st.subheader("Random Forest Classifier")

# Features and Target
X = df.drop("Category", axis=1)
y = df["Category"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Prediction
y_pred = rf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

st.write("Accuracy Score :", round(accuracy * 100, 2), "%")

# User Prediction
st.subheader("Predict Waste Category")

weight2 = st.number_input("Weight (kg)", 0.0, key="rf_weight")
moisture2 = st.number_input("Moisture (%)", 0.0, key="rf_moisture")

if st.button("Predict Category"):
    sample = [[1, 0, 0, weight2, moisture2, 1, 1, 0]]
    category = rf.predict(sample)
    st.success(f"Predicted Category : {category[0]}") 

#K-Means Cluster

st.subheader("K-Means Clustering")

# Select Features
X_cluster = df[["Weight_kg", "Moisture_%"]]

# Create K-Means Model
kmeans = KMeans(n_clusters=3, random_state=42)

# Train Model
df["Cluster"] = kmeans.fit_predict(X_cluster)

# Display Dataset with Cluster
st.write(df[["Weight_kg", "Moisture_%", "Cluster"]].head())

# Scatter Plot
fig, ax = plt.subplots()

ax.scatter(
    df["Weight_kg"],
    df["Moisture_%"],
    c=df["Cluster"]
)

ax.set_xlabel("Weight (kg)")
ax.set_ylabel("Moisture (%)")

st.pyplot(fig)

# Cluster Centers
st.subheader("Cluster Centers")

st.write(kmeans.cluster_centers_) 

#Model Comparison and Final Prediction
st.subheader("Model Comparison")

# Create Comparison Table
comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "Performance": [round(score, 2), round(accuracy * 100, 2)],
    "Metric": ["R² Score", "Accuracy (%)"]
})

st.dataframe(comparison)

# Best Model
if accuracy * 100 > score * 100:
    best = "Random Forest"
else:
    best = "Linear Regression"

st.success(f"Best Model : {best}")