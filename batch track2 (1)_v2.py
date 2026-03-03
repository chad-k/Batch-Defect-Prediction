# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 14:51:20 2025

@author: ckaln
"""


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Title
st.title("Batch Defect Prediction Dashboard")

# Generate synthetic data
np.random.seed(42)
num_batches = 100
df = pd.DataFrame({
    "BatchID": [f"B{str(i).zfill(3)}" for i in range(num_batches)],
    "MaterialLot": np.random.choice(['MLOT1', 'MLOT2', 'MLOT3', 'MLOT4'], size=num_batches),
    "Machine": np.random.choice(['M1', 'M2', 'M3'], size=num_batches),
    "Operator": np.random.choice(['O1', 'O2', 'O3', 'O4'], size=num_batches),
    "Shift": np.random.choice(['Day', 'Night'], size=num_batches),
    "Temperature": np.random.normal(loc=200, scale=10, size=num_batches).round(2),
    "Pressure": np.random.normal(loc=5, scale=1.0, size=num_batches).round(2),
    "Defective": np.random.choice([0, 1], size=num_batches, p=[0.85, 0.15])
})

# Encode data
df_encoded = pd.get_dummies(df.drop(columns=["BatchID"]), drop_first=True)
X = df_encoded.drop(columns=["Defective"])
y = df_encoded["Defective"]

# Train/test split and model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Add predictions to original dataframe
risk_scores = model.predict_proba(X)[:, 1] * 100
df["Defect Risk (%)"] = risk_scores.round(2)

# Sidebar to select batch
selected_batch = st.sidebar.selectbox("Select Batch ID", df["BatchID"].unique())

# Display selected batch info
selected_info = df[df["BatchID"] == selected_batch]
st.subheader(f"Details for {selected_batch}")
st.write(selected_info.drop(columns=["Defective"]))

# Display defect risk
risk = selected_info["Defect Risk (%)"].values[0]
st.metric(label="Predicted Defect Risk", value=f"{risk:.2f} %", delta=None)

# Show table and chart
st.subheader("All Batch Risk Scores")
st.dataframe(df[["BatchID", "Defect Risk (%)"]].sort_values(by="Defect Risk (%)", ascending=False))

st.line_chart(df.set_index("BatchID")["Defect Risk (%)"])
