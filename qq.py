import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap, io, base64


# Load the model
model = joblib.load('ET.pkl')

# Define the names of the prediction variables
# Define the names of the prediction variables in the desired order
feature_names = [
    "WBC", "Albumin", "ALT", "CR", "uWBC",
    "Surgical_Duration", "Stone_burden", "Double-J_stent_duration"
]

# Set up the Streamlit web interface
st.title("Sepsis Risk Predictor")

# Collect user input
WBC = st.number_input("White Blood Cell Count (WBC):", min_value=0, max_value=100, value=10)
Albumin = st.number_input("Albumin (g/dL):", min_value=1, max_value=100, value=4)
ALT = st.number_input("Alanine transaminase (ALT):", min_value=0, max_value=500, value=35)
CR = st.number_input("Creatinine (CR):", min_value=0, max_value=10000, value=90)
uWBC = st.number_input("Urinary WBC (uWBC):", min_value=0, max_value=10000, value=100)
Surgical_Duration = st.number_input("Surgical Duration (minutes):", min_value=0, max_value=600, value=90)
Stone_burden = st.number_input("Stone burden (π multiplied by the longest radius and width mm^2):", min_value=0, max_value=1000, value=50)
Double_J_stent_duration = st.number_input("Double-J stent duration (days):", min_value=0, max_value=1000, value=30)

# Convert the input features to an array for model processing
feature_values = [WBC, Albumin, ALT, CR, uWBC, Surgical_Duration, Stone_burden, Double_J_stent_duration]

features = np.array([feature_values])
df_features = pd.DataFrame([feature_values], columns=feature_names)

# Make predictions when the user clicks "Predict"
if st.button("Predict"):
    # Predict the class (sepsis or no sepsis)
    predicted_class = model.predict(features)[0]

    # Predict the probabilities
    predicted_proba = model.predict_proba(features)[0]

    # Display the prediction results
    st.write(f"**Predicted Class:** {'Sepsis' if predicted_class == 1 else 'No Sepsis'}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Provide advice based on the prediction
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of sepsis. "
            f"The model predicts that your probability of having sepsis is {probability:.1f}%. "
            "Please consult your doctor for further evaluation and potential treatments."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of sepsis. "
            f"The model predicts that your probability of not having sepsis is {probability:.1f}%. "
            "Keep monitoring your health and consult a doctor if you have any concerns."
        )
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(features)
    shap_plot = shap.force_plot(explainer.expected_value, shap_values[0].T[1], df_features.iloc[0, :], show=False, matplotlib=True)

    save_file = io.BytesIO()
    plt.savefig(save_file, format='png', bbox_inches="tight")
    save_file_base64 = base64.b64encode(save_file.getvalue()).decode('utf8')

    st.image(f"data:image/png;base64,{save_file_base64}")
    
    #shap.save_html("force_plot.html", shap_plot)
