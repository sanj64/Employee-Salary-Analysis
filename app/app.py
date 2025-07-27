import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- Streamlit App Interface (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Salary Range Predictor", layout="centered")

# --- Configuration & File Paths ---
# Get the absolute path to the directory where this script (app.py) is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(CURRENT_DIR, '..', 'models')


# Now define your model, scaler, and encoder paths using MODELS_DIR
MODEL_PATH = os.path.join(MODELS_DIR, 'best_salary_predictor_xgb_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'minmax_scaler.pkl')
INCOME_ENCODER_PATH = os.path.join(MODELS_DIR, 'income_label_encoder.pkl')


# --- Load Pre-trained Models and Scaler ---
@st.cache_resource # Use st.cache_resource for heavy objects like models to load once
def load_assets():


    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        income_encoder = joblib.load(INCOME_ENCODER_PATH)

        # --- Load all INPUT LabelEncoders ---
        input_encoders = {}
        # List all categorical features that were encoded in your notebook
        categorical_input_features = [
            'workclass', 'marital.status', 'occupation', 'relationship',
            'race', 'sex', 'native.country'
        ]

        for feature in categorical_input_features:
            # Construct path for each individual encoder
            encoder_path = os.path.join(MODELS_DIR, f'{feature}_encoder.pkl')
            if os.path.exists(encoder_path):
                input_encoders[feature] = joblib.load(encoder_path)
            else:
                raise FileNotFoundError(f"Error: Missing {feature}_encoder.pkl at {encoder_path}. "
                                        "Please ensure you saved all input feature encoders from your notebook.")

        # --- Define the exact order of features ---
        feature_columns = ['age', 'workclass', 'fnlwgt', 'education.num', 'marital.status', 'occupation',
                           'relationship', 'race', 'sex', 'capital.gain', 'capital.loss',
                           'hours.per.week', 'native.country']


        return model, scaler, income_encoder, input_encoders, feature_columns
    except FileNotFoundError as e:
        st.error(f"Error loading model assets: {e}. Please ensure all .pkl files are in the 'models/' directory and paths are correct.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading assets: {e}")
        st.stop()

# Load all assets when the app starts
model, scaler, income_encoder, input_encoders, feature_columns = load_assets()

# --- Rest of your Streamlit App Interface ---
st.markdown("<h1 style='text-align: center; color: #1E90FF;'>Employee Salary Range Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predicting optimal employee salary ranges for fairness and benchmarking.</p>", unsafe_allow_html=True)

st.divider()

# --- Input Fields ---
st.header("Employee Details")

col1, col2 = st.columns(2)
with col1:
    age = st.slider('Age', 17, 90, 30)
    fnlwgt = st.number_input('Final Weight (fnlwgt)', min_value=10000, max_value=1500000, value=200000, help="The number of people the census believes the entry represents. This is a demographic weighting feature.")
    education_num = st.slider('Education Years (education.num)', 1, 16, 9, help="Numerical representation of education level. E.g., 9 for HS-grad, 13 for Bachelors.")
    hours_per_week = st.slider('Hours per Week', 1, 99, 40)
    capital_gain = st.number_input('Capital Gain', min_value=0, max_value=100000, value=0)
    capital_loss = st.number_input('Capital Loss', min_value=0, max_value=5000, value=0)

with col2:
    # Use loaded encoder's classes for selectbox options
    workclass = st.selectbox('Workclass', input_encoders['workclass'].classes_.tolist())
    # Ensure this matches the feature name in your dataset AND saved encoder file
    marital_status = st.selectbox('Marital Status', input_encoders['marital.status'].classes_.tolist())
    occupation = st.selectbox('Occupation', input_encoders['occupation'].classes_.tolist())
    relationship = st.selectbox('Relationship', input_encoders['relationship'].classes_.tolist())
    race = st.selectbox('Race', input_encoders['race'].classes_.tolist())
    sex = st.selectbox('Gender', input_encoders['sex'].classes_.tolist())
    # Ensure this matches the feature name in your dataset AND saved encoder file
    native_country = st.selectbox('Native Country', input_encoders['native.country'].classes_.tolist())


st.divider()

# --- Prediction Button ---
if st.button('Predict Salary Range', help="Click to predict the employee's salary range (<=50K or >50K)"):
    # Create a DataFrame from the user inputs, ensuring column order matches feature_columns
    input_df = pd.DataFrame([[
        age, workclass, fnlwgt, education_num, marital_status, occupation,
        relationship, race, sex, capital_gain, capital_loss,
        hours_per_week, native_country
    ]], columns=feature_columns)

    # --- Preprocessing the input ---
    # Apply LabelEncoding to categorical features using the *same fitted encoders* as during training
    for col, encoder in input_encoders.items():
        if col in input_df.columns: # Check if the column exists in input_df
            # Handle unseen categories: If a category is not in the encoder's classes,
            # it will raise an error. A robust solution needs a strategy (e.g., map to 0, or most frequent, or 'Other').
            # For now, we assume input values will be in known classes.
            input_df[col] = encoder.transform(input_df[col])

    # Ensure the order of columns matches the training data's X.columns
    input_df = input_df[feature_columns]

    # Apply MinMaxScaler
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction_encoded = model.predict(input_scaled)

    predicted_salary_range = income_encoder.inverse_transform(prediction_encoded)[0]

    st.markdown(f"## Predicted Salary Range:")
    if predicted_salary_range == '>50K':
        st.success(f"{predicted_salary_range}")
        st.balloons()
    else:
        st.info(f"{predicted_salary_range}")
    st.write("This prediction helps HR in compensation benchmarking and fairness analysis.")

st.divider()