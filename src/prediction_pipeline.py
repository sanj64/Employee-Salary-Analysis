# --- src/prediction_pipeline.py ---
import pandas as pd
import joblib
import os
import numpy as np

def load_prediction_assets(model_path, scaler_path, income_encoder_path, input_encoders_dir='models/'):
    """
    Loads the trained model, scaler, income label encoder,
    and all input feature label encoders.
    """
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        income_encoder = joblib.load(income_encoder_path)

        # Load all input feature encoders dynamically
        input_encoders = {}
        # List of your categorical features that were encoded (from your notebook's X.columns)
        categorical_features = ['workclass', 'marital-status', 'occupation',
                                'relationship', 'race', 'sex', 'native-country']

        for feature in categorical_features:
            encoder_file = os.path.join(input_encoders_dir, f"{feature}_encoder.pkl")
            if os.path.exists(encoder_file):
                input_encoders[feature] = joblib.load(encoder_file)
            else:
                raise FileNotFoundError(f"Missing encoder for {feature} at {encoder_file}. Please ensure all input feature encoders are saved.")

        return model, scaler, income_encoder, input_encoders
    except Exception as e:
        print(f"Error loading prediction assets: {e}")
        return None, None, None, None

def preprocess_new_data(input_df, scaler, input_encoders, feature_columns):
    """
    Preprocesses new raw input data using loaded encoders and scaler.
    Ensures column order matches training data.
    """
    df_processed = input_df.copy()

    # Apply LabelEncoding to categorical features
    for col, encoder in input_encoders.items():
        if col in df_processed.columns:
            # Handle unseen categories: map to a known class or a default (e.g., 'Others' if you had one)
            # This requires careful handling. For simplicity here, assume input values are in encoder's classes_.
            # A more robust approach might be to use a dictionary mapping if 'Others' was handled during training.
            try:
                df_processed[col] = encoder.transform(df_processed[col])
            except ValueError as e:
                print(f"Warning: Category in '{col}' not seen during training. {e}")
                # You might want to map unseen categories to a specific value (e.g., 0)
                # or raise a more specific error. This depends on your original preprocessing strategy.
                df_processed[col] = df_processed[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else 0)


    # Ensure the order of columns matches the training data's X.columns
    # This is CRITICAL for model prediction correctness
    df_processed = df_processed[feature_columns]

    # Apply MinMaxScaler
    input_scaled = scaler.transform(df_processed)

    return input_scaled

def predict_salary_range(model, processed_input, income_encoder):
    """
    Makes a prediction using the trained model and decodes the result.
    """
    prediction_encoded = model.predict(processed_input)
    predicted_label = income_encoder.inverse_transform(prediction_encoded)[0]
    return predicted_label