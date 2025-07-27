# --- src/data_preprocessing.py ---
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib # To save/load encoders

def load_data(file_path="data/adult.csv"):
    """Loads the adult income dataset."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Please ensure the file is in the correct directory.")
        return None

def clean_and_filter_data(df):
    """
    Performs initial data cleaning and filtering based on common sense rules.
    Assumes '?' in categorical columns and specific outliers for age, workclass, education.
    """
    if df is None:
        return None

    # Replace '?' with 'Others'
    df.replace('?', 'Others', inplace=True)

    # Age outlier handling
    df = df[(df['age'] <= 75) & (df['age'] >= 17)]

    # Remove irrelevant 'workclass' categories
    df = df[df['workclass'] != 'Without-pay']
    df = df[df['workclass'] != 'Never-worked']

    # Remove very low education categories
    df = df[df['education'] != '1st-4th']
    df = df[df['education'] != '5th-6th']
    df = df[df['education'] != 'Preschool']

    # Drop the 'education' column as 'education-num' is kept
    df.drop(columns=['education'], inplace=True)

    return df

def encode_features(df, save_encoders=False, encoders_path='models/'):
    """
    Encodes categorical features using LabelEncoder.
    Optionally saves the fitted encoders.
    Returns the processed DataFrame and a dictionary of fitted encoders.
    """
    if df is None:
        return None, {}

    df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
    
    # Identify categorical columns, excluding 'income' which is the target
    categorical_cols = df_copy.select_dtypes(include='object').columns.tolist()
    if 'income' in categorical_cols:
        categorical_cols.remove('income')

    fitted_encoders = {}
    for col in categorical_cols:
        encoder = LabelEncoder()
        df_copy[col] = encoder.fit_transform(df_copy[col])
        fitted_encoders[col] = encoder
        if save_encoders:
            joblib.dump(encoder, f"{encoders_path}{col}_encoder.pkl")
            print(f"Saved {col}_encoder.pkl to {encoders_path}")

    return df_copy, fitted_encoders

def scale_features(X, scaler=None, save_scaler=False, scaler_path='models/'):
    """
    Scales numerical features using MinMaxScaler.
    If scaler is None, fits a new one. Otherwise, uses the provided scaler.
    Optionally saves the fitted scaler.
    Returns the scaled DataFrame/array and the fitted scaler.
    """
    if scaler is None:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    if save_scaler:
        joblib.dump(scaler, f"{scaler_path}minmax_scaler.pkl")
        print(f"Saved minmax_scaler.pkl to {scaler_path}")
        
    return X_scaled, scaler