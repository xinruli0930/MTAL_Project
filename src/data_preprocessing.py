import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os

def load_ihdp_data(file_path):
    # Load IHDP dataset (assumes npz format)
    data = np.load(file_path)
    X = data['x']        # Covariates
    T = data['t']        # Treatment assignment
    y_factual = data['yf']   # Observed outcome
    y_cfactual = data['ycf'] # Counterfactual outcome
    
    return X, T, y_factual, y_cfactual

def preprocess_data(X):
    # Handle missing values by imputing with mean
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Normalize features to have zero mean and unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled

def create_final_dataframe(X, T, y_factual, y_cfactual):
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
    df['treatment'] = T
    df['y_factual'] = y_factual
    df['y_counterfactual'] = y_cfactual

    return df

def main():
    # Define path to your IHDP dataset file
    ihdp_data_file = "ihdp_npci_1-1000.train.npz"
    
    if not os.path.exists(ihdp_data_file):
        raise FileNotFoundError(f"The file {ihdp_data_file} does not exist.")

    # Load data
    X, T, y_factual, y_cfactual = load_ihdp_data(ihdp_data_file)
    
    # Preprocess features
    X_processed = preprocess_data(X)
    
    # Create final dataframe
    df_final = create_final_dataframe(X_processed, T, y_factual, y_cfactual)

    # Optionally save to CSV
    df_final.to_csv('ihdp_preprocessed.csv', index=False)

    print(df_final.head())

if __name__ == "__main__":
    main()
