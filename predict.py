import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import argparse


MODEL_TO_USE = 'models/BiLSTM_Attention_Network.keras' 

FULL_DATA_FILE = 'data/vmcloud_performance.csv'

SEQUENCE_LENGTH = 24
MINIMUM_AVG_CPU_USAGE = 15.0 

def run_prediction(input_file_path=None):
    """
    A self-contained function that loads data and models, prepares a consistent
    input sample, and makes a prediction.
    
    Args:
        input_file_path (str, optional): Path to a user-provided 24-hour CSV. 
                                         If None, a sample is found automatically.
    """
    print(f"Loading full dataset from: {FULL_DATA_FILE}")
    try:
        df_full = pd.read_csv(FULL_DATA_FILE, parse_dates=['timestamp'])
    except FileNotFoundError:
        print(f"ERROR: The full dataset '{FULL_DATA_FILE}' was not found.")
        return
    print("Dataset loaded successfully.")

    print("Creating and fitting scalers based on the full dataset...")
    features = ['cpu_usage', 'memory_usage', 'network_traffic']
    temporal_features = ['hour', 'day', 'month', 'day_of_week']
    all_features = features + temporal_features

    df_full['hour'] = df_full['timestamp'].dt.hour
    df_full['day'] = df_full['timestamp'].dt.day
    df_full['month'] = df_full['timestamp'].dt.month
    df_full['day_of_week'] = df_full['timestamp'].dt.dayofweek
    
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_x.fit(df_full[all_features])
    scaler_y.fit(df_full[features])
    print("Scalers fitted successfully.")


    if input_file_path:
        print(f"Loading user-provided input sequence from: {input_file_path}")
        try:
            input_df = pd.read_csv(input_file_path, parse_dates=['timestamp'])
            if len(input_df) != SEQUENCE_LENGTH:
                print(f"Error: Provided file must have exactly {SEQUENCE_LENGTH} rows, but it has {len(input_df)}.")
                return
        except FileNotFoundError:
            print(f"Error: The file '{input_file_path}' was not found.")
            return
    else:
        print("Searching for a suitable, active input sequence...")
        vm_counts = df_full['vm_id'].value_counts()
        eligible_vms = vm_counts[vm_counts >= SEQUENCE_LENGTH].index
        
        input_df = None
        for vm_id in eligible_vms:
            vm_data = df_full[df_full['vm_id'] == vm_id]
            for i in range(len(vm_data) - SEQUENCE_LENGTH + 1):
                window = vm_data.iloc[i:i+SEQUENCE_LENGTH]
                if window['cpu_usage'].mean() >= MINIMUM_AVG_CPU_USAGE:
                    input_df = window
                    print(f"Found an active sample from VM ID: {vm_id}")
                    break
            if input_df is not None:
                break

        if input_df is None:
            raise RuntimeError("Could not find any 24-hour window with sufficient activity.")

    print("Preparing input sequence for the model...")
    input_df['hour'] = input_df['timestamp'].dt.hour
    input_df['day'] = input_df['timestamp'].dt.day
    input_df['month'] = input_df['timestamp'].dt.month
    input_df['day_of_week'] = input_df['timestamp'].dt.dayofweek
    
    scaled_sequence = scaler_x.transform(input_df[all_features])
    input_for_model = np.expand_dims(scaled_sequence, axis=0)
    print("Input sequence prepared successfully.")

    print(f"\nLoading model: {MODEL_TO_USE}")
    model = tf.keras.models.load_model(MODEL_TO_USE)
    
    print("Making prediction...")
    scaled_prediction = model.predict(input_for_model)
    actual_prediction = scaler_y.inverse_transform(scaled_prediction)

    print("\n--- FORECAST FOR THE NEXT HOUR ---")
    print(f"Predicted CPU Usage:     {actual_prediction[0][0]:.2f} %")
    print(f"Predicted Memory Usage:  {actual_prediction[0][1]:.2f} %")
    print(f"Predicted Network Traffic: {actual_prediction[0][2]:.2f} KB/s")
    print("----------------------------------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict VM resource usage for the next hour.")
    parser.add_argument('--input_file', type=str, help="Optional: Path to a CSV file with 24 hours of data for prediction.")
    args = parser.parse_args()

    if not os.path.exists(MODEL_TO_USE) or not os.path.exists(FULL_DATA_FILE):
        print("Error: Ensure the model file and the full data file ('vmcloud_performance.csv') exist.")
    else:               
        run_prediction(input_file_path=args.input_file)
