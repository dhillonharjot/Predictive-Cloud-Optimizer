import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, LSTM, GRU, Conv1D, MaxPooling1D, 
                                     Bidirectional, Dropout, Flatten, Concatenate, 
                                     BatchNormalization, LayerNormalization, Add, 
                                     MultiHeadAttention, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm
import time
import traceback

SEQUENCE_LENGTH = 24
BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 0.001

def load_and_prepare_data(filepath):
    print("--- Loading and Preparing Data ---")
    start_time = time.time()
    

    if not os.path.exists(filepath):
        print(f"Warning: Data file not found at '{filepath}'")
        print("Generating synthetic data for demonstration purposes...")
        n_synthetic = 1000
        all_features = ['cpu_usage', 'memory_usage', 'network_traffic', 'hour', 'day', 'month', 'day_of_week']
        targets = ['cpu_usage', 'memory_usage', 'network_traffic']
        feature_dim = len(all_features)
        target_dim = len(targets)
        
        X = np.random.rand(n_synthetic, SEQUENCE_LENGTH, feature_dim)
        y = np.random.rand(n_synthetic, target_dim)
        
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        scaler_x.fit(np.random.rand(n_synthetic, feature_dim))
        scaler_y.fit(np.random.rand(n_synthetic, target_dim))

    else:
        print(f"Loading performance dataset from '{filepath}'...")
        df = pd.read_csv(filepath)
        print(f"Dataset loaded with shape: {df.shape}")

        print("Processing timestamps and features...")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['vm_id', 'timestamp'])
        
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        features = ['cpu_usage', 'memory_usage', 'network_traffic']
        temporal_features = ['hour', 'day', 'month', 'day_of_week']
        all_features = features + temporal_features
        targets = ['cpu_usage', 'memory_usage', 'network_traffic']

        print("Handling missing values...")
        for col in all_features + targets:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        print("Scaling features...")
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        df_scaled = df.copy()
        df_scaled[all_features] = scaler_x.fit_transform(df[all_features])
        df_scaled[targets] = scaler_y.fit_transform(df[targets])

        print("Creating sequences for time series modeling...")
        X, y = [], []
        vm_ids = df['vm_id'].unique()
        progress_bar = tqdm(total=len(vm_ids), desc="Processing VMs")
        
        for vm_id in vm_ids:
            vm_data = df_scaled[df_scaled['vm_id'] == vm_id]
            if len(vm_data) > SEQUENCE_LENGTH:
                for i in range(len(vm_data) - SEQUENCE_LENGTH):
                    X.append(vm_data[all_features].iloc[i:i+SEQUENCE_LENGTH].values)
                    y.append(vm_data[targets].iloc[i+SEQUENCE_LENGTH].values)
            progress_bar.update(1)
        progress_bar.close()

        X = np.array(X)
        y = np.array(y)

    print(f"Created {len(X)} sequences.")
    
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
    
    total_time = time.time() - start_time
    print(f"Data preparation completed in {total_time:.2f} seconds")
    return X_train, X_test, y_train, y_test, scaler_y



def build_lstm_cnn_hybrid(input_shape, output_shape):
    print("Building LSTM-CNN Hybrid model...")
    inputs = Input(shape=input_shape)
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = MaxPooling1D(pool_size=2)(conv1)
    lstm1 = LSTM(100, return_sequences=False)(conv1)
    dense1 = Dense(50, activation='relu')(lstm1)
    outputs = Dense(output_shape)(dense1)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mae'])
    return model

def build_temporal_convolutional_network(input_shape, output_shape):
    print("Building Temporal Convolutional Network model...")
    def residual_block(x, dilation_rate, filters):
        res = x
        conv = Conv1D(filters=filters, kernel_size=3, padding='causal', dilation_rate=dilation_rate, activation='relu')(x)
        conv = BatchNormalization()(conv)
        conv = Dropout(0.2)(conv)
        if res.shape[-1] != filters:
            res = Conv1D(filters=filters, kernel_size=1, padding='same')(res)
        return Add()([res, conv])

    inputs = Input(shape=input_shape)
    x = Conv1D(filters=32, kernel_size=3, padding='causal')(inputs)
    x = residual_block(x, dilation_rate=1, filters=32)
    x = residual_block(x, dilation_rate=2, filters=64)
    x = residual_block(x, dilation_rate=4, filters=128)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(output_shape)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mae'])
    return model

def build_bilstm_attention_network(input_shape, output_shape):
    print("Building BiLSTM-Attention Network model...")
    inputs = Input(shape=input_shape)
    bilstm1 = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    bilstm1 = LayerNormalization()(bilstm1)
    attention = MultiHeadAttention(num_heads=4, key_dim=16)(bilstm1, bilstm1)
    pooled = GlobalAveragePooling1D()(attention)
    dense = Dense(64, activation='relu')(pooled)
    dense = Dropout(0.3)(dense)
    outputs = Dense(output_shape)(dense)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mae'])
    return model

def build_deep_performance_net(input_shape, output_shape):
    print("Building DeepPerformanceNet model...")
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
    x = BatchNormalization()(x)
    skip = x
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    if skip.shape[-1] != x.shape[-1]:
        skip = Conv1D(filters=64, kernel_size=1, padding='same')(skip)
    x = Add()([x, skip])
    x = tf.keras.layers.Activation('relu')(x)
    gru = GRU(128)(x)
    dense1 = Dense(64, activation='relu')(gru)
    outputs = Dense(output_shape)(dense1)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mae'])
    return model


def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test):
    print(f"\n--- Training {model_name} ---")
    start_time = time.time()
    os.makedirs('models', exist_ok=True)
    
    print("Model summary:")
    model.summary(line_length=100)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001),
        ModelCheckpoint(f'models/{model_name}.keras', monitor='val_loss', save_best_only=True),
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.2f} seconds")
    
    print("\nEvaluating model on test data...")
    metrics = model.evaluate(X_test, y_test, verbose=0)
    print(f"-> {model_name} | Test Loss (MSE): {metrics[0]:.4f} | Test MAE: {metrics[1]:.4f}")
    
    print("Generating training history plots...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title(f'{model_name} MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'models/{model_name}_history.png')
    plt.close()
    
    print(f"Training history saved to 'models/{model_name}_history.png'")
    return history, metrics


def analyze_models(results, models):
    print("\n" + "="*80)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*80)
    analysis_dir = "model_analysis"
    os.makedirs(analysis_dir, exist_ok=True)
    
    model_names = list(results.keys())
    test_maes = [results[name]['test_mae'] for name in model_names]
    
    metrics_df = pd.DataFrame({
        'Model': model_names,
        'Test MAE': test_maes,
        'Convergence (epochs)': [np.argmin(results[name]['history'].history['val_loss']) + 1 for name in model_names],
        'Model Size (params)': [models[name].count_params() for name in model_names]
    })
    
    metrics_df = metrics_df.sort_values('Test MAE')
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(f"{analysis_dir}/model_metrics_comparison.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    plt.bar(metrics_df['Model'], metrics_df['Test MAE'], color='skyblue')
    plt.title('Model Comparison - Test MAE (lower is better)')
    plt.ylabel('Mean Absolute Error')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{analysis_dir}/model_comparison.png")
    plt.close()
    print(f"\nComparative visualization saved to {analysis_dir}/model_comparison.png")


def main():
    overall_start_time = time.time()
    print("="*80)
    print("VM CLOUD PERFORMANCE PREDICTION - DEEP LEARNING MODELS")
    print("="*80)


    performance_file = "data/vmcloud_performance.csv"
    
    X_train, X_test, y_train, y_test, scaler_y = load_and_prepare_data(performance_file)
    
    if X_train is None:
        print("Could not load or generate data. Exiting.")
        return

    input_shape = X_train.shape[1:]
    output_shape = y_train.shape[1]
    print(f"\nInput shape: {input_shape}, Output shape: {output_shape}\n")
    
    model_builders = {
        "LSTM_CNN_Hybrid": build_lstm_cnn_hybrid,
        "Temporal_Convolutional_Network": build_temporal_convolutional_network,
        "BiLSTM_Attention_Network": build_bilstm_attention_network,
        "DeepPerformanceNet": build_deep_performance_net
    }
    
    results = {}
    trained_models = {}
    
    for name, builder_func in model_builders.items():
        model = builder_func(input_shape, output_shape)
        trained_models[name] = model
        
        history, metrics = train_and_evaluate(model, name, X_train, X_test, y_train, y_test)
        
        results[name] = {
            'history': history,
            'test_loss': metrics[0],
            'test_mae': metrics[1]
        }
    
    analyze_models(results, trained_models)
    
    best_model_name = min(results, key=lambda x: results[x]['test_mae'])
    print("\n" + "="*80)
    print(f"BEST PERFORMING MODEL: {best_model_name}")
    print(f"  -> Test MAE: {results[best_model_name]['test_mae']:.4f}")
    print("="*80)
    
    overall_time = time.time() - overall_start_time
    print(f"\nTotal execution time: {overall_time / 60:.2f} minutes")

if __name__ == "__main__":
    try:
        main()
        print("\nAll models successfully trained and evaluated!")
    except Exception as e:
        print(f"\nAn error occurred during execution: {str(e)}")
        traceback.print_exc()