'''
This script reads the FD001 dataset, performs feature engineering, generates sequences for time series prediction, 
and trains multiple machine learning and deep learning models to predict the Remaining Useful Life (RUL) of engines. 
The models include Random Forest, XGBoost, LSTM, GRU, and 1D CNN. The performance of each model is evaluated using 
RMSE, MAE, R2 score, and a custom NASA scoring function. The results are logged using MLflow for experiment tracking 
and comparison.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import joblib
import os
from pathlib import Path 
from featureeng import feature_eng
from train import train_and_log

data_read=pd.read_csv("E:/P2/Python/TurbofanEngine/git/data/FD001.csv")
data, feature_columns = feature_eng(data_read)



window_size = 30
X_list, y_list = [], []

def gen_seq(group, window_size, feature_columns, rul_capped_col):
    data_matrix = group[feature_columns].values
    target_matrix = group[rul_capped_col].values
    num_elements = data_matrix.shape[0]
    X, y = [], []

    for start in range(num_elements - window_size + 1):
        end = start + window_size
        X.append(data_matrix[start:end])
        y.append(target_matrix[end - 1])
    return np.array(X) , np.array(y)
    


for eng_id,group in data.groupby("Engine"):
    if len(group)>=window_size:
        X_seq, y_seq = gen_seq(group, window_size, feature_columns, "RUL_Capped")
        X_list.append(X_seq)
        y_list.append(y_seq)

#print(X_list) 
X = np.concatenate(X_list)
y = np.concatenate(y_list)

print("Preprocessed Training Data Shapes:")
print("X (Samples, Window Size, Features):", X.shape)
print("y (Samples,):", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_samples_tr, window_size, num_features = X_train.shape
X_train_flat = X_train.reshape(num_samples_tr, window_size * num_features)

num_samples_val = X_test.shape[0]
X_test_flat = X_test.reshape(num_samples_val, window_size * num_features)


# i=12
# while i<=12:

#     db_path = Path("E:/P2/Python/TurbofanEngine/git/mlflow.db").resolve().as_uri()
#     # This converts the path to: sqlite:///E:/P2/Python/TurbofanEngine/mlflow.db under the hood correctly or file:/// equivalent for MLflow backend config
#     mlflow.set_tracking_uri(f"sqlite:///{db_path.split(':///')[1]}")
#     mlflow.create_experiment(f"cmapss_turbofan_experiment_{i}", artifact_location="E:/P2/Python/TurbofanEngine/git/mlruns")
#     mlflow.set_experiment(f"cmapss_turbofan_experiment_{i}")

#     rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
#     train_and_log(rf, "RandomForest", X_train_flat, y_train, X_test_flat, y_test, {"model_type": "RF"}, is_dl=False)

#     xg_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
#     train_and_log(xg_model, "XGBoost", X_train_flat, y_train, X_test_flat, y_test, {"model_type": "XGB"}, is_dl=False)

#     lstm = Sequential([
#         LSTM(64, input_shape=(window_size, num_features), return_sequences=False),
#         Dropout(0.2),
#         Dense(32, activation='relu'),
#         Dense(1)
#     ])
#     lstm.compile(optimizer='adam', loss='mse')
#     train_and_log(lstm, "LSTM", X_train, y_train, X_test, y_test, {"epochs": 15, "batch_size": 64}, is_dl=True)


#     gru = Sequential([
#         GRU(64, input_shape=(window_size, num_features), return_sequences=False),
#         Dropout(0.2),
#         Dense(32, activation='relu'),
#         Dense(1)
#     ])
#     gru.compile(optimizer='adam', loss='mse')
#     train_and_log(gru, "GRU", X_train, y_train, X_test, y_test, {"epochs": 15, "batch_size": 64}, is_dl=True)


#     cnn = Sequential([
#         Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(window_size, num_features)),
#         MaxPooling1D(pool_size=2),
#         Conv1D(filters=32, kernel_size=3, activation='relu'),
#         Flatten(),
#         Dense(32, activation='relu'),
#         Dense(1)
#     ])
#     cnn.compile(optimizer='adam', loss='mse')
#     train_and_log(cnn, "CNN_1D", X_train, y_train, X_test, y_test, {"epochs": 15, "batch_size": 64}, is_dl=True)

#     i=i+1