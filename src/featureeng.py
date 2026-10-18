import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import os

def feature_eng(data):
    max_cycles = data.groupby("Engine")["Cycles"].max()


    data = data.merge(max_cycles, on="Engine")
    #print(data)

    data["RUL"]=data["Cycles_y"] - data["Cycles_x"]
    data["RUL_Capped"]=data["RUL"].clip(upper=125)
    data.drop(["Sensor_1", "Sensor_5", "Sensor_6", "Sensor_10", "Sensor_16", "Sensor_18", "Sensor_19", "RUL"], axis=1, inplace=True)



    feature_columns=["OP_Setting 1", "OP_Setting 2", "OP_Setting 3", "Sensor_2", "Sensor_3", "Sensor_4", "Sensor_7", "Sensor_8", "Sensor_9", "Sensor_11", "Sensor_12", "Sensor_13", "Sensor_14", "Sensor_15", "Sensor_17"]
    scaler = MinMaxScaler()
    data[feature_columns] = scaler.fit_transform(data[feature_columns])

    joblib.dump(scaler, "scaler.save")
    print("Absolute path of scaler.save:", os.path.abspath("scaler.save"))

    return data, feature_columns
